#!/bin/bash
# Shared setup for every BarkNet-AMIL job on Rorqual. Sourced, never executed directly.
#
# Contract (set these before sourcing, or accept the defaults):
#   REPO_DIR      path to the git checkout            (default $HOME/BarkNet_ML)
#   CONFIG        base config yaml                    (default $REPO_DIR/configs/config_cluster.yaml)
#   PATCH_SIZE    224 | 288 | 384                     (no default)
#   PATCH_TAR     tarball of the cut patches          (default $SCRATCH/data/barknet_patches_${PATCH_SIZE}.tar)
#   HF_HOME       huggingface cache with the weights  (default $HOME/.cache/huggingface)
#
# Exports: PATCH_ROOT, NUM_WORKERS, PY (the venv python), and the offline env vars.

# ---------------------------------------------------------------- fail fast --
: "${PATCH_SIZE:?set PATCH_SIZE (224|288|384)}"
REPO_DIR="${REPO_DIR:-$HOME/BarkNet_ML}"
CONFIG="${CONFIG:-$REPO_DIR/config/config_cluster.yaml}"
PATCH_TAR="${PATCH_TAR:-$SCRATCH/data/barknet_patches_${PATCH_SIZE}.tar}"

[ -d "$REPO_DIR" ] || { echo "REPO_DIR does not exist: $REPO_DIR"; exit 1; }
[ -f "$CONFIG" ]   || { echo "CONFIG does not exist: $CONFIG"; exit 1; }
[ -f "$PATCH_TAR" ] || { echo "PATCH_TAR does not exist: $PATCH_TAR"; exit 1; }

echo "=============================================================="
echo " job     : ${SLURM_JOB_ID:-interactive}  on $(hostname)"
echo " repo    : $REPO_DIR"
echo " config  : $CONFIG"
echo " patches : $PATCH_TAR"
echo " started : $(date)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ------------------------------------------------------------ environment ----
# OpenCV comes from the MODULE, not from pip: the `opencv_python` wheel in the
# Alliance wheelhouse is a placeholder stub that installs nothing usable.
module load python/3.11 opencv/4.13.0
module list 2>&1 | head -20

# Compute nodes have NO internet. timm must resolve ImageNet weights from the cache
# that 00_prefetch_weights.sh warmed on the login node.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# One thread per process: DataLoader workers are processes, and letting OpenCV/OpenMP
# each spawn their own pool inside every worker oversubscribes the cores SLURM gave us.
export OMP_NUM_THREADS=1
export OPENCV_NUM_THREADS=1
export MKL_NUM_THREADS=1

CPUS="${SLURM_CPUS_PER_TASK:-4}"
NUM_WORKERS=$(( CPUS > 1 ? CPUS - 1 : 0 ))   # leave one core for the main process
export NUM_WORKERS

# ------------------------------------------------------------------- venv ----
echo "[$(date +%T)] building venv in \$SLURM_TMPDIR ..."
virtualenv --no-download "$SLURM_TMPDIR/venv" >/dev/null
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --quiet --upgrade pip
pip install --no-index --quiet torch torchvision timm pyyaml pandas scikit-learn openpyxl tqdm
PY="$SLURM_TMPDIR/venv/bin/python"
export PY

# Fail now, not after 20 minutes of data staging.
$PY - <<'PYCHK'
import sys
import torch, timm, cv2, yaml, pandas, sklearn, openpyxl  # noqa: F401
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | timm {timm.__version__} "
      f"| cv2 {cv2.__version__} | cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("CUDA not visible -- did the job actually get a GPU?")
PYCHK

# Offline weight check for the size we are about to train. Cheap, and the failure mode it
# guards against is a 3-hour queue wait followed by an instant crash on a compute node.
#
# Two supported ways to have ImageNet weights available without internet:
#   (a) HF cache warmed on the login node  -> nothing more to do (default)
#   (b) PRETRAINED_FILE=/path/to/weights   -> passed through to model.pretrained_file
if [ -n "${PRETRAINED_FILE:-}" ]; then
  [ -f "$PRETRAINED_FILE" ] || { echo "PRETRAINED_FILE not found: $PRETRAINED_FILE"; exit 1; }
  MODEL_SIZE_CHECK="${MODEL_SIZE:-pico}" WEIGHT_FILE="$PRETRAINED_FILE" $PY - <<'PYCHK'
import os, timm
tag = f"convnextv2_{os.environ['MODEL_SIZE_CHECK']}.fcmae_ft_in1k"
timm.create_model(tag, pretrained=True, num_classes=0,
                  pretrained_cfg_overlay=dict(file=os.environ["WEIGHT_FILE"]))
print(f"ImageNet weights OK from local file: {tag} <- {os.environ['WEIGHT_FILE']}")
PYCHK
else
  MODEL_SIZE_CHECK="${MODEL_SIZE:-pico}" $PY - <<'PYCHK'
import os, timm
tag = f"convnextv2_{os.environ['MODEL_SIZE_CHECK']}.fcmae_ft_in1k"
timm.create_model(tag, pretrained=True, num_classes=0)
print(f"offline ImageNet weights OK from the HF cache: {tag}")
PYCHK
fi

# ------------------------------------------------------------- stage data ----
echo "[$(date +%T)] extracting $PATCH_TAR -> \$SLURM_TMPDIR ..."
mkdir -p "$SLURM_TMPDIR/patches"
time tar -xf "$PATCH_TAR" -C "$SLURM_TMPDIR/patches"

# The tar's internal layout is not guaranteed (patches/, patches/train/, ...). Find a
# species directory and take its parent -- that is the patch_root the loader wants.
# Species-agnostic: find any patch file, and the patch_root is its grandparent
# (patch_root/<SPECIES>/<file>.jpg).
FIRST_JPG=$(find "$SLURM_TMPDIR/patches" -name '*.jpg' -print -quit)
[ -n "$FIRST_JPG" ] || { echo "No .jpg patches found inside $PATCH_TAR"; exit 1; }
PATCH_ROOT=$(dirname "$(dirname "$FIRST_JPG")")
export PATCH_ROOT

N_SPECIES=$(find "$PATCH_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l)
[ "$N_SPECIES" -ge 15 ] || { echo "Only $N_SPECIES species dirs under $PATCH_ROOT -- wrong layer?"; exit 1; }

echo "[$(date +%T)] patch_root = $PATCH_ROOT"
echo "    $N_SPECIES species dirs | $(du -sh "$PATCH_ROOT" | cut -f1) on node-local SSD"
echo "    workers = $NUM_WORKERS (of $CPUS cpus)"

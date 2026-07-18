#!/bin/bash
# PARALLEL PATCH CUTTING -- one CPU array task per patch size.
#
# Cuts each size into node-local SSD ($SLURM_TMPDIR), then tars the result back to
# $SCRATCH/data/barknet_patches_<N>.tar with the SAME internal layout the training jobs
# expect: patches_<N>/{train,test}/<species>/*.jpg  (common.sh finds the first .jpg and
# takes its grandparent as PATCH_ROOT, so the patches_<N>/train/ nesting is required).
#
# Edit PATCH_SIZES below, then:
#   sbatch cluster/job_cut_patches.sh
# The array range is set to match the number of sizes. If you change how many sizes are in
# PATCH_SIZES, update --array to 0-(count-1).
#
# CPU job on def-oberman_cpu -- cutting is CPU-bound and the CPU partition queues far
# faster than GPU. Do NOT waste a GPU allocation on this.
#SBATCH --account=def-oberman_cpu
#SBATCH --job-name=bark_cut
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out

set -euo pipefail
mkdir -p "$SCRATCH/logs"

# ---- EDIT THIS LIST. --array above must be 0-(N-1) where N = number of sizes. ----
PATCH_SIZES=(96 160 512)
# ----------------------------------------------------------------------------------
PS=${PATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

REPO_DIR="$HOME/BarkNet_ML"
RAW_SRC="$SCRATCH/data/barknet_raw"          # the 23 species dirs live here
OUT_TAR="$SCRATCH/data/barknet_patches_${PS}.tar"

# Refuse to clobber an existing tar (224/288/384 already exist). Remove by hand to recut.
if [ -f "$OUT_TAR" ]; then
  echo "$OUT_TAR already exists -- skipping patch $PS. Remove it by hand to recut."
  exit 0
fi

# Locate the real cut_patches.py rather than assuming a path (past runs broke on a wrong
# hard-coded path). Prefer src/, fall back to a find.
CUTTER="$REPO_DIR/src/cut_patches.py"
[ -f "$CUTTER" ] || CUTTER=$(find "$REPO_DIR" -name cut_patches.py -print -quit)
[ -n "$CUTTER" ] && [ -f "$CUTTER" ] || { echo "cut_patches.py not found under $REPO_DIR"; exit 1; }
echo "cutter: $CUTTER"

# ---- environment: opencv MODULE (not a wheel) ----
# cut_patches.py imports only cv2 (from the module) and PIL (Pillow). cv2 comes from the
# opencv module. If Pillow isn't in the module stack, build a tiny --no-download venv for
# it from the Alliance wheelhouse (Pillow-SIMD is pulled -- faster JPEG encode, a bonus
# here). The venv inherits the module's cv2 via system-site-packages.
module load StdEnv/2023 python/3.11 opencv/4.13.0

if ! python -c "import PIL" 2>/dev/null; then
  echo "Pillow not in module stack -- building a minimal venv for it"
  virtualenv --no-download --system-site-packages "$SLURM_TMPDIR/cutvenv"
  source "$SLURM_TMPDIR/cutvenv/bin/activate"
  pip install --no-index --upgrade pip >/dev/null
  pip install --no-index Pillow >/dev/null
fi
# sanity: both deps importable before we spend an hour cutting
python -c "import cv2, PIL; print('cv2', cv2.__version__, '| PIL ok')"

# THE #1 SLURM CUTTING BUG: cut_patches --workers defaults to os.cpu_count(), which on a
# shared node reports the WHOLE node's cores, oversubscribing the process pool. Pass the
# job's real allocation, and pin OpenCV/OMP to 1 thread so cv2 threads don't fight the pool.
export OPENCV_NUM_THREADS=1
export OMP_NUM_THREADS=1
WORKERS=${SLURM_CPUS_PER_TASK:-16}

# Cut into node-local SSD, into a dir named patches_<N> so the tar has the right top level.
WORK="$SLURM_TMPDIR/patches_${PS}"
mkdir -p "$WORK"

echo "[$(date +%T)] cutting patch $PS  (src=$RAW_SRC  workers=$WORKERS) -> $WORK"
python "$CUTTER" "$RAW_SRC" "$WORK" \
  --patch-size "$PS" \
  --method minimal_overlap \
  --test-ratio 0.0 \
  --split-seed 42 \
  --quality 95 \
  --workers "$WORKERS"

# Tar the patches_<N> directory itself (cd into TMPDIR so the tar's top entry is
# "patches_<N>/..."), then move to $SCRATCH. Tar to node-local first, then copy -- writing
# the tar straight to Lustre while cutting is I/O-contended.
echo "[$(date +%T)] tarring -> $OUT_TAR"
cd "$SLURM_TMPDIR"
tar -cf "$SLURM_TMPDIR/barknet_patches_${PS}.tar" "patches_${PS}"
cp "$SLURM_TMPDIR/barknet_patches_${PS}.tar" "$OUT_TAR"

echo "[$(date +%T)] done patch $PS"
ls -lh "$OUT_TAR"
echo "verify internal layout (first few entries):"
tar -tf "$OUT_TAR" || true

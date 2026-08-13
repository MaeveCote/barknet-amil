#!/bin/bash
# SINGLE-STAGE AMIL -- 5-fold CV at patch 224. Trains backbone + classifier + attention
# head together from ImageNet init (no Stage-1 pretraining), 40 epochs, uniform LR.
#
# CONTROLLED COMPARISON: the ONLY intended difference from the 2-stage runs is the absence
# of backbone pretraining. Auxiliary loss is kept at the tuned 2-stage value (0.641) so a
# difference here is attributable to staging, not to the aux loss. Uniform LR (lr_multiplier
# =1) is used because there is no pretrained backbone to protect with a lower rate -- this
# co-varies with staging by necessity and should be stated as such in the paper.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_1stage
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

FOLD=$SLURM_ARRAY_TASK_ID
export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
PATCH_SIZE=224; MODEL_SIZE=nano; INPUT_SIZE=224
RUN_NAME="abl_1stage_nano_224_f${FOLD}"
OUT="$SCRATCH/runs/$RUN_NAME"
mkdir -p "$OUT"

# ---- environment (mirror common.sh) ----
module load StdEnv/2023 python/3.11 opencv/4.13.0
export HF_HOME="$HOME/.cache/huggingface"; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
virtualenv --no-download "$SLURM_TMPDIR/venv"; source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch torchvision timm optuna pandas scikit-learn openpyxl pillow numpy >/dev/null

# ---- stage data (patch 224 tar -> node-local SSD) ----
TAR="$SCRATCH/data/barknet_patches_224.tar"
mkdir -p "$SLURM_TMPDIR/patches"
echo "[$(date +%T)] extracting $TAR ..."
tar -xf "$TAR" -C "$SLURM_TMPDIR/patches"
FIRST_JPG=$(find "$SLURM_TMPDIR/patches" -name '*.jpg' -print -quit)
PATCH_ROOT=$(dirname "$(dirname "$FIRST_JPG")")
echo "[$(date +%T)] patch_root = $PATCH_ROOT"

CPUS=${SLURM_CPUS_PER_TASK:-12}; NUM_WORKERS=$(( CPUS>1 ? CPUS-1 : 0 ))
cd "$REPO_DIR/src"; export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

# ---- SINGLE-STAGE TRAIN: no --backbone-checkpoint => ImageNet init, head fresh, all train
# together. Overrides that define the single-stage condition:
#   model.weights=default        -> ImageNet FCMAE init (not from scratch, not a Stage-1 ckpt)
#   train.optimal_params=null    -> use inlined values, then override below
#   train.lr_multiplier=1        -> UNIFORM LR (no differential)
#   train.base_lr=1.006e-4       -> the FROM-IMAGENET rate (Stage-1's), NOT Stage-2's 1e-5
#                                   which assumes an already-trained backbone and is far too
#                                   low to learn a backbone from ImageNet in 40 epochs
#   train.instance_loss_weight=0.641 -> KEEP the tuned aux loss (the controlled variable is
#                                       staging, not the aux loss)
echo "[$(date +%T)] SINGLE-STAGE training fold $FOLD (40 epochs, uniform LR, aux=0.641)"
python train_model.py \
  -c "$CONFIG" \
  --patch-root "$PATCH_ROOT" \
  --output-dir "$OUT/train" \
  --model-size "$MODEL_SIZE" \
  --input-size "$INPUT_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda:0 \
  --fold "$FOLD" \
  --epochs 40 \
  --set data.split.n_folds=5 \
  --set model.weights=default \
  --set model.backbone_checkpoint=null \
  --set train.optimal_params=null \
  --set train.lr_multiplier=1 \
  --set train.base_lr=1.0062785194709649e-4 \
  --set train.instance_loss_weight=0.6412880283086111 \
  --set train.warmup_ratio=0.16 \
  --set train.early_stopping_patience=999 \
  --set train.min_epochs=999

[ -f "$OUT/train/best_model.pth" ] || { echo "no model produced"; exit 1; }

# ---- TEST: --mode amil gives amil + amil_vote (both from the trained model). We SKIP the
# separate "vote" baseline because it needs a standalone Stage-1 backbone, which single-
# stage does not produce. amil_vote (majority vote over the model's own patch head) is the
# headline comparison anyway -- same weights, isolates aggregation.
echo "[$(date +%T)] testing fold $FOLD"
python test_model.py \
  -c "$CONFIG" \
  --patch-root "$PATCH_ROOT" \
  --output-dir "$OUT/test" \
  --checkpoint "$OUT/train/best_model.pth" \
  --model-size "$MODEL_SIZE" \
  --input-size "$INPUT_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda:0 \
  --fold "$FOLD" \
  --mode amil --bags both \
  --set data.split.n_folds=5

echo "[$(date +%T)] DONE fold $FOLD -> $OUT"
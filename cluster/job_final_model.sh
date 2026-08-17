#!/bin/bash
# FINAL "BEST MODEL" -- single-stage AMIL, nano, 5-fold CV. The headline model compared
# against prior work (Carpentier et al.). Trains backbone + classifier + attention head
# together from ImageNet (no Stage-1 pretraining), 40 epochs, uniform LR, aux loss kept.
#
# PATCH SIZE: set below. You asked for 512; note from your own results that 512 gives only
# ~38 patches/bag (thin tail) while 384 gives ~64 (top of your stated 32-64 range) at
# essentially identical accuracy (0.9489 vs 0.9466) and a LARGER, more meaningful AMIL gap
# (+0.21 vs +0.10 pp). Change PATCH_SIZE=384 if you prefer that; everything else is identical.
#
#   sbatch cluster/job_final_model.sh
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_final
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

# ============================ EDIT HERE =====================================
PATCH_SIZE=512          # you asked for 512; set 384 for ~64 patches/bag (see header)
# ============================================================================
MODEL_SIZE=nano
INPUT_SIZE=224
EPOCHS=40
FOLD=$SLURM_ARRAY_TASK_ID
export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
RUN_NAME="final_${MODEL_SIZE}_p${PATCH_SIZE}_1stage_f${FOLD}"
OUT="$SCRATCH/runs/$RUN_NAME"
mkdir -p "$OUT"

# ---- environment ----
module load StdEnv/2023 python/3.11 opencv/4.13.0
export HF_HOME="$HOME/.cache/huggingface"; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
virtualenv --no-download "$SLURM_TMPDIR/venv"; source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch torchvision timm optuna pandas scikit-learn openpyxl pillow numpy >/dev/null

# ---- stage the patch tar for THIS patch size ----
TAR="$SCRATCH/data/barknet_patches_${PATCH_SIZE}.tar"
[ -f "$TAR" ] || { echo "Missing $TAR"; exit 1; }
mkdir -p "$SLURM_TMPDIR/patches"
echo "[$(date +%T)] extracting $TAR ..."
tar -xf "$TAR" -C "$SLURM_TMPDIR/patches"
FIRST_JPG=$(find "$SLURM_TMPDIR/patches" -name '*.jpg' -print -quit)
PATCH_ROOT=$(dirname "$(dirname "$FIRST_JPG")")
echo "[$(date +%T)] patch_root = $PATCH_ROOT"

CPUS=${SLURM_CPUS_PER_TASK:-12}; NUM_WORKERS=$(( CPUS>1 ? CPUS-1 : 0 ))
cd "$REPO_DIR/src"; export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

# ---- SINGLE-STAGE TRAIN (no --backbone-checkpoint => ImageNet init, all trained together) ----
#   weights=default            -> ImageNet FCMAE init
#   lr_multiplier=1            -> uniform LR (no differential; nothing pretrained to protect)
#   base_lr=1.006e-4           -> from-ImageNet rate (Stage-1's), NOT Stage-2's 1e-5
#   instance_loss_weight=0.641 -> keep the tuned aux loss (the operative regularizer)
#   n_folds=5                  -> real CV (config ships null -> would silently be holdout)
echo "[$(date +%T)] SINGLE-STAGE train: nano p${PATCH_SIZE} fold ${FOLD}, ${EPOCHS} epochs"
python train_model.py \
  -c "$CONFIG" \
  --patch-root "$PATCH_ROOT" \
  --output-dir "$OUT/train" \
  --model-size "$MODEL_SIZE" \
  --input-size "$INPUT_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda:0 \
  --fold "$FOLD" \
  --epochs "$EPOCHS" \
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

# ---- TEST: amil + amil_vote (single-stage has no standalone backbone for the vote baseline)
echo "[$(date +%T)] testing fold ${FOLD}"
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

echo "[$(date +%T)] DONE fold ${FOLD} -> $OUT"
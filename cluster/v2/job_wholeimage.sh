#!/bin/bash
# v2 A1 -- whole-image baseline on the IMAGE-level folds. One array task per fold.
#
# Same recipe as the paper's tree-level whole-image baseline (cluster/run_wholeimage.ps1 ->
# cluster/eval_wholeimage.py); the ONLY change is the split. Settings copied from that run's
# resolved_config.yaml so the tree-vs-image delta is not confounded:
#   ConvNeXt-V2 nano, ImageNet init, full photo resized to 224, single instance,
#   40 epochs (early stopping off), batch 64, checkpoint on val_acc,
#   rotation 15 deg (the local config used for that run; the patch runs used 30).
# Inner validation is image-level too (helper/image_folds.partition).
#
# Output: $V2_ROOT/runs/v2img_whole_nano_f<k>/{pretrain/, wholeimage_preds.csv}
#SBATCH --job-name=bark_v2_whole
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"
HERE="${V2_DIR:-$HOME/BarkNet_ML/cluster/v2}"
# shellcheck source=v2_env.sh
source "$HERE/v2_env.sh"

export FOLD="${FOLD:-$SLURM_ARRAY_TASK_ID}"
export MODEL_SIZE=nano INPUT_SIZE=224 EPOCHS_S1=40
export RUN_NAME="v2img_whole_nano_f$FOLD"
OUT="$V2_ROOT/runs/$RUN_NAME"
if [ -f "$OUT/wholeimage_preds.summary.csv" ]; then
  echo "$OUT already complete -- not overwriting."; exit 0
fi
[ -f "$FOLD_FILE" ] || { echo "missing fold file $FOLD_FILE -- run 00_preflight.sh"; exit 1; }

# common.sh stages a tar to node-local SSD and takes the grandparent of the first .jpg as
# the root -- for the raw tar that is the folder of species dirs, which is what we want.
export PATCH_SIZE=whole PATCH_TAR="$RAW_TAR"
source "$REPO_DIR/cluster/common.sh"
source "$REPO_DIR/cluster/provenance.sh"
IMAGE_ROOT="$PATCH_ROOT"
export EXTRA_ARGS="data.split.unit=image data.filename.layout=raw pretrain.batch_size=64 augmentation.rotation.degrees=15"
write_provenance "$OUT" "v2 A1 whole-image fold $FOLD"

cd "$REPO_DIR/src"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

if [ ! -f "$OUT/pretrain/pretrain_summary.json" ]; then
  echo "[$(date +%T)] ===== A1 train (whole images, image-level fold $FOLD) ====="
  $PY pretrain_backbone.py -c "$CONFIG" \
      --patch-root "$IMAGE_ROOT" \
      --output-dir "$OUT/pretrain" \
      --model-size "$MODEL_SIZE" \
      --input-size "$INPUT_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --fold "$FOLD" \
      --device cuda:0 \
      --epochs "$EPOCHS_S1" \
      --set data.split.unit=image \
      --set "data.split.fold_file=$FOLD_FILE" \
      --set data.filename.layout=raw \
      --set pretrain.batch_size=64 \
      --set pretrain.checkpoint_metric=val_acc \
      --set augmentation.rotation.degrees=15
fi
[ -f "$OUT/pretrain/best_backbone.pth" ] || { echo "no backbone produced"; exit 1; }

echo "[$(date +%T)] ===== A1 eval (one prediction per whole image) ====="
$PY "$REPO_DIR/cluster/eval_wholeimage.py" \
    --image-root "$IMAGE_ROOT" \
    --backbone "$OUT/pretrain/best_backbone.pth" \
    --repo-root "$REPO_DIR" \
    --config "$CONFIG" \
    --fold "$FOLD" --n-folds 5 --seed 42 \
    --model-size "$MODEL_SIZE" --input-size "$INPUT_SIZE" \
    --split image --fold-file "$FOLD_FILE" \
    --out "$OUT/wholeimage_preds.csv"
cat "$OUT/wholeimage_preds.summary.csv"
echo "[$(date +%T)] DONE $RUN_NAME"

#!/bin/bash
# v2 Part B -- INFERENCE ONLY. One array task re-scores one existing tree-level checkpoint
# with every aggregator test_model.py now reports (attention MIL, uniform attention,
# soft vote x2, majority vote x2, per-patch accuracy). No training, no checkpoint is
# written or selected: the run's own best_model.pth (picked on VALIDATION during training)
# is loaded as-is and scored on its own held-out test fold.
#
# Submitted by submit_v2.sh with MANIFEST=<tsv> and --array=0-(N-1). Each manifest line:
#   name  source_run_dir  patch_size  model_size  fold  mode(both|amil)
# Output: $V2_ROOT/eval/<name>/test/  (refuses to overwrite an existing result).
#SBATCH --job-name=bark_v2_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"
HERE="${V2_DIR:-$HOME/BarkNet_ML/cluster/v2}"
# shellcheck source=v2_env.sh
source "$HERE/v2_env.sh"

: "${MANIFEST:?MANIFEST not set (submit via submit_v2.sh)}"
LINE=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$MANIFEST")
[ -n "$LINE" ] || { echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 1; }
read -r NAME SRC PATCH_SIZE MODEL_SIZE FOLD MODE <<< "$LINE"
export PATCH_SIZE MODEL_SIZE FOLD RUN_NAME="$NAME"

OUT="$V2_ROOT/eval/$NAME"
if [ -f "$OUT/test/test_summary.json" ]; then
  echo "$OUT/test/test_summary.json exists -- not overwriting. Done."; exit 0
fi
MODEL="$SRC/train/best_model.pth"
BACKBONE="$SRC/pretrain/best_backbone.pth"
[ -f "$MODEL" ] || { echo "missing $MODEL"; exit 1; }

# venv, offline weights, patch tar -> node-local SSD; exports PY, PATCH_ROOT, NUM_WORKERS
source "$REPO_DIR/cluster/common.sh"
source "$REPO_DIR/cluster/provenance.sh"
write_provenance "$OUT" "v2 eval of $SRC (mode=$MODE)"

if [ "$MODE" = both ]; then
  [ -f "$BACKBONE" ] || { echo "missing $BACKBONE"; exit 1; }
  CKPT_ARGS=(--backbone-checkpoint "$BACKBONE" --mode both)
else
  CKPT_ARGS=(--set model.backbone_checkpoint=null --mode amil)
fi

cd "$REPO_DIR/src"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
echo "[$(date +%T)] eval $NAME <- $SRC  (patch $PATCH_SIZE, $MODEL_SIZE, fold $FOLD, $MODE)"
$PY test_model.py -c "$CONFIG" \
    --patch-root "$PATCH_ROOT" \
    --model-size "$MODEL_SIZE" \
    --input-size 224 \
    --num-workers "$NUM_WORKERS" \
    --fold "$FOLD" \
    --device cuda:0 \
    --checkpoint "$MODEL" \
    "${CKPT_ARGS[@]}" \
    --bags full \
    --output-dir "$OUT/test"
echo "$SRC" > "$OUT/source_run.txt"
cat "$OUT/test/test_summary.json"
echo "[$(date +%T)] DONE $NAME"

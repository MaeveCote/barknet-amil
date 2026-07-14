#!/bin/bash
# Stage 1 (patch pretraining) -> Stage 2 (AMIL fine-tune) -> test (AMIL vs majority vote),
# chained in a single job so the Stage-1 backbone never has to survive a queue wait.
#
# All outputs go straight to $SCRATCH (NOT $SLURM_TMPDIR): if the job is killed on
# wall-clock, $SLURM_TMPDIR is wiped and anything written there is gone. Checkpoints are
# written once per epoch, which is far too little I/O to bother the shared filesystem.
#
# Works both under sbatch and inside an salloc session.
#
#   PATCH_SIZE=224 MODEL_SIZE=nano EPOCHS_S1=2 EPOCHS_S2=2 RUN_NAME=smoke \
#     bash cluster/run_chain.sh
set -euo pipefail

# ---- knobs ------------------------------------------------------------------
PATCH_SIZE="${PATCH_SIZE:?set PATCH_SIZE}"
MODEL_SIZE="${MODEL_SIZE:-pico}"
# Network input size. Default = native (the patch is fed at the size it was cut), which
# is what the patch-size ablation is about: bigger patch = wider field of view AND more
# compute. To ablate field of view at CONSTANT compute instead, set INPUT_SIZE=224 for
# all three runs so the 288/384 patches get downscaled to 224.
INPUT_SIZE="${INPUT_SIZE:-$PATCH_SIZE}"
EPOCHS_S1="${EPOCHS_S1:-20}"
EPOCHS_S2="${EPOCHS_S2:-15}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-${MODEL_SIZE}_${PATCH_SIZE}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OUT="${OUT_ROOT:-$SCRATCH/runs}/$RUN_NAME"
mkdir -p "$OUT"

# ---- environment, venv, data staging ---------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_SIZE
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

cd "$REPO_DIR"

# If the ImageNet weights live in a plain directory rather than the HF cache, point
# PRETRAINED_FILE at the weight file and it is threaded into every stage.
WEIGHT_ARGS=()
if [ -n "${PRETRAINED_FILE:-}" ]; then
  WEIGHT_ARGS=(--set "model.pretrained_file=$PRETRAINED_FILE")
fi

COMMON=(-c "$CONFIG"
        --patch-root "$PATCH_ROOT"
        --model-size "$MODEL_SIZE"
        --input-size "$INPUT_SIZE"
        --num-workers "$NUM_WORKERS"
        --fold "$FOLD"
        --device cuda:0
        "${WEIGHT_ARGS[@]+"${WEIGHT_ARGS[@]}"}")

echo
echo "##############################################################"
echo "# RUN $RUN_NAME | convnextv2_$MODEL_SIZE | patch $PATCH_SIZE -> input $INPUT_SIZE"
echo "# epochs: stage1=$EPOCHS_S1  stage2=$EPOCHS_S2  fold=$FOLD"
echo "# out   : $OUT"
echo "##############################################################"

# ---- Stage 1: patch-level backbone pretraining ------------------------------
echo; echo "[$(date +%T)] ===== STAGE 1 ====="
$PY pretrain_backbone.py "${COMMON[@]}" \
    --output-dir "$OUT/pretrain" \
    --epochs "$EPOCHS_S1" \
    $EXTRA_ARGS

BACKBONE="$OUT/pretrain/best_backbone.pth"
[ -f "$BACKBONE" ] || { echo "Stage 1 produced no backbone at $BACKBONE"; exit 1; }

# ---- Stage 2: image-level AMIL fine-tuning ----------------------------------
echo; echo "[$(date +%T)] ===== STAGE 2 ====="
$PY train_model.py "${COMMON[@]}" \
    --output-dir "$OUT/train" \
    --backbone-checkpoint "$BACKBONE" \
    --epochs "$EPOCHS_S2" \
    $EXTRA_ARGS

MODEL="$OUT/train/best_model.pth"
[ -f "$MODEL" ] || { echo "Stage 2 produced no model at $MODEL"; exit 1; }

# ---- Test: AMIL vs. hard majority voting, on the same held-out trees ---------
echo; echo "[$(date +%T)] ===== TEST (AMIL vs majority vote) ====="
$PY test_model.py "${COMMON[@]}" \
    --output-dir "$OUT/test" \
    --checkpoint "$MODEL" \
    --backbone-checkpoint "$BACKBONE" \
    --mode both \
    $EXTRA_ARGS

echo
echo "[$(date +%T)] DONE. Artefacts under $OUT"
cat "$OUT/test/test_summary.json" || true
echo
echo "Right-size the next job with:  seff ${SLURM_JOB_ID:-<jobid>}"

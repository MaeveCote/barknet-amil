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
# all | 1 | 2   -- "1" is Stage 1 only; "2" is Stage 2 + test, resuming the backbone that
# a previous "1" job left on $SCRATCH. Lets a long Stage 1 sit in the 24h partition while
# Stage 2 + test go in a short one, chained with --dependency=afterok.
STAGES="${STAGES:-all}"

OUT="${OUT_ROOT:-$SCRATCH/runs}/$RUN_NAME"
mkdir -p "$OUT"

# ---- environment, venv, data staging ---------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_SIZE
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

SRC_DIR="${SRC_DIR:-$REPO_DIR/src}"
[ -f "$SRC_DIR/pretrain_backbone.py" ] || { echo "No pretrain_backbone.py under $SRC_DIR"; exit 1; }
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
cd "$SRC_DIR"

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

BACKBONE="$OUT/pretrain/best_backbone.pth"
MODEL="$OUT/train/best_model.pth"

# ---- Stage 1: patch-level backbone pretraining ------------------------------
if [ "$STAGES" = "all" ] || [ "$STAGES" = "1" ]; then
  echo; echo "[$(date +%T)] ===== STAGE 1 ($EPOCHS_S1 epochs, early stopping) ====="
  $PY pretrain_backbone.py "${COMMON[@]}" \
      --output-dir "$OUT/pretrain" \
      --epochs "$EPOCHS_S1" \
      $EXTRA_ARGS
  [ -f "$BACKBONE" ] || { echo "Stage 1 produced no backbone at $BACKBONE"; exit 1; }
  echo; echo "[$(date +%T)] Stage-1 summary:"; cat "$OUT/pretrain/pretrain_summary.json" || true
fi

# ---- Stage 2: image-level AMIL fine-tuning ----------------------------------
if [ "$STAGES" = "all" ] || [ "$STAGES" = "2" ]; then
  [ -f "$BACKBONE" ] || { echo "No Stage-1 backbone at $BACKBONE -- run STAGES=1 first."; exit 1; }

  echo; echo "[$(date +%T)] ===== STAGE 2 ====="
  $PY train_model.py "${COMMON[@]}" \
      --output-dir "$OUT/train" \
      --backbone-checkpoint "$BACKBONE" \
      --epochs "$EPOCHS_S2" \
      $EXTRA_ARGS
  [ -f "$MODEL" ] || { echo "Stage 2 produced no model at $MODEL"; exit 1; }

  # ---- Test: 3 predictors x {full, capped} bags ------------------------------
  echo; echo "[$(date +%T)] ===== TEST ====="
  $PY test_model.py "${COMMON[@]}" \
      --output-dir "$OUT/test" \
      --checkpoint "$MODEL" \
      --backbone-checkpoint "$BACKBONE" \
      --mode both --bags both \
      $EXTRA_ARGS
  cat "$OUT/test/test_summary.json" || true
fi

echo
echo "[$(date +%T)] DONE (stages=$STAGES). Artefacts under $OUT"
echo
echo "Right-size the next job with:  seff ${SLURM_JOB_ID:-<jobid>}"

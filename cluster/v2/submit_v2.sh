#!/bin/bash
# v2 -- submit everything. LOGIN NODE. Run 00_preflight.sh first.
#
#   bash cluster/v2/submit_v2.sh           # DRY RUN: prints every sbatch call, submits nothing
#   GO=1 bash cluster/v2/submit_v2.sh      # submits
#
# MODE (default: preflight's RECOMMENDED_MODE)
#   full     A1 whole-image (5) + A2 image-level 512 two-stage (5 Stage-1 -> 5 Stage-2,
#            chained per fold with aftercorr). A2's own test step already runs every
#            aggregator, which IS B3. Part B on the existing tree-level checkpoints:
#            B1 (5 patch sizes x 5 folds), B2 (single-stage 224 x 5), B4 (pico/tiny x 5,
#            if complete; INCLUDE_B4=0 to skip). Then one CPU job compiles the tables.
#   minimal  (tree-level checkpoints purged) A1 + A2 as above, plus RETRAIN tree-level nano
#            512 two-stage and nano 224 single-stage (5 folds each, paper settings); their
#            own test steps give Part B. No other patch sizes, no B4.
#
# Every job refuses to overwrite a finished result, and this script refuses to run twice
# (lock file) unless FORCE=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=v2_env.sh
source "$HERE/v2_env.sh"

GO="${GO:-0}"
case "$ACCOUNT_GPU$ACCOUNT_CPU" in
  *YOUR_ACCOUNT*) echo "Set your SLURM allocation first, e.g.:"
                  echo "  export ACCOUNT_GPU=def-someone_gpu ACCOUNT_CPU=def-someone_cpu"; exit 1 ;;
esac
STATUS="$V2_ROOT/preflight/status.env"
[ -f "$STATUS" ] || { echo "No $STATUS -- run: bash cluster/v2/00_preflight.sh"; exit 1; }
# shellcheck disable=SC1090
source "$STATUS"
MODE="${MODE:-$RECOMMENDED_MODE}"
INCLUDE_B4="${INCLUDE_B4:-1}"
[ "$DATA_OK" = 1 ]  || { echo "preflight: data missing ($PREFLIGHT_REPORT)"; exit 1; }
[ "$FOLDS_OK" = 1 ] && [ -f "$FOLD_FILE" ] || { echo "preflight: no fold file $FOLD_FILE"; exit 1; }
case "$MODE" in
  full)    [ "$B1_OK" = 1 ] && [ "$B2_OK" = 1 ] || {
             echo "MODE=full but B1/B2 checkpoints are incomplete (see $PREFLIGHT_REPORT)."
             echo "Use MODE=minimal, or restore the checkpoints and re-run the preflight."; exit 1; } ;;
  minimal) ;;
  *)       echo "MODE must be full or minimal, got '$MODE'"; exit 1 ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
LOCK="$V2_ROOT/submitted.lock"
LOG="$V2_ROOT/submitted_jobs_$STAMP.txt"
if [ "$GO" = 1 ]; then
  if [ -f "$LOCK" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "Already submitted once (see $LOCK). Check squeue / the logs; FORCE=1 to submit again."
    exit 1
  fi
  mkdir -p "$V2_ROOT/runs" "$V2_ROOT/eval" "$SCRATCH/logs"
  { echo "mode=$MODE stamp=$STAMP commit=$(git -C "$REPO_DIR" rev-parse HEAD)"; } > "$LOG"
fi

JOBS=()
# sub LABEL [VAR=value ...] -- <sbatch args>   -> echoes the job id (or a placeholder)
sub() {
  local label=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  if [ "$GO" = 1 ]; then
    local jid
    jid=$(env "${envs[@]}" sbatch --parsable "$@")
    echo "$label $jid  [${envs[*]}] $*" >> "$LOG"
    echo "  submitted $label -> $jid" >&2
    echo "$jid"
  else
    printf '  [dry] %-14s %s sbatch %s\n' "$label" "${envs[*]}" "$*" >&2
    echo "<$label>"
  fi
}

TRAIN="$REPO_DIR/cluster/job_train.sh"
IMG_ARGS="--set data.split.unit=image --set data.split.fold_file=$FOLD_FILE"
echo "v2 submission  MODE=$MODE  GO=$GO  ->  $V2_ROOT"
echo

# ---- A1: whole image, image-level folds (5 jobs) ------------------------------------
echo "A1 whole-image baseline, image-level (5):"
A1=$(sub A1_whole V2_DIR="$HERE" -- --account="$ACCOUNT_GPU" "$HERE/job_wholeimage.sh")
JOBS+=("$A1")

# ---- A2: 512 px two-stage, image-level folds (5 + 5 jobs) ---------------------------
echo "A2 nano 512 two-stage, image-level (5 Stage-1 -> 5 Stage-2 + test incl. B3):"
A2_ENV=(PATCH_SIZE=512 MODEL_SIZE=nano RUN_PREFIX=v2img_p512_nano OUT_ROOT="$V2_ROOT/runs"
        "EXTRA_ARGS=$IMG_ARGS")
A2S1=$(sub A2_stage1 "${A2_ENV[@]}" STAGES=1 -- --account="$ACCOUNT_GPU" --time=24:00:00 \
       --job-name=bark_v2_a2s1 "$TRAIN")
A2S2=$(sub A2_stage2 "${A2_ENV[@]}" STAGES=2 -- --account="$ACCOUNT_GPU" --time=8:00:00 \
       --job-name=bark_v2_a2s2 --dependency="aftercorr:$A2S1" "$TRAIN")
JOBS+=("$A2S1" "$A2S2")

# ---- Part B --------------------------------------------------------------------------
if [ "$MODE" = full ]; then
  MANIFEST="$V2_ROOT/eval/manifest_$STAMP.tsv"
  TMP_MANIFEST=$(mktemp)
  add() {  # name src_run patch model mode  (one line per fold)
    for k in "${FOLDS[@]}"; do
      [ -f "$V2_ROOT/eval/${1}_f$k/test/test_summary.json" ] && continue   # already done
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${1}_f$k" "$TREE_RUNS/${2}_f$k" "$3" "$4" "$k" "$5" \
        >> "$TMP_MANIFEST"
    done
  }
  for ps in "${B1_PATCH_SIZES[@]}"; do add "tree_p${ps}_nano" "abl_p${ps}_nano" "$ps" nano both; done
  add tree_1stage_nano_224 "$B2_RUN_PREFIX" 224 nano amil
  if [ "$INCLUDE_B4" = 1 ] && [ "$B4_OK" = 1 ]; then
    add tree_msize_pico_224 abl_msize_pico_224 224 pico both
    add tree_msize_tiny_224 abl_msize_tiny_224 224 tiny both
  fi
  N=$(wc -l < "$TMP_MANIFEST")
  echo "Part B inference-only on existing tree-level checkpoints ($N evaluations):"
  if [ "$N" -gt 0 ]; then
    if [ "$GO" = 1 ]; then cp "$TMP_MANIFEST" "$MANIFEST"; else MANIFEST="$TMP_MANIFEST"; fi
    sed 's/^/    /' "$TMP_MANIFEST" | cut -f1,2,6 >&2
    B=$(sub B_eval V2_DIR="$HERE" MANIFEST="$MANIFEST" -- --account="$ACCOUNT_GPU" \
        --array="0-$(( N - 1 ))%10" "$HERE/job_eval.sh")
    JOBS+=("$B")
  fi
  [ "$GO" = 1 ] && rm -f "$TMP_MANIFEST"
else
  echo "Part B via RETRAINED tree-level checkpoints (purge fallback):"
  R_ENV=(MODEL_SIZE=nano OUT_ROOT="$V2_ROOT/runs")
  R1=$(sub R_p512_s1 "${R_ENV[@]}" PATCH_SIZE=512 RUN_PREFIX=retrain_abl_p512_nano STAGES=1 -- \
       --account="$ACCOUNT_GPU" --time=24:00:00 --job-name=bark_v2_r512s1 "$TRAIN")
  R2=$(sub R_p512_s2 "${R_ENV[@]}" PATCH_SIZE=512 RUN_PREFIX=retrain_abl_p512_nano STAGES=2 -- \
       --account="$ACCOUNT_GPU" --time=8:00:00 --job-name=bark_v2_r512s2 \
       --dependency="aftercorr:$R1" "$TRAIN")
  R3=$(sub R_1stage224 "${R_ENV[@]}" PATCH_SIZE=224 SINGLE_STAGE=1 \
       RUN_PREFIX=retrain_abl_1stage_nano_224 -- \
       --account="$ACCOUNT_GPU" --time=24:00:00 --job-name=bark_v2_r1st "$TRAIN")
  JOBS+=("$R1" "$R2" "$R3")
fi

# ---- compile the tables once everything has finished (pass or fail) -----------------
echo "Compile tables (CPU, after all of the above):"
DEP=$(IFS=:; echo "${JOBS[*]}")
C=$(sub compile V2_DIR="$HERE" -- --account="$ACCOUNT_CPU" --dependency="afterany:$DEP" \
    "$HERE/job_compile.sh")

echo
if [ "$GO" = 1 ]; then
  cp "$LOG" "$LOCK"
  echo "Submitted. Job ids: $LOG"
  echo "Watch:  squeue -u $USER"
  echo "When the compile job ($C) finishes, the tables are in $V2_ROOT/tables/"
else
  echo "Dry run only -- nothing submitted. Re-run with GO=1 to submit."
fi

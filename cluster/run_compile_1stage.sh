#!/bin/bash
# Compile the SINGLE-STAGE ablation and put it side-by-side with the 2-stage nano@224
# baseline, so the 1-vs-2-stage difference is readable in one master file. LOGIN NODE.
#
#   abl_1stage_nano_224_f*   <- single-stage (this ablation)
#   abl_p224_nano_f*         <- 2-stage nano@224 (from the patch study = the baseline)
#
# All metrics recomputed from each run's per-bag rows (sheet Pred_full). No GPU.
# Single-stage was tested with --mode amil, so it has amil + amil_vote but no standalone
# 'vote' predictor -- the compiler handles the missing column gracefully.
#
#   bash run_compile_1stage.sh
# Outputs -> $SCRATCH/compiled_1stage/ : per_fold_results.csv, master_summary.csv (+.json)
set -euo pipefail
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/runs}"
OUT_DIR="${OUT_DIR:-$SCRATCH/compiled_1stage}"
SCRIPT="${SCRIPT:-$HOME/BarkNet_ML/cluster/compile_results.py}"

module load StdEnv/2023 python/3.11 2>/dev/null || module load python/3.11
VENV="/tmp/compile_venv_$$"; virtualenv --no-download "$VENV" >/dev/null; source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index pandas numpy openpyxl scipy scikit-learn >/dev/null

# Stage both groups into one temp dir of symlinks for a single compile pass.
STAGE="/tmp/1stage_stage_$$"; mkdir -p "$STAGE"
for d in "$RUNS_ROOT"/abl_1stage_nano_224_f* "$RUNS_ROOT"/abl_p224_nano_f*; do
  [ -d "$d" ] && ln -s "$d" "$STAGE/$(basename "$d")"
done
echo "Staged $(ls "$STAGE" | wc -l) run dirs (single-stage + 2-stage nano@224)"

python "$SCRIPT" --runs-root "$STAGE" --pattern "*" --sheet Pred_full --out-dir "$OUT_DIR"

rm -rf "$STAGE"; deactivate; rm -rf "$VENV"
echo
echo "Two arms to compare in master_summary.csv:"
echo "  onestage_nano  (single-stage)   vs   patch_224 (2-stage nano@224)"
echo
echo "Pull to your PC:"
echo "  scp $USER@rorqual.alliancecan.ca:$OUT_DIR/'*.csv' ."
#!/bin/bash
# Compile all ablation results into a master summary -- runs on the LOGIN NODE.
#
# No GPU, no SLURM job, no re-inference: everything is recomputed from the per-bag rows
# that test_model.py already wrote into each run's classification_results.xlsx
# (sheet Pred_full = uncapped bags). Reading 25 spreadsheets is a few seconds of CPU.
#
#   bash cluster/run_compile.sh                      # patch-size ablation
#   PATTERN='abl_msize_*_f*' bash cluster/run_compile.sh   # model-size ablation
#   SHEET=Pred_capped_128 bash cluster/run_compile.sh      # capped-bag robustness check
#
# Outputs land in $SCRATCH/compiled/:
#   per_fold_results.csv   one row per fold, every metric
#   master_summary.csv     one row per arm, mean + std across folds
#   master_summary.json    both, machine-readable
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/runs}"
PATTERN="${PATTERN:-abl_p*_nano_f*}"
SHEET="${SHEET:-Pred_full}"
OUT_DIR="${OUT_DIR:-$SCRATCH/compiled}"
SCRIPT="${SCRIPT:-$HOME/BarkNet_ML/cluster/compile_results.py}"

module load StdEnv/2023 python/3.11 2>/dev/null || module load python/3.11

VENV="/tmp/compile_venv_$$"
echo "[$(date +%T)] building venv at $VENV ..."
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
# openpyxl is needed to read the .xlsx; scipy gives the exact binomial McNemar
pip install --no-index pandas numpy openpyxl scipy >/dev/null
echo "[$(date +%T)] pandas $(python -c 'import pandas;print(pandas.__version__)')"

python "$SCRIPT" --runs-root "$RUNS_ROOT" --pattern "$PATTERN" \
                 --sheet "$SHEET" --out-dir "$OUT_DIR"

deactivate
rm -rf "$VENV"

echo
echo "[$(date +%T)] done. Pull the results to your PC with:"
echo "  scp $USER@rorqual.alliancecan.ca:$OUT_DIR/'*.csv' ."
echo "  scp $USER@rorqual.alliancecan.ca:$OUT_DIR/master_summary.json ."

#!/bin/bash
# Compile the MODEL-SIZE ablation (pico / nano / tiny @ 224) into a master file -- LOGIN NODE.
#
# nano@224 was produced by the PATCH study (abl_p224_nano_f*), not a separate msize run, so
# this pulls all three sizes with a glob that matches both naming schemes:
#   abl_msize_pico_224_f*   abl_msize_tiny_224_f*   abl_p224_nano_f*
# Everything is recomputed from the per-bag rows already in each run's
# classification_results.xlsx (sheet Pred_full). No GPU, no re-inference.
#
#   bash run_compile_msize.sh
#
# Outputs -> $SCRATCH/compiled_msize/ : per_fold_results.csv, master_summary.csv (+.json)
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/runs}"
OUT_DIR="${OUT_DIR:-$SCRATCH/compiled_msize}"
SCRIPT="${SCRIPT:-$HOME/BarkNet_ML/cluster/compile_msize.py}"

module load StdEnv/2023 python/3.11 2>/dev/null || module load python/3.11
VENV="/tmp/compile_venv_$$"
virtualenv --no-download "$VENV" >/dev/null
source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index pandas numpy openpyxl scipy scikit-learn >/dev/null

# Two globs (pico+tiny use one scheme, nano the patch scheme) -> stage into one temp dir of
# symlinks so a single compile pass sees all three sizes.
STAGE="/tmp/msize_stage_$$"; mkdir -p "$STAGE"
for d in "$RUNS_ROOT"/abl_msize_pico_224_f* "$RUNS_ROOT"/abl_msize_tiny_224_f* "$RUNS_ROOT"/abl_p224_nano_f*; do
  [ -d "$d" ] && ln -s "$d" "$STAGE/$(basename "$d")"
done
echo "Staged $(ls "$STAGE" | wc -l) run dirs (pico + nano + tiny)"

python "$SCRIPT" --runs-root "$STAGE" --pattern "*" --sheet Pred_full --out-dir "$OUT_DIR"

rm -rf "$STAGE"; deactivate; rm -rf "$VENV"
echo
echo "Pull to your PC:"
echo "  scp $USER@rorqual.alliancecan.ca:$OUT_DIR/'*.csv' ."
#!/bin/bash
# Compile the FINAL "best model" (single-stage nano @ 512, 5-fold CV) into the headline
# numbers for the prior-work comparison. LOGIN NODE, no GPU.
#
#   bash run_compile_final.sh
#
# All metrics recomputed from each fold's per-bag rows (sheet Pred_full = uncapped, the
# honest evaluation). The final model was tested with --mode amil, so it has amil +
# amil_vote but no standalone 'vote' -- handled gracefully (vote shows NaN).
#
# Outputs -> $SCRATCH/compiled_final/ : per_fold_results.csv, master_summary.csv (+.json)
set -euo pipefail
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/runs}"
OUT_DIR="${OUT_DIR:-$SCRATCH/compiled_final}"
SCRIPT="${SCRIPT:-$HOME/BarkNet_ML/cluster/compile_results.py}"

module load StdEnv/2023 python/3.11 2>/dev/null || module load python/3.11
VENV="/tmp/compile_venv_$$"; virtualenv --no-download "$VENV" >/dev/null; source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index pandas numpy openpyxl scipy scikit-learn >/dev/null

python "$SCRIPT" --runs-root "$RUNS_ROOT" --pattern "final_*_f*" --sheet Pred_full --out-dir "$OUT_DIR"

# ---- headline summary for the prior-work comparison ----
python - "$OUT_DIR/master_summary.csv" << 'PY'
import sys, pandas as pd
m = pd.read_csv(sys.argv[1])
if m.empty:
    print("\nNo final-model runs found yet."); sys.exit()
r = m.iloc[0]
def g(k): return r[k] if k in m.columns and pd.notna(r[k]) else float('nan')
print("\n" + "="*64)
print("FINAL MODEL  (single-stage nano @ 512px, 5-fold tree-level CV)")
print("="*64)
print(f"  folds                : {int(g('n_folds'))}")
print(f"  image-level AMIL acc : {g('amil_accuracy_mean'):.4f} +/- {g('amil_accuracy_std'):.4f}")
print(f"  image-level macro-F1 : {g('amil_macro_f1_mean'):.4f} +/- {g('amil_macro_f1_std'):.4f}")
print(f"  TREE-level AMIL acc  : {g('tree_amil_accuracy_mean'):.4f} +/- {g('tree_amil_accuracy_std'):.4f}")
print(f"  AMIL - vote gap (pp) : {g('amil_vs_amil_vote_delta_pp_mean'):+.3f} +/- {g('amil_vs_amil_vote_delta_pp_std'):.3f}")
print()
print("  Prior work (leakage-safe, tree-level split):")
print("    Carpentier et al.  : image-vote 93.88% | tree-vote 97.81%")
print("  Report YOUR number as CV mean +/- std, framed as 'comparable to prior")
print("  leakage-safe work' -- the tree-level accuracy is the direct comparison.")
PY

deactivate; rm -rf "$VENV"
echo
echo "Pull to your PC:"
echo "  scp $USER@rorqual.alliancecan.ca:$OUT_DIR/'*.csv' ."
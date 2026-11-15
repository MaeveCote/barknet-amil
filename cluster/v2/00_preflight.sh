#!/bin/bash
# v2 STEP 0 -- preflight. LOGIN NODE, no GPU, a few minutes. Read-only on existing runs.
#
#   bash cluster/v2/00_preflight.sh
#
# 1. Inventory of the tree-level checkpoints Part B needs (exists? size? days since last
#    touch -- /scratch purges files untouched for ~60 days) and of any saved per-patch logit
#    files (none are expected: test_model.py never wrote them, so the soft-vote / uniform
#    baselines are recomputed on GPU).
# 2. Data: the patch tars and the raw photos.
# 3. Builds the image-level fold file (once) + the per-fold leakage sanity check.
# 4. Writes $V2_ROOT/preflight/status.env, which submit_v2.sh reads to pick full vs minimal.
#
# Nothing is submitted. Read the report, then run submit_v2.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=v2_env.sh
source "$HERE/v2_env.sh"

PRE="$V2_ROOT/preflight"
mkdir -p "$PRE"
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$PRE/preflight_$STAMP.txt"
INV="$PRE/checkpoint_inventory_$STAMP.csv"
exec > >(tee "$REPORT") 2>&1

NOW=$(date +%s)
echo "BarkNet v2 preflight  $(date)  on $(hostname)"
echo "repo   : $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')" \
     "$( [ -z "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ] && echo clean || echo DIRTY)"
echo "config : $CONFIG"
echo "runs   : $TREE_RUNS (read-only)"
echo "v2 out : $V2_ROOT"
echo

# ---------------------------------------------------------------- 1. checkpoints ----
echo "group,run,file,exists,size_mb,days_since_access,days_since_modify" > "$INV"
check() {   # group run relpath  -> prints 1 if present
  local f="$TREE_RUNS/$2/$3"
  if [ -f "$f" ]; then
    read -r size atime mtime < <(stat -c '%s %X %Y' "$f")
    echo "$1,$2,$3,1,$(( size / 1048576 )),$(( (NOW - atime) / 86400 )),$(( (NOW - mtime) / 86400 ))" >> "$INV"
    echo 1
  else
    echo "$1,$2,$3,0,,," >> "$INV"
    echo 0
  fi
}
declare -A GROUP_OK=([B1]=1 [B2]=1 [B4]=1)
need() {    # group run relpath...
  local g=$1 r=$2; shift 2
  for rel in "$@"; do
    [ "$(check "$g" "$r" "$rel")" = 1 ] || GROUP_OK[$g]=0
  done
}
for ps in "${B1_PATCH_SIZES[@]}"; do for k in "${FOLDS[@]}"; do
  need B1 "abl_p${ps}_nano_f$k" train/best_model.pth pretrain/best_backbone.pth \
       pretrain/pretrain_summary.json test/test_summary.json
done; done
for k in "${FOLDS[@]}"; do
  need B2 "${B2_RUN_PREFIX}_f$k" train/best_model.pth test/test_summary.json
done
for pre in "${B4_RUN_PREFIXES[@]}"; do for k in "${FOLDS[@]}"; do
  need B4 "${pre}_f$k" train/best_model.pth pretrain/best_backbone.pth
done; done

echo "== 1. Tree-level checkpoints (full list: $INV)"
for g in B1 B2 B4; do
  n=$(awk -F, -v g=$g '$1==g' "$INV" | wc -l); ok=$(awk -F, -v g=$g '$1==g && $4==1' "$INV" | wc -l)
  printf '   %s: %3d / %3d files present  -> %s\n' "$g" "$ok" "$n" \
    "$( [ "${GROUP_OK[$g]}" = 1 ] && echo OK || echo INCOMPLETE)"
done
missing=$(awk -F, 'NR>1 && $4==0 {print "     MISSING " $2 "/" $3}' "$INV")
[ -n "$missing" ] && echo "$missing"
oldest=$(awk -F, 'NR>1 && $4==1 {print $6}' "$INV" | sort -n | tail -1)
echo "   oldest last-access among present checkpoints: ${oldest:-n/a} days (purge after ~60)"
if [ -n "${oldest:-}" ] && [ "$oldest" -ge 45 ]; then
  echo "   !! within ~2 weeks of the purge window -- back them up to /project now (see README)."
fi
PURGE_LIST=""
for td in "/scratch/to_delete/$USER" "$SCRATCH/../to_delete/$USER"; do
  [ -e "$td" ] && { PURGE_LIST=$(readlink -f "$td"); break; }
done
N_NEEDED_LISTED=0
if [ -n "$PURGE_LIST" ]; then
  echo "   purge candidate list found: $PURGE_LIST"
  echo "     files under the runs folder listed for deletion: $(grep -c "$(readlink -f "$TREE_RUNS")" "$PURGE_LIST" 2>/dev/null || echo 0)"
  # which of the checkpoints v2 needs are on it?
  while IFS=, read -r g r f ok _; do
    [ "$ok" = 1 ] || continue
    if grep -qF "$(readlink -f "$TREE_RUNS/$r/$f")" "$PURGE_LIST" 2>/dev/null; then
      N_NEEDED_LISTED=$(( N_NEEDED_LISTED + 1 ))
    fi
  done < <(tail -n +2 "$INV")
  echo "     of the $(awk -F, 'NR>1 && $4==1' "$INV" | wc -l) checkpoint/summary files v2 needs, on the list: $N_NEEDED_LISTED"
  if [ "$N_NEEDED_LISTED" -gt 0 ]; then
    echo "   !! BACK UP $TREE_RUNS TO /project BEFORE SUBMITTING (cluster/v2/README.md, step 0)."
  fi
fi

echo
echo "== Saved per-patch logit files under $TREE_RUNS"
n_logits=$(find "$TREE_RUNS" -type f \( -iname '*logit*' -o -name '*.npy' -o -name '*.npz' \) 2>/dev/null | wc -l)
echo "   $n_logits found (expected 0 -- soft vote / uniform attention will be re-inferred on GPU)"

# ---------------------------------------------------------------- 2. data -----------
echo
echo "== 2. Data"
DATA_OK=1
for ps in "${B1_PATCH_SIZES[@]}"; do
  t="$DATA_DIR/barknet_patches_${ps}.tar"
  if [ -f "$t" ]; then printf '   %-40s %s\n' "$(basename "$t")" "$(du -h "$t" | cut -f1)"
  else echo "   MISSING $t"; DATA_OK=0; fi
done
if [ -f "$RAW_TAR" ]; then printf '   %-40s %s\n' "$(basename "$RAW_TAR")" "$(du -h "$RAW_TAR" | cut -f1)"
else echo "   MISSING $RAW_TAR (needed by the whole-image jobs)"; DATA_OK=0; fi
if [ -d "$RAW_DIR" ]; then
  echo "   $RAW_DIR: $(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) species dirs," \
       "$(find "$RAW_DIR" -type f -iname '*.jpg' | wc -l) jpg"
else echo "   MISSING $RAW_DIR (needed to build the fold file)"; DATA_OK=0; fi

# ---------------------------------------------------------------- 3. folds ----------
echo
echo "== 3. Image-level fold file"
login_venv
trap login_venv_cleanup EXIT
if [ -f "$FOLD_FILE" ]; then
  echo "   exists, NOT rebuilt: $FOLD_FILE"
  python - "$FOLD_FILE" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
print(f"   {len(d)} images | {d.groupby(['species','tree_id']).ngroups} trees | "
      f"{d.species.nunique()} species | per fold: {d.fold.value_counts().sort_index().tolist()}")
PY
elif [ -d "$RAW_DIR" ]; then
  python "$REPO_DIR/src/data_preparation/make_image_folds.py" \
      --raw-root "$RAW_DIR" -c "$CONFIG" --out-dir "$(dirname "$FOLD_FILE")" \
      --expect-images "$EXPECT_IMAGES"
else
  echo "   cannot build: raw photos missing"
fi
FOLDS_OK=$([ -f "$FOLD_FILE" ] && echo 1 || echo 0)
if [ -f "$(dirname "$FOLD_FILE")/image_folds_leakage.csv" ]; then
  echo "   leakage sanity check: $(dirname "$FOLD_FILE")/image_folds_leakage.csv"
fi

# ---------------------------------------------------------------- 4. verdict --------
if [ "${GROUP_OK[B1]}" = 1 ] && [ "${GROUP_OK[B2]}" = 1 ]; then MODE=full; else MODE=minimal; fi
cat > "$PRE/status.env" <<EOF
# written by 00_preflight.sh at $STAMP -- read by submit_v2.sh
B1_OK=${GROUP_OK[B1]}
B2_OK=${GROUP_OK[B2]}
B4_OK=${GROUP_OK[B4]}
DATA_OK=$DATA_OK
FOLDS_OK=$FOLDS_OK
RECOMMENDED_MODE=$MODE
PREFLIGHT_REPORT=$REPORT
EOF
echo
echo "== VERDICT"
if [ "$MODE" = full ]; then
  echo "   All B1/B2 checkpoints present -> MODE=full: 15 training jobs + inference-only Part B"
  echo "   on the existing checkpoints (B4 $( [ "${GROUP_OK[B4]}" = 1 ] && echo included || echo 'SKIPPED: incomplete'))."
else
  echo "   Some B1/B2 checkpoints are gone -> MODE=minimal: 15 training jobs + RETRAIN tree-level"
  echo "   nano 512 two-stage and nano 224 single-stage (5 folds each); Part B runs on those."
  echo "   (If you have a backup of the missing checkpoints, restore them under \$TREE_RUNS --"
  echo "   or point TREE_RUNS at the backup -- and re-run this preflight.)"
fi
[ "$DATA_OK" = 1 ] || echo "   !! data missing -- fix before submitting."
[ "$FOLDS_OK" = 1 ] || echo "   !! no fold file -- fix before submitting."
echo
echo "Report: $REPORT"
echo "Next  : bash cluster/v2/submit_v2.sh            (dry run: prints the plan)"
echo "        GO=1 bash cluster/v2/submit_v2.sh       (submits)"

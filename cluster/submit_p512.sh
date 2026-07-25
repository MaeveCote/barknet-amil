#!/bin/bash
# Submit patch-512: Stage-1 array, then Stage-2 array chained per-fold via aftercorr.
# aftercorr pairs S2 task N with S1 task N (same fold), so each fold's S2 resumes its own
# backbone and starts as soon as that fold's S1 finishes -- independently of other folds.
set -euo pipefail
CL="${REPO_DIR:-$HOME/BarkNet_ML}/cluster"
S1=$(sbatch --parsable "$CL/job_s1_p512.sh")
echo "patch 512 Stage-1 array: $S1"
S2=$(sbatch --parsable --dependency=aftercorr:"$S1" "$CL/job_s2_p512.sh")
echo "patch 512 Stage-2 array: $S2  (aftercorr on $S1)"

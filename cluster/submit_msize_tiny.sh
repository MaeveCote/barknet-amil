#!/bin/bash
# tiny @ 224 model-size ablation: Stage-1 array -> Stage-2 array chained per-fold (aftercorr).
set -euo pipefail
CL="${REPO_DIR:-$HOME/BarkNet_ML}/cluster"
S1=$(sbatch --parsable "$CL/job_s1_tiny_224.sh")
echo "tiny Stage-1 array: $S1"
S2=$(sbatch --parsable --dependency=aftercorr:"$S1" "$CL/job_s2_tiny_224.sh")
echo "tiny Stage-2 array: $S2  (aftercorr on $S1)"

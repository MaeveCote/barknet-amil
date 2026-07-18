#!/bin/bash
# Submit the FULL 6-size x 5-fold nano patch ablation, hands-off.
#
#   bash cluster/submit_cv_ablation.sh
#
# Two tiers by walltime: small (96,160) @ 48h Stage-1, large (224,288,384,512) @ 24h.
# Each tier's Stage-2+test array is chained behind its Stage-1 with aftercorr, so Stage-2
# task N waits on Stage-1 task N (same size,fold) and resumes that exact backbone.
# Concurrency is capped at %5 inside every array. 30 folds total; nothing overwrites
# anything because RUN_NAME carries the fold (abl_p<size>_nano_f<fold>).
set -euo pipefail
CL="${REPO_DIR:-$HOME/BarkNet_ML}/cluster"

S1SM=$(sbatch --parsable "$CL/job_cv_stage1_small.sh")
echo "Stage-1 SMALL (96,160)   array $S1SM  [48h, 10 tasks, %5]"
S2SM=$(sbatch --parsable --dependency=aftercorr:"$S1SM" "$CL/job_cv_stage2_small.sh")
echo "Stage-2 SMALL            array $S2SM  (aftercorr on $S1SM)"

S1LG=$(sbatch --parsable "$CL/job_cv_stage1_large.sh")
echo "Stage-1 LARGE (224..512) array $S1LG  [24h, 20 tasks, %5]"
S2LG=$(sbatch --parsable --dependency=aftercorr:"$S1LG" "$CL/job_cv_stage2_large.sh")
echo "Stage-2 LARGE            array $S2LG  (aftercorr on $S1LG)"

echo
echo "Watch:    squeue -u \$USER"
echo "Results:  \$SCRATCH/runs/abl_p<size>_nano_f<fold>/test/test_summary.json"
echo "Note: a Stage-1 task that fails leaves its Stage-2 in DependencyNeverSatisfied;"
echo "      scancel that one task by hand (aftercorr does not auto-clean)."

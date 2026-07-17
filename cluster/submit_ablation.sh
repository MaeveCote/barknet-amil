#!/bin/bash
# Submit the two-stage patch-size ablation with per-task dependency chaining.
#
#   bash cluster/submit_ablation.sh
#
# Submits the Stage-1 array (3 tasks, 24h each), then the Stage-2 array (3 tasks, 8h),
# with EACH Stage-2 task depending on the SAME-INDEX Stage-1 task via aftercorr. So
# Stage-2 task 0 (patch 224) starts only when Stage-1 task 0 succeeds, independently of
# tasks 1 and 2. A patch size that fails Stage 1 skips its Stage 2 without blocking the
# others.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/BarkNet_ML}"
CLUSTER="$REPO_DIR/cluster"

S1=$(sbatch --parsable "$CLUSTER/job_abl_stage1.sh")
echo "Stage-1 array submitted: $S1  (tasks 0,1,2 = patch 224,288,384)"

# aftercorr:<jobid> ties element N of this array to element N of the dependency array --
# exactly the per-patch pairing we want, not "wait for the whole Stage-1 array".
S2=$(sbatch --parsable --dependency=aftercorr:"$S1" "$CLUSTER/job_abl_stage2.sh")
echo "Stage-2 array submitted: $S2  (each task waits on its matching Stage-1 task)"

echo
echo "Watch:   squeue -u \$USER"
echo "Results: \$SCRATCH/runs/abl_patch{224,288,384}_nano/test/test_summary.json"

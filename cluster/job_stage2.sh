#!/bin/bash
# STAGE 2 + TEST on an EXISTING Stage-1 backbone.
#
# run_chain.sh with STAGES=2 skips Stage 1 and resumes $OUT/pretrain/best_backbone.pth,
# then fine-tunes AMIL and runs the test (AMIL vs voting, full + capped bags). Use this to
# get the image-level number from a backbone a previous STAGES=1 job already produced --
# e.g. the epoch-30 nano backbone from the s1probe run.
#
#   # against the probe backbone (default):
#   sbatch cluster/job_stage2.sh
#
#   # against any other finished Stage-1 run, override RUN_NAME to its directory name:
#   RUN_NAME=abl_patch224_nano sbatch cluster/job_stage2.sh
#
# RUN_NAME MUST match the directory the backbone lives in:
#   $SCRATCH/runs/<RUN_NAME>/pretrain/best_backbone.pth
# and MODEL_SIZE / PATCH_SIZE / INPUT_SIZE MUST match how that backbone was trained, or the
# transfer silently mismatches. Watch the log for "(N tensors transferred)" -- N ~= 160,
# never 0.
#
# Budget: Stage 2 ~15 epochs x ~22 min ~= 5.5 h + test ~= 6.5 h + staging. 8 h is safe.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_s2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.out

set -euo pipefail
mkdir -p "$SCRATCH/logs"

export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_cluster.yaml"

# These MUST match the backbone being resumed.
export PATCH_SIZE=224
export MODEL_SIZE=nano
export INPUT_SIZE=224          # backbone was trained at 224
export FOLD=0                  # same split as Stage 1, or val/test leak differently

export STAGES=2               # Stage 2 + test only; resumes $OUT/pretrain/best_backbone.pth
export EPOCHS_S2=15

# The run directory that already holds .../pretrain/best_backbone.pth.
export RUN_NAME="${RUN_NAME:-s1probe_nano_224}"

# Fail before queuing work if the backbone isn't actually there.
BACKBONE="$SCRATCH/runs/$RUN_NAME/pretrain/best_backbone.pth"
if [ ! -f "$BACKBONE" ]; then
  echo "No backbone at $BACKBONE"
  echo "Set RUN_NAME to the directory holding pretrain/best_backbone.pth."
  exit 1
fi
echo "Resuming backbone: $BACKBONE"

bash "$REPO_DIR/cluster/run_chain.sh"

echo
echo "=============================================================="
echo " Image-level result:"
echo "   cat \$SCRATCH/runs/$RUN_NAME/test/test_summary.json"
echo " Key fields: evaluations.full.amil_accuracy,"
echo "             evaluations.full.amil_vs_amil_vote (the headline McNemar)"
echo "   seff $SLURM_JOB_ID"
echo "=============================================================="

#!/bin/bash
# STAGE 2 + TEST -- patch 384, all 5 folds. Resumes the backbone that job_s1_p384.sh
# produced. Submitted by submit_384.sh with an aftercorr dependency so task N waits on
# Stage-1 task N (same fold).
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_s2_p384
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

FOLD=$SLURM_ARRAY_TASK_ID
export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=384
export MODEL_SIZE=nano
export INPUT_SIZE=224
export FOLD=$FOLD
export STAGES=2
export EPOCHS_S2=15
export RUN_NAME="abl_p384_nano_f${FOLD}"
export EXTRA_ARGS="--set data.split.n_folds=5"

BACKBONE="$SCRATCH/runs/$RUN_NAME/pretrain/best_backbone.pth"
if [ ! -f "$BACKBONE" ]; then
  echo "No backbone at $BACKBONE -- Stage 1 (fold $FOLD) did not finish. Skipping."
  exit 1
fi
echo "Resuming $BACKBONE"
bash "$REPO_DIR/cluster/run_chain.sh"

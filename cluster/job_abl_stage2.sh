#!/bin/bash
# PATCH-SIZE ABLATION -- STAGE 2 + TEST (array, 3 tasks). Each task depends on the matching
# Stage-1 task and resumes its backbone. Submitted by submit_ablation.sh, not directly.
#
# Reads PS from the array index (same order as Stage 1), resumes
# $SCRATCH/runs/abl_patch<PS>_nano/pretrain/best_backbone.pth, fine-tunes AMIL (15 epochs),
# and runs the test (AMIL vs both voting baselines, full + capped bags).
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_abl_s2
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out

set -euo pipefail
mkdir -p "$SCRATCH/logs"

PATCH_SIZES=(224 288 384)
PS=${PATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=$PS
export MODEL_SIZE=nano
export FOLD=0
export INPUT_SIZE=224          # must match Stage 1
export STAGES=2                # Stage 2 + test; resumes the Stage-1 backbone
export EPOCHS_S2=15
export RUN_NAME="abl_patch${PS}_${MODEL_SIZE}"

BACKBONE="$SCRATCH/runs/$RUN_NAME/pretrain/best_backbone.pth"
if [ ! -f "$BACKBONE" ]; then
  echo "No backbone at $BACKBONE -- Stage 1 for patch $PS did not finish."
  exit 1
fi

bash "$REPO_DIR/cluster/run_chain.sh"

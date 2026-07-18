#!/bin/bash
# FULL CV PATCH ABLATION -- STAGE 2 + TEST, small patches. 8h walltime.
# Mirrors job_cv_stage1_small.sh index-for-index so --dependency=aftercorr pairs each
# Stage-2 task with its OWN Stage-1 backbone. Resumes the fold-qualified RUN_NAME. Do not
# submit directly; submit_cv_ablation.sh chains this behind Stage 1.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_cv_s2sm
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

SIZES=(96 160)
PS=${SIZES[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))

export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=$PS
export MODEL_SIZE=nano
export INPUT_SIZE=224
export FOLD=$FOLD
export STAGES=2
export EPOCHS_S2=15
export RUN_NAME="abl_p${PS}_${MODEL_SIZE}_f${FOLD}"

BACKBONE="$SCRATCH/runs/$RUN_NAME/pretrain/best_backbone.pth"
if [ ! -f "$BACKBONE" ]; then
  echo "No backbone at $BACKBONE -- Stage 1 ($RUN_NAME) did not finish. Skipping."
  exit 1
fi

# GUARANTEE k-fold mode. --fold sets fold_index, but n_folds lives only in the config;
# if the config still says n_folds:null every "fold" silently trains the SAME holdout
# split. Force it here so fold rotation actually happens regardless of the config file.
export EXTRA_ARGS="--set data.split.n_folds=5"

bash "$REPO_DIR/cluster/run_chain.sh"

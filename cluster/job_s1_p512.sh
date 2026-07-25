#!/bin/bash
# STAGE 1 ONLY -- patch 512, all 5 folds. Pretrains the backbone; Stage 2 is a separate
# array (job_s2_p512.sh) chained behind this via aftercorr in submit_512.sh.
# Walltime = measured 40-epoch Stage-1 time + margin, kept short for gpubackfill.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_s1_p512
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=7:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

FOLD=$SLURM_ARRAY_TASK_ID
export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=512
export MODEL_SIZE=nano
export INPUT_SIZE=224
export FOLD=$FOLD
export STAGES=1
export EPOCHS_S1=40
export RUN_NAME="abl_p512_nano_f${FOLD}"
export EXTRA_ARGS="--set data.split.n_folds=5"

bash "$REPO_DIR/cluster/run_chain.sh"

#!/bin/bash
# MODEL-SIZE ABLATION -- Stage 1, tiny @ patch 224, 5 folds. Uses the SAME 224 patch tar as
# the patch-size study, so field of view is fixed and BACKBONE CAPACITY is the only variable.
# Epochs: 22 (tiny converges within this; tiny trimmed to fit the 24h wall).
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_s1_tiny224
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

FOLD=$SLURM_ARRAY_TASK_ID
export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=224
export MODEL_SIZE=tiny
export INPUT_SIZE=224
export FOLD=$FOLD
export STAGES=1
export EPOCHS_S1=22
export RUN_NAME="abl_msize_tiny_224_f${FOLD}"
export EXTRA_ARGS="--set data.split.n_folds=5 --set pretrain.drop_path_rate=0.35"

bash "$REPO_DIR/cluster/run_chain.sh"

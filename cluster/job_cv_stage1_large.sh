#!/bin/bash
# FULL CV PATCH ABLATION -- STAGE 1, LARGE patches (224, 288, 384, 512). 24h walltime.
# 4 sizes x 5 folds = 20 tasks, max 5 concurrent. Do not submit directly.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_cv_s1lg
#SBATCH --array=0-19%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

SIZES=(224 288 384 512)
PS=${SIZES[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))

export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=$PS
export MODEL_SIZE=nano
export INPUT_SIZE=224
export FOLD=$FOLD
export STAGES=1
export EPOCHS_S1=40
export RUN_NAME="abl_p${PS}_${MODEL_SIZE}_f${FOLD}"

# GUARANTEE k-fold mode. --fold sets fold_index, but n_folds lives only in the config;
# if the config still says n_folds:null every "fold" silently trains the SAME holdout
# split. Force it here so fold rotation actually happens regardless of the config file.
export EXTRA_ARGS="--set data.split.n_folds=5"

bash "$REPO_DIR/cluster/run_chain.sh"

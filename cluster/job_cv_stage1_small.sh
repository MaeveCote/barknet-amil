#!/bin/bash
# FULL CV PATCH ABLATION -- STAGE 1, SMALL patches (96, 160). 48h walltime.
# Small patches produce far more patches/image -> more per-epoch work -> 48h (224 uses the
# full 24h, so 96/160 get double). 2 sizes x 5 folds = 10 tasks, max 5 concurrent.
# Do not submit directly; use submit_cv_ablation.sh.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_cv_s1sm
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
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
export INPUT_SIZE=224            # ablation knob: all sizes resized to 224, only FOV varies
export FOLD=$FOLD
export STAGES=1
export EPOCHS_S1=40
# Fold-qualified RUN_NAME -- THIS is what stops fold N overwriting fold N-1. run_chain.sh
# writes to $SCRATCH/runs/$RUN_NAME, so the fold MUST be in the name.
export RUN_NAME="abl_p${PS}_${MODEL_SIZE}_f${FOLD}"

# GUARANTEE k-fold mode. --fold sets fold_index, but n_folds lives only in the config;
# if the config still says n_folds:null every "fold" silently trains the SAME holdout
# split. Force it here so fold rotation actually happens regardless of the config file.
export EXTRA_ARGS="--set data.split.n_folds=5"

bash "$REPO_DIR/cluster/run_chain.sh"

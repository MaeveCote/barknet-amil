#!/bin/bash
# FULL CHAIN (Stage 1 + Stage 2 + test) -- patch 512, all 5 folds. These sizes never ran
# Stage 1, so there's no backbone to resume: STAGES=all does everything in one job, which
# also means no aftercorr dependency and no orphan risk.
#
# Walltime right-sized from your measured 40-epoch single-split times (sacct 16499217):
#   288 -> 14:09 measured -> 18h ; 384 -> 07:56 -> 11h ; 512 -> ~4.5h est -> 7h
# Short enough to be gpubackfill-eligible, unlike the old 24h requests.
#
#   sbatch cluster/job_full_p512.sh
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_full_p512
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
export STAGES=all
export EPOCHS_S1=40
export EPOCHS_S2=15
export RUN_NAME="abl_p512_nano_f${FOLD}"
export EXTRA_ARGS="--set data.split.n_folds=5"

bash "$REPO_DIR/cluster/run_chain.sh"

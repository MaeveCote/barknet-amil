#!/bin/bash
# PATCH-SIZE ABLATION -- STAGE 1 (array, 3 tasks: 224 / 288 / 384).
#
# Do not submit this directly. Use submit_ablation.sh, which chains each task's Stage 2
# behind it with --dependency. (Submitting alone just runs the three Stage-1 backbones.)
#
# Each task trains one nano backbone for the full 40-epoch cosine (early stopping disabled
# in the ablation config), resized to INPUT_SIZE=224 so field of view is the only variable.
# ~21h at 31.4 min/epoch; 24h wall gives margin. best_backbone.pth is checkpointed every
# epoch, so a timeout still leaves a usable backbone.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_abl_s1
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out

set -euo pipefail
mkdir -p "$SCRATCH/logs"

PATCH_SIZES=(224 288 384)
PS=${PATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

export REPO_DIR="$HOME/BarkNet_ML"
# Ablation config: 40 epochs, early stopping disabled.
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=$PS
export MODEL_SIZE=nano
export FOLD=0
export INPUT_SIZE=224          # THE ABLATION KNOB: all arms resized to 224; only FOV varies
export STAGES=1                # Stage 1 only; Stage 2 is the dependent job
export EPOCHS_S1=40
export RUN_NAME="abl_patch${PS}_${MODEL_SIZE}"

bash "$REPO_DIR/cluster/run_chain.sh"

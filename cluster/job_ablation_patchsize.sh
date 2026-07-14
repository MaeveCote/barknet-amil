#!/bin/bash
# PATCH-SIZE ABLATION -- three identical nano runs that differ only in the patch size the
# data was cut at (224 / 288 / 384). Everything else (model, seed, split, fold, epochs,
# hyperparameters) is held fixed, so any difference in the final numbers is attributable
# to the patch size.
#
# Submit ONLY after job_smoke.sh has passed and you have re-set --time/--mem from `seff`.
#
#   sbatch cluster/job_ablation_patchsize.sh
#
# Each array task is one independent single-GPU job. %3 lets all three run at once; drop
# to %1 if you would rather be gentle on fair-share.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_patchabl
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out

set -euo pipefail

PATCH_SIZES=(224 288 384)
PS=${PATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

export REPO_DIR="$HOME/BarkNet_ML"
export PATCH_SIZE=$PS
export MODEL_SIZE=nano
export FOLD=0
export EPOCHS_S1=20
export EPOCHS_S2=15
export RUN_NAME="abl_patch${PS}_${MODEL_SIZE}"

# Native resolution: a 384px patch is fed to the network at 384px. That means the three
# runs differ in BOTH field of view and compute. If you want to isolate field of view at
# constant compute, uncomment the next line so every patch is resized to 224.
# export INPUT_SIZE=224

# Bigger patches -> fewer patches per image -> smaller bags, but each patch costs ~2.9x
# the activations at 384 vs 224. If 384 OOMs, lower the bag cap for that task only:
# [ "$PS" = "384" ] && export EXTRA_ARGS="--set data.max_patches_per_bag=64"

bash "$REPO_DIR/cluster/run_chain.sh"

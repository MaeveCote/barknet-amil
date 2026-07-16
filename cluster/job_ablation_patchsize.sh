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
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out

set -euo pipefail
mkdir -p "$SCRATCH/logs"

PATCH_SIZES=(224 288 384)
PS=${PATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_cluster.yaml"
export PATCH_SIZE=$PS
export MODEL_SIZE=nano
export FOLD=0
# Set EPOCHS_S1 from the job_stage1_probe.sh result (stop_epoch + patience), NOT this
# placeholder. All three arms MUST use the same value or the ablation is confounded.
export EPOCHS_S1=20
export EPOCHS_S2=15
export RUN_NAME="abl_patch${PS}_${MODEL_SIZE}"

# THE ABLATION KNOB. Resize every patch to 224 so the ONLY thing that varies across the
# three arms is field of view (how much trunk a patch covers), not compute or feature
# scale. Per your decision: bigger patches resized to 224. Comment this out only if you
# deliberately want native-resolution (field-of-view AND compute both vary).
export INPUT_SIZE=224

# Bigger patches -> fewer patches per image -> smaller bags, but each patch costs ~2.9x
# the activations at 384 vs 224. If 384 OOMs, lower the bag cap for that task only:
# [ "$PS" = "384" ] && export EXTRA_ARGS="--set data.max_patches_per_bag=64"

bash "$REPO_DIR/cluster/run_chain.sh"

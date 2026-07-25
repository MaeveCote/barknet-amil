#!/bin/bash
# STAGE 2 + TEST -- patch 160, all 5 folds. Resumes the EXISTING Stage-1 backbones in
# $SCRATCH/runs/abl_p160_nano_f<fold>/pretrain/best_backbone.pth (already trained).
#
# No Stage 1, no dependency: the backbones are on disk. No concurrency cap (user choice:
# burn fair-share fast on the exponential-decay recovery). Distinct array per patch size so
# each size's completion is visible separately in squeue.
#
#   sbatch cluster/job_s2_p160.sh
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_s2_p160
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

FOLD=$SLURM_ARRAY_TASK_ID          # 0..4, one task per fold

export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_ablation.yaml"
export PATCH_SIZE=160
export MODEL_SIZE=nano
export INPUT_SIZE=224
export FOLD=$FOLD
export STAGES=2                     # resume backbone, fine-tune AMIL, test
export EPOCHS_S2=15
export RUN_NAME="abl_p160_nano_f${FOLD}"

# MUST re-assert kfold mode: config has n_folds:null, so without this the Stage-2/test split
# falls back to holdout and would NOT match the fold the backbone was pretrained on.
export EXTRA_ARGS="--set data.split.n_folds=5"

BACKBONE="$SCRATCH/runs/$RUN_NAME/pretrain/best_backbone.pth"
if [ ! -f "$BACKBONE" ]; then
  echo "No backbone at $BACKBONE -- cannot run Stage 2 for $RUN_NAME."
  exit 1
fi
echo "Resuming $BACKBONE"

bash "$REPO_DIR/cluster/run_chain.sh"

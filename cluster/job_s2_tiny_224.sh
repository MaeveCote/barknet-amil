#!/bin/bash
# MODEL-SIZE ABLATION -- Stage 2 + test, tiny @ patch 224, 5 folds. Resumes the Stage-1
# backbone; chained via aftercorr in submit_msize_tiny.sh.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_s2_tiny224
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
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
export STAGES=2
export EPOCHS_S2=15
export RUN_NAME="abl_msize_tiny_224_f${FOLD}"
export EXTRA_ARGS="--set data.split.n_folds=5 --set pretrain.drop_path_rate=0.35"

BACKBONE="$SCRATCH/runs/$RUN_NAME/pretrain/best_backbone.pth"
if [ ! -f "$BACKBONE" ]; then
  echo "No backbone at $BACKBONE -- Stage 1 (tiny fold $FOLD) did not finish. Skipping."
  exit 1
fi
echo "Resuming $BACKBONE"
bash "$REPO_DIR/cluster/run_chain.sh"

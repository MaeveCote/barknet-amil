#!/bin/bash
# SMOKE TEST -- 2 epochs per stage. Proves the whole chain on real data before any
# expensive run: venv builds, offline weights load, tar stages, patch_root resolves, the
# tree-level split holds, Stage 1 -> Stage 2 transfer works, test writes its report.
# It is NOT meant to produce a meaningful accuracy.
#
#   sbatch cluster/job_smoke.sh
#
# Note on --time: partitions are tiered, and <= 3h jobs are eligible for the most nodes
# (plus gpubackfill). Keeping this at 3h is the single biggest lever on queue wait.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail

export REPO_DIR="$HOME/BarkNet_ML"
export PATCH_SIZE=224
export MODEL_SIZE=nano          # the size the ablation will use
export EPOCHS_S1=2
export EPOCHS_S2=2
export RUN_NAME="smoke_${MODEL_SIZE}_${PATCH_SIZE}"

bash "$REPO_DIR/cluster/run_chain.sh"

#!/bin/bash
# STAGE-1 EPOCH-BUDGET PROBE  --  run this ONE job before anything else.
#
# Purpose: find out where Stage 1 ACTUALLY early-stops. Everything downstream (the patch
# ablation, the 5-fold CV, every --time estimate) is currently budgeted against a guess of
# 50 epochs, borrowed from Cui et al. -- who trained ConvNeXt FROM SCRATCH. This project
# fine-tunes from FCMAE ImageNet weights, which converges far faster: the local run peaked
# at epoch 9, and the cluster smoke test was already at 81.9% after epoch 1.
#
# If this stops at epoch ~15, the whole 24h scheduling problem evaporates and the ablation
# gets 3x cheaper. If it really does run to 50, at least the number is measured.
#
#   sbatch cluster/job_stage1_probe.sh
#
# Budget: measured 34.5 min/epoch (nano @ 224, bf16 off). 50 epochs = ~29 h, so this asks
# for 24 h and relies on early stopping to finish sooner. If it TIMEOUTs, the best backbone
# is still on $SCRATCH (checkpointed every epoch) and Stage 2 can proceed from it.
#SBATCH --account=def-oberman_gpu
#SBATCH --job-name=bark_s1probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.out

set -euo pipefail
mkdir -p "$SCRATCH/logs"

export REPO_DIR="$HOME/BarkNet_ML"
export CONFIG="$REPO_DIR/configs/config_cluster.yaml"
export PATCH_SIZE=224
export MODEL_SIZE=nano
export FOLD=0

export STAGES=1            # Stage 1 ONLY. Stage 2 is a separate, much shorter job.
export EPOCHS_S1=35        # ceiling, not a target -- early stopping should fire first

export RUN_NAME="s1probe_${MODEL_SIZE}_${PATCH_SIZE}"

bash "$REPO_DIR/cluster/run_chain.sh"

echo
echo "=============================================================="
echo " NEXT: read the epoch where it stopped, then set EPOCHS_S1 in"
echo " job_ablation_patchsize.sh from it (stop_epoch + patience)."
echo "   cat \$SCRATCH/runs/$RUN_NAME/pretrain/pretrain_summary.json"
echo "   seff $SLURM_JOB_ID"
echo "=============================================================="

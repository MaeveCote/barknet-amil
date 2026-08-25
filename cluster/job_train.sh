#!/bin/bash
# Generic SLURM array job: Stage 1 -> Stage 2 -> test, chained via run_chain.sh.
#
# One parametrized script for every training ablation in the paper (patch size, model
# size, 1- vs 2-stage, CV fold) instead of a near-duplicate script per experiment. Set
# the knobs via --export at submit time; everything not set falls back to
# configs/config.yaml's defaults (5-fold CV, nano @ 224).
#
# EXAMPLES (5-fold array, one task per fold: SLURM_ARRAY_TASK_ID -> FOLD)
#
#   # patch-size ablation, nano @ patch 288, full 40-epoch cosine (early stopping off,
#   # matching the paper's ablation methodology instead of config.yaml's local default)
#   sbatch --export=ALL,PATCH_SIZE=288,MODEL_SIZE=nano,\
#     EXTRA_ARGS="--set pretrain.early_stopping_patience=999 --set pretrain.min_epochs=999" \
#     cluster/job_train.sh
#
#   # model-size ablation, tiny @ 224 (pico/nano the same, swap MODEL_SIZE)
#   sbatch --export=ALL,PATCH_SIZE=224,MODEL_SIZE=tiny cluster/job_train.sh
#
#   # single-stage control: no Stage-1 pretraining, uniform LR straight from ImageNet
#   # into Stage 2 (see helper/model_wrapper.py's docstring for why the from-ImageNet
#   # base_lr is used here rather than Stage-2's normal fine-tune rate)
#   sbatch --export=ALL,PATCH_SIZE=224,MODEL_SIZE=nano,STAGES=2,EPOCHS_S2=40,\
#     EXTRA_ARGS="--set model.backbone_checkpoint=null --set train.optimizer.lr_multiplier=1 \
#                 --set train.optimizer.base_lr=1.0062785194709649e-4" \
#     cluster/job_train.sh
#
#   # "final model" run: single-stage nano @ 512 (see the example above for the
#   # single-stage overrides), just change PATCH_SIZE=512
#
#   # smoke test: 2 epochs per stage, proves the chain end-to-end
#   sbatch --export=ALL,PATCH_SIZE=224,MODEL_SIZE=nano,EPOCHS_S1=2,EPOCHS_S2=2,RUN_NAME=smoke \
#     --array=0 cluster/job_train.sh
#
#   # sweep every patch size in the paper (loop on the login node, one submission each):
#   for ps in 96 160 224 288 384 512; do
#     sbatch --export=ALL,PATCH_SIZE=$ps,MODEL_SIZE=nano cluster/job_train.sh
#   done
#
#   # split Stage 1 (long) and Stage 2 (short) across queue tiers, same fold pairing --
#   # STAGES=2 resumes $SCRATCH/runs/$RUN_NAME/pretrain/best_backbone.pth, so RUN_NAME
#   # (which defaults from MODEL_SIZE/PATCH_SIZE/FOLD) must match between the two:
#   S1=$(sbatch --parsable --export=ALL,PATCH_SIZE=224,STAGES=1 cluster/job_train.sh)
#   sbatch --dependency=aftercorr:"$S1" --export=ALL,PATCH_SIZE=224,STAGES=2 cluster/job_train.sh
#
# Walltime defaults to a conservative 48h covering the slowest configuration (small
# patches -> more patches/image -> longer epochs). Pass --time to tighten it once you've
# measured your own epoch time for a given size (the paper used 17-48h depending on size).
#
# Before submitting: replace YOUR_ACCOUNT below (or `sbatch --account=<you> ...`), and run
# 00_prefetch_weights.sh once on a login node (compute nodes have no internet).
#SBATCH --account=YOUR_ACCOUNT_gpu
#SBATCH --job-name=bark_train
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.out
set -euo pipefail
mkdir -p "$SCRATCH/logs"

export REPO_DIR="${REPO_DIR:-$HOME/BarkNet_ML}"
export CONFIG="${CONFIG:-$REPO_DIR/configs/config.yaml}"
export PATCH_SIZE="${PATCH_SIZE:?set PATCH_SIZE (e.g. 96,160,224,288,384,512)}"
export MODEL_SIZE="${MODEL_SIZE:-nano}"
export INPUT_SIZE="${INPUT_SIZE:-224}"
export FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-0}}"
export STAGES="${STAGES:-all}"
export EPOCHS_S1="${EPOCHS_S1:-40}"
export EPOCHS_S2="${EPOCHS_S2:-15}"
export RUN_NAME="${RUN_NAME:-${MODEL_SIZE}_p${PATCH_SIZE}_f${FOLD}}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

bash "$REPO_DIR/cluster/run_chain.sh"

#!/bin/bash
# Generic SLURM array job: Stage 1 -> Stage 2 -> test, chained via run_chain.sh.
#
# One parametrized script for every training ablation in the paper (patch size, model
# size, 1- vs 2-stage, CV fold) instead of a near-duplicate script per experiment. Set
# the knobs via --export at submit time; everything not set falls back to
# configs/config.yaml's defaults (5-fold CV, nano @ 224, the paper's hyperparameters).
#
# RUN NAMES. Each array task writes to $OUT_ROOT/<RUN_PREFIX>_f<FOLD> (OUT_ROOT defaults
# to $SCRATCH/runs). cluster/compile_results.py only picks up the paper's naming patterns,
# so the examples below set RUN_PREFIX to the name the paper's runs used.
# !! Those names already exist on the cluster from the original runs: to REPRODUCE a row,
# !! point OUT_ROOT somewhere new (e.g. OUT_ROOT=$SCRATCH/runs_repro) or the original
# !! checkpoints and results are overwritten.
#
# EXAMPLES (5-fold array, one task per fold: SLURM_ARRAY_TASK_ID -> FOLD). Each one
# reproduces the settings of the corresponding paper run (input always resized to 224):
#
#   # patch-size ablation, nano @ 288 (96/160/224/384/512 the same, swap PATCH_SIZE)
#   sbatch --export=ALL,PATCH_SIZE=288,RUN_PREFIX=abl_p288_nano cluster/job_train.sh
#
#   # model-size ablation @ 224. Stage-1 drop-path follows capacity (0.15 pico,
#   # 0.25 nano = the config default, 0.35 tiny); Stage 2 keeps model.drop_path_rate.
#   # The paper's tiny Stage 1 ran 22 epochs (walltime), not 40.
#   sbatch --export=ALL,PATCH_SIZE=224,MODEL_SIZE=pico,RUN_PREFIX=abl_msize_pico_224,\
#   EXTRA_ARGS="--set pretrain.drop_path_rate=0.15" cluster/job_train.sh
#   sbatch --export=ALL,PATCH_SIZE=224,MODEL_SIZE=tiny,EPOCHS_S1=22,RUN_PREFIX=abl_msize_tiny_224,\
#   EXTRA_ARGS="--set pretrain.drop_path_rate=0.35" cluster/job_train.sh
#
#   # single-stage control (H3): no Stage 1, AMIL trained from ImageNet for 40 epochs.
#   # See the SINGLE_STAGE block in run_chain.sh for exactly what this sets (and why the
#   # learning rate is the config's Stage-2 one, not a uniform 1e-4).
#   sbatch --export=ALL,PATCH_SIZE=224,SINGLE_STAGE=1,RUN_PREFIX=abl_1stage_nano_224 \
#     --time=24:00:00 cluster/job_train.sh
#
#   # "final model": single-stage nano @ 512 (the paper's recommended configuration)
#   sbatch --export=ALL,PATCH_SIZE=512,SINGLE_STAGE=1,RUN_PREFIX=final_nano_p512_1stage \
#     --time=16:00:00 cluster/job_train.sh
#
#   # smoke test: 2 epochs per stage, proves the chain end-to-end
#   sbatch --export=ALL,PATCH_SIZE=224,EPOCHS_S1=2,EPOCHS_S2=2,RUN_PREFIX=smoke \
#     --array=0 cluster/job_train.sh
#
#   # split Stage 1 (long) and Stage 2 (short) across queue tiers, same fold pairing --
#   # STAGES=2 resumes $OUT_ROOT/$RUN_NAME/pretrain/best_backbone.pth, so RUN_PREFIX
#   # must match between the two (this is how the paper's two-stage runs were submitted):
#   S1=$(sbatch --parsable --export=ALL,PATCH_SIZE=224,RUN_PREFIX=abl_p224_nano,STAGES=1 cluster/job_train.sh)
#   sbatch --dependency=aftercorr:"$S1" --time=8:00:00 \
#     --export=ALL,PATCH_SIZE=224,RUN_PREFIX=abl_p224_nano,STAGES=2 cluster/job_train.sh
#
# Walltime defaults to a conservative 48h covering the slowest configuration (small
# patches -> more patches/image -> longer epochs). Pass --time to tighten it once you've
# measured your own epoch time for a given size (the paper used 8-48h depending on stage
# and size).
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
export SINGLE_STAGE="${SINGLE_STAGE:-0}"
export EPOCHS_S1="${EPOCHS_S1:-40}"
if [ "$SINGLE_STAGE" = "1" ]; then
  export EPOCHS_S2="${EPOCHS_S2:-40}"
else
  export EPOCHS_S2="${EPOCHS_S2:-15}"
fi
export RUN_PREFIX="${RUN_PREFIX:-${MODEL_SIZE}_p${PATCH_SIZE}}"
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_f${FOLD}}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

bash "$REPO_DIR/cluster/run_chain.sh"

#!/bin/bash
# Shared settings for every v2 script. Sourced, never executed. Override any of these in
# the environment before running a v2 script.
#
# All NEW outputs go under $V2_ROOT. The original tree-level runs under $TREE_RUNS are
# only ever READ (checkpoints, summaries); nothing in this folder writes there.

export REPO_DIR="${REPO_DIR:-$HOME/BarkNet_ML}"
export CONFIG="${CONFIG:-$REPO_DIR/configs/config.yaml}"

export V2_ROOT="${V2_ROOT:-$SCRATCH/results/v2}"
export TREE_RUNS="${TREE_RUNS:-$SCRATCH/runs}"           # the paper's tree-level runs (read-only)
export DATA_DIR="${DATA_DIR:-$SCRATCH/data}"
export RAW_DIR="${RAW_DIR:-$DATA_DIR/barknet_raw}"         # 23 species folders of raw photos
export RAW_TAR="${RAW_TAR:-$DATA_DIR/barknet_raw.tar}"     # same photos, staged to node SSD by jobs

export FOLD_FILE="${FOLD_FILE:-$V2_ROOT/folds/image_folds.csv}"
export EXPECT_IMAGES="${EXPECT_IMAGES:-23271}"             # BarkNet 1.0, 20 species

export ACCOUNT_GPU="${ACCOUNT_GPU:-def-oberman_gpu}"
export ACCOUNT_CPU="${ACCOUNT_CPU:-def-oberman_cpu}"

# The tree-level checkpoints Part B re-scores (inference only).
B1_PATCH_SIZES=(160 224 288 384 512)                     # abl_p<N>_nano_f<k>   two-stage
B2_RUN_PREFIX="abl_1stage_nano_224"                       # single-stage, no Stage-1 backbone
B4_RUN_PREFIXES=(abl_msize_pico_224 abl_msize_tiny_224)  # optional capacity runs
FOLDS=(0 1 2 3 4)

# Python on the LOGIN node (no GPU): a throwaway venv from the Alliance wheelhouse.
login_venv() {
  module load StdEnv/2023 python/3.11 2>/dev/null || module load python/3.11
  LOGIN_VENV="/tmp/barknet_v2_venv_$$"
  virtualenv --no-download "$LOGIN_VENV" >/dev/null
  # shellcheck disable=SC1091
  source "$LOGIN_VENV/bin/activate"
  pip install --no-index --quiet --upgrade pip
  pip install --no-index --quiet numpy pandas scipy scikit-learn pyyaml openpyxl
}
login_venv_cleanup() {
  deactivate 2>/dev/null || true
  [ -n "${LOGIN_VENV:-}" ] && rm -rf "$LOGIN_VENV"
}

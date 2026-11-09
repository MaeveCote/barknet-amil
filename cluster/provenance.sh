#!/bin/bash
# Sourced, never executed. Defines write_provenance <out_dir> [label]: appends who/what/when
# for one job step to <out_dir>/provenance.txt -- git commit (+ dirty files), config path
# and checksum, seed, every pipeline knob, and library versions. Each stage also dumps its
# fully-resolved config (resolved_config.yaml) next to its outputs, which carries the seed
# and every hyperparameter actually used.

write_provenance() {
  # Bookkeeping must never kill a job: run in a subshell without errexit/pipefail.
  ( set +e +o pipefail; _write_provenance "$@" ) || true
}

_write_provenance() {
  local out="$1" label="${2:-run}"
  mkdir -p "$out"
  local repo="${REPO_DIR:-$HOME/BarkNet_ML}" cfg="${CONFIG:-}"
  {
    echo "=== $label | $(date -Iseconds) | host $(hostname) ==="
    echo "slurm_job        : ${SLURM_JOB_ID:-none}  array ${SLURM_ARRAY_JOB_ID:-}-${SLURM_ARRAY_TASK_ID:-}"
    echo "git_commit       : $(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_branch       : $(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    local dirty
    dirty=$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null | tr '\n' ' ')
    echo "git_dirty        : ${dirty:-clean}"
    echo "config           : $cfg  md5 $(md5sum "$cfg" 2>/dev/null | cut -d' ' -f1)"
    echo "seed (config)    : $(grep -m1 -E '^\s*seed:' "$cfg" 2>/dev/null | awk '{print $2}')"
    for v in RUN_NAME PATCH_SIZE MODEL_SIZE INPUT_SIZE FOLD STAGES SINGLE_STAGE EPOCHS_S1 \
             EPOCHS_S2 PATCH_TAR PATCH_ROOT FOLD_FILE EXTRA_ARGS; do
      printf '%-17s: %s\n' "$v" "${!v:-}"
    done
    if [ -n "${PY:-}" ]; then
      echo "libs             : $($PY -c 'import sys,torch,timm;print("python",sys.version.split()[0],"torch",torch.__version__,"timm",timm.__version__)' 2>/dev/null)"
    fi
    echo
  } >> "$out/provenance.txt"
}

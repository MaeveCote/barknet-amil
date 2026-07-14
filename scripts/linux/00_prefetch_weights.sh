#!/bin/bash
# Run this ON THE LOGIN NODE (it needs internet). Compute nodes on Rorqual/Narval have none.
#
#   bash cluster/00_prefetch_weights.sh
#
# Downloads the ConvNeXt-V2 ImageNet (FCMAE ft_in1k) weights into the HF cache under
# $HOME, so the jobs can load them with HF_HUB_OFFLINE=1. Idempotent: re-running is a
# no-op once the files are cached.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

module load python/3.11

# A throwaway venv on the login node just for the download (small, fast).
TMPENV=$(mktemp -d)
virtualenv --no-download "$TMPENV/venv" >/dev/null
source "$TMPENV/venv/bin/activate"
pip install --no-index --quiet --upgrade pip
pip install --no-index --quiet torch timm

python - <<'PY'
import os, timm
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
for size in ["pico", "nano", "tiny", "base"]:
    tag = f"convnextv2_{size}.fcmae_ft_in1k"
    m = timm.create_model(tag, pretrained=True, num_classes=0)
    print(f"cached {tag}  (num_features={m.num_features})")
PY

deactivate
rm -rf "$TMPENV"
echo
echo "Weights cached under $HF_HOME"
echo "Jobs must export: HF_HOME=$HF_HOME  HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1"

#!/bin/bash
# Run ONCE on the LOGIN NODE (which has internet) before submitting the model-size ablation.
# Caches pico + tiny ImageNet weights into $HF_HOME so the offline compute nodes can load
# them. Without this, every job dies at model construction with a network error (compute
# nodes have no internet).
set -euo pipefail
module load StdEnv/2023 python/3.11 opencv/4.13.0 2>/dev/null || module load python/3.11
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Use the same venv the jobs use if it exists, else a quick temp one. timm must be importable.
python - << 'PY'
import timm
for tag in ["convnextv2_pico.fcmae_ft_in1k", "convnextv2_tiny.fcmae_ft_in1k"]:
    print("caching", tag, "...")
    timm.create_model(tag, pretrained=True)
    print("  ok")
print("pico + tiny weights cached into HF_HOME")
PY

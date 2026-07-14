"""Config loading with environment-variable expansion and CLI overrides.

Two problems this solves:

1. **YAML cannot see the environment.** On SLURM, the patch data lives in
   ``$SLURM_TMPDIR`` (node-local SSD, path differs per job), so no static config can
   name it. Every string in the config is passed through :func:`os.path.expandvars`,
   so ``patch_root: ${SLURM_TMPDIR}/patches_224/train`` now resolves at load time.
   Undefined variables are left verbatim (Windows-safe).

2. **Per-run overrides without rewriting configs.** ``--patch-root``, ``--epochs``,
   ``--output-dir`` etc. override the loaded config in memory, plus a generic
   ``--set a.b.c=value`` escape hatch for anything else. The config file on disk is
   never modified; the *resolved* config is dumped into the run's output dir so a run
   is always reproducible from its own artefacts.
"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Stage -> where its output_dir lives in the config tree.
_STAGE_OUTPUT_KEY = {
    "pretrain": "pretrain.output_dir",
    "train": "train.output_dir",
    "test": "test.output_dir",
    "tune": "tune.output_dir",
}
_STAGE_EPOCH_KEY = {
    "pretrain": "pretrain.epochs",
    "train": "train.epochs",
}


# --------------------------------------------------------------------------- #
# Env expansion + dotted access
# --------------------------------------------------------------------------- #
def expand_env(obj: Any) -> Any:
    """Recursively expand ``$VAR`` / ``${VAR}`` inside every string of a config."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    return obj


def set_in(cfg: Dict, dotted: str, value: Any) -> None:
    """Set ``cfg["a"]["b"]["c"] = value`` from the dotted path ``"a.b.c"``."""
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def get_in(cfg: Dict, dotted: str, default=None) -> Any:
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# --------------------------------------------------------------------------- #
# Argparse plumbing
# --------------------------------------------------------------------------- #
def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags shared by pretrain_backbone.py / train_model.py / test_model.py."""
    parser.add_argument("-c", "--config", default="configs/config.yaml", type=str)
    parser.add_argument("--patch-root", default=None,
                        help="Override data.patch_root (e.g. $SLURM_TMPDIR/patches_224/train).")
    parser.add_argument("--output-dir", default=None,
                        help="Override this stage's output_dir. When given, the directory is "
                             "used as-is (no auto-incrementing suffix).")
    parser.add_argument("--epochs", default=None, type=int, help="Override this stage's epoch count.")
    parser.add_argument("--model-size", default=None, type=str, help="Override model.size.")
    parser.add_argument("--input-size", default=None, type=int, help="Override augmentation.input_size.")
    parser.add_argument("--num-workers", default=None, type=int, help="Override data.num_workers.")
    parser.add_argument("--device", default=None, type=str, help="Override project.device.")
    parser.add_argument("--seed", default=None, type=int, help="Override project.seed.")
    parser.add_argument("--fold", default=None, type=int,
                        help="Override data.split.fold_index (cross-validation fold to hold out).")
    parser.add_argument("--backbone-checkpoint", default=None,
                        help="Override model.backbone_checkpoint (Stage-1 -> Stage-2 transfer).")
    parser.add_argument("--checkpoint", default=None, help="Override test.checkpoint.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Generic dotted override, repeatable. Value is parsed as YAML: "
                             "--set data.max_patches_per_bag=128 --set model.weights=none")
    return parser


def load_config(args: argparse.Namespace, stage: str) -> Dict:
    """Load the YAML, expand env vars, then apply CLI overrides for ``stage``."""
    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = expand_env(cfg)

    named = {
        "data.patch_root": getattr(args, "patch_root", None),
        "model.size": getattr(args, "model_size", None),
        "augmentation.input_size": getattr(args, "input_size", None),
        "data.num_workers": getattr(args, "num_workers", None),
        "project.device": getattr(args, "device", None),
        "project.seed": getattr(args, "seed", None),
        "data.split.fold_index": getattr(args, "fold", None),
        "model.backbone_checkpoint": getattr(args, "backbone_checkpoint", None),
        "test.checkpoint": getattr(args, "checkpoint", None),
    }
    if getattr(args, "output_dir", None) and stage in _STAGE_OUTPUT_KEY:
        named[_STAGE_OUTPUT_KEY[stage]] = args.output_dir
    if getattr(args, "epochs", None) and stage in _STAGE_EPOCH_KEY:
        named[_STAGE_EPOCH_KEY[stage]] = args.epochs

    applied = []
    for key, value in named.items():
        if value is not None:
            set_in(cfg, key, expand_env(value))
            applied.append(f"{key}={value}")

    for item in getattr(args, "overrides", []) or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item!r}")
        key, raw = item.split("=", 1)
        value = expand_env(yaml.safe_load(raw))
        set_in(cfg, key.strip(), value)
        applied.append(f"{key.strip()}={value!r}")

    if applied:
        print("CLI overrides applied: " + ", ".join(applied))

    cfg["_meta"] = {"config_path": str(config_path), "stage": stage}
    return cfg


def dump_config(cfg: Dict, result_dir: Path, name: str = "resolved_config.yaml") -> Path:
    """Write the fully-resolved config next to the run's outputs (reproducibility)."""
    out = Path(result_dir) / name
    payload = {k: v for k, v in cfg.items() if k != "_meta"}
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out

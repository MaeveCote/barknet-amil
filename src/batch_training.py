"""batch_training.py

Orchestrates the full BarkNet-AMIL pipeline across a sweep of (model size x patch
size) experiments:

    Stage-1 search -> Stage-1 pretrain -> Stage-2 search -> Stage-2 fine-tune -> test

Each step's output (tuned hyperparameters, checkpoints) is fed into the next step
through one evolving temp config per experiment. The base config on disk is never
modified; everything goes through ``<experiment_dir>/temp_config.yaml``.

Each step can be switched on/off independently via ``steps:`` in the batch config,
so you can e.g. tune+train the backbone once and reuse it across several Stage-2
runs, or skip Stage-1 entirely (set ``train_backbone: false``) to ablate against
single-stage ImageNet-initialised AMIL training.

Usage:
    python batch_training.py -b batch_training_config.yaml
"""
import argparse
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
HP_SCRIPT = SCRIPT_DIR / "hyperparameter_tuning.py"
PRETRAIN_SCRIPT = SCRIPT_DIR / "pretrain_backbone.py"
TRAIN_SCRIPT = SCRIPT_DIR / "train_model.py"
TEST_SCRIPT = SCRIPT_DIR / "test_model.py"

REQUIRED_STEPS = ["tune_backbone", "train_backbone", "tune_board", "train_board", "test"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run_step(script: Path, config_path: Path, extra_args=None, label=""):
    if not script.exists():
        raise FileNotFoundError(f"Expected script not found: {script}")
    cmd = [sys.executable, str(script), "-c", str(config_path)] + (extra_args or [])
    print(f"\n{'='*78}\n>>> {label}\n    {' '.join(cmd)}\n{'='*78}")
    subprocess.run(cmd, check=True)


def newest_optimal_params(tune_dir: Path, stage: str) -> Path:
    """hyperparameter_tuning.py suffixes the export file (_1, _2, ...) if a file
    with the same name already exists, so the most recently written match is the
    one we just produced."""
    matches = sorted(tune_dir.glob(f"optimal_params_*{stage}*.yaml"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No optimal_params_*{stage}*.yaml found in {tune_dir}.")
    return matches[-1]


def find_checkpoint(output_dir: Path, filename: str) -> Path:
    """Locate a checkpoint, accounting for train_model.py's auto-incrementing
    output directory (output_dir, output_dir_1, output_dir_2, ...). Always compares
    mtimes across output_dir AND its numbered siblings rather than trusting
    output_dir first -- a previous successful run can leave a real, but stale,
    checkpoint sitting at the direct path while the run that just finished wrote
    to an incremented sibling instead."""
    candidates = []
    direct = output_dir / filename
    if direct.exists():
        candidates.append(direct)
    candidates += [
        d / filename for d in output_dir.parent.glob(f"{output_dir.name}_*")
        if d.is_dir() and (d / filename).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {filename} in {output_dir} or any "
            f"{output_dir.name}_N sibling directory."
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    if latest != direct:
        print(f"    note: {output_dir} already had a {filename} from a previous "
              f"run; using the newer {latest.parent} instead.")
    return latest


def resolve_path(explicit, template, patch_size):
    """Explicit per-experiment value wins; otherwise fill {patch_size} into the
    global template; otherwise None (caller decides if that's an error)."""
    if explicit:
        return explicit
    if template:
        return template.format(patch_size=patch_size)
    return None


def write_temp_config(cfg, temp_path: Path):
    with open(temp_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# --------------------------------------------------------------------------- #
# Per-experiment config assembly
# --------------------------------------------------------------------------- #
def build_experiment_config(base_cfg, exp, batch_cfg, exp_dir: Path):
    """Deep-copy the base config and apply the structural overrides for one
    (size, patch_size) experiment: model size, patch source, and every step's
    output directory. Hyperparameter/checkpoint paths are filled in later, once
    the corresponding step has actually produced them."""
    cfg = copy.deepcopy(base_cfg)

    size = str(exp["size"]).lower()
    patch_size = int(exp["patch_size"])

    patch_root = resolve_path(
        exp.get("patch_root"), batch_cfg.get("patch_root_template"), patch_size
    )
    if not patch_root:
        raise ValueError(
            f"No patch_root for size={size} patch_size={patch_size}: set "
            f"'patch_root' on the experiment or 'patch_root_template' globally."
        )
    test_data_root = resolve_path(
        exp.get("test_data_root"), batch_cfg.get("test_data_root_template"), patch_size
    )

    cfg["model"]["size"] = size
    cfg["data"]["patch_root"] = patch_root
    if "input_size" in exp:  # optional per-experiment override; defaults to inherited
        cfg["augmentation"]["input_size"] = exp["input_size"]

    cfg["tune"]["output_dir"] = str(exp_dir / "tune")
    cfg["pretrain"]["output_dir"] = str(exp_dir / "pretrain")
    cfg["train"]["output_dir"] = str(exp_dir / "train")
    cfg["test"]["output_dir"] = str(exp_dir / "test")
    if test_data_root:
        cfg["test"]["data_root"] = test_data_root

    return cfg


def run_experiment(exp, base_cfg, batch_cfg, result_root: Path, steps: dict):
    size = str(exp["size"]).lower()
    patch_size = int(exp["patch_size"])
    prefix = batch_cfg.get("prefix", "exp")
    exp_name = f"{prefix}_{size}_{patch_size}"
    exp_dir = result_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*78}\n# EXPERIMENT: {exp_name}\n{'#'*78}")

    cfg = build_experiment_config(base_cfg, exp, batch_cfg, exp_dir)
    temp_config_path = exp_dir / "temp_config.yaml"
    write_temp_config(cfg, temp_config_path)

    # ---- Stage 1: hyperparameter search for the backbone ---------------------
    if steps["tune_backbone"]:
        run_step(HP_SCRIPT, temp_config_path, ["--stage", "pretrain"],
                 f"[{exp_name}] Stage-1 hyperparameter search")
        tuned = newest_optimal_params(Path(cfg["tune"]["output_dir"]), "pretrain")
        cfg["pretrain"]["optimal_params"] = str(tuned)
        write_temp_config(cfg, temp_config_path)
        print(f"    -> tuned Stage-1 params: {tuned}")

    # ---- Stage 1: backbone pretraining ----------------------------------------
    if steps["train_backbone"]:
        run_step(PRETRAIN_SCRIPT, temp_config_path, label=f"[{exp_name}] Stage-1 backbone pretraining")
        backbone_ckpt = find_checkpoint(Path(cfg["pretrain"]["output_dir"]), "best_backbone.pth")
        cfg["model"]["backbone_checkpoint"] = str(backbone_ckpt)
        write_temp_config(cfg, temp_config_path)
        print(f"    -> backbone checkpoint: {backbone_ckpt}")

    # ---- Stage 2: hyperparameter search for the AMIL fine-tune ----------------
    if steps["tune_board"]:
        run_step(HP_SCRIPT, temp_config_path, ["--stage", "board"],
                 f"[{exp_name}] Stage-2 hyperparameter search")
        tuned = newest_optimal_params(Path(cfg["tune"]["output_dir"]), "board")
        cfg["train"]["optimal_params"] = str(tuned)
        write_temp_config(cfg, temp_config_path)
        print(f"    -> tuned Stage-2 params: {tuned}")

    # ---- Stage 2: AMIL fine-tuning ---------------------------------------------
    if steps["train_board"]:
        run_step(TRAIN_SCRIPT, temp_config_path, label=f"[{exp_name}] Stage-2 AMIL fine-tuning")
        model_ckpt = find_checkpoint(Path(cfg["train"]["output_dir"]), "best_model.pth")
        cfg["test"]["checkpoint"] = str(model_ckpt)
        write_temp_config(cfg, temp_config_path)
        print(f"    -> AMIL checkpoint: {model_ckpt}")

    # ---- Test --------------------------------------------------------------------
    if steps["test"]:
        run_step(TEST_SCRIPT, temp_config_path, label=f"[{exp_name}] Testing")

    print(f"\n[{exp_name}] Done. Final config: {temp_config_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Sweep BarkNet-AMIL training across model/patch sizes.")
    parser.add_argument("-c", "--batch-config", default="batch_training_config.yaml", type=str)
    args = parser.parse_args()

    batch_path = Path(args.batch_config).resolve()
    with open(batch_path, "r") as f:
        batch_cfg = yaml.safe_load(f)

    base_config_path = Path(batch_cfg["base_config"]).resolve()
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    result_root = Path(batch_cfg["result_dir"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(batch_path, result_root)

    steps = batch_cfg["steps"]
    missing = [k for k in REQUIRED_STEPS if k not in steps]
    if missing:
        raise KeyError(f"batch_training_config.yaml: steps.{missing[0]} is required "
                       f"(all of {REQUIRED_STEPS} must be set true/false).")

    experiments = batch_cfg["experiments"]
    # Matches the Carbotech batch script's behaviour: a failed experiment is logged
    # and skipped, the sweep keeps going. Set stop_on_error: true for hard failure.
    stop_on_error = batch_cfg.get("stop_on_error", False)

    print(f"Loaded {len(experiments)} experiment(s). Steps enabled: "
          f"{', '.join(k for k, v in steps.items() if v) or 'none'}")

    failures = []
    for exp in experiments:
        tag = f"size={exp.get('size')} patch_size={exp.get('patch_size')}"
        try:
            run_experiment(exp, base_cfg, batch_cfg, result_root, steps)
        except Exception as exc:
            msg = f"{tag} FAILED: {exc}"
            print(f"\n!!! {msg}\n")
            failures.append(msg)
            if stop_on_error:
                raise

    print(f"\n{'='*78}\nBatch sweep complete. "
          f"{len(experiments) - len(failures)}/{len(experiments)} experiment(s) succeeded.")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()

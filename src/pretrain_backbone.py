"""Stage 1 - patch-level backbone pretraining.

Trains the ConvNeXt backbone + a shared classifier as a plain per-patch classifier
(one label per patch), like the original bark papers. The best checkpoint is saved as
``best_backbone.pth`` and later loaded by train_model.py to initialise the AMIL model
for Stage-2 fine-tuning.

The selection criterion is explicit (``pretrain.checkpoint_metric``: ``val_loss`` or
``val_acc``) because on this dataset the two disagree: label-smoothing overconfidence
pushes val loss up while val accuracy is still climbing. Whatever you pick, every fold
must use the same one.

Usage:
    python pretrain_backbone.py -c configs/config.yaml
    python pretrain_backbone.py -c configs/config_cluster.yaml \
        --patch-root $SLURM_TMPDIR/patches_224/train \
        --output-dir $SCRATCH/runs/smoke/pretrain --epochs 2
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

import helper.data_loader as dt
from helper.config_cli import add_config_args, dump_config, load_config
from helper.early_stopping import EarlyStopping
from helper.optimizer_scheduler import build_scheduler, build_uniform_optimizer
from helper.timing import timer
from helper.model_wrapper import ConvNeXtAMIL, unwrap

CHECKPOINT_METRICS = {"val_loss", "val_acc"}


def load_optimal_params(path):
    """Load a tuned-parameter file (yaml) emitted by hyperparameter_tuning.py."""
    if not path:
        return {}
    with open(path, "r") as f:
        payload = yaml.safe_load(f)
    return payload.get("optimal_parameters", payload)


def pick(*values, default=None):
    """First value that is not None (0.0 and False are legitimate values)."""
    for v in values:
        if v is not None:
            return v
    return default


def main(cfg, result_dir):
    device = cfg["project"]["device"]
    seed = cfg["project"].get("seed", 42)
    perf_cfg = cfg["project"].get("perf", {}) or {}
    species = cfg["data"]["species"]
    model_cfg = cfg["model"]
    pre = cfg["pretrain"]

    # Stage-local hyperparameters win over the shared model.* defaults, so Stage 1 and
    # Stage 2 can carry the different dropouts their searches found without two configs.
    base_lr = pre["base_lr"]
    weight_decay = pre["weight_decay"]
    batch_size = pre["batch_size"]
    class_dropout = pick(pre.get("class_dropout"), model_cfg.get("class_dropout"), default=0.4)
    drop_path_rate = pick(pre.get("drop_path_rate"), model_cfg.get("drop_path_rate"), default=0.2)

    tuned = load_optimal_params(pre.get("optimal_params"))
    if tuned:
        base_lr = tuned.get("base_lr", base_lr)
        weight_decay = tuned.get("weight_decay", weight_decay)
        batch_size = tuned.get("batch_size", batch_size)
        class_dropout = tuned.get("class_dropout", class_dropout)
        drop_path_rate = tuned.get("drop_path_rate", drop_path_rate)
        print("Applied tuned hyperparameters from", pre["optimal_params"])

    metric = str(pre.get("checkpoint_metric", "val_loss")).lower()
    if metric not in CHECKPOINT_METRICS:
        raise ValueError(f"pretrain.checkpoint_metric must be one of {sorted(CHECKPOINT_METRICS)}")

    # Resume from the best backbone in this output dir if requested.
    resume_ckpt = None
    if pre.get("resume"):
        candidate = result_dir / "best_backbone.pth"
        if candidate.exists():
            resume_ckpt = str(candidate)
            print(f"Resuming Stage-1 from {candidate}")
        else:
            print(f"resume requested but {candidate} not found; starting fresh.")

    wrapper = ConvNeXtAMIL(
        device=device,
        species=species,
        model_size=model_cfg["size"],
        weights=model_cfg["weights"],
        backbone_checkpoint=resume_ckpt,
        class_dropout=class_dropout,
        drop_path_rate=drop_path_rate,
        label_smoothing=model_cfg.get("label_smoothing", 0.0),
        pretrained_file=model_cfg.get("pretrained_file"),
        # --- performance (see project.perf in the config) --------------------
        amp_dtype=perf_cfg.get("amp_dtype", "bf16"),
        channels_last=perf_cfg.get("channels_last", True),
        # Stage 1 has a fixed batch shape, so torch.compile can actually cache a graph.
        compile_model=perf_cfg.get("compile_stage1", False),
        save_data=True,
    )
    model, criterion = wrapper.get_patch_model()

    optimizer = build_uniform_optimizer(model, base_lr, weight_decay)
    scheduler = build_scheduler(
        optimizer, pre["epochs"], pre["warmup_ratio"], eta_min=base_lr * 0.01
    )

    train_loader, val_loader = dt.build_patch_dataloaders(cfg, batch_size=batch_size, seed=seed)
    scaler = wrapper.make_scaler()   # disabled automatically when amp_dtype=bf16

    csv_path = result_dir / "pretrain_results.csv"
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"Stage-1 pretraining convnextv2_{model_cfg['size']} -> {result_dir}")
    print(f"  batch_size={batch_size} base_lr={base_lr:.3e} wd={weight_decay:.3e} "
          f"class_dropout={class_dropout:.3f} drop_path={drop_path_rate:.3f}")
    print(f"  checkpoint criterion: {metric}")

    best = {"val_loss": float("inf"), "val_acc": -1.0, "epoch": -1}

    with timer("Stage 1 Pretraining"):
        early_stopper = EarlyStopping(patience=pre["early_stopping_patience"])

        for epoch in range(pre["epochs"]):
            train_loss, train_acc = wrapper.train_patch_epoch(
                model, criterion, optimizer, scaler, train_loader, epoch
            )
            if "cuda" in str(device):
                torch.cuda.empty_cache()

            val_loss, val_acc = wrapper.validate_patch_epoch(model, criterion, val_loader)
            if "cuda" in str(device):
                torch.cuda.empty_cache()

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            improved = (val_loss < best["val_loss"]) if metric == "val_loss" \
                else (val_acc > best["val_acc"])
            if improved:
                best = {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch}
                torch.save(unwrap(model).state_dict(), result_dir / "best_backbone.pth")
                print(f"New best backbone ({metric}): val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

            # Always keep the last epoch too: it is the fallback if the criterion above
            # turns out to be the wrong one, and it costs one file.
            torch.save(unwrap(model).state_dict(), result_dir / "last_backbone.pth")

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            print(f"[Stage 1] Epoch {epoch} | {row}", flush=True)

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}.")
                break

    summary = {
        "stage": "pretrain",
        "model_size": model_cfg["size"],
        "input_size": cfg["augmentation"]["input_size"],
        "patch_root": cfg["data"]["patch_root"],
        "checkpoint_metric": metric,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "checkpoint": str(result_dir / "best_backbone.pth"),
    }
    with open(result_dir / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Stage-1 done. Best epoch {best['epoch']} "
          f"(val_loss={best['val_loss']:.4f}, val_acc={best['val_acc']:.2f}%). "
          f"Backbone -> {result_dir / 'best_backbone.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="barknet-AMIL Stage-1 pretraining")
    add_config_args(parser)
    args = parser.parse_args()

    cfg = load_config(args, stage="pretrain")

    # Stable output dir (so --resume and train_model.py's backbone path stay consistent).
    result_dir = Path(cfg["pretrain"]["output_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, result_dir)

    main(cfg, result_dir)

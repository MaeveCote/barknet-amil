"""Stage 2 - image-level AMIL fine-tuning on pre-cut bark patches.

Loads the Stage-1 backbone (extractor.* + classifier.*, strict=False, attention head
fresh) and fine-tunes end-to-end with the auxiliary instance loss and differential LRs.

Usage:
    python train_model.py -c configs/config.yaml
    python train_model.py -c configs/config_cluster.yaml \
        --patch-root $SLURM_TMPDIR/patches_224/train \
        --backbone-checkpoint $SCRATCH/runs/smoke/pretrain/best_backbone.pth \
        --output-dir $SCRATCH/runs/smoke/train --epochs 2
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
from helper.optimizer_scheduler import build_optimizer_and_scheduler
from helper.timing import timer
from helper.model_wrapper import ConvNeXtAMIL, unwrap


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
    train_cfg = cfg["train"]

    # Optimizer/training values: config defaults, optionally replaced by tuned params.
    opt_cfg = train_cfg.get("optimizer", {}) or {}
    base_lr = opt_cfg.get("base_lr")
    weight_decay = opt_cfg.get("weight_decay")
    lr_multiplier = opt_cfg.get("lr_multiplier")
    accumulation_steps = train_cfg["accumulation_steps"]
    instance_loss_weight = train_cfg["instance_loss_weight"]
    # Stage-local dropouts override the shared model.* defaults (Stage 1 and Stage 2
    # were tuned separately and want different values).
    attn_dropout = pick(train_cfg.get("attn_dropout"), model_cfg.get("attn_dropout"), default=0.1)
    class_dropout = pick(train_cfg.get("class_dropout"), model_cfg.get("class_dropout"), default=0.4)
    drop_path_rate = pick(train_cfg.get("drop_path_rate"), model_cfg.get("drop_path_rate"), default=0.2)

    tuned = load_optimal_params(train_cfg.get("optimal_params"))
    if tuned:
        base_lr = tuned.get("base_lr", base_lr)
        weight_decay = tuned.get("weight_decay", weight_decay)
        lr_multiplier = tuned.get("lr_multiplier", lr_multiplier)
        accumulation_steps = tuned.get("accumulation_steps", accumulation_steps)
        instance_loss_weight = tuned.get("instance_loss_weight", instance_loss_weight)
        attn_dropout = tuned.get("attn_dropout", attn_dropout)
        class_dropout = tuned.get("class_dropout", class_dropout)
        print("Applied tuned hyperparameters from", train_cfg["optimal_params"])

    wrapper = ConvNeXtAMIL(
        device=device,
        species=species,
        model_size=model_cfg["size"],
        weights=model_cfg["weights"],
        pretrained_checkpoint=model_cfg.get("pretrained_checkpoint"),
        backbone_checkpoint=model_cfg.get("backbone_checkpoint"),
        attn_dropout=attn_dropout,
        class_dropout=class_dropout,
        drop_path_rate=drop_path_rate,
        label_smoothing=model_cfg.get("label_smoothing", 0.0),
        instance_loss_weight=instance_loss_weight,
        pretrained_file=model_cfg.get("pretrained_file"),
        # --- performance (see project.perf in the config) --------------------
        amp_dtype=perf_cfg.get("amp_dtype", "bf16"),
        channels_last=perf_cfg.get("channels_last", True),
        # NOT compiled: Stage-2 bags are [N, C, H, W] with N varying per image, so
        # torch.compile would recompile on nearly every batch and run SLOWER.
        compile_model=False,
        # Per-epoch full-model dumps are useful locally, wasteful on a cluster filesystem.
        save_data=bool(train_cfg.get("save_epoch_states", False)),
    )
    model, criterion = wrapper.get_model()

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        weights=model_cfg["weights"],
        num_epochs=train_cfg["epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        base_lr=base_lr,
        weight_decay=weight_decay,
        lr_multiplier=lr_multiplier,
    )

    train_loader, val_loader = dt.build_dataloaders(cfg, seed=seed)
    scaler = wrapper.make_scaler()   # disabled automatically when amp_dtype=bf16

    csv_path = result_dir / f"convnextv2_{model_cfg['size']}_results.csv"
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"Stage-2 AMIL fine-tuning convnextv2_{model_cfg['size']} -> {result_dir}")
    print(f"  accumulation_steps={accumulation_steps} lambda_inst={instance_loss_weight} "
          f"attn_dropout={attn_dropout} class_dropout={class_dropout}")

    best = {"val_acc": 0.0, "val_loss": float("inf"), "epoch": -1}

    with timer("Model Training"):
        early_stopper = EarlyStopping(patience=train_cfg["early_stopping_patience"])

        for epoch in range(train_cfg["epochs"]):
            # Redraw the stochastic per-bag subset. Must happen BEFORE the loader is
            # iterated, since that is when the (non-persistent) workers fork and inherit
            # the dataset's epoch.
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

            train_loss, train_acc = wrapper.train_epoch(
                model, criterion, optimizer, scaler, train_loader, epoch,
                result_dir=result_dir, accumulation_steps=accumulation_steps,
            )
            if "cuda" in str(device):
                torch.cuda.empty_cache()

            val_loss, val_acc = wrapper.validate_epoch(model, criterion, val_loader)
            if "cuda" in str(device):
                torch.cuda.empty_cache()

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
                torch.save(unwrap(model).state_dict(), result_dir / "best_model.pth")
                print(f"New best model saved: {val_acc:.2f}% image-level val accuracy")
            torch.save(unwrap(model).state_dict(), result_dir / "last_model.pth")

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
            print(f"[Stage 2] Epoch {epoch} | {row}", flush=True)

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}.")
                break

    summary = {
        "stage": "train",
        "model_size": model_cfg["size"],
        "input_size": cfg["augmentation"]["input_size"],
        "patch_root": cfg["data"]["patch_root"],
        "backbone_checkpoint": model_cfg.get("backbone_checkpoint"),
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"],
        "checkpoint": str(result_dir / "best_model.pth"),
    }
    with open(result_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Stage-2 done. Best epoch {best['epoch']} ({best['val_acc']:.2f}% val acc). "
          f"Model -> {result_dir / 'best_model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="barknet-AMIL Stage-2 training")
    add_config_args(parser)
    args = parser.parse_args()

    cfg = load_config(args, stage="train")

    base_dir = Path(cfg["train"]["output_dir"])
    if args.output_dir:
        # Explicit dir (cluster jobs): use it verbatim so the next step knows where to look.
        result_dir = base_dir
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Legacy local behaviour: never overwrite an existing run.
        result_dir = base_dir
        count = 0
        while result_dir.exists():
            count += 1
            result_dir = base_dir.with_name(f"{base_dir.name}_{count}")
        result_dir.mkdir(parents=True)

    dump_config(cfg, result_dir)
    main(cfg, result_dir)

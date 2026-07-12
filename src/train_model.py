"""Train ConvNeXt + AMIL on pre-cut bark patches.

Usage:
    python train_model.py -c configs/config.yaml
"""
import argparse
import csv
import shutil
from pathlib import Path

import torch
import yaml

import helper.data_loader as dt
from helper.early_stopping import EarlyStopping
from helper.optimizer_scheduler import build_optimizer_and_scheduler
from helper.timing import timer
from helper.model_wrapper import ConvNeXtAMIL

def load_optimal_params(path):
    """Load a tuned-parameter file (yaml or json) emitted by hyperparameter_tuning.py; return its dict."""
    if not path:
        return {}
    with open(path, "r") as f:
        payload = yaml.safe_load(f)
    return payload.get("optimal_parameters", payload)

def main(cfg, result_dir):
    device = cfg["project"]["device"]
    seed = cfg["project"].get("seed", 42)
    species = cfg["data"]["species"]

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    # Optimizer/training overrides: config defaults, optionally replaced by tuned params.
    opt_cfg = train_cfg.get("optimizer", {})
    base_lr = opt_cfg.get("base_lr")
    weight_decay = opt_cfg.get("weight_decay")
    lr_multiplier = opt_cfg.get("lr_multiplier")
    accumulation_steps = train_cfg["accumulation_steps"]
    instance_loss_weight = train_cfg["instance_loss_weight"]
    attn_dropout = model_cfg["attn_dropout"]
    class_dropout = model_cfg["class_dropout"]

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
        drop_path_rate=model_cfg["drop_path_rate"],
        label_smoothing=model_cfg.get("label_smoothing", 0.0),
        instance_loss_weight=instance_loss_weight,
        save_data=True,
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
    scaler = torch.amp.GradScaler(device=wrapper.amp_device)

    csv_path = result_dir / f"convnext_{model_cfg['size']}_results.csv"
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"Training convnext_{model_cfg['size']} -> {result_dir}")
    print(f"Effective batch (accumulation steps): {accumulation_steps}")

    with timer("Model Training"):
        best_val_acc = 0.0
        early_stopper = EarlyStopping(patience=train_cfg["early_stopping_patience"])

        for epoch in range(train_cfg["epochs"]):
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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), result_dir / "best_model.pth")
                print(f"New best model saved: {val_acc:.2f}% val accuracy")

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
            print(f"Epoch {epoch} | {row}")

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="barknet-AMIL training")
    parser.add_argument("-c", "--config", default="configs/config.yaml", type=str)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["train"]["output_dir"])
    result_dir = base_dir
    count = 0
    while result_dir.exists():
        count += 1
        result_dir = base_dir.with_name(f"{base_dir.name}_{count}")
    result_dir.mkdir(parents=True)
    shutil.copy(config_path, result_dir)

    main(cfg, result_dir)
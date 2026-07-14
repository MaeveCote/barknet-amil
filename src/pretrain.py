"""Stage 1 - patch-level backbone pretraining.

Trains the ConvNeXt backbone + a shared classifier as a plain per-patch classifier
(one label per patch), like the original bark papers. The best checkpoint by
patch-level validation loss is saved as ``best_backbone.pth`` and later loaded by
train.py to initialise the AMIL model for Stage-2 fine-tuning.

Usage:
    python pretrain.py -c configs/config.yaml
    python pretrain.py -c configs/config.yaml   # set pretrain.resume: true to continue
"""
import argparse
import csv
import shutil
from pathlib import Path

import torch
import yaml

from helper import data as dt
from helper.early_stopping import EarlyStopping
from helper.optimizer_scheduler import build_scheduler, build_uniform_optimizer
from helper.timing import timer
from helper.wrapper import ConvNeXtAMIL


def main(cfg, result_dir):
    device = cfg["project"]["device"]
    seed = cfg["project"].get("seed", 42)
    species = cfg["data"]["species"]
    model_cfg = cfg["model"]
    pre = cfg["pretrain"]

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
        class_dropout=model_cfg["class_dropout"],
        drop_path_rate=model_cfg["drop_path_rate"],
        label_smoothing=model_cfg.get("label_smoothing", 0.0),
        save_data=True,
    )
    model, criterion = wrapper.get_patch_model()

    optimizer = build_uniform_optimizer(model, pre["base_lr"], pre["weight_decay"])
    scheduler = build_scheduler(
        optimizer, pre["epochs"], pre["warmup_ratio"], eta_min=pre["base_lr"] * 0.01
    )

    train_loader, val_loader = dt.build_patch_dataloaders(
        cfg, batch_size=pre["batch_size"], seed=seed
    )
    scaler = torch.amp.GradScaler(device=wrapper.amp_device)

    csv_path = result_dir / "pretrain_results.csv"
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"Stage-1 pretraining convnext_{model_cfg['size']} -> {result_dir}")

    with timer("Stage 1 Pretraining"):
        best_val_loss = float("inf")
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

            # Best checkpoint by patch-level validation loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), result_dir / "best_backbone.pth")
                print(f"New best backbone: {val_loss:.4f} val loss ({val_acc:.2f}% acc)")

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
            print(f"[Stage 1] Epoch {epoch} | {row}")

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Stage-1 backbone saved to {result_dir / 'best_backbone.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BarkNet-AMIL Stage-1 pretraining")
    parser.add_argument("-c", "--config", default="configs/config.yaml", type=str)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Stable output dir (so --resume and train.py's backbone path stay consistent).
    result_dir = Path(cfg["pretrain"]["output_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, result_dir)

    main(cfg, result_dir)

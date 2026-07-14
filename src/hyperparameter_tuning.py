"""Optuna hyperparameter search for ConvNeXt + AMIL.

A single study (board-level fine-tuning) minimising validation loss, with median
pruning. The search space adapts to whether weights are pretrained or from scratch.
Best parameters are written to a YAML file that train.py can consume via
``train.optimal_params``.

Usage:
    python tune.py -c configs/config.yaml
"""
import argparse
from pathlib import Path

import optuna
import torch
import yaml

import helper.data as dt
from helper.early_stopping import EarlyStopping
from helper.optimizer_scheduler import build_optimizer_and_scheduler
from helper.timing import timer
from helper.wrapper import ConvNeXtAMIL


class Objective:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["project"]["device"]
        self.seed = cfg["project"].get("seed", 42)
        self.species = cfg["data"]["species"]
        self.model_size = str(cfg["model"]["size"]).lower()
        self.weights = str(cfg["model"]["weights"]).lower()
        self.drop_path_rate = cfg["model"]["drop_path_rate"]
        self.label_smoothing = cfg["model"].get("label_smoothing", 0.0)
        self.backbone_checkpoint = cfg["model"].get("backbone_checkpoint")

        tune_cfg = cfg["tune"]
        self.num_epochs = tune_cfg["epoch_limit"]
        self.subset_ratio = tune_cfg.get("train_subset_ratio", 1.0)
        self.warmup_ratio = cfg["train"]["warmup_ratio"]

        # Build loaders once and reuse across trials.
        self.train_loader, self.val_loader = dt.build_dataloaders(
            cfg, seed=self.seed, train_subset_ratio=self.subset_ratio
        )

    def __call__(self, trial):
        accumulation_steps = trial.suggest_categorical("accumulation_steps", [8, 16, 32])
        attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.3)
        class_dropout = trial.suggest_float("class_dropout", 0.1, 0.6)
        instance_loss_weight = trial.suggest_float("instance_loss_weight", 0.1, 1.0)

        if self.weights == "none":
            base_lr = trial.suggest_float("base_lr", 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            lr_multiplier = 1.0
        else:
            base_lr = trial.suggest_float("base_lr", 1e-5, 2e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            lr_multiplier = trial.suggest_categorical("lr_multiplier", [5, 10, 20])

        wrapper = ConvNeXtAMIL(
            device=self.device,
            species=self.species,
            model_size=self.model_size,
            weights=self.weights,
            backbone_checkpoint=self.backbone_checkpoint,
            attn_dropout=attn_dropout,
            class_dropout=class_dropout,
            drop_path_rate=self.drop_path_rate,
            label_smoothing=self.label_smoothing,
            instance_loss_weight=instance_loss_weight,
            save_data=False,
        )
        model, criterion = wrapper.get_model()

        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            weights=self.weights,
            num_epochs=self.num_epochs,
            warmup_ratio=self.warmup_ratio,
            base_lr=base_lr,
            weight_decay=weight_decay,
            lr_multiplier=lr_multiplier,
        )

        scaler = torch.amp.GradScaler(device=wrapper.amp_device)
        early_stopper = EarlyStopping(patience=5)
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            wrapper.train_epoch(
                model, criterion, optimizer, scaler, self.train_loader, epoch,
                result_dir=None, accumulation_steps=accumulation_steps,
            )
            val_loss, _ = wrapper.validate_epoch(model, criterion, self.val_loader)
            scheduler.step()

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_val_loss = min(best_val_loss, val_loss)
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stop at epoch {epoch}.")
                break

        return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BarkNet-AMIL hyperparameter search")
    parser.add_argument("-c", "--config", default="configs/config.yaml", type=str)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tune_cfg = cfg["tune"]
    output_dir = Path(tune_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    size = str(cfg["model"]["size"]).lower()
    weights = str(cfg["model"]["weights"]).lower()
    study_name = f"convnext_amil_{weights}_{size}"
    db_uri = f"sqlite:///{(output_dir / f'{study_name}.db').as_posix()}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=db_uri,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    with timer("Hyperparameter Optimization"):
        study.optimize(Objective(cfg), n_trials=tune_cfg["trials"])

    best = study.best_trial
    print(f"\nBest val loss: {best.value:.4f}")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    export = {
        "study_name": study_name,
        "weights": weights,
        "best_val_loss": best.value,
        "optimal_parameters": best.params,
    }
    out_path = output_dir / f"optimal_params_{study_name}.yaml"
    count = 0
    while out_path.exists():
        count += 1
        out_path = output_dir / f"optimal_params_{study_name}_{count}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(export, f, sort_keys=False)
    print(f"Saved best parameters to {out_path}")

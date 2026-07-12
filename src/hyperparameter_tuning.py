"""Optuna hyperparameter search for ConvNeXt + AMIL — Stage 1 or Stage 2.

Two independent studies are supported, selected by ``tune.stage`` in the config
(or the ``--stage`` CLI flag, which overrides the config):

* ``pretrain``: Stage-1 patch-level backbone search (plain per-patch classifier,
  no attention, no bags). Tunes ``base_lr``, ``weight_decay``, ``class_dropout``,
  ``drop_path_rate``, ``batch_size``. Validation is subset by its own
  ``val_subset_ratio`` (separate from ``train_subset_ratio``), stratified per class,
  and automatically raised to keep at least ``min_val_images_per_class`` instances
  of the rarest class.
* ``board``:    Stage-2 image-level AMIL fine-tuning search (unchanged from before).
  Tunes ``base_lr``, ``weight_decay``, ``lr_multiplier``, ``attn_dropout``,
  ``class_dropout``, ``instance_loss_weight``, ``accumulation_steps``.

Each stage reads its own ``tune.pretrain`` / ``tune.board`` block for trial count,
epoch limit, and training-subset ratio, so the two searches can be configured
(and re-run) independently without touching code.

Best parameters are written to a stage-tagged YAML file, e.g.
``optimal_params_convnext_amil_pretrain_default_pico.yaml``.

Usage:
    python hyperparameter_tuning.py -c configs/config.yaml                # uses tune.stage
    python hyperparameter_tuning.py -c configs/config.yaml --stage pretrain
    python hyperparameter_tuning.py -c configs/config.yaml --stage board
"""
import argparse
import gc
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

import helper.data_loader as dt
from helper.early_stopping import EarlyStopping
from helper.optimizer_scheduler import (
    build_optimizer_and_scheduler,
    build_scheduler,
    build_uniform_optimizer,
)
from helper.timing import timer
from helper.model_wrapper import ConvNeXtAMIL


def _is_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def _per_class_counts(patches):
    """Count validation instances per class label."""
    counts = defaultdict(int)
    for _, label in patches:
        counts[label] += 1
    return dict(counts)


def _min_viable_val_ratio(counts, min_per_class):
    """Smallest uniform subset ratio that still yields >= ``min_per_class`` instances
    for the least-represented class. Driven by the smallest class, since that's the
    binding constraint -- any larger class is automatically satisfied at that ratio."""
    smallest = min(counts.values())
    return min_per_class / smallest


def _stratified_subset(patches, ratio, seed):
    """Take the same fraction of every class. Unlike slicing the first N batches of a
    class-ordered loader (which only covers the leading class), this keeps every class
    represented proportionally. Deterministic given ``seed``."""
    by_class = defaultdict(list)
    for item in patches:
        by_class[item[1]].append(item)
    subset = []
    for label in sorted(by_class):
        items = list(by_class[label])
        random.Random(seed + int(label)).shuffle(items)
        n = min(len(items), max(1, math.ceil(ratio * len(items))))
        subset.extend(items[:n])
    return subset


# --------------------------------------------------------------------------- #
# Stage 1 — patch-level backbone search
# --------------------------------------------------------------------------- #
class PretrainObjective:
    """Plain per-patch classifier search (no attention, no bags) — Stage 1."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["project"]["device"]
        self.seed = cfg["project"].get("seed", 42)
        self.species = cfg["data"]["species"]
        self.model_size = str(cfg["model"]["size"]).lower()
        self.weights = str(cfg["model"]["weights"]).lower()
        self.label_smoothing = cfg["model"].get("label_smoothing", 0.0)
        self.warmup_ratio = cfg["pretrain"]["warmup_ratio"]

        search_cfg = cfg["tune"]["pretrain"]
        self.num_epochs = search_cfg["epoch_limit"]
        self.subset_ratio = search_cfg.get("train_subset_ratio", 1.0)
        self.batch_size_choices = search_cfg.get(
            "batch_size_choices", [cfg["pretrain"]["batch_size"]]
        )

        # Scan the disk and split by tree exactly ONCE, regardless of how many
        # trials run. With a multi-million-patch dataset, re-scanning every trial
        # (the previous design) dominates wall-clock time far more than training
        # itself — only the DataLoader wrapper actually needs to change per trial,
        # since batch_size is the tuned dimension, not the underlying file list.
        data_cfg = cfg["data"]
        aug = cfg["augmentation"]
        train_bags, val_bags = dt.split_by_tree(
            patch_root=data_cfg["patch_root"],
            species=data_cfg["species"],
            fcfg=data_cfg.get("filename", {}),
            val_ratio=data_cfg["val_ratio"],
            seed=self.seed,
        )
        self.train_patches = dt.flatten_bags(train_bags)
        self.val_patches = dt.flatten_bags(val_bags)
        self.train_transform = dt.build_train_transform(aug)
        self.eval_transform = dt.build_eval_transform(aug)
        self.num_workers = data_cfg.get("num_workers", 8)

        # ---- Validation subset: own ratio + class-coverage floor ----------------
        # Validation is subset to keep the search fast, but with its OWN ratio,
        # independent of train_subset_ratio (val often needs a larger fraction to
        # stay representative). Two things matter:
        #   1. It must be stratified per class. A naive "first N batches" slice of
        #      the class-ordered val set only covers the leading species, so the
        #      val metric would not reflect the rare classes at all.
        #   2. Too small a ratio can still drop the rarest class below usefulness,
        #      so the ratio is raised to the smallest value that keeps at least
        #      `min_val_images_per_class` instances of the least-represented class.
        self.val_min_per_class = search_cfg.get("min_val_images_per_class", 1)
        configured_val_ratio = search_cfg.get("val_subset_ratio", self.subset_ratio)

        val_counts = _per_class_counts(self.val_patches)
        present = len(val_counts)
        if present < len(self.species):
            print(f"[val subset] warning: only {present}/{len(self.species)} classes have "
                  f"validation patches; classes with none cannot be represented.")

        min_viable = _min_viable_val_ratio(val_counts, self.val_min_per_class)
        if min_viable > 1.0:
            smallest_label = min(val_counts, key=val_counts.get)
            print(f"[val subset] warning: smallest class (label {smallest_label}) has only "
                  f"{val_counts[smallest_label]} val patches, fewer than "
                  f"min_val_images_per_class={self.val_min_per_class}; using 100% of validation.")
        effective_val_ratio = min(1.0, max(configured_val_ratio, min_viable))

        if effective_val_ratio > configured_val_ratio + 1e-9:
            smallest_label = min(val_counts, key=val_counts.get)
            print(f"[val subset] configured val_subset_ratio={configured_val_ratio:.4f} is below "
                  f"the minimum {min_viable:.4f} needed to keep >= {self.val_min_per_class} "
                  f"val patch(es) for the smallest class (label {smallest_label}, "
                  f"{val_counts[smallest_label]} val patches). Overriding to "
                  f"{effective_val_ratio:.4f} for this run.")

        self.effective_val_ratio = effective_val_ratio
        self.val_subset = _stratified_subset(self.val_patches, effective_val_ratio, self.seed)
        print(f"[val subset] using {len(self.val_subset):,} of {len(self.val_patches):,} val "
              f"patches ({effective_val_ratio:.4f}, stratified across {present} classes).")

    def __call__(self, trial):
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_choices)
        base_lr = trial.suggest_float("base_lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        class_dropout = trial.suggest_float("class_dropout", 0.1, 0.6)
        drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.4)

        # Cheap: wraps the already-scanned patch lists in fresh DataLoaders at this
        # trial's batch_size. No disk scan happens here.
        #
        # persistent_workers is deliberately OFF here, unlike the main training
        # scripts. persistent_workers=True means "keep worker processes alive
        # indefinitely for reuse" -- correct for one long-lived DataLoader iterated
        # across many epochs, but wrong here: a brand new DataLoader is created
        # every trial (batch_size is the thing being tuned), so there is nothing
        # to reuse workers across, only orphaned processes to accumulate. Combined
        # with Windows' spawn-based multiprocessing (each worker is a full new
        # process holding its own pickled copy of the patch-path list) and OOM
        # exceptions interrupting iteration mid-batch -- exactly the scenario most
        # likely to leave persistent workers uncleanly torn down -- this was
        # measured to leak to 50+ GB of committed RAM over the course of a search.
        train_loader = DataLoader(
            dt.PatchDataset(self.train_patches, self.train_transform),
            batch_size=batch_size, shuffle=True, num_workers=self.num_workers,
            persistent_workers=False, pin_memory=True, drop_last=True,
        )
        # Validation uses the pre-computed stratified subset (see __init__): a fixed,
        # class-balanced sample, identical across trials, iterated in full each epoch.
        val_loader = DataLoader(
            dt.PatchDataset(self.val_subset, self.eval_transform),
            batch_size=batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers=False, pin_memory=True,
        )

        try:
            return self._run_trial(
                trial, batch_size, base_lr, weight_decay, class_dropout,
                drop_path_rate, train_loader, val_loader,
            )
        finally:
            # Explicit, unconditional teardown -- runs whether the trial completes,
            # raises TrialPruned, or raises anything else. Drops the only references
            # to these loaders immediately rather than waiting on whatever Python's
            # GC gets around to next, which matters across a long Optuna search.
            del train_loader, val_loader
            gc.collect()

    def _run_trial(self, trial, batch_size, base_lr, weight_decay, class_dropout,
                   drop_path_rate, train_loader, val_loader):
        # Training is capped per epoch via train_subset_ratio (the search only needs
        # enough signal to RANK trials). Validation is NOT capped here -- val_loader
        # is already the fixed, stratified, class-balanced subset built in __init__,
        # so it's iterated in full for a representative, trial-comparable metric.
        max_train_batches = max(1, int(len(train_loader) * self.subset_ratio))

        wrapper = ConvNeXtAMIL(
            device=self.device,
            species=self.species,
            model_size=self.model_size,
            weights=self.weights,
            class_dropout=class_dropout,
            drop_path_rate=drop_path_rate,
            label_smoothing=self.label_smoothing,
            save_data=False,
        )
        model, criterion = wrapper.get_patch_model()

        optimizer = build_uniform_optimizer(model, base_lr, weight_decay)
        scheduler = build_scheduler(
            optimizer, self.num_epochs, self.warmup_ratio, eta_min=base_lr * 0.01
        )
        scaler = torch.amp.GradScaler(device=wrapper.amp_device)
        early_stopper = EarlyStopping(patience=3)
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            limited_train = itertools.islice(train_loader, max_train_batches)
            try:
                wrapper.train_patch_epoch(model, criterion, optimizer, scaler, limited_train, epoch,
                                          total=max_train_batches)
                val_loss, _ = wrapper.validate_patch_epoch(model, criterion, val_loader,
                                                           total=len(val_loader))
            except RuntimeError as exc:
                if _is_oom(exc):
                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()
                    print(f"  OOM at batch_size={batch_size} — pruning trial.")
                    raise optuna.exceptions.TrialPruned()
                raise

            scheduler.step()

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_val_loss = min(best_val_loss, val_loss)
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"  Early stop at epoch {epoch}.")
                break

        return best_val_loss


# --------------------------------------------------------------------------- #
# Stage 2 — image-level AMIL search
# --------------------------------------------------------------------------- #
class BoardObjective:
    """Attention-pooled image-level fine-tuning search — Stage 2 (unchanged)."""

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
        self.warmup_ratio = cfg["train"]["warmup_ratio"]

        search_cfg = cfg["tune"]["board"]
        self.num_epochs = search_cfg["epoch_limit"]
        self.subset_ratio = search_cfg.get("train_subset_ratio", 1.0)

        # Bag-level loaders don't depend on a tuned batch_size, so build once.
        self.train_loader, self.val_loader = dt.build_dataloaders(
            cfg, seed=self.seed, train_subset_ratio=self.subset_ratio
        )

    def __call__(self, trial):
        accumulation_steps = trial.suggest_categorical("accumulation_steps", [8, 16, 32])
        attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.5)
        class_dropout = trial.suggest_float("class_dropout", 0.1, 0.6)
        instance_loss_weight = trial.suggest_float("instance_loss_weight", 0.1, 1.0)

        if self.weights == "none":
            base_lr = trial.suggest_float("base_lr", 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            lr_multiplier = 1.0
        else:
            base_lr = trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
            lr_multiplier = trial.suggest_categorical("lr_multiplier", [5, 10, 15, 20])

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
            try:
                wrapper.train_epoch(
                    model, criterion, optimizer, scaler, self.train_loader, epoch,
                    result_dir=None, accumulation_steps=accumulation_steps,
                )
                val_loss, _ = wrapper.validate_epoch(model, criterion, self.val_loader)
            except RuntimeError as exc:
                if _is_oom(exc):
                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()
                    print("  OOM — pruning trial.")
                    raise optuna.exceptions.TrialPruned()
                raise

            scheduler.step()

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_val_loss = min(best_val_loss, val_loss)
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"  Early stop at epoch {epoch}.")
                break

        return best_val_loss


STAGES = {"pretrain": PretrainObjective, "board": BoardObjective}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="barknet-AMIL hyperparameter search")
    parser.add_argument("-c", "--config", default="configs/config.yaml", type=str)
    parser.add_argument(
        "--stage", choices=list(STAGES), default=None,
        help="which stage to tune; overrides tune.stage in the config if given",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    stage = args.stage or cfg["tune"].get("stage", "board")
    if stage not in STAGES:
        raise ValueError(f"tune.stage must be one of {list(STAGES)}, got '{stage}'.")

    output_dir = Path(cfg["tune"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    search_cfg = cfg["tune"][stage]
    size = str(cfg["model"]["size"]).lower()
    weights = str(cfg["model"]["weights"]).lower()
    study_name = f"convnext_amil_{stage}_{weights}_{size}"
    db_uri = f"sqlite:///{(output_dir / f'{study_name}.db').as_posix()}"

    print(f"Tuning stage: {stage}  (study: {study_name})")

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=db_uri,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    objective = STAGES[stage](cfg)
    with timer(f"Hyperparameter Optimization [{stage}]"):
        study.optimize(objective, n_trials=search_cfg["trials"])

    best = study.best_trial
    print(f"\nBest val loss ({stage}): {best.value:.4f}")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    export = {
        "study_name": study_name,
        "stage": stage,
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

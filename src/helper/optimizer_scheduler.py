"""AdamW optimizer + linear-warmup -> cosine-decay scheduler.

Warmup length is expressed as a *fraction of total epochs* (``warmup_ratio``) so it
scales automatically when the epoch budget changes.

Differential learning rates apply when fine-tuning the AMIL model on a pretrained
backbone: the backbone moves slowly while the freshly initialised attention and
classifier heads move ``lr_multiplier`` times faster. From scratch, a single global
LR is used.
"""
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def warmup_epochs_from_ratio(num_epochs, warmup_ratio):
    """Convert a warmup fraction into an integer epoch count (>=1, <=num_epochs)."""
    return max(1, min(round(warmup_ratio * num_epochs), num_epochs))


def build_scheduler(optimizer, num_epochs, warmup_ratio, eta_min):
    """Linear warmup for ``warmup_ratio`` of the run, then cosine decay for the rest."""
    we = warmup_epochs_from_ratio(num_epochs, warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=we)
    cosine = CosineAnnealingLR(optimizer, T_max=max(num_epochs - we, 1), eta_min=eta_min)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[we])


def build_uniform_optimizer(model, base_lr, weight_decay):
    """Plain AdamW over all parameters (used for Stage-1 patch pretraining)."""
    return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)


def build_optimizer_and_scheduler(
    model,
    weights,
    num_epochs,
    warmup_ratio=0.1,
    base_lr=None,
    weight_decay=None,
    lr_multiplier=None,
):
    """Stage-2 (AMIL) optimizer + scheduler, with optional differential LRs."""
    from_scratch = str(weights).lower() == "none"

    if base_lr is None:
        base_lr = 1e-3 if from_scratch else 5e-5
    if weight_decay is None:
        weight_decay = 5e-2
    if lr_multiplier is None:
        lr_multiplier = 1 if from_scratch else 5

    if from_scratch:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(
            [
                {"params": model.extractor.parameters(), "lr": base_lr},
                {"params": model.attention.parameters(), "lr": base_lr * lr_multiplier},
                {"params": model.classifier.parameters(), "lr": base_lr * lr_multiplier},
            ],
            weight_decay=weight_decay,
        )

    scheduler = build_scheduler(optimizer, num_epochs, warmup_ratio, eta_min=base_lr * 0.01)
    return optimizer, scheduler

"""Early stopping on a monitored validation metric.

Supports either direction so it can watch the SAME quantity the checkpointer selects on:
label smoothing makes val_loss and val_acc disagree on this dataset (loss drifts up while
acc still climbs), so stopping on loss while checkpointing on acc would cut runs short for
the wrong reason. Construct with ``mode="max"`` when monitoring accuracy.
"""
import numpy as np


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Args:
        patience:   epochs to wait after the last improvement before stopping.
        min_delta:  minimum change counting as an improvement (always a positive number;
                    direction is handled by ``mode``).
        min_epochs: never stop before this many epochs. Set it above the warmup length so
                    the patience counter can't run out while the LR is still ramping and
                    the val metric is legitimately flat.
        mode:       "min" -> lower is better (val_loss); "max" -> higher is better (val_acc).
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0,
                 min_epochs: int = 10, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.min_epochs = min_epochs
        self.mode = mode
        self.counter = 0
        self.best = np.inf if mode == "min" else -np.inf
        self.early_stop = False
        self.epoch_count = 0

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def __call__(self, value: float) -> bool:
        self.epoch_count += 1

        # Track improvement every epoch so best/counter are always current, but only ARM
        # the stop after min_epochs -- otherwise a flat warmup plateau trips it.
        if self._is_improvement(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        if self.epoch_count > self.min_epochs and self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

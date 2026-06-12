"""Early stopping on a monitored validation metric (lower is better)."""
import numpy as np

class EarlyStopping:
    """Stop training when the monitored loss stops improving.

    Args:
        patience:  epochs to wait after the last improvement before stopping.
        min_delta: minimum decrease in the monitored loss that counts as an improvement.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
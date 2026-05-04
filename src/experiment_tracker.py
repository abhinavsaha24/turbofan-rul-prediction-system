"""
Lightweight experiment tracker.

Saves a structured JSON log after each training run containing:
  - Timestamp and duration
  - Full hyperparameter snapshot
  - Per-epoch train/val losses
  - Best metrics
  - System info (device, PyTorch version)

Logs are appended to outputs/experiments.jsonl — one JSON object per line,
making it trivial to parse, diff across runs, or load into a DataFrame.
"""

import json
import time
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import (
    OUTPUTS_DIR,
    SEQUENCE_LENGTH,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    ATTENTION_HEADS,
    DROPOUT,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    VAL_SPLIT,
    RANDOM_SEED,
    RUL_CAP,
    MC_SAMPLES,
)
from src.utils import get_logger

log = get_logger(__name__)

EXPERIMENTS_FILE = OUTPUTS_DIR / "experiments.jsonl"


def _system_info() -> dict:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "os": platform.system(),
        "machine": platform.machine(),
    }


def _hyperparameters() -> dict:
    return {
        "sequence_length": SEQUENCE_LENGTH,
        "rul_cap": RUL_CAP,
        "cnn_filters": CNN_FILTERS,
        "cnn_kernel_size": CNN_KERNEL_SIZE,
        "lstm_hidden": LSTM_HIDDEN,
        "lstm_layers": LSTM_LAYERS,
        "attention_heads": ATTENTION_HEADS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "mc_samples": MC_SAMPLES,
    }


class ExperimentTracker:
    """
    Context-manager–style tracker for a single training run.

    Usage:
        tracker = ExperimentTracker(model_name="cnn_lstm_attention")
        tracker.start()
        for epoch in range(N):
            ...
            tracker.log_epoch(epoch, train_loss, val_loss)
        tracker.finish(best_val_loss=..., test_rmse=..., test_mae=...)
    """

    def __init__(self, model_name: str = "cnn_lstm_attention"):
        self.model_name = model_name
        self.start_time = None
        self.epoch_log: list[dict] = []
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def start(self):
        self.start_time = time.time()
        log.info("Experiment %s started (model=%s)", self.run_id, self.model_name)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float = None):
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if lr is not None:
            entry["learning_rate"] = lr
        self.epoch_log.append(entry)

    def finish(self, **metrics):
        """
        Finalize the experiment: compute duration, assemble the record,
        and append it to the experiments JSONL file.
        """
        duration = time.time() - self.start_time if self.start_time else 0

        record = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 1),
            "epochs_completed": len(self.epoch_log),
            "hyperparameters": _hyperparameters(),
            "system": _system_info(),
            "metrics": metrics,
            "epoch_history": self.epoch_log,
        }

        # Append as a single JSON line
        with open(EXPERIMENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        log.info(
            "Experiment %s saved to %s (%.1fs, %d epochs)",
            self.run_id, EXPERIMENTS_FILE, duration, len(self.epoch_log),
        )
        return record

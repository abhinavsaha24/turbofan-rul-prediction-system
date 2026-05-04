"""
PyTorch Dataset for sliding-window sequence generation.

Each sample is a (sequence, target) pair where:
  - sequence: tensor of shape (SEQUENCE_LENGTH, n_features)
  - target:   scalar RUL at the end of that window
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import SEQUENCE_LENGTH
from src.utils import get_logger

log = get_logger(__name__)


class RULDataset(Dataset):
    """
    Generates sliding windows over per-engine time series.

    For engines whose total length is shorter than the window size,
    we pad with zeros on the left — this mirrors the real scenario
    where an engine has been running for fewer cycles than our
    lookback requires.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.seq_len = sequence_length
        self.samples: list[tuple[np.ndarray, float]] = []

        for eid, grp in df.groupby("engine_id"):
            features = grp[feature_cols].values  # (T, F)
            targets = grp["rul"].values           # (T,)

            # Slide a window across each engine's timeline
            for i in range(len(grp)):
                end = i + 1
                start = max(0, end - self.seq_len)
                window = features[start:end]

                # Left-pad if needed
                if window.shape[0] < self.seq_len:
                    pad = np.zeros((self.seq_len - window.shape[0], window.shape[1]))
                    window = np.vstack([pad, window])

                self.samples.append((window, targets[i]))

        log.info("Built %d sequences (window=%d)", len(self.samples), self.seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, target = self.samples[idx]
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def build_test_last_cycle_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[Dataset, np.ndarray]:
    """
    For test evaluation we only care about the *last* cycle of each engine.
    Returns a Dataset of those final windows and the true RUL array.
    """
    last_rows = df.groupby("engine_id").last().reset_index()
    engine_ids = last_rows["engine_id"].values

    samples = []
    true_ruls = []

    for eid in engine_ids:
        grp = df[df["engine_id"] == eid]
        features = grp[feature_cols].values
        rul_val = grp["rul"].values[-1]

        window = features[-sequence_length:]
        if window.shape[0] < sequence_length:
            pad = np.zeros((sequence_length - window.shape[0], window.shape[1]))
            window = np.vstack([pad, window])

        samples.append((window, rul_val))
        true_ruls.append(rul_val)

    class _TestDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            seq, target = self.data[idx]
            return (
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
            )

    return _TestDataset(samples), np.array(true_ruls)

"""
Data ingestion and preprocessing for NASA CMAPSS FD001.

Responsibilities:
  1. Parse the whitespace-delimited raw text files.
  2. Compute piece-wise linear RUL targets for training data.
  3. Merge ground-truth RUL into test data.
  4. Drop constant / uninformative sensors.
  5. Normalize features using min-max scaling (fitted on train only).
  6. Persist processed DataFrames to data/processed/ for downstream use.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.config import (
    DATASET_DIR,
    PROCESSED_DIR,
    ALL_COLS,
    SETTING_COLS,
    SENSOR_COLS,
    DROP_SENSORS,
    DROP_SETTINGS,
    RUL_CAP,
)
from src.utils import get_logger

log = get_logger(__name__)


# ── Raw file loading ────────────────────────────────────────────────────────

def _load_txt(filename: str) -> pd.DataFrame:
    """Read a CMAPSS whitespace-separated file into a DataFrame."""
    path = DATASET_DIR / filename
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == len(ALL_COLS):
        df.columns = ALL_COLS
    elif df.shape[1] == 1:
        # RUL file — single column
        df.columns = ["rul"]
    else:
        raise ValueError(
            f"Unexpected column count {df.shape[1]} in {filename}"
        )
    return df


def load_raw_data():
    """Return (train_df, test_df, rul_df) with proper column names."""
    train = _load_txt("train_FD001.txt")
    test = _load_txt("test_FD001.txt")
    rul = _load_txt("RUL_FD001.txt")
    log.info(
        "Raw data loaded — train: %s, test: %s, rul: %s",
        train.shape, test.shape, rul.shape,
    )
    return train, test, rul


# ── RUL computation ────────────────────────────────────────────────────────

def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each engine, RUL = max_cycle - current_cycle, then clipped
    to RUL_CAP so the model doesn't waste capacity on far-future values.
    """
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=RUL_CAP)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def attach_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row in the RUL file gives the remaining life at the *last* cycle
    of the corresponding engine in the test set.  We compute per-row RUL
    by working backwards from that final value.
    """
    max_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    rul_df = rul_df.copy()
    rul_df["engine_id"] = rul_df.index + 1  # 1-indexed engine IDs

    test_df = test_df.merge(max_cycles, on="engine_id")
    test_df = test_df.merge(rul_df, on="engine_id")

    test_df["rul"] = test_df["rul"] + test_df["max_cycle"] - test_df["cycle"]
    test_df["rul"] = test_df["rul"].clip(upper=RUL_CAP)
    test_df.drop(columns=["max_cycle"], inplace=True)
    return test_df


# ── Feature engineering ─────────────────────────────────────────────────────

def _feature_columns() -> list[str]:
    """Return the sensor + setting columns we actually keep."""
    keep = [c for c in SENSOR_COLS + SETTING_COLS
            if c not in DROP_SENSORS and c not in DROP_SETTINGS]
    return keep


def drop_and_normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    1. Drop constant/uninformative columns.
    2. Fit a MinMaxScaler on training features, transform both splits.
    3. Save the scaler for later inference.
    """
    feat_cols = _feature_columns()
    scaler = MinMaxScaler()

    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    # Persist scaler so the Streamlit app can reuse it
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    log.info("Scaler saved to %s", scaler_path)

    return train_df, test_df, feat_cols


# ── Main preprocessing pipeline ────────────────────────────────────────────

def run_preprocessing():
    """
    End-to-end preprocessing pipeline.  Returns processed DataFrames and
    the list of feature column names used for modelling.
    """
    train_df, test_df, rul_df = load_raw_data()

    # Compute targets
    train_df = compute_rul(train_df)
    test_df = attach_test_rul(test_df, rul_df)

    # Normalize
    train_df, test_df, feat_cols = drop_and_normalize(train_df, test_df)

    # Save
    train_df.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)
    log.info(
        "Processed data saved — train: %s, test: %s, features: %d",
        train_df.shape, test_df.shape, len(feat_cols),
    )
    return train_df, test_df, feat_cols


if __name__ == "__main__":
    run_preprocessing()

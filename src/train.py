"""
Training loop for RUL prediction models.

Supports:
  - CNN-BiLSTM-Attention (default)
  - Transformer (--model transformer)
  - Engine-level train/val split (no data leakage)
  - Epoch-level logging with experiment tracking
  - Early stopping with patience
  - Best-model checkpointing
  - ReduceLROnPlateau scheduling

Usage:
    python -m src.train                    # train CNN-LSTM-Attention
    python -m src.train --model transformer  # train Transformer
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    VAL_SPLIT,
    RANDOM_SEED,
    MODELS_DIR,
    BEST_MODEL_NAME,
    SEQUENCE_LENGTH,
)
from src.preprocess import run_preprocessing
from src.dataset import RULDataset
from src.model import CNNLSTMAttention
from src.transformer_model import TransformerRUL
from src.experiment_tracker import ExperimentTracker
from src.utils import seed_everything, get_device, get_logger

log = get_logger(__name__)

MODEL_REGISTRY = {
    "cnn_lstm_attention": CNNLSTMAttention,
    "transformer": TransformerRUL,
}

CHECKPOINT_NAMES = {
    "cnn_lstm_attention": BEST_MODEL_NAME,
    "transformer": "best_transformer.pt",
}


def _split_by_engine(df, val_frac: float = VAL_SPLIT):
    """
    Split engines (not rows) into train and validation sets.
    This prevents any temporal leakage between the two splits.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    engine_ids = df["engine_id"].unique()
    rng.shuffle(engine_ids)
    split = int(len(engine_ids) * (1 - val_frac))
    train_ids = engine_ids[:split]
    val_ids = engine_ids[split:]
    return (
        df[df["engine_id"].isin(train_ids)].copy(),
        df[df["engine_id"].isin(val_ids)].copy(),
    )


def train(model_name: str = "cnn_lstm_attention"):
    seed_everything()
    device = get_device()
    log.info("Device: %s", device)
    log.info("Model: %s", model_name)

    # ── Data ────────────────────────────────────────────────────────────
    train_df, _test_df, feat_cols = run_preprocessing()
    train_sub, val_sub = _split_by_engine(train_df)

    train_ds = RULDataset(train_sub, feat_cols, SEQUENCE_LENGTH)
    val_ds = RULDataset(val_sub, feat_cols, SEQUENCE_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # ── Model ───────────────────────────────────────────────────────────
    n_features = len(feat_cols)
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{total_params:,}")

    # ── Experiment tracking ─────────────────────────────────────────────
    tracker = ExperimentTracker(model_name=model_name)
    tracker.start()

    # ── Training loop ───────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    ckpt_name = CHECKPOINT_NAMES.get(model_name, f"best_{model_name}.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # — Train —
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # — Validate —
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Track
        tracker.log_epoch(epoch, train_loss, val_loss, lr)

        log.info(
            "Epoch %3d/%d  |  train_loss: %.4f  |  val_loss: %.4f  |  lr: %.2e  |  %.1fs",
            epoch, EPOCHS, train_loss, val_loss, lr, elapsed,
        )

        # — Early stopping —
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = MODELS_DIR / ckpt_name
            torch.save(model.state_dict(), ckpt_path)
            log.info("  ↳ best model saved (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log.info("Early stopping triggered at epoch %d", epoch)
                break

    log.info("Training complete. Best val_loss: %.4f", best_val_loss)

    # ── Save experiment log ─────────────────────────────────────────────
    tracker.finish(
        best_val_loss=float(best_val_loss),
        total_params=total_params,
        epochs_trained=len(history["train_loss"]),
    )

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RUL prediction model")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="cnn_lstm_attention",
        help="Which architecture to train (default: cnn_lstm_attention)",
    )
    args = parser.parse_args()
    train(model_name=args.model)

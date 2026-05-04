"""
CNN + BiLSTM + Multi-Head Attention model for RUL prediction.

Architecture overview:
  1. 1-D CNN extracts local temporal patterns from the sensor window.
  2. Bidirectional LSTM captures long-range sequential dependencies.
  3. Multi-head self-attention re-weights time steps so the model can
     focus on the most informative part of the degradation curve.
  4. A small MLP head maps the attended representation to a scalar RUL.

Dropout is applied at multiple stages so we can use Monte Carlo Dropout
at inference time to quantify predictive uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    ATTENTION_HEADS,
    DROPOUT,
)


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()

        # ── 1-D Convolutional feature extractor ────────────────────────
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=CNN_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            padding=CNN_KERNEL_SIZE // 2,
        )
        self.bn1 = nn.BatchNorm1d(CNN_FILTERS)
        self.conv2 = nn.Conv1d(
            in_channels=CNN_FILTERS,
            out_channels=CNN_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            padding=CNN_KERNEL_SIZE // 2,
        )
        self.bn2 = nn.BatchNorm1d(CNN_FILTERS)
        self.cnn_drop = nn.Dropout(DROPOUT)

        # ── Bidirectional LSTM ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=CNN_FILTERS,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(DROPOUT)

        # ── Multi-head self-attention ──────────────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=LSTM_HIDDEN * 2,  # bidirectional doubles the dim
            num_heads=ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(LSTM_HIDDEN * 2)

        # ── Regression head ────────────────────────────────────────────
        self.fc1 = nn.Linear(LSTM_HIDDEN * 2, 64)
        self.fc_drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            rul: (batch,)
        """
        # CNN expects (batch, channels, seq_len)
        c = x.permute(0, 2, 1)
        c = self.cnn_drop(F.relu(self.bn1(self.conv1(c))))
        c = self.cnn_drop(F.relu(self.bn2(self.conv2(c))))
        c = c.permute(0, 2, 1)  # back to (batch, seq_len, filters)

        # LSTM
        lstm_out, _ = self.lstm(c)        # (batch, seq_len, hidden*2)
        lstm_out = self.lstm_drop(lstm_out)

        # Self-attention (query = key = value)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # residual

        # Use the last time-step's representation
        last = attn_out[:, -1, :]         # (batch, hidden*2)

        out = F.relu(self.fc1(last))
        out = self.fc_drop(out)
        out = self.fc2(out).squeeze(-1)   # (batch,)
        return out

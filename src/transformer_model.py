"""
Time-Series Transformer for RUL prediction.

A pure-attention architecture that replaces the LSTM backbone with
stacked Transformer encoder layers.  The key idea: sensor degradation
is a sequence-to-one mapping, and self-attention over the full window
can capture both local transitions and long-range trends without the
sequential bottleneck of recurrence.

Architecture:
  1. Learnable positional encoding (not sinusoidal — the positions have
     physical meaning as cycle offsets, so learned embeddings fit better).
  2. Linear projection from raw feature dim to model dim.
  3. N Transformer encoder layers with pre-norm (more stable training).
  4. Global average pooling over the time axis (more robust than
     taking just the last token).
  5. Regression head with dropout for MC Dropout compatibility.

Design notes:
  - Causal masking is NOT used.  Unlike language, future sensor readings
    within the window are legitimate context for predicting RUL.
  - Dropout is placed at every residual path so Monte Carlo Dropout
    works out of the box with the same inference code as the LSTM model.
"""

import torch
import torch.nn as nn

from src.config import (
    SEQUENCE_LENGTH,
    DROPOUT,
    ATTENTION_HEADS,
)

# ── Transformer-specific hyperparameters ───────────────────────────────────
# These can be moved to config.py if needed, but keeping them here
# avoids polluting the shared config with model-specific knobs.
TRANSFORMER_DIM = 128          # internal model dimension
TRANSFORMER_LAYERS = 4         # number of encoder blocks
TRANSFORMER_FFN_DIM = 256      # feed-forward expansion dim


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings for the sensor window."""

    def __init__(self, d_model: int, max_len: int = SEQUENCE_LENGTH):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1), :]


class TransformerRUL(nn.Module):
    """
    Transformer encoder for Remaining Useful Life regression.

    Compatible with the existing dataset pipeline — expects input of
    shape (batch, seq_len, n_features) and outputs (batch,).
    """

    def __init__(self, n_features: int):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, TRANSFORMER_DIM),
            nn.LayerNorm(TRANSFORMER_DIM),
            nn.Dropout(DROPOUT),
        )

        self.pos_enc = PositionalEncoding(TRANSFORMER_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=TRANSFORMER_FFN_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(TRANSFORMER_DIM),
        )

        self.head = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            rul: (batch,)
        """
        # Project to model dimension
        h = self.input_proj(x)              # (B, T, D)
        h = self.pos_enc(h)

        # Transformer encoder (no causal mask — full attention is valid)
        h = self.encoder(h)                 # (B, T, D)

        # Global average pool over the time axis
        h = h.mean(dim=1)                   # (B, D)

        return self.head(h).squeeze(-1)     # (B,)

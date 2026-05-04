"""
FastAPI service for RUL prediction with uncertainty.

Exposes a POST /predict endpoint that accepts a sensor sequence
and returns the predicted RUL with Monte Carlo Dropout confidence
intervals.

Run:
    uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    MODELS_DIR,
    PROCESSED_DIR,
    BEST_MODEL_NAME,
    SEQUENCE_LENGTH,
    MC_SAMPLES,
    SENSOR_COLS,
    SETTING_COLS,
    DROP_SENSORS,
    DROP_SETTINGS,
)
from src.model import CNNLSTMAttention
from src.utils import get_device

# ── App setup ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model and scaler on startup."""
    _load_resources()
    yield


app = FastAPI(
    title="Turbofan RUL Prediction API",
    description=(
        "Predict the Remaining Useful Life (RUL) of turbofan engines "
        "from sensor time-series data, with Monte Carlo Dropout "
        "uncertainty quantification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Feature columns (must match training) ───────────────────────────────────

def _feature_columns() -> list[str]:
    return [c for c in SENSOR_COLS + SETTING_COLS
            if c not in DROP_SENSORS and c not in DROP_SETTINGS]

FEAT_COLS = _feature_columns()
N_FEATURES = len(FEAT_COLS)


# ── Model loading (singleton) ──────────────────────────────────────────────

_model = None
_device = None
_scaler = None


def _load_resources():
    global _model, _device, _scaler

    if _model is not None:
        return

    _device = get_device()

    # Load model
    _model = CNNLSTMAttention(N_FEATURES)
    ckpt = MODELS_DIR / BEST_MODEL_NAME
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt}. Run `python -m src.train` first."
        )
    _model.load_state_dict(
        torch.load(ckpt, map_location=_device, weights_only=True)
    )
    _model.to(_device)

    # Load scaler
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"No scaler found at {scaler_path}. Run `python -m src.preprocess` first."
        )
    _scaler = joblib.load(scaler_path)


# ── Request / Response schemas ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input: a 2D array of sensor readings.
    Shape: (n_timesteps, n_features) where n_features = len(FEAT_COLS).

    If n_timesteps < SEQUENCE_LENGTH, the sequence will be left-padded.
    If n_timesteps > SEQUENCE_LENGTH, only the last SEQUENCE_LENGTH steps
    are used.

    Feature order must match the training pipeline:
    sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9,
    sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17,
    sensor_20, sensor_21, setting_1, setting_2
    """
    sequence: list[list[float]] = Field(
        ...,
        description=f"2D array of shape (timesteps, {N_FEATURES})",
    )
    mc_samples: int = Field(
        default=MC_SAMPLES,
        ge=1,
        le=500,
        description="Number of Monte Carlo Dropout forward passes",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to apply MinMax scaling (set False if already normalized)",
    )


class PredictResponse(BaseModel):
    predicted_rul: float = Field(..., description="Mean RUL prediction (cycles)")
    uncertainty_std: float = Field(..., description="Standard deviation across MC passes")
    ci_lower_95: float = Field(..., description="Lower bound of 95% CI")
    ci_upper_95: float = Field(..., description="Upper bound of 95% CI")
    mc_samples_used: int = Field(..., description="Number of MC Dropout passes used")
    sequence_length_used: int = Field(..., description="Effective window length after padding/trimming")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    n_features: int
    sequence_length: int


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if the API is healthy and the model is loaded."""
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model_loaded=_model is not None,
        device=str(_device) if _device else "unknown",
        n_features=N_FEATURES,
        sequence_length=SEQUENCE_LENGTH,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Predict RUL for a single engine given its recent sensor history.

    Returns the mean prediction, standard deviation, and 95% confidence
    interval from Monte Carlo Dropout inference.
    """
    _load_resources()

    # ── Validate input shape ────────────────────────────────────────────
    seq = np.array(req.sequence, dtype=np.float64)
    if seq.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 2D array, got shape {seq.shape}",
        )
    if seq.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {N_FEATURES} features per timestep, "
                f"got {seq.shape[1]}. Feature order: {FEAT_COLS}"
            ),
        )

    # ── Normalize if requested ──────────────────────────────────────────
    if req.normalize:
        seq = _scaler.transform(seq)

    # ── Window extraction ───────────────────────────────────────────────
    if seq.shape[0] > SEQUENCE_LENGTH:
        seq = seq[-SEQUENCE_LENGTH:]
    elif seq.shape[0] < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - seq.shape[0], N_FEATURES))
        seq = np.vstack([pad, seq])

    effective_len = seq.shape[0]

    # ── MC Dropout inference ────────────────────────────────────────────
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(_device)
    _model.train()  # keep dropout active

    preds = []
    with torch.no_grad():
        for _ in range(req.mc_samples):
            pred = _model(tensor).cpu().item()
            preds.append(pred)

    preds_arr = np.array(preds)
    mean_rul = float(preds_arr.mean())
    std_rul = float(preds_arr.std())

    return PredictResponse(
        predicted_rul=round(mean_rul, 2),
        uncertainty_std=round(std_rul, 2),
        ci_lower_95=round(max(0.0, mean_rul - 2 * std_rul), 2),
        ci_upper_95=round(mean_rul + 2 * std_rul, 2),
        mc_samples_used=req.mc_samples,
        sequence_length_used=effective_len,
    )


@app.get("/model/info")
async def model_info():
    """Return model architecture metadata."""
    _load_resources()
    total_params = sum(p.numel() for p in _model.parameters())
    trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    return {
        "architecture": "CNN-BiLSTM-Attention",
        "total_parameters": total_params,
        "trainable_parameters": trainable,
        "feature_columns": FEAT_COLS,
        "sequence_length": SEQUENCE_LENGTH,
        "mc_dropout_default": MC_SAMPLES,
    }

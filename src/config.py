"""
Central configuration for the Turbofan RUL Prediction System.

All paths, hyperparameters, and feature definitions live here so that
every other module can import a single source of truth.
"""

from pathlib import Path

# ── Project root (two levels up from src/) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ──────────────────────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Create dirs lazily so imports alone don't fail on a fresh clone
for _dir in [PROCESSED_DIR, MODELS_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── CMAPSS column schema ────────────────────────────────────────────────────
INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors that are practically constant across all engines and provide
# zero predictive signal — identified during EDA.
DROP_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# Settings columns are operational conditions, not degradation features
DROP_SETTINGS = ["setting_3"]

# ── RUL cap ─────────────────────────────────────────────────────────────────
# Piecewise-linear RUL: cap the maximum target at 125 cycles.
# Beyond this horizon the engine is essentially healthy and predicting
# exact remaining life is neither possible nor useful.
RUL_CAP = 125

# ── Sequence modelling ──────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30  # sliding-window width (time steps)

# ── Model architecture ──────────────────────────────────────────────────────
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
ATTENTION_HEADS = 4
DROPOUT = 0.3

# ── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 80
PATIENCE = 12          # early-stopping patience
VAL_SPLIT = 0.2        # fraction of training engines held out
RANDOM_SEED = 42

# ── Monte Carlo Dropout ────────────────────────────────────────────────────
MC_SAMPLES = 50  # number of stochastic forward passes at inference

# ── Best model checkpoint name ──────────────────────────────────────────────
BEST_MODEL_NAME = "best_cnn_lstm_attention.pt"

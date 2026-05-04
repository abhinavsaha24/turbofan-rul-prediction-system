"""
Streamlit app for interactive RUL prediction with uncertainty.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DATASET_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    BEST_MODEL_NAME,
    ALL_COLS,
    SENSOR_COLS,
    SEQUENCE_LENGTH,
    MC_SAMPLES,
    DROP_SENSORS,
    DROP_SETTINGS,
    SETTING_COLS,
)
from src.model import CNNLSTMAttention
from src.utils import get_device

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="wide",
)

# ── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 16px; border-radius: 12px; }
    .stMetric label { color: #e0e0e0 !important; }
    .stMetric [data-testid="stMetricValue"] { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    feat_cols = _feature_columns()
    n_features = len(feat_cols)
    device = get_device()
    model = CNNLSTMAttention(n_features)
    ckpt = MODELS_DIR / BEST_MODEL_NAME
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)
        return model, device
    return None, device


def _feature_columns():
    keep = [c for c in SENSOR_COLS + SETTING_COLS
            if c not in DROP_SENSORS and c not in DROP_SETTINGS]
    return keep


def prepare_engine_data(df: pd.DataFrame):
    """Normalize and create sliding window for the last cycle."""
    feat_cols = _feature_columns()
    scaler_path = PROCESSED_DIR / "scaler.pkl"

    if not scaler_path.exists():
        st.error("Scaler not found. Please run the training pipeline first.")
        return None, None

    scaler = joblib.load(scaler_path)
    df[feat_cols] = scaler.transform(df[feat_cols])

    windows = {}
    for eid in df["engine_id"].unique():
        grp = df[df["engine_id"] == eid]
        features = grp[feat_cols].values
        window = features[-SEQUENCE_LENGTH:]
        if window.shape[0] < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - window.shape[0], window.shape[1]))
            window = np.vstack([pad, window])
        windows[eid] = window

    return windows, feat_cols


def mc_predict(model, window: np.ndarray, device, n_samples: int = MC_SAMPLES):
    """MC Dropout forward passes for a single window."""
    model.train()  # dropout active
    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(tensor).cpu().item()
            preds.append(pred)
    return np.mean(preds), np.std(preds)


# ── Main app ────────────────────────────────────────────────────────────────

def main():
    st.title("✈️ Turbofan Engine — Remaining Useful Life Predictor")
    st.markdown(
        "Predict how many operational cycles an engine has left before failure, "
        "with uncertainty quantification via Monte Carlo Dropout."
    )

    model, device = load_model()
    if model is None:
        st.warning(
            "⚠️ No trained model found. Run `python -m src.train` first, "
            "then restart this app."
        )
        st.stop()

    st.sidebar.header("Data Source")
    source = st.sidebar.radio(
        "Choose input", ["Use existing test data (FD001)", "Upload custom file"]
    )

    if source == "Use existing test data (FD001)":
        test_path = DATASET_DIR / "test_FD001.txt"
        if not test_path.exists():
            st.error(f"Test file not found at {test_path}")
            st.stop()
        df = pd.read_csv(test_path, sep=r"\s+", header=None)
        df.columns = ALL_COLS
    else:
        uploaded = st.sidebar.file_uploader("Upload CMAPSS-format .txt", type=["txt", "csv"])
        if uploaded is None:
            st.info("Upload a CMAPSS-format file to get started.")
            st.stop()
        df = pd.read_csv(uploaded, sep=r"\s+", header=None)
        df.columns = ALL_COLS

    windows, feat_cols = prepare_engine_data(df)
    if windows is None:
        st.stop()

    engine_ids = sorted(windows.keys())

    st.sidebar.header("Settings")
    n_mc = st.sidebar.slider("MC Dropout passes", 10, 200, MC_SAMPLES)

    # ── Predict all engines ─────────────────────────────────────────────
    if st.button("🔮 Run Predictions", type="primary"):
        results = []
        progress = st.progress(0.0)
        for i, eid in enumerate(engine_ids):
            mean_rul, std_rul = mc_predict(model, windows[eid], device, n_mc)
            results.append({
                "Engine": int(eid),
                "Predicted RUL (mean)": round(mean_rul, 1),
                "Uncertainty (±std)": round(std_rul, 1),
                "Lower 95% CI": round(max(0, mean_rul - 2 * std_rul), 1),
                "Upper 95% CI": round(mean_rul + 2 * std_rul, 1),
            })
            progress.progress((i + 1) / len(engine_ids))

        results_df = pd.DataFrame(results)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Engines Evaluated", len(results))
        col2.metric("Avg Predicted RUL", f"{results_df['Predicted RUL (mean)'].mean():.1f} cycles")
        col3.metric("Avg Uncertainty", f"±{results_df['Uncertainty (±std)'].mean():.1f} cycles")

        # Table
        st.subheader("Per-Engine Predictions")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Plot
        st.subheader("Predicted RUL with Confidence Intervals")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(
            results_df["Engine"],
            results_df["Predicted RUL (mean)"],
            yerr=2 * results_df["Uncertainty (±std)"],
            capsize=2,
            color="#667eea",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlabel("Engine Unit")
        ax.set_ylabel("Predicted RUL (cycles)")
        ax.set_title("RUL Predictions with 95% Confidence Intervals")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()

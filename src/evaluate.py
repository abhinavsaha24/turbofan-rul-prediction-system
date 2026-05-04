"""
Evaluation module: metrics, Monte Carlo Dropout inference, and plot generation.

Produces:
  - RMSE, MAE, and NASA scoring function on the test set
  - MC Dropout uncertainty estimates (mean ± std for each engine)
  - Predicted vs Actual RUL with confidence intervals
  - Error distribution with statistical annotation
  - Sensor degradation trends (multi-engine overlay)
  - Uncertainty calibration plot
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    BATCH_SIZE,
    MC_SAMPLES,
    MODELS_DIR,
    PLOTS_DIR,
    BEST_MODEL_NAME,
    SEQUENCE_LENGTH,
    DATASET_DIR,
    ALL_COLS,
    SENSOR_COLS,
)
from src.preprocess import run_preprocessing
from src.dataset import build_test_last_cycle_dataset
from src.model import CNNLSTMAttention
from src.utils import get_device, get_logger

log = get_logger(__name__)

# ── Plot styling ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "grid.alpha": 0.6,
})

PALETTE = {
    "primary": "#2563eb",
    "secondary": "#f97316",
    "accent": "#10b981",
    "error": "#ef4444",
    "muted": "#6b7280",
    "fill": "#dbeafe",
}


# ── Metrics ─────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA's asymmetric scoring function for PHM08.
    Late predictions (under-estimating RUL) are penalized exponentially
    harder than early ones (over-estimating).
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


# ── MC Dropout inference ────────────────────────────────────────────────────

def mc_dropout_predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = MC_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model n_samples times with dropout enabled.
    Returns (mean_predictions, std_predictions).
    """
    model.train()  # keep dropout active
    all_preds = []

    for _ in range(n_samples):
        batch_preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(device)
                pred = model(X)
                batch_preds.append(pred.cpu().numpy())
        all_preds.append(np.concatenate(batch_preds))

    stacked = np.stack(all_preds, axis=0)  # (n_samples, n_engines)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ── Visualization ──────────────────────────────────────────────────────────

def plot_pred_vs_actual(y_true, y_pred, y_std=None):
    """Publication-quality predicted vs actual with confidence bands."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})

    # — Left: per-engine comparison —
    ax = axes[0]
    engines = np.arange(1, len(y_true) + 1)

    ax.plot(engines, y_true, "o-", color=PALETTE["primary"], label="Actual RUL",
            alpha=0.85, markersize=3.5, linewidth=1.2)
    ax.plot(engines, y_pred, "s-", color=PALETTE["secondary"], label="Predicted RUL",
            alpha=0.85, markersize=3.5, linewidth=1.2)

    if y_std is not None:
        ax.fill_between(
            engines,
            np.maximum(0, y_pred - 2 * y_std),
            y_pred + 2 * y_std,
            alpha=0.15,
            color=PALETTE["secondary"],
            label="95% CI (MC Dropout)",
        )

    ax.set_xlabel("Engine Unit")
    ax.set_ylabel("Remaining Useful Life (cycles)")
    ax.set_title("Predicted vs Actual RUL — FD001 Test Set")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True)
    ax.set_xlim(0, len(y_true) + 1)

    # — Right: scatter with perfect-prediction line —
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.6, s=25, c=PALETTE["primary"], edgecolors="white", linewidth=0.5)
    lims = [0, max(y_true.max(), y_pred.max()) + 10]
    ax2.plot(lims, lims, "--", color=PALETTE["error"], linewidth=1.2, label="Perfect prediction")
    ax2.set_xlabel("Actual RUL (cycles)")
    ax2.set_ylabel("Predicted RUL (cycles)")
    ax2.set_title("Prediction Accuracy Scatter")
    ax2.legend(fontsize=9)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect("equal")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pred_vs_actual.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved pred_vs_actual.png")


def plot_error_distribution(y_true, y_pred):
    """Error distribution with statistical annotations."""
    errors = y_pred - y_true

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(errors, bins=25, kde=True, ax=ax, color=PALETTE["primary"],
                 edgecolor="white", linewidth=0.8, alpha=0.7)

    # Reference line
    ax.axvline(0, color=PALETTE["error"], linestyle="--", linewidth=1.5, label="Zero error")

    # Statistical annotations
    mean_err = errors.mean()
    median_err = np.median(errors)
    ax.axvline(mean_err, color=PALETTE["accent"], linestyle="-.", linewidth=1.2,
               label=f"Mean = {mean_err:.1f}")
    ax.axvline(median_err, color=PALETTE["secondary"], linestyle=":", linewidth=1.2,
               label=f"Median = {median_err:.1f}")

    # Text box with stats
    textstr = (
        f"μ = {mean_err:.2f}\n"
        f"σ = {errors.std():.2f}\n"
        f"|errors| ≤ 10: {(np.abs(errors) <= 10).sum()}/{len(errors)}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    ax.text(0.97, 0.95, textstr, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    ax.set_xlabel("Prediction Error (Predicted − Actual)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution — Test Set")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "error_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved error_distribution.png")


def plot_sensor_degradation():
    """Multi-engine sensor degradation overlay — shows consistency of failure patterns."""
    path = DATASET_DIR / "train_FD001.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = ALL_COLS

    sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]
    n_engines = 8  # overlay count

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.ravel()

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_engines))

    for i, sensor in enumerate(sensors):
        ax = axes[i]
        for j in range(1, n_engines + 1):
            eng = df[df["engine_id"] == j]
            ax.plot(eng["cycle"], eng[sensor], linewidth=0.7, alpha=0.7, color=cmap[j - 1])
        ax.set_title(sensor, fontsize=12, fontweight="bold")
        ax.set_ylabel("Sensor Value")
        ax.grid(True)
        if i >= 3:
            ax.set_xlabel("Cycle")

    fig.suptitle(
        "Sensor Degradation Trends — Engines 1–8 (run-to-failure)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sensor_degradation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved sensor_degradation.png")


def plot_uncertainty_calibration(y_true, y_pred, y_std):
    """
    Calibration plot: for engines sorted by predicted uncertainty,
    do higher-uncertainty predictions actually have larger errors?
    """
    abs_err = np.abs(y_pred - y_true)
    sort_idx = np.argsort(y_std)
    sorted_std = y_std[sort_idx]
    sorted_err = abs_err[sort_idx]

    # Rolling average for smoothing
    window = max(5, len(y_true) // 10)
    rolling_err = np.convolve(sorted_err, np.ones(window) / window, mode="valid")
    rolling_std = np.convolve(sorted_std, np.ones(window) / window, mode="valid")
    x_axis = np.arange(len(rolling_err))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, rolling_err, color=PALETTE["error"], linewidth=1.5,
            label="Rolling |Error|")
    ax.plot(x_axis, rolling_std, color=PALETTE["primary"], linewidth=1.5,
            label="Rolling Uncertainty (σ)")
    ax.fill_between(x_axis, rolling_std, alpha=0.15, color=PALETTE["primary"])

    ax.set_xlabel("Engines (sorted by increasing uncertainty)")
    ax.set_ylabel("Cycles")
    ax.set_title("Uncertainty Calibration — Does Higher σ Mean Higher Error?")
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "uncertainty_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved uncertainty_calibration.png")


# ── Main evaluation pipeline ───────────────────────────────────────────────

def evaluate():
    device = get_device()
    log.info("Device: %s", device)

    # Load processed data
    _train_df, test_df, feat_cols = run_preprocessing()

    # Build test dataset (last cycle per engine)
    test_ds, true_ruls = build_test_last_cycle_dataset(test_df, feat_cols, SEQUENCE_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load best model
    n_features = len(feat_cols)
    model = CNNLSTMAttention(n_features).to(device)
    ckpt = MODELS_DIR / BEST_MODEL_NAME
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    log.info("Loaded checkpoint: %s", ckpt)

    # ── Deterministic evaluation ────────────────────────────────────────
    model.eval()
    preds_det = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            pred = model(X)
            preds_det.append(pred.cpu().numpy())
    preds_det = np.concatenate(preds_det)

    test_rmse = rmse(true_ruls, preds_det)
    test_mae = mae(true_ruls, preds_det)
    test_score = nasa_score(true_ruls, preds_det)
    log.info("Test RMSE: %.4f", test_rmse)
    log.info("Test MAE:  %.4f", test_mae)
    log.info("NASA Score: %.2f", test_score)

    # ── MC Dropout evaluation ───────────────────────────────────────────
    mc_mean, mc_std = mc_dropout_predict(model, test_loader, device)
    mc_rmse = rmse(true_ruls, mc_mean)
    log.info("MC Dropout RMSE: %.4f (mean of %d passes)", mc_rmse, MC_SAMPLES)

    # ── Plots ───────────────────────────────────────────────────────────
    plot_pred_vs_actual(true_ruls, mc_mean, mc_std)
    plot_error_distribution(true_ruls, mc_mean)
    plot_sensor_degradation()
    plot_uncertainty_calibration(true_ruls, mc_mean, mc_std)

    return {
        "rmse": test_rmse,
        "mae": test_mae,
        "nasa_score": test_score,
        "mc_rmse": mc_rmse,
        "mc_std_mean": float(mc_std.mean()),
    }


if __name__ == "__main__":
    results = evaluate()
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")
    print("=" * 50)

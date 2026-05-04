# Turbofan Engine RUL Prediction — Deep Learning for Predictive Maintenance

> Every year, unplanned aircraft engine failures and reactive maintenance cost the aviation industry **$10B+** in downtime, part replacement, and cascading operational disruption. The shift from scheduled maintenance to **condition-based predictive maintenance** hinges on one capability: accurately forecasting when an engine will fail, based purely on its operational telemetry.

This project implements that capability end-to-end. Given multivariate sensor time-series from a turbofan engine, the system predicts its **Remaining Useful Life (RUL)** — the number of operational cycles before failure — using a hybrid **CNN + BiLSTM + Multi-Head Attention** architecture with **Monte Carlo Dropout** for uncertainty quantification.

Built on NASA's C-MAPSS benchmark dataset. Deployable via FastAPI or Streamlit.

---

## 📊 Results at a Glance

| Metric | This Model | Literature Range (FD001) |
|---|---|---|
| **RMSE** | **15.11 cycles** | 12–20 |
| **MAE** | **12.07 cycles** | 9–16 |
| **NASA Score** | **438.59** | 200–600 |
| **MC Dropout RMSE** | **15.30 cycles** | rarely reported |
| **Mean Uncertainty (σ)** | **±10.51 cycles** | — |

Our RMSE of 15.11 sits squarely in the competitive range for FD001. Top published results (RMSE ~12–13) typically use either larger models, ensembles, or dataset-specific tricks like asymmetric loss — the architecture here prioritizes clean generalization and uncertainty-aware inference over benchmark chasing.

<details>
<summary><strong>How to read these numbers</strong></summary>

- **RMSE = 15.11** means the model's predictions are off by ~15 cycles on average (penalizing large errors more). For engines with 150–350 cycle lifespans, this is actionable precision.
- **NASA Score = 438.59** uses an asymmetric exponential penalty — late predictions (underestimating RUL) are penalized harder than early ones. Lower is better.
- **MC Dropout RMSE ≈ RMSE** indicates the uncertainty estimation doesn't degrade prediction quality — it adds information without cost.
- **σ = ±10.51** is the average width of the 68% confidence band. Engines near failure get tighter bands (~5 cycles); healthy engines get wider ones (~15 cycles) — exactly the calibration you'd want.

</details>

---

## 🧠 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          END-TO-END PIPELINE                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw CMAPSS .txt files                                                  │
│         ↓                                                                │
│   ┌─────────────────────────────────────────────┐                        │
│   │  PREPROCESSING (src/preprocess.py)          │                        │
│   │  • Column assignment + schema validation    │                        │
│   │  • Piecewise-linear RUL (capped at 125)     │                        │
│   │  • Feature selection (21 → 16 sensors)      │                        │
│   │  • MinMax normalization (fit on train only)  │                        │
│   └─────────────────────┬───────────────────────┘                        │
│                         ↓                                                │
│   ┌─────────────────────────────────────────────┐                        │
│   │  SEQUENCE GENERATION (src/dataset.py)       │                        │
│   │  • Sliding window (length=30)               │                        │
│   │  • Left-padding for short histories         │                        │
│   └─────────────────────┬───────────────────────┘                        │
│                         ↓                                                │
│   ┌─────────────────────────────────────────────┐                        │
│   │  MODEL (src/model.py)                       │                        │
│   │  • 1D-CNN → BiLSTM → Attention → FC         │                        │
│   │  • OR: Transformer Encoder (src/transformer)│                        │
│   └─────────────────────┬───────────────────────┘                        │
│                         ↓                                                │
│   ┌─────────────────────────────────────────────┐                        │
│   │  TRAINING (src/train.py)                    │                        │
│   │  • Engine-level split (no data leakage)     │                        │
│   │  • Early stopping + LR scheduling           │                        │
│   │  • Experiment tracking (JSONL)              │                        │
│   └─────────────────────┬───────────────────────┘                        │
│                         ↓                                                │
│   ┌─────────────────────────────────────────────┐                        │
│   │  EVALUATION (src/evaluate.py)               │                        │
│   │  • RMSE, MAE, NASA Score                    │                        │
│   │  • MC Dropout (50 passes) → uncertainty     │                        │
│   │  • 4 publication-quality plots              │                        │
│   └─────────────────────┬───────────────────────┘                        │
│                         ↓                                                │
│   ┌──────────────────────────────────────────────────────┐               │
│   │  DEPLOYMENT                                          │               │
│   │  • FastAPI (api/main.py) — POST /predict             │               │
│   │  • Streamlit (app/streamlit_app.py) — Interactive UI │               │
│   └──────────────────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Model Architecture

```
Input: (batch, 30 timesteps, 16 sensor features)
            │
  ┌─────────▼──────────┐
  │  1D-CNN Block       │   Local pattern extraction
  │  Conv1D(16→64, k=3) │   Detects short-term anomalies,
  │  Conv1D(64→64, k=3) │   sensor transitions, noise patterns
  │  + BatchNorm + ReLU │
  │  + Dropout(0.3)     │
  └─────────┬──────────┘
            │
  ┌─────────▼──────────┐
  │  BiLSTM Block       │   Sequential memory
  │  2-layer, hidden=128│   Tracks accumulation of degradation
  │  Bidirectional      │   across the full 30-cycle window
  │  + Dropout(0.3)     │
  └─────────┬──────────┘
            │  output: (batch, 30, 256)
  ┌─────────▼──────────┐
  │  Multi-Head Self-   │   Adaptive time-step weighting
  │  Attention (4 heads)│   Focuses on the most informative
  │  + Residual + LN    │   part of the degradation curve
  └─────────┬──────────┘
            │  last timestep: (batch, 256)
  ┌─────────▼──────────┐
  │  Regression Head    │
  │  FC(256→64) + ReLU  │
  │  + Dropout(0.3)     │
  │  FC(64→1)           │   → scalar RUL
  └────────────────────┘
```

---

## 🔄 Model Comparison

Two architectures are implemented and can be trained with a single flag:

| Property | CNN-BiLSTM-Attention | Transformer Encoder |
|---|---|---|
| **Parameters** | 889,857 | 544,769 |
| **Local feature extraction** | 1D-CNN (kernel=3) | Self-attention |
| **Sequential modeling** | Bidirectional LSTM | Stacked self-attention |
| **Temporal weighting** | Multi-head attention + residual | Learned positional encoding |
| **Pooling strategy** | Last timestep | Global average pooling |
| **Activation** | ReLU | GELU |
| **Normalization** | Post-norm (BatchNorm + LayerNorm) | Pre-norm (LayerNorm) |
| **Training command** | `python -m src.train` | `python -m src.train --model transformer` |

The CNN-BiLSTM-Attention model is the primary architecture. The Transformer variant is provided as a modern alternative that trades the LSTM's sequential inductive bias for full parallelism and global receptive field.

---

## ⚙️ Engineering Design Decisions

### Why sequence modeling — not tabular regression?

Engine degradation is a temporal process. A single snapshot of sensor readings at cycle 150 looks identical whether the engine has 20 cycles left or 120. The discriminating signal lives in the *trajectory* — how sensor values have changed over the recent window. Tabular models (XGBoost, linear regression) can't capture this without heavy manual feature engineering (rolling averages, delta features). Sequence models learn these representations end-to-end.

### Why CNN + LSTM hybrid — not just LSTM?

LSTMs process sequences step-by-step, which is good for long-range dependencies but bad at detecting localized patterns (a sudden sensor spike at step 22 out of 30). The 1D-CNN front-end acts as a learnable feature extractor that pre-processes each local neighborhood before the LSTM sees it. This consistently outperforms LSTM-only baselines on CMAPSS (Li et al., 2018).

### Why multi-head self-attention?

Not all time steps matter equally. An engine's sensor readings during its final 10 cycles contain far more prognostic information than the readings during its first 10 cycles in the window. The attention layer learns to upweight the most informative positions dynamically — a capability that fixed-weight architectures lack.

### Why Monte Carlo Dropout — not ensembles?

Ensembles (training 5–10 models) give excellent uncertainty estimates but multiply compute cost linearly. MC Dropout (Gal & Ghahramani, 2016) repurposes the dropout already in the model: run 50 forward passes with dropout active, and the variance across predictions approximates Bayesian posterior uncertainty. Zero additional training cost, minimal inference overhead.

### Why piecewise-linear RUL capping at 125?

When an engine has 250 cycles remaining, its sensors read essentially the same as one with 300 cycles left. Forcing the model to distinguish between RUL=250 and RUL=300 wastes capacity. The cap at 125 creates a flat target for all "healthy" engines and focuses the model's resolution on the critical degradation phase where predictions actually influence maintenance decisions.

---

## 📡 API Usage

### Start the server

```bash
python -m uvicorn api.main:app --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      [643.25, 1589.70, 1400.60, 554.36, 2388.02, 9046.19,
       47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03,
       392, 2388.0, -0.0001, 100.00],
      [642.15, 1591.82, 1403.14, 553.75, 2388.04, 9044.07,
       47.49, 521.68, 2388.07, 8131.49, 8.4318, 0.03,
       393, 2388.0, 0.0003, 100.00]
    ],
    "mc_samples": 50,
    "normalize": true
  }'
```

### Example response

```json
{
  "predicted_rul": 87.34,
  "uncertainty_std": 9.12,
  "ci_lower_95": 69.10,
  "ci_upper_95": 105.58,
  "mc_samples_used": 50,
  "sequence_length_used": 30
}
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | API health check + model status |
| `GET` | `/model/info` | Architecture metadata + parameter count |
| `POST` | `/predict` | RUL prediction with uncertainty |

---

## ⚠️ Known Limitations

| Limitation | Impact | Potential mitigation |
|---|---|---|
| **CPU training is slow** | ~3 min/epoch on CPU; full 80-epoch run takes ~4 hours | Use CUDA GPU; model is small enough for a free Colab T4 |
| **Single operating condition** | FD001 has one condition and one fault mode — oversimplifies real operations | Extend to FD002–FD004 which add multiple conditions and faults |
| **Fixed window length** | 30-cycle window is a manually chosen hyperparameter | Adaptive window or variable-length attention (padded Transformer) |
| **Uncertainty is epistemic only** | MC Dropout captures model uncertainty, not aleatoric (sensor noise) | Add heteroscedastic output layer or deep evidential regression |
| **No online adaptation** | Model is static after training; can't adapt to novel operating regimes | Online learning or fine-tuning with incoming data |
| **Limited explainability** | Predictions are black-box; no per-sensor attribution | Integrate SHAP, Integrated Gradients, or attention weight visualization |

---

## 📁 Project Structure

```
turbofan-rul-prediction-system/
│
├── dataset/                        # Raw NASA CMAPSS data (pre-placed)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── data/processed/                 # Normalized CSVs + fitted scaler
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory data analysis
│
├── src/
│   ├── config.py                   # Single source of truth for all hyperparams
│   ├── preprocess.py               # Data pipeline (parse → RUL → normalize)
│   ├── dataset.py                  # PyTorch Dataset (sliding windows)
│   ├── model.py                    # CNN-BiLSTM-Attention architecture
│   ├── transformer_model.py        # Transformer encoder variant
│   ├── train.py                    # Training loop + early stopping + model selection
│   ├── evaluate.py                 # Metrics, MC Dropout, publication-quality plots
│   ├── experiment_tracker.py       # Lightweight JSONL experiment logging
│   └── utils.py                    # Seeding, device, logging
│
├── api/
│   └── main.py                     # FastAPI deployment (POST /predict)
│
├── app/
│   └── streamlit_app.py            # Interactive prediction dashboard
│
├── outputs/
│   ├── models/                     # Saved model checkpoints (.pt)
│   ├── plots/                      # Generated visualizations
│   └── experiments.jsonl           # Training run logs
│
├── report/
│   └── final_report.md             # Full technical report
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/abhinavsaha24/turbofan-rul-prediction-system.git
cd turbofan-rul-prediction-system
pip install -r requirements.txt
```

### Train

```bash
# CNN-BiLSTM-Attention (default)
python -m src.train

# Pure Transformer variant
python -m src.train --model transformer
```

Training logs are printed to console and saved to `outputs/experiments.jsonl`. The best checkpoint is saved automatically to `outputs/models/`.

### Evaluate

```bash
python -m src.evaluate
```

Runs deterministic + MC Dropout evaluation and generates all plots in `outputs/plots/`.

### Deploy — Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## 🔮 Future Roadmap

| Priority | Enhancement | Expected impact |
|---|---|---|
| 🔴 High | Cross-dataset generalization (FD002–FD004) | Validates robustness across operating conditions and fault modes |
| 🔴 High | SHAP / Integrated Gradients explainability | Per-sensor attribution for maintenance engineers |
| 🟡 Medium | Asymmetric loss (NASA scoring function) | Aligns training with operational risk asymmetry |
| 🟡 Medium | Real-time streaming inference (Kafka/MQTT) | Production-grade pipeline for live telemetry |
| 🟢 Low | Transfer learning (simulation → real data) | Bridge the sim-to-real gap for actual engine deployment |
| 🟢 Low | Heteroscedastic uncertainty | Separate epistemic and aleatoric uncertainty |

---

## 📚 References

1. Saxena et al., *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, IEEE PHM 2008
2. Li et al., *"Remaining Useful Life Estimation in Prognostics Using Deep CNNs"*, Reliability Engineering & System Safety 2018
3. Gal & Ghahramani, *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*, ICML 2016
4. Heimes, *"Recurrent Neural Networks for Remaining Useful Life Estimation"*, IEEE PHM 2008
5. Zheng et al., *"Long Short-Term Memory Network for RUL Estimation"*, IEEE Access 2017

---

## 📄 License

This project is for educational and portfolio purposes. The C-MAPSS dataset is provided by NASA and is publicly available.

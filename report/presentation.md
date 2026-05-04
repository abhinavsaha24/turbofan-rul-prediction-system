# Turbofan RUL Prediction — Presentation Slides

> **Subject:** AI for Engineering  
> **Topic:** Remaining Useful Life Prediction of Turbofan Engines Using Deep Learning  
> **Duration:** 12–15 minutes

---

## Slide 1 — Title

### Remaining Useful Life Prediction of Turbofan Engines Using Multi-Sensor Time-Series Data

**Approach:** CNN + BiLSTM + Multi-Head Attention with Monte Carlo Dropout Uncertainty Quantification

**Dataset:** NASA C-MAPSS (FD001)

**Key Result:** RMSE = 15.11 cycles | Uncertainty-aware predictions via MC Dropout

---

## Slide 2 — Introduction

### Why Predictive Maintenance Matters

- Unplanned engine failures cost the aviation industry **$10B+ annually**
- Traditional maintenance is **schedule-based** — parts replaced regardless of actual condition
- Airlines discard components with **30–50% useful life remaining**
- **Predictive maintenance** uses sensor telemetry to forecast failures before they happen
- Enables the shift from *"when should we inspect?"* to *"when will it fail?"*

> **Speaker Notes:** Open by emphasizing the real-world economic impact. Predictive maintenance is not a research curiosity — it's a multi-billion dollar operational problem. Frame RUL prediction as the core technical capability that enables this shift.

---

## Slide 3 — Problem Statement

### The RUL Prediction Problem

**Given:** Multivariate time-series from 21 sensors and 3 operational settings, recorded once per operational cycle

**Predict:** Number of remaining operational cycles before engine failure (scalar regression)

**Challenges:**
- Degradation is gradual, noisy, and engine-specific
- Two engines at the same age can have vastly different remaining lifespans
- Must handle variable-length histories (30 to 300+ cycles)
- Uncertainty quantification is mandatory for safety-critical decisions

> **Speaker Notes:** Emphasize that this is NOT a classification problem — we're predicting a continuous value. The difficulty lies in the fact that degradation is subtle and progressive, not a sudden event.

---

## Slide 4 — Dataset Overview

### NASA C-MAPSS — FD001 Subset

| Property | Value |
|---|---|
| Training engines | 100 (run-to-failure) |
| Test engines | 100 (cut-off before failure) |
| Operating conditions | 1 (sea level) |
| Fault modes | 1 (HPC degradation) |
| Sensors | 21 (7 constant → dropped) |
| Total training rows | 20,631 |
| Engine lifespans | 128 – 362 cycles |

**Key insight:** Engine lifespans vary by 3×, meaning the model cannot rely on chronological age — it must learn degradation *signatures* from sensor trajectories.

> **Speaker Notes:** Mention that FD001 is the simplest subset (one condition, one fault). This is deliberate — it isolates the modeling challenge from operational complexity. Scaling to FD002–FD004 is a natural extension.

---

## Slide 5 — Data Preprocessing

### Pipeline: Raw Text → Model-Ready Sequences

1. **Parse** whitespace-delimited CMAPSS text files → structured DataFrames
2. **Compute RUL** targets: `RUL(t) = max_cycle - current_cycle`, capped at 125
3. **Feature selection**: Drop 7 constant sensors + 1 zero-variance setting → **16 features**
4. **Normalize**: MinMaxScaler fitted on training data only (scaler persisted for deployment)
5. **Sequence generation**: Sliding window (length=30) with left-padding for short histories

**Why cap at 125?** Sensors for engines with RUL=200 and RUL=300 are statistically indistinguishable. The cap focuses learning on the critical degradation phase.

**Why engine-level splitting?** Splitting by individual rows would leak temporal information. We split by engine to prevent data leakage.

> **Speaker Notes:** Stress the scaler persistence point — this is a deployment detail that's easy to miss but silently breaks real-world systems. The RUL cap is a well-established practice in the CMAPSS literature (Saxena et al., 2008).

---

## Slide 6 — System Architecture

### End-to-End Pipeline

```
Raw .txt → Preprocessing → Sliding Windows → Model → Evaluation → Deployment
              ↓                                  ↓
         Scaler.pkl                        MC Dropout (×50)
              ↓                                  ↓
         Reused by                         Mean prediction +
         API & Streamlit                   Confidence intervals
```

**Deployment options:**
- **FastAPI** — REST API with POST `/predict` endpoint
- **Streamlit** — Interactive dashboard with per-engine visualization
- **Experiment tracking** — JSONL logs with hyperparameters, system info, per-epoch history

> **Speaker Notes:** Emphasize that this is a *system*, not just a model. The preprocessing, training, evaluation, and deployment are all modular and connected.

---

## Slide 7 — Model Design (CNN-BiLSTM-Attention)

### Hybrid Architecture — 889,857 Parameters

| Component | Role | Detail |
|---|---|---|
| **1D-CNN** (2 layers) | Local pattern extraction | Detects short-term anomalies and transitions |
| **BiLSTM** (2 layers) | Sequential memory | Captures long-range degradation accumulation |
| **Multi-Head Attention** (4 heads) | Adaptive time-step weighting | Focuses on most informative cycles |
| **Regression Head** (2 FC layers) | Output mapping | Maps to scalar RUL prediction |

**Dropout (p=0.3)** at every stage → enables Monte Carlo Dropout at inference

**Why this combination?** No single architecture handles local patterns, sequential memory, and adaptive weighting simultaneously. The hybrid composes all three.

> **Speaker Notes:** Walk through the data flow: CNN extracts local features, LSTM captures the temporal evolution, attention selects which timesteps matter. The dropout at every stage is critical — it enables uncertainty estimation for free.

---

## Slide 8 — Training Process

### Robust Training Protocol

| Setting | Value | Why |
|---|---|---|
| Optimizer | Adam (lr=1e-3, wd=1e-5) | Adaptive rates + light L2 regularization |
| LR Schedule | ReduceLROnPlateau (factor=0.5) | Automatic plateau detection |
| Gradient Clipping | max_norm=1.0 | Prevents exploding gradients in LSTM |
| Early Stopping | Patience=12 epochs | Prevents overfitting |
| Batch Size | 256 | Balances gradient noise with speed |
| Train/Val Split | 80/20 by engine | Prevents temporal data leakage |

**Experiment tracking:** Every run logs hyperparameters, system info, per-epoch losses, and final metrics to `experiments.jsonl`.

> **Speaker Notes:** Highlight the engine-level split as a key engineering decision. If rows from the same engine appeared in train and val, the model would memorize individual degradation curves.

---

## Slide 9 — Results

### Quantitative Performance on FD001 Test Set (100 engines)

| Metric | Value | Meaning |
|---|---|---|
| **RMSE** | 15.11 cycles | Root-mean-square error; penalizes large errors |
| **MAE** | 12.07 cycles | Average absolute prediction error |
| **NASA Score** | 438.59 | Asymmetric penalty (late predictions penalized 2× more) |
| **MC Dropout RMSE** | 15.30 cycles | Uncertainty estimation adds no cost to accuracy |
| **Mean σ** | ±10.51 cycles | Average 68% confidence band width |

**Interpretation:** Predictions are off by ~12 cycles on average. For engines with 150–350 cycle lifespans, this is sufficient for scheduling-level maintenance decisions. 62% of predictions fall within ±10 cycles of actual RUL.

> **Speaker Notes:** Show the prediction vs actual plot. Point out that the model tracks the degradation trend well, with uncertainty widening for healthy engines (where prediction is inherently harder).

---

## Slide 10 — Model Comparison

### CNN-BiLSTM-Attention vs. Transformer Encoder

| Property | CNN-BiLSTM-Attention | Transformer |
|---|---|---|
| Parameters | 889,857 | 544,769 |
| Architecture | CNN → BiLSTM → Attention | Linear projection → Transformer encoder |
| Positional info | Implicit (LSTM memory) | Learned positional embeddings |
| Pooling | Last timestep | Global average pooling |
| Training command | `python -m src.train` | `python -m src.train --model transformer` |

### Benchmark Comparison (FD001 Literature)

| Method | RMSE |
|---|---|
| Linear regression | ~25–30 |
| Standard LSTM | 16–18 |
| CNN-LSTM hybrids | 13–16 |
| Deep ensembles | 11–13 |
| **This project** | **15.11** |

> **Speaker Notes:** Position the result honestly — we're competitive with published CNN-LSTM hybrids, but not state-of-the-art. The gap is explainable: single model (not ensemble), symmetric loss (not NASA scoring), no dataset-specific tuning.

---

## Slide 11 — Uncertainty Estimation

### Monte Carlo Dropout — Bayesian Uncertainty for Free

**Method:** Run the model 50 times with dropout active at inference. The variance across predictions estimates epistemic uncertainty.

**Calibration result:** Engines with higher predicted uncertainty *do* have larger actual errors — the uncertainty is informative, not random noise.

**Practical example:**
- Engine #17: RUL = 8 cycles (95% CI: 3–13) → **Immediate maintenance**
- Engine #42: RUL = 35 cycles (95% CI: 14–56) → **Schedule monitoring**

**Why this matters:** Point predictions alone are unsafe for maintenance decisions. Confidence intervals enable risk-aware scheduling.

> **Speaker Notes:** Show the uncertainty calibration plot. Emphasize that MC Dropout is a single-line code change (model.train() at inference) — it repurposes existing dropout without any additional training.

---

## Slide 12 — Deployment

### Two Deployment Interfaces

**FastAPI (REST API):**
- POST `/predict` — accepts sensor sequence, returns RUL + uncertainty
- GET `/health` — system status
- GET `/model/info` — architecture metadata
- Auto-generated OpenAPI docs at `/docs`

**Streamlit (Interactive Dashboard):**
- Load existing test data or upload custom files
- Per-engine predictions with confidence intervals
- Configurable MC Dropout passes
- Real-time progress tracking

> **Speaker Notes:** The API is production-oriented — it validates input shapes, handles normalization, and returns structured JSON with confidence intervals. The Streamlit app is designed for non-technical stakeholders (maintenance engineers).

---

## Slide 13 — Limitations

### Honest Assessment

| Limitation | Why it matters |
|---|---|
| Single operating condition (FD001) | Real engines operate under varying loads and altitudes |
| CPU training is slow (~4 hours) | Needs GPU for rapid experimentation |
| Fixed 30-cycle window | May miss very long-range degradation trends |
| Epistemic uncertainty only | Doesn't separate model uncertainty from sensor noise |
| No explainability | Cannot tell which sensor drives each prediction |
| Static model | Cannot adapt to new operating regimes post-deployment |

> **Speaker Notes:** Be upfront about limitations — it demonstrates critical thinking and awareness of production constraints. Each limitation has a known mitigation path.

---

## Slide 14 — Future Work & Conclusion

### Immediate Next Steps
- **Multi-dataset generalization** — FD002–FD004 (multiple operating conditions and faults)
- **Explainability** — SHAP or Integrated Gradients for per-sensor attribution

### Medium-Term
- **Asymmetric loss training** — align with NASA scoring function's risk asymmetry
- **Real-time streaming** — Kafka/MQTT pipeline for live engine telemetry

### Research Directions
- Heteroscedastic uncertainty (separate epistemic vs. aleatoric)
- Neural ODE-based continuous-time degradation models
- Transfer learning: simulation → real engine data

### Conclusion

This project demonstrates a complete, production-oriented pipeline for turbofan RUL prediction — from raw data to deployable API — with competitive accuracy (RMSE=15.11) and well-calibrated uncertainty estimates. The modular architecture allows swapping model backbones (CNN-LSTM vs. Transformer) with a single command-line flag.

> **Speaker Notes:** End by connecting back to the opening — this system enables condition-based maintenance, which reduces unplanned downtime by 20–30%. The engineering is as important as the modeling.

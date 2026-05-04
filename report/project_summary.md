# Project Summary — Turbofan RUL Prediction System

## What Was Built

A complete, production-oriented machine learning system that predicts the Remaining Useful Life (RUL) of turbofan engines from multi-sensor time-series data. The system spans the full ML lifecycle: data preprocessing, model training, evaluation with uncertainty quantification, and deployment via both REST API and interactive dashboard.

## Key Achievements

### Modeling
- Implemented a hybrid **CNN + BiLSTM + Multi-Head Attention** architecture (889K params) that achieves **RMSE = 15.11 cycles** on the NASA C-MAPSS FD001 benchmark — competitive with published CNN-LSTM results.
- Built an alternative **Transformer encoder** variant (544K params) trainable via a single `--model transformer` flag.
- Integrated **Monte Carlo Dropout** (50 stochastic passes) for uncertainty quantification, producing calibrated confidence intervals at zero additional training cost.

### Engineering
- **Zero data leakage:** Engine-level train/validation splitting prevents temporal information leakage.
- **Deployment consistency:** MinMaxScaler serialized and reused across evaluation, Streamlit, and FastAPI.
- **Experiment tracking:** Every training run logs hyperparameters, system info, per-epoch losses, and final metrics to structured JSONL.
- **Modular codebase:** Clean separation of concerns — preprocessing, dataset, model, training, evaluation, and serving are independent modules.

### Deployment
- **FastAPI REST API** with `POST /predict`, `GET /health`, and `GET /model/info` endpoints.
- **Streamlit dashboard** for interactive per-engine prediction with configurable MC Dropout passes.

### Documentation
- **README** with architecture diagrams, model comparison, benchmark context, API examples, and limitations.
- **Technical report** (12 sections) with academic tone suitable for AI for Engineering coursework.
- **14-slide presentation** with speaker notes for viva delivery.

## Final Results

| Metric | Value |
|---|---|
| Test RMSE | 15.11 cycles |
| Test MAE | 12.07 cycles |
| NASA Score | 438.59 |
| MC Dropout RMSE | 15.30 cycles |
| Mean Uncertainty (σ) | ±10.51 cycles |
| Predictions within ±10 cycles | 62/100 engines |
| Model parameters | 889,857 (CNN-LSTM) / 544,769 (Transformer) |

## Why It Matters

Predictive maintenance represents a **$10B+ annual opportunity** in aviation alone. This system demonstrates that a moderately sized deep learning model, trained on standard benchmark data, can provide actionable RUL predictions with calibrated uncertainty — the two essential capabilities for transitioning from schedule-based to condition-based maintenance.

## Project Structure

```
17 source files | 9 modules | 2 deployment interfaces | 4 visualization plots
```

| Component | Files |
|---|---|
| Core ML | config.py, preprocess.py, dataset.py, model.py, transformer_model.py |
| Training | train.py, experiment_tracker.py, utils.py |
| Evaluation | evaluate.py (4 publication-quality plots) |
| Deployment | api/main.py (FastAPI), app/streamlit_app.py (Streamlit) |
| Documentation | README.md, report/final_report.md, report/presentation.md |
| Infrastructure | requirements.txt, .gitignore, notebooks/EDA.ipynb |

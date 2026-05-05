# 🔥 Turbofan Engine Remaining Useful Life (RUL) Prediction

Predictive maintenance is rapidly transforming how aerospace and heavy industries manage their most critical physical assets. Instead of relying on rigid, schedule-based maintenance—which often leads to unnecessary downtime or catastrophic mid-flight failures—data-driven approaches allow engineers to target maintenance exactly when it's needed.

This project implements an end-to-end Machine Learning system for predicting the Remaining Useful Life (RUL) of turbofan engines using multi-sensor time-series data. Built on the NASA CMAPSS dataset, the system processes raw sensor telemetry, models degradation patterns using a hybrid sequence architecture, and serves real-time predictions through a FastAPI backend and a Streamlit dashboard.

---

## 🎯 Problem Statement

**Remaining Useful Life (RUL)** is the number of operational cycles an engine has left before it reaches a critical state of failure. 

Accurately forecasting RUL is a complex challenge because engine degradation is highly non-linear. Early in an engine's lifecycle, sensor readings remain stable. As wear and tear accumulate, slight sensor deviations cascade into rapid failure trajectories. By capturing these temporal dependencies, we can alert operators well in advance, minimizing downtime and optimizing maintenance schedules.

---

## ⚙️ System Pipeline

The architecture is designed to handle raw sensor logs and transform them into actionable intelligence:

1. **Dataset Ingestion:** Raw CMAPSS text files containing engine ID, operational cycles, and 21 sensor readings are loaded.
2. **Preprocessing:** Constant and uninformative sensors are dropped. Features are normalized to ensure stable gradient flow during training.
3. **Sequence Generation:** Time-series data is converted into sliding windows (sequence length of 30) to capture temporal context rather than relying on isolated snapshots.
4. **Model Inference:** Sequences are passed through a hybrid neural network designed for spatial and temporal feature extraction.
5. **Prediction & Uncertainty:** The model outputs a continuous RUL value alongside Monte Carlo Dropout bounds to quantify predictive uncertainty.
6. **Deployment:** The inference engine is wrapped in a FastAPI backend and visualized via a Streamlit dashboard.

---

## 🧠 Model Architecture

The core of the system is a **CNN + BiLSTM + Attention** neural network implemented in PyTorch. 

* **Convolutional Neural Network (CNN):** A 1D convolutional layer scans across the sequence windows to extract localized, short-term sensor features (spatial representation).
* **Bidirectional LSTM (BiLSTM):** The CNN feature maps are fed into a stacked Bidirectional LSTM. This captures long-term dependencies and degradation trends from both past and future contexts within the window.
* **Attention Mechanism:** An attention layer weights the LSTM outputs, forcing the model to focus on the specific time steps where critical degradation signatures occur.
* **Monte Carlo Dropout:** Dropout layers remain active during inference to generate a distribution of predictions, providing a confidence interval (mean ± std) rather than a fragile point estimate.

---

## 📊 Results

The model was evaluated on the held-out test engines (FD001 dataset).

* **RMSE (Root Mean Square Error): ~15.62**
* **MAE (Mean Absolute Error): ~11.43**

**Why these metrics?** RMSE penalizes large errors heavily, which is crucial in predictive maintenance (predicting an engine will last 50 cycles longer than it actually will is a fatal error). An MAE of ~11 cycles means the model's predictions are, on average, within 11 flights of the exact failure point—highly viable for operational planning.

---

## 📈 Visualizations

The evaluation script automatically generates several analytical plots in the `outputs/plots/` directory:

* **Predicted vs Actual RUL:** Visualizes how closely the model tracks the true degradation curve over an engine's lifecycle.
* **Error Distribution:** A histogram showing the spread of prediction residuals, verifying that errors are normally distributed around zero.
* **Sensor Degradation Trends:** Plots showing how critical sensors (e.g., temperature, pressure) behave as the engine approaches failure.
* **Uncertainty Calibration:** Highlights the Monte Carlo confidence bounds around the predictions.

---

## 🌐 Deployment Overview

The project is structured for immediate deployment and interaction:

* **FastAPI Backend (`api/main.py`):** Provides a high-performance REST API. The `/predict` endpoint accepts JSON payloads of sensor sequences and returns the computed RUL and variance.
* **Streamlit Frontend (`app/streamlit_app.py`):** A clean, interactive dashboard. Users can select an engine ID, view real-time sensor streams, and run model inference with one click.

---

## 🚀 Local Deployment Guide

You can run the entire system locally from scratch by following these steps.

### 🔹 Step 1 — Clone Repo
```bash
git clone https://github.com/abhinavsaha24/turbofan-rul-prediction-system.git
cd turbofan-rul-prediction-system
```

### 🔹 Step 2 — Create Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 🔹 Step 3 — Install Requirements
```bash
pip install -r requirements.txt
```

### 🔹 Step 4 — Set PYTHONPATH (Windows users)
```bash
set PYTHONPATH=.
```
*(Mac/Linux users can run `export PYTHONPATH=.`)*

### 🔹 Step 5 — Run Pipeline
Execute the machine learning pipeline sequentially:
```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

### 🔹 Step 6 — Run API
Start the FastAPI server:
```bash
python -m uvicorn api.main:app --reload
```
Open the interactive documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 🔹 Step 7 — Run UI
Start the Streamlit dashboard:
```bash
python -m streamlit run app/streamlit_app.py
```
The dashboard will automatically open at: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```text
turbofan-rul-prediction-system/
├── api/                  # FastAPI deployment code
├── app/                  # Streamlit dashboard UI
├── data/                 # Processed numpy sequences and scalers
├── dataset/              # Raw CMAPSS text files (train, test, RUL)
├── notebooks/            # Exploratory Data Analysis (EDA)
├── outputs/              # Saved model weights and evaluation plots
├── src/                  # Core ML modules (config, dataset, model, train)
└── requirements.txt      # Project dependencies
```

---

## ⚠️ Limitations

* **Hardware constraints:** The current training loop defaults to CPU execution. While sufficient for the FD001 dataset, scaling to larger CMAPSS subsets will require GPU acceleration.
* **Operational scope:** This model is trained exclusively on single-operating-condition engines (FD001). Engines operating across multiple fault modes or varying altitudes/mach numbers require dataset expansion.
* **Inference latency:** Monte Carlo Dropout requires 50 stochastic forward passes per prediction, slightly increasing inference time compared to standard deterministic models.

---

## 🚀 Future Work

* **Transformer Integration:** Upgrading the sequence processing layer from BiLSTMs to a full Time-Series Transformer architecture.
* **Multi-Dataset Training:** Extending the pipeline to handle FD002, FD003, and FD004, enabling the model to generalize across complex, multi-regime operational conditions.
* **Edge Deployment:** Quantizing the model weights via ONNX to run directly on edge hardware embedded near the physical sensors.

---

## 👨‍💻 Contributors

* **Abhinav Kumar** (1024031186)
* **Aditya Singh** (1024031177)
* **Suryansh Chauhan** (1024031017)

**Batch:** 2C81  
**Subject:** AI for Engineering

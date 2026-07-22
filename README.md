# ML-IDS — Machine-Learning Intrusion Detection System

A machine-learning intrusion detection system built and evaluated on the
**CSE-CIC-IDS2018** benchmark (16,232,943 flows, 15 classes, ~155,000:1 imbalance),
with a real-time scoring dashboard.

## Highlights

- **Hybrid feature space** — the 40 most discriminative flow features (selected by
  Mutual Information, to capture non-linear dependencies a linear filter would miss)
  concatenated with a 16-dimensional autoencoder latent embedding.
- **Imbalance handling** — capped SMOTE oversampling + balanced class weighting, with
  strictly leakage-safe preprocessing, splitting, and per-class threshold calibration
  (statistics fitted on train/validation only).
- **Five models compared** — Random Forest, XGBoost, MLP (supervised) plus a
  benign-trained Autoencoder and Isolation Forest (unsupervised), scored by
  macro-averaged F1.
- **Best model** — Random Forest, **macro-F1 0.836** (95% CI [0.815, 0.852]) at a
  binary attack-vs-benign false-positive rate of **0.63%**.
- **Live monitor** — a FastAPI + WebSocket dashboard that scores CICFlowMeter flows in
  real time (see [`web/`](web/)).

## Repository layout

```
notebooks/    data exploration → preprocessing → feature engineering →
              training → evaluation → explainability (+ inference scripts)
web/          FastAPI + WebSocket live monitoring dashboard
models/        trained scalers, autoencoders, configs (large .pkl excluded)
results/       figures, confusion matrices, metric CSVs, SHAP plots
report_methodology_results_UPDATED.md   methodology & results manuscript
DATA.md        how to obtain the dataset and regenerate excluded artifacts
```

## Getting started

The raw dataset and the largest trained models are **not** in this repo (they exceed
GitHub's size limits). See **[DATA.md](DATA.md)** for how to download the
CSE-CIC-IDS2018 data and regenerate everything by running the notebooks in order.

To run the live dashboard, see **[web/README.md](web/README.md)**.

## Environment

Trained and scored with **scikit-learn 1.7.2** (pin this version, or the saved models
fail to load). Dashboard dependencies are in [`web/requirements.txt`](web/requirements.txt).

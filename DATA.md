# Data & Model Artifacts

This repository contains **all code, notebooks, reports, and result figures**, but
**not** the multi-gigabyte datasets or the largest trained models — GitHub caps files
at 100 MB and these total ~13 GB. This page explains what is excluded and how to
regenerate it.

## What's excluded (see `.gitignore`)

| Path | Size | What it is |
|------|------|------------|
| `data/raw/*.csv` | ~1.7 GB + | Raw CSE-CIC-IDS2018 flow CSVs (10 capture days) |
| `data/processed/`, `data/features/` | ~9 GB | Cleaned / split / hybrid-feature `.npy` arrays |
| `models/random_forest.pkl` | ~2 GB | Trained Random Forest (best model) |
| `models/xgboost.pkl` | ~117 MB | Trained XGBoost classifier |
| `models/feat_sel_X_sample.npy` | ~123 MB | Feature-selection working sample |
| `results/*.npy` | ~340 MB | Per-flow prediction / probability arrays |

Small artifacts **are** kept: scalers, imputer, autoencoders (`.pth`), Isolation
Forest, feature-column JSONs, class/ensemble weights, and every figure (`.png`),
CSV summary, and JSON in `results/`.

## Getting the raw dataset

The project uses the **CSE-CIC-IDS2018** benchmark from the Canadian Institute for
Cybersecurity:

- <https://www.unb.ca/cic/datasets/ids-2018.html>

Download the "Processed Traffic Data for ML Algorithms" CSVs and place them in
`data/raw/` with these filenames:

```
data/raw/02-14-2018.csv  02-15-2018.csv  02-16-2018.csv  02-20-2018.csv
         02-21-2018.csv  02-22-2018.csv  02-23-2018.csv  02-28-2018.csv
         03-01-2018.csv  03-02-2018.csv
```

> Note: 7 of the 10 official CSVs are truncated at the Excel row limit
> (1,048,575 rows) in the upstream CIC distribution — this is expected, not a
> download error.

## Regenerating everything

Run the notebooks in order — each writes the artifacts the next one consumes:

1. `notebooks/data_exploration.ipynb` — sanity checks on the raw CSVs
2. `notebooks/preprocessing.ipynb` — cleaning, splitting → `data/processed/`
3. `notebooks/feature_engineering.ipynb` — MI selection + autoencoder latent → `data/features/`
4. `notebooks/model_training.ipynb` — trains RF / XGBoost / MLP → `models/`
5. `notebooks/evaluation.ipynb` — metrics, confusion matrices → `results/`
6. `notebooks/explainability.ipynb` — SHAP attribution figures

## Environment

Models were trained/scored with **scikit-learn 1.7.2** — pinning this version is
required or the saved `.pkl` models will fail to load/score. See `web/requirements.txt`
for the live-dashboard stack.

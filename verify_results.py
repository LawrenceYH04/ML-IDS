#!/usr/bin/env python
"""
verify_results.py - regenerate the report's headline tables from stored artifacts.

Recomputes Tables 4.2, 4.2b and 4.3 of the report directly from the held-out test
labels (data/processed/y_test.npy) and the per-flow predictions saved at evaluation
time (results/y_pred_*.npy). Nothing is retrained and no number is hard-coded: every
value printed is derived here, so the output either matches the report or it does not.

Usage:
    python verify_results.py              # tables only (a few seconds)
    python verify_results.py --bootstrap  # also runs the paired bootstrap (slow)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parent
MODELS = ["rf", "xgb", "mlp", "ensemble"]
LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP",
          "ensemble": "RF + XGB ensemble"}

# The values printed in the report, for side-by-side comparison.
REPORTED = {
    "rf":       dict(acc=0.9815, mp=0.8384, mr=0.8417, mf1=0.8360, wf1=0.9804),
    "xgb":      dict(acc=0.9820, mp=0.7717, mr=0.8564, mf1=0.7850, wf1=0.9808),
    "mlp":      dict(acc=0.9608, mp=0.6480, mr=0.8030, mf1=0.6381, wf1=0.9627),
    "ensemble": dict(acc=0.9812, mp=0.8367, mr=0.8399, mf1=0.8339, wf1=0.9803),
}


def metrics(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true, y_pred),
        mp=precision_score(y_true, y_pred, average="macro", zero_division=0),
        mr=recall_score(y_true, y_pred, average="macro", zero_division=0),
        mf1=f1_score(y_true, y_pred, average="macro"),
        wf1=f1_score(y_true, y_pred, average="weighted"),
    )


def table_4_2(y_test, preds):
    print("\nTable 4.2 - Multi-class performance on the held-out test set")
    print(f"{'Model':<20}{'Acc':>8}{'MacroP':>9}{'MacroR':>9}{'MacroF1':>10}"
          f"{'WtdF1':>9}   {'vs report':>10}")
    print("-" * 76)
    for m in MODELS:
        got = metrics(y_test, preds[m])
        exp = REPORTED[m]
        drift = max(abs(got[k] - exp[k]) for k in exp)
        flag = "match" if drift < 5e-4 else f"DIFF {drift:.4f}"
        print(f"{LABELS[m]:<20}{got['acc']:>8.4f}{got['mp']:>9.4f}{got['mr']:>9.4f}"
              f"{got['mf1']:>10.4f}{got['wf1']:>9.4f}   {flag:>10}")


def table_4_2b(y_test, preds, class_names):
    print("\nTable 4.2b - Random Forest per-class results (calibrated)")
    print(classification_report(y_test, preds["rf"], target_names=class_names,
                                digits=3, zero_division=0))


def table_4_3(y_test, preds, benign_idx):
    """Collapse all attack classes into one and report the operational view."""
    print("\nTable 4.3 - Binary attack-vs-benign detection")
    print(f"{'Model':<20}{'Prec':>9}{'Recall':>9}{'F1':>9}{'BenignFPR':>12}")
    print("-" * 59)
    y_bin = (y_test != benign_idx).astype(int)
    benign_mask = y_bin == 0
    for m in MODELS:
        p_bin = (preds[m] != benign_idx).astype(int)
        fpr = p_bin[benign_mask].mean()
        print(f"{LABELS[m]:<20}"
              f"{precision_score(y_bin, p_bin, zero_division=0):>9.4f}"
              f"{recall_score(y_bin, p_bin, zero_division=0):>9.4f}"
              f"{f1_score(y_bin, p_bin):>9.4f}{fpr:>12.4f}")


def paired_bootstrap(y_test, preds, a="rf", b="xgb", n_boot=2000, seed=42):
    """Paired bootstrap of the macro-F1 difference: both models scored on the
    identical resample each time, so resample-to-resample variance cancels."""
    rng = np.random.default_rng(seed)
    n = len(y_test)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_test[idx]
        deltas[i] = (f1_score(yt, preds[a][idx], average="macro")
                     - f1_score(yt, preds[b][idx], average="macro"))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"\nPaired bootstrap ({n_boot} resamples): {LABELS[a]} - {LABELS[b]}")
    print(f"  delta macro-F1 = {deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  P({a} > {b}) = {(deltas > 0).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true",
                    help="also run the paired bootstrap (slow: minutes)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    y_test = np.load(ROOT / "data/processed/y_test.npy")
    preds = {m: np.load(ROOT / f"results/y_pred_{m}.npy") for m in MODELS}

    import joblib
    le = joblib.load(ROOT / "models/label_encoder.pkl")
    class_names = list(le.classes_)
    benign_idx = int(np.where(le.classes_ == "Benign")[0][0])

    print(f"Test set: {len(y_test):,} held-out flows, {len(class_names)} classes")
    print(f"Benign share: {(y_test == benign_idx).mean():.1%}")
    for m in MODELS:
        assert len(preds[m]) == len(y_test), f"{m}: prediction/label length mismatch"
    print("All prediction arrays align with the test labels.")

    table_4_2(y_test, preds)
    table_4_2b(y_test, preds, class_names)
    table_4_3(y_test, preds, benign_idx)
    if args.bootstrap:
        paired_bootstrap(y_test, preds, "rf", "xgb", args.n_boot)
        paired_bootstrap(y_test, preds, "rf", "ensemble", args.n_boot)


if __name__ == "__main__":
    main()

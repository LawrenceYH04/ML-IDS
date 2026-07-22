#!/usr/bin/env python3
"""
Ablation study: does the HYBRID feature space beat its parts?

Standalone & non-destructive. Reads only saved feature arrays; writes only
results/ablation_hybrid.csv + results/ablation_hybrid.png. Touches no model,
scaler, threshold, or the retrained anomaly AE.

Feature layout in the saved 56-dim hybrid vector:
  cols  0..39  -> 40 MI-selected hand-crafted flow features
  cols 40..55  -> 16 autoencoder latent dimensions

Design: hold everything constant (same 2M-row train subset, same fast model,
same full test set, same seed) and vary ONLY the feature columns. That isolates
the contribution of each feature group. Uses the 2M-row subset + a single fast
classifier by design, so this is a controlled *relative* comparison, not a
re-training of the final 18M-row Random Forest.
"""
import os, time, json
import numpy as np
from pathlib import Path

# ---- project root -----------------------------------------------------------
for cand in (Path.cwd(), *Path.cwd().parents):
    if (cand / 'data').is_dir() and (cand / 'models').is_dir():
        os.chdir(cand); break
print('Project root:', os.getcwd())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import joblib

SEED = 42
CONFIGS = {
    'Hand-crafted only (40)': slice(0, 40),
    'Latent only (16)':       slice(40, 56),
    'Hybrid (56)':            slice(0, 56),
}

# ---- load saved arrays (no recompute) --------------------------------------
# Use the CANONICAL hybrid train array (the exact 18.4M-row SMOTE-balanced array
# the deployed models were trained on) so columns are guaranteed aligned with the
# test set. Sample a fixed random subset for speed (memmap + sorted indices).
SUBSET = 1_500_000
_rng = np.random.default_rng(SEED)
_Xfull = np.load('data/features/X_train_hybrid.npy', mmap_mode='r')  # (18412356, 56)
_yfull = np.load('data/processed/y_train.npy')
_idx = np.sort(_rng.choice(_Xfull.shape[0], SUBSET, replace=False))
Xtr = np.asarray(_Xfull[_idx])
ytr = _yfull[_idx]
Xte = np.load('data/features/X_test_hybrid.npy').astype(np.float32)  # (2,434,942, 56)
yte = np.load('data/processed/y_test.npy')
le  = joblib.load('models/label_encoder.pkl')
classes = list(le.classes_)
infil_idx = classes.index('Infilteration')  # raw dataset spelling
print(f'Train subset {Xtr.shape}, Test {Xte.shape}, classes={len(classes)}')
print(f'Infiltration class index = {infil_idx}')

# balanced weights so no config is unfairly biased to Benign
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
cw_map = {c: w for c, w in zip(np.unique(ytr), cw)}

rows = []
for name, cols in CONFIGS.items():
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=120, max_depth=30, min_samples_leaf=2,
        max_samples=0.5, class_weight=cw_map, n_jobs=-1, random_state=SEED,
    )
    clf.fit(Xtr[:, cols], ytr)
    yp = clf.predict(Xte[:, cols])
    macro_f1 = f1_score(yte, yp, average='macro', zero_division=0)
    macro_p  = precision_score(yte, yp, average='macro', zero_division=0)
    macro_r  = recall_score(yte, yp, average='macro', zero_division=0)
    acc      = accuracy_score(yte, yp)
    per_class_f1 = f1_score(yte, yp, average=None, labels=range(len(classes)),
                            zero_division=0)
    infil_f1 = per_class_f1[infil_idx]
    dt = time.time() - t0
    print(f'\n=== {name} ===  ({dt:.0f}s)')
    print(f'  Accuracy   : {acc:.4f}')
    print(f'  Macro-F1   : {macro_f1:.4f}   (P {macro_p:.4f} / R {macro_r:.4f})')
    print(f'  Infiltration F1: {infil_f1:.4f}')
    rows.append({
        'config': name, 'n_features': (cols.stop - cols.start),
        'accuracy': round(acc, 4), 'macro_precision': round(macro_p, 4),
        'macro_recall': round(macro_r, 4), 'macro_f1': round(macro_f1, 4),
        'infiltration_f1': round(float(infil_f1), 4),
    })

# ---- save table -------------------------------------------------------------
import csv
out_csv = 'results/ablation_hybrid.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'\nSaved {out_csv}')

# ---- bar chart --------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [r['config'] for r in rows]
    mf1 = [r['macro_f1'] for r in rows]
    inf = [r['infiltration_f1'] for r in rows]
    x = np.arange(len(labels)); wdt = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - wdt/2, mf1, wdt, label='Macro-F1 (all 15 classes)')
    b2 = ax.bar(x + wdt/2, inf, wdt, label='Infiltration F1 (hardest class)')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel('F1'); ax.set_ylim(0, 1)
    ax.set_title('Ablation: hand-crafted vs latent vs hybrid feature space')
    ax.legend()
    for b in (b1, b2):
        for r in b:
            ax.annotate(f'{r.get_height():.3f}', (r.get_x()+r.get_width()/2, r.get_height()),
                        ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig('results/ablation_hybrid.png', dpi=130)
    print('Saved results/ablation_hybrid.png')
except Exception as e:
    print('Plot skipped:', e)

print('\nDONE — non-destructive; no model/scaler/threshold was modified.')

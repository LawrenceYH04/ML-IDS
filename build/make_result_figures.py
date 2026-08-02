#!/usr/bin/env python
"""
Generate the additional Chapter 4 (Results) figures from the saved test-set
predictions and probabilities. Nothing is re-trained: every number is read from
results/*.npy and data/processed/y_test.npy.

Run:  python build/make_result_figures.py
Out:  ML-IDS/results/fig_*.png  (+ a printed audit of the numbers quoted in the report)
"""
import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
SKY = "#56B4E9"
GREY = "#5A5A5A"
LGREY = "#D9D9D9"
INK = "#1A1A1A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GREY, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

y = np.load(os.path.join(HERE, "data/processed/y_test.npy"))
le = joblib.load(os.path.join(HERE, "models/label_encoder.pkl"))
CLASSES = [c.replace("Infilteration", "Infiltration").replace("Brute Force -", "Brute Force-")
           for c in le.classes_]
K = len(CLASSES)
BENIGN = CLASSES.index("Benign")


def save(fig, name):
    p = os.path.join(RES, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", p)


def per_class_f1(y_true, y_pred, k=K):
    """Per-class F1 from a confusion matrix built with bincount (fast)."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return f1, cm


# ---------------------------------------------------------------------------
# A. Per-class F1 across the three supervised models
# ---------------------------------------------------------------------------
def fig_per_class_f1():
    preds = {n: np.load(os.path.join(RES, f"y_pred_{n}.npy"))
             for n in ("rf", "xgb", "mlp")}
    f1s = {n: per_class_f1(y, p)[0] for n, p in preds.items()}
    support = np.bincount(y, minlength=K)
    order = np.argsort(-support)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ypos = np.arange(K)
    h = 0.26
    for off, (name, lab, col) in zip(
            (h, 0.0, -h),
            [("rf", "Random Forest", BLUE), ("xgb", "XGBoost", GREEN),
             ("mlp", "MLP", ORANGE)]):
        ax.barh(ypos + off, f1s[name][order], height=h * 0.92, color=col,
                label=lab, edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{CLASSES[i]}  (n={support[i]:,})" for i in order],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("per-class F1 on the held-out test set (calibrated thresholds)")
    ax.axvline(1.0, color=LGREY, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LGREY, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    # direct-label the two hardest classes
    for cls, note in (("Infiltration", "benign-like: the macro-F1 bottleneck"),
                      ("DoS attacks-SlowHTTPTest", "confused with FTP-BruteForce")):
        i = list(order).index(CLASSES.index(cls))
        ax.text(max(f1s["rf"][CLASSES.index(cls)],
                    f1s["xgb"][CLASSES.index(cls)]) + 0.03,
                i, note, fontsize=7.4, color=VERM, va="center")
    ax.set_title("Per-class F1 by model (classes ordered by test support)",
                 fontsize=10, pad=8)
    save(fig, "fig_per_class_f1.png")
    return f1s, support


# ---------------------------------------------------------------------------
# B. Effect of per-class threshold calibration (one-vs-rest F1 vs threshold)
# ---------------------------------------------------------------------------
def fig_threshold_calibration():
    proba = np.load(os.path.join(RES, "proba_rf.npy"), mmap_mode="r")
    thr = np.load(os.path.join(HERE, "models/thresholds_rf.npy"))
    argmax_pred = np.load(os.path.join(RES, "y_pred_rf.npy"))  # calibrated
    show = ["Brute Force-Web", "Brute Force-XSS", "DoS attacks-SlowHTTPTest"]
    grid = np.arange(0.04, 0.97, 0.02)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for cls, col in zip(show, (BLUE, GREEN, ORANGE)):
        k = CLASSES.index(cls)
        p = np.asarray(proba[:, k])
        pos = (y == k)
        npos = pos.sum()
        f1s = []
        for t in grid:
            pred = p >= t
            tp = np.count_nonzero(pred & pos)
            fp = np.count_nonzero(pred) - tp
            f1s.append(2 * tp / (2 * tp + fp + (npos - tp)) if tp else 0.0)
        f1s = np.array(f1s)
        ax.plot(grid, f1s, color=col, lw=2.0, label=f"{cls} (n={npos:,})")
        ax.plot([thr[k]], [f1s[np.argmin(np.abs(grid - thr[k]))]], "o",
                color=col, ms=7, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"tuned $\\tau$={thr[k]:.2f}",
                    xy=(thr[k], f1s[np.argmin(np.abs(grid - thr[k]))]),
                    xytext=(thr[k] + 0.03, f1s[np.argmin(np.abs(grid - thr[k]))] - 0.11),
                    fontsize=7.5, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
        del p
    ax.axvline(0.5, color=GREY, lw=1.0, ls="--")
    ax.text(0.51, 0.02, "naive 0.5 / argmax region", fontsize=7.5, color=GREY)
    ax.set_xlabel("decision threshold applied to the class probability")
    ax.set_ylabel("one-vs-rest F1 (test set)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.02, 0.98)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=LGREY, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Why per-class thresholds matter: F1 of a rare class is highly "
                 "threshold-sensitive", fontsize=9.5, pad=8)
    save(fig, "fig_threshold_calibration.png")


# ---------------------------------------------------------------------------
# C. Paired bootstrap of macro-F1 differences
#    (exact: resample the joint (y, pred_a, pred_b) contingency table)
# ---------------------------------------------------------------------------
def _macro_f1_from_cells(counts, ya, pa):
    """counts: (C,) cell counts; ya/pa: (C,) true and predicted labels per cell."""
    cm = np.bincount(ya * K + pa, weights=counts, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return f1.mean()


def bootstrap_delta(pred_a, pred_b, n_boot=2000, seed=0):
    key = (y * K + pred_a) * K + pred_b
    counts = np.bincount(key)
    nz = np.nonzero(counts)[0]
    c = counts[nz].astype(np.int64)
    ya = nz // (K * K)
    pa = (nz // K) % K
    pb = nz % K
    n = c.sum()
    p = c / n
    rng = np.random.default_rng(seed)
    obs = _macro_f1_from_cells(c, ya, pa) - _macro_f1_from_cells(c, ya, pb)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        rc = rng.multinomial(n, p)
        deltas[i] = (_macro_f1_from_cells(rc, ya, pa)
                     - _macro_f1_from_cells(rc, ya, pb))
    return obs, deltas


def fig_bootstrap_delta():
    rf = np.load(os.path.join(RES, "y_pred_rf.npy"))
    xgb = np.load(os.path.join(RES, "y_pred_xgb.npy"))
    ens = np.load(os.path.join(RES, "y_pred_ensemble.npy"))

    obs1, d1 = bootstrap_delta(rf, xgb, seed=1)
    obs2, d2 = bootstrap_delta(rf, ens, seed=2)
    ci1 = np.percentile(d1, [2.5, 97.5])
    ci2 = np.percentile(d2, [2.5, 97.5])
    print(f"RF - XGB : delta={obs1:+.4f} CI=[{ci1[0]:+.4f}, {ci1[1]:+.4f}] "
          f"P(RF>XGB)={(d1 > 0).mean():.3f}")
    print(f"RF - ENS : delta={obs2:+.4f} CI=[{ci2[0]:+.4f}, {ci2[1]:+.4f}] "
          f"P(RF>ENS)={(d2 > 0).mean():.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    bins = np.linspace(min(d1.min(), d2.min()) - 0.005,
                       max(d1.max(), d2.max()) + 0.005, 60)
    ax.hist(d1, bins=bins, color=BLUE, alpha=0.75, label="RF − XGBoost",
            edgecolor="white", lw=0.4)
    ax.hist(d2, bins=bins, color=ORANGE, alpha=0.75, label="RF − RF/XGB ensemble",
            edgecolor="white", lw=0.4)
    ax.axvline(0, color=INK, lw=1.2, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.34)
    ymax = ax.get_ylim()[1]
    ax.text(0.0015, ymax * 0.60, "no difference", fontsize=7.5, color=INK)
    for ci, col, yfrac in ((ci1, BLUE, 0.97), (ci2, ORANGE, 0.85)):
        ax.plot(ci, [ymax * yfrac] * 2, color=col, lw=2.6,
                solid_capstyle="butt")
        ax.text(np.mean(ci), ymax * (yfrac - 0.055),
                f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]", ha="center",
                fontsize=7.5, color=col)
    ax.set_xlabel("paired bootstrap difference in macro-F1 (2,000 resamples of the test set)")
    ax.set_ylabel("frequency")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.set_title("Random Forest beats XGBoost significantly, but ties with the ensemble",
                 fontsize=9.5, pad=8)
    save(fig, "fig_bootstrap_delta.png")
    return (obs1, ci1), (obs2, ci2)


# ---------------------------------------------------------------------------
# D. Audit: print every headline number the report quotes
# ---------------------------------------------------------------------------
def audit(f1s, support):
    print("\n--- per-class F1 (calibrated) ---")
    for i in np.argsort(-support):
        print(f"{CLASSES[i]:28s} n={support[i]:>9,}  RF={f1s['rf'][i]:.3f} "
              f"XGB={f1s['xgb'][i]:.3f} MLP={f1s['mlp'][i]:.3f}")

    # argmax-vs-calibrated for RF
    proba = np.load(os.path.join(RES, "proba_rf.npy"), mmap_mode="r")
    am = np.asarray(proba).argmax(1).astype(np.int64)
    f1_am, _ = per_class_f1(y, am)
    f1_cal = f1s["rf"]
    print("\n--- RF: argmax vs calibrated ---")
    print(f"macro-F1 argmax={f1_am.mean():.4f}  calibrated={f1_cal.mean():.4f}")
    for cls in ("Brute Force-Web", "Brute Force-XSS", "SQL Injection",
                "Infiltration"):
        k = CLASSES.index(cls)
        print(f"{cls:20s} argmax={f1_am[k]:.3f} -> calibrated={f1_cal[k]:.3f}")

    # Infiltration -> Benign leakage
    rf = np.load(os.path.join(RES, "y_pred_rf.npy"))
    k = CLASSES.index("Infiltration")
    inf = (y == k)
    print(f"\nInfiltration flows misclassified as Benign (RF): "
          f"{np.count_nonzero(rf[inf] == BENIGN):,} of {inf.sum():,}")

    # anomaly detectors
    ae = np.load(os.path.join(RES, "anomaly_flags_ae.npy"))
    iso = np.load(os.path.join(RES, "anomaly_flags.npy"))
    yb = (y != BENIGN)
    for name, fl in (("AE", ae), ("IsolationForest", iso)):
        fl = fl.astype(bool)
        tp = np.count_nonzero(fl & yb)
        rec = tp / yb.sum()
        fpr = np.count_nonzero(fl & ~yb) / (~yb).sum()
        prec = tp / max(np.count_nonzero(fl), 1)
        f1 = 2 * prec * rec / (prec + rec)
        print(f"{name:16s} recall={rec:.4f} FPR={fpr:.4f} prec={prec:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    f1s, support = fig_per_class_f1()
    fig_threshold_calibration()
    fig_bootstrap_delta()
    audit(f1s, support)

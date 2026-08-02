#!/usr/bin/env python
"""
Regenerate the older notebook-produced figures in the same visual style as the
Chapter 3 diagrams (Okabe-Ito colour-blind-safe palette, consistent typography,
recessive grid), so the whole report looks like one document.

Nothing is re-trained and no number is invented: every figure is rebuilt from
the arrays already saved under models/ , results/ and data/processed/. The
originals are backed up to results/legacy_style/ before being overwritten.

Also prints an audit of the §4.4 numbers (Expected Calibration Error and the
cost-sensitive totals) that were previously carried over from the notebooks.

Run:  python build/restyle_legacy_figures.py
"""
import os
import shutil
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
BAK = os.path.join(RES, "legacy_style")
os.makedirs(BAK, exist_ok=True)

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERM, SKY, PURPLE = "#D55E00", "#56B4E9", "#CC79A7"
GREY, LGREY, INK = "#5A5A5A", "#D9D9D9", "#1A1A1A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GREY, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.titlesize": 10, "legend.frameon": False,
})

# single-hue sequential ramps matching the palette (light -> dark)
CMAPS = {
    "blue": LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#DCEBF5", "#0072B2", "#00456B"]),
    "green": LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#DCF0E9", "#009E73", "#005F45"]),
    "orange": LinearSegmentedColormap.from_list("o", ["#FFFFFF", "#FBEBD2", "#E69F00", "#8A6000"]),
    "purple": LinearSegmentedColormap.from_list("p", ["#FFFFFF", "#F4E4EE", "#CC79A7", "#7A4864"]),
}


def save(fig, name):
    dst = os.path.join(RES, name)
    if os.path.exists(dst) and not os.path.exists(os.path.join(BAK, name)):
        shutil.copy2(dst, os.path.join(BAK, name))
    fig.savefig(dst, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("restyled", name)


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


# ---------------------------------------------------------------- data ------
y_test = np.load(os.path.join(HERE, "data/processed/y_test.npy"))
le = joblib.load(os.path.join(HERE, "models/label_encoder.pkl"))
CLASSES = [c.replace("Infilteration", "Infiltration").replace("Brute Force -", "Brute Force-")
           for c in le.classes_]
K = len(CLASSES)
BENIGN = CLASSES.index("Benign")
SHORT = [c.replace("DDoS attacks-", "DDoS-").replace("DDOS attack-", "DDoS-")
          .replace("DoS attacks-", "DoS-") for c in CLASSES]


# ---------------------------------------------------------------------------
# 1. Overall class distribution (linear + log)
# ---------------------------------------------------------------------------
def class_distribution():
    counts = np.zeros(K, dtype=np.int64)
    for f in ("y_train_original", "y_val", "y_test"):
        counts += np.bincount(np.load(os.path.join(HERE, f"data/processed/{f}.npy")),
                              minlength=K)
    order = np.argsort(-counts)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, log in zip(axes, (False, True)):
        cols = [BLUE if i == BENIGN else VERM for i in order]
        ax.bar(range(K), counts[order], color=cols, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(K))
        ax.set_xticklabels([SHORT[i] for i in order], rotation=55, ha="right", fontsize=7)
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("number of flows (log scale)")
            ax.set_title("Log scale — reveals the true 6-order-of-magnitude spread",
                         color=GREY, fontsize=9)
            ax.annotate(f"{counts[order][0]:,}", xy=(0, counts[order][0]),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=7, color=BLUE)
            ax.annotate(f"{counts[order][-1]:,}", xy=(K - 1, counts[order][-1]),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=7, color=VERM)
        else:
            ax.set_ylabel("number of flows")
            ax.set_title("Linear scale — every attack class is flattened to ~0",
                         color=GREY, fontsize=9)
            ax.ticklabel_format(axis="y", style="plain")
        ax.grid(axis="y", color=LGREY, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        despine(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=VERM)]
    axes[0].legend(handles, ["Benign (majority)", "Attack classes (minority)"],
                   fontsize=8, loc="upper right")
    fig.suptitle("CSE-CIC-IDS2018 class distribution (16,232,943 flows)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "class_distribution.png")


# ---------------------------------------------------------------------------
# 2. Class distribution before vs after capped SMOTE
# ---------------------------------------------------------------------------
def smote_distribution():
    before = np.bincount(np.load(os.path.join(HERE, "data/processed/y_train_original.npy")),
                         minlength=K)
    after = np.bincount(np.load(os.path.join(HERE, "data/processed/y_train.npy")),
                        minlength=K)
    order = np.argsort(-before)
    x = np.arange(K)
    w = 0.4
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.bar(x - w / 2, before[order], w, color=GREY, label="before SMOTE",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, after[order], w, color=ORANGE, label="after capped SMOTE",
           edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[i] for i in order], rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("training flows (log scale)")
    ax.grid(axis="y", color=LGREY, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(fontsize=8.5, loc="upper right")
    k = CLASSES.index("SQL Injection")
    i = list(order).index(k)
    ax.annotate(f"{before[k]:,} → {after[k]:,}\n(50× cap)", xy=(i + w / 2, after[k]),
                xytext=(i - 1.6, after[k] * 14), fontsize=7.5, color=VERM, ha="center",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=0.9))
    ax.set_title(f"Capped SMOTE: {before.sum():,} → {after.sum():,} training flows — "
                 "deliberately still imbalanced", fontsize=10)
    save(fig, "smote_distribution.png")


# ---------------------------------------------------------------------------
# 3. MI vs ANOVA vs variance
# ---------------------------------------------------------------------------
def feature_selection_comparison():
    mi = np.load(os.path.join(HERE, "models/feat_sel_mi_scores.npy"))
    f = np.load(os.path.join(HERE, "models/feat_sel_f_scores.npy"))
    X = np.load(os.path.join(HERE, "models/feat_sel_X_sample.npy"))
    var = X.var(axis=0)
    nrm = lambda a: (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)
    mi_n, f_n, v_n = nrm(mi), nrm(np.nan_to_num(f)), nrm(var)
    top = lambda a, n=40: set(np.argsort(-np.nan_to_num(a))[:n])
    t_mi, t_f, t_v = top(mi), top(f_n), top(v_n)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for ax, other, name, col in ((axes[0], v_n, "Variance", ORANGE),
                                 (axes[1], f_n, "ANOVA F-score", GREEN)):
        ax.scatter(other, mi_n, s=22, color=col, alpha=0.75, edgecolor="white", lw=0.4)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY)
        ax.set_xlabel(f"{name} (normalised)")
        ax.set_ylabel("Mutual Information (normalised)")
        ax.set_title(f"MI vs {name}", color=GREY, fontsize=9)
        ax.grid(color=LGREY, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        despine(ax)
        ax.text(0.03, 0.95, "above the line =\nMI keeps, linear filter drops",
                transform=ax.transAxes, fontsize=7.2, color=col, va="top")

    ax = axes[2]
    M = np.array([[40, len(t_mi & t_f), len(t_mi & t_v)],
                  [len(t_mi & t_f), 40, len(t_f & t_v)],
                  [len(t_mi & t_v), len(t_f & t_v), 40]])
    im = ax.imshow(M, cmap=CMAPS["blue"], vmin=0, vmax=40)
    labs = ["MI (ours)", "ANOVA F", "Variance"]
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labs, fontsize=8); ax.set_yticklabels(labs, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 26 else INK)
    ax.set_title("Shared features in each top-40", color=GREY, fontsize=9)
    fig.tight_layout()
    save(fig, "feature_selection_comparison.png")
    return mi, f_n


# ---------------------------------------------------------------------------
# 4. The non-linear demo feature
# ---------------------------------------------------------------------------
def mi_nonlinear_demo(mi, f_n):
    """Show a feature MI ranks highly and ANOVA does not.

    Network features are extremely right-skewed (>75% of mass in one bin), so a
    raw histogram is unreadable. Values are therefore shown on a pooled
    percentile-rank axis, which preserves *shape* differences between classes
    while making them visible; the raw class means are printed alongside to
    demonstrate that an F-test, which compares means, sees almost nothing.
    """
    X = np.load(os.path.join(HERE, "models/feat_sel_X_sample.npy"))
    ys = np.load(os.path.join(HERE, "models/feat_sel_y_sample.npy"))
    cols = json.load(open(os.path.join(HERE, "models/feature_cols.json")))
    mi_rank = np.argsort(np.argsort(-np.nan_to_num(mi)))
    f_rank = np.argsort(np.argsort(-np.nan_to_num(f_n)))
    cand = np.where((mi_rank < 20) & (f_rank >= 40))[0]
    j = int(cand[np.argmin(mi_rank[cand])]) if len(cand) else int(np.argmax(mi))

    show = ["Benign", "DoS attacks-Hulk", "DDoS attacks-LOIC-HTTP", "DDOS attack-HOIC"]
    palette = [BLUE, VERM, GREEN, ORANGE]
    v = X[:, j]
    pct = np.empty_like(v, dtype=float)
    pct[np.argsort(v)] = np.linspace(0, 100, len(v))      # pooled percentile rank

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    data, labels, means = [], [], []
    for cls in show:
        k = list(le.classes_).index(cls)
        m = ys == k
        if m.sum() < 30:
            continue
        data.append(pct[m])
        labels.append(cls.replace("DDoS attacks-", "DDoS-").replace("DDOS attack-", "DDoS-")
                         .replace("DoS attacks-", "DoS-") + f"\n(n={int(m.sum()):,})")
        means.append(v[m].mean())
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for body, col in zip(parts["bodies"], palette):
        body.set_facecolor(col); body.set_alpha(0.55); body.set_edgecolor(col)
        body.set_linewidth(1.2)
    for i, d in enumerate(data, start=1):
        ax.plot([i], [np.median(d)], "o", color="white", ms=5, mec=INK, mew=1.1, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"{cols[j]} — pooled percentile rank")
    ax.set_ylim(-4, 104)
    ax.grid(axis="y", color=LGREY, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Distributions differ in SHAPE — MI detects this", color=GREY, fontsize=9)

    lo, hi = min(means), max(means)
    pad = max((hi - lo) * 0.45, 1e-9)
    base = lo - pad
    ax2.barh(range(len(means)), [m - base for m in means], left=base,
             color=palette[:len(means)], edgecolor="white", lw=0.6)
    for i, m in enumerate(means):
        ax2.text(m, i, f"  {m:.6f}", va="center", fontsize=7.6, color=INK)
    ax2.set_yticks(range(len(means)))
    ax2.set_yticklabels([l.split("\n")[0] for l in labels], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(base, hi + pad * 4.5)
    ax2.set_xticks([])
    ax2.set_xlabel(f"class mean (axis zoomed to a {(hi-lo):.0e} range)", fontsize=8.5)
    despine(ax2, keep=("left",))
    ax2.set_title("…but the class MEANS agree to 5 decimal\nplaces — an F-test sees almost nothing",
                  color=GREY, fontsize=9)
    fig.suptitle(f"{cols[j]}: MI rank {mi_rank[j]+1} of 77, ANOVA F rank {f_rank[j]+1} of 77",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "mi_nonlinear_demo.png")
    return cols[j], int(mi_rank[j] + 1), int(f_rank[j] + 1)


# ---------------------------------------------------------------------------
# 5. Feature-space ablation
# ---------------------------------------------------------------------------
def ablation():
    import csv
    rows = list(csv.DictReader(open(os.path.join(RES, "ablation_hybrid.csv"))))
    labs = [r["config"] for r in rows]
    macro = [float(r["macro_f1"]) for r in rows]
    infil = [float(r["infiltration_f1"]) for r in rows]
    x = np.arange(len(rows)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    b1 = ax.bar(x - w / 2, macro, w, color=BLUE, label="Macro-F1 (all 15 classes)",
                edgecolor="white", lw=0.6)
    b2 = ax.bar(x + w / 2, infil, w, color=ORANGE, label="Infiltration F1 (hardest class)",
                edgecolor="white", lw=0.6)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("F1")
    ax.grid(axis="y", color=LGREY, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    despine(ax)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.annotate("", xy=(2 - w / 2, macro[2] + 0.055), xytext=(0 - w / 2, macro[0] + 0.055),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.text(1 - w / 2, macro[2] + 0.075, f"+{macro[2]-macro[0]:.3f} macro-F1",
            ha="center", fontsize=8, color=GREEN)
    ax.set_title("Ablation: hand-crafted vs latent vs hybrid feature space\n"
                 "(identical model, sample and seed — only the feature columns change)",
                 fontsize=9.5)
    save(fig, "ablation_hybrid.png")


# ---------------------------------------------------------------------------
# 6. Random Forest feature importance
# ---------------------------------------------------------------------------
def feature_importance():
    imp = np.load(os.path.join(RES, "importances_rf.npy"))
    cols = json.load(open(os.path.join(HERE, "models/feature_cols.json")))
    sel = np.load(os.path.join(HERE, "models/selected_indices.npy"))
    names = [cols[i] for i in sel] + [f"Latent_{i}" for i in range(1, 17)]
    order = np.argsort(-imp)[:20][::-1]
    cols_bar = [ORANGE if names[i].startswith("Latent") else BLUE for i in order]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.barh(range(20), imp[order], color=cols_bar, edgecolor="white", lw=0.5)
    ax.set_yticks(range(20))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("Gini importance")
    ax.grid(axis="x", color=LGREY, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    despine(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    n_lat = sum(1 for i in order if names[i].startswith("Latent"))
    ax.legend(handles, ["hand-crafted (MI-selected)", "autoencoder latent"],
              fontsize=8.5, loc="lower right")
    ax.set_title(f"Random Forest — top 20 feature importances "
                 f"({n_lat} of 20 are latent dimensions)", fontsize=9.5)
    save(fig, "feature_importance.png")


# ---------------------------------------------------------------------------
# 7. Confusion matrices
# ---------------------------------------------------------------------------
def confusion_matrices():
    specs = [("rf", "Random Forest", "blue"), ("xgb", "XGBoost", "green"),
             ("mlp", "MLP", "purple"), ("ensemble", "RF + XGB Ensemble (w=0.80)", "orange")]
    for key, title, cmap in specs:
        p = np.load(os.path.join(RES, f"y_pred_{key}.npy"))
        cm = np.bincount(y_test * K + p, minlength=K * K).reshape(K, K).astype(float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        im = ax.imshow(cmn, cmap=CMAPS[cmap], vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(SHORT, rotation=55, ha="right", fontsize=7.5)
        ax.set_yticklabels([f"{s}  ({int(cm[i].sum()):,})" for i, s in enumerate(SHORT)],
                           fontsize=7.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual  (test support)")
        for i in range(K):
            for j in range(K):
                v = cmn[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                            color="white" if v > 0.55 else INK)
        ax.set_xticks(np.arange(-.5, K, 1), minor=True)
        ax.set_yticks(np.arange(-.5, K, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="minor", length=0)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(
            "row-normalised (diagonal = recall)", fontsize=8)
        ax.set_title(f"{title} — confusion matrix (row-normalised)", fontsize=10, pad=10)
        save(fig, f"{key}_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 8. Reliability diagrams  (+ recompute ECE)
# ---------------------------------------------------------------------------
def reliability():
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, (key, title, col) in zip(axes, [("rf", "Random Forest", BLUE),
                                            ("xgb", "XGBoost", GREEN),
                                            ("mlp", "MLP", ORANGE)]):
        path = os.path.join(RES, f"proba_{key}.npy")
        if not os.path.exists(path):
            ax.axis("off"); continue
        proba = np.load(path, mmap_mode="r")
        conf = np.asarray(proba.max(1)).astype(np.float64)
        pred = np.asarray(proba.argmax(1))
        correct = (pred == y_test)
        NB = 15                      # matches the binning used for the reported ECE
        bins = np.linspace(0, 1, NB + 1)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, NB - 1)
        ece, xs, ys_, ns = 0.0, [], [], []
        for b in range(NB):
            m = idx == b
            n = int(m.sum())
            if not n:
                continue
            acc, cf = correct[m].mean(), conf[m].mean()
            ece += n / len(conf) * abs(acc - cf)
            xs.append(cf); ys_.append(acc); ns.append(n)
        out[key] = ece
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY, label="perfect calibration")
        ax.plot(xs, ys_, "-o", color=col, lw=1.8, ms=5, mec="white", mew=1,
                label=f"{title}  (ECE={ece:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean confidence"); ax.set_ylabel("observed accuracy")
        ax.grid(color=LGREY, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
        despine(ax)
        ax.legend(fontsize=7.6, loc="upper left")
        ax.set_title(title, color=GREY, fontsize=9)
        del proba, conf, pred
    fig.suptitle("Reliability diagrams (15 equal-width confidence bins) — does stated confidence match observed accuracy?",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "calibration_reliability.png")
    return out


# ---------------------------------------------------------------------------
# 9. Cost-sensitive curve (+ recompute totals)
# ---------------------------------------------------------------------------
def cost_sensitive():
    yb = (y_test != BENIGN)
    ratios = np.arange(1, 51)
    totals = {}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for key, title, col in [("rf", "Random Forest", BLUE), ("xgb", "XGBoost", GREEN),
                            ("mlp", "MLP", ORANGE), ("ensemble", "Ensemble", VERM)]:
        p = np.load(os.path.join(RES, f"y_pred_{key}.npy"))
        pb = (p != BENIGN)
        fp = int(np.count_nonzero(pb & ~yb)); fn = int(np.count_nonzero(~pb & yb))
        cost = fp + ratios * fn
        totals[key] = (fp, fn, int(fp + 10 * fn))
        ax.plot(ratios, cost / 1e6, color=col, lw=2.0, label=f"{title}")
    ax.axvline(10, color=GREY, lw=1.0, ls="--")
    ax.text(10.6, ax.get_ylim()[1] * 0.06, "reference: a missed attack\ncosts 10× a false alarm",
            fontsize=7.5, color=GREY)
    ax.set_xlabel("cost of a false negative, relative to a false alarm")
    ax.set_ylabel("total operational cost  (millions)")
    ax.grid(color=LGREY, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    despine(ax)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Cost-sensitive comparison — lower is better", fontsize=9.5)
    save(fig, "cost_sensitive.png")
    return totals


if __name__ == "__main__":
    class_distribution()
    smote_distribution()
    mi, f_n = feature_selection_comparison()
    mi_nonlinear_demo(mi, f_n)
    ablation()
    feature_importance()
    confusion_matrices()
    ece = reliability()
    costs = cost_sensitive()

    print("\n=== audit of the §4.4 numbers ===")
    for k, v in ece.items():
        print(f"ECE {k:9s} = {v:.4f}")
    print("\nmodel      FP      FN      cost @10x")
    for k, (fp, fn, c) in costs.items():
        print(f"{k:9s} {fp:7,} {fn:7,} {c:12,}")
    print("\noriginals backed up to results/legacy_style/")

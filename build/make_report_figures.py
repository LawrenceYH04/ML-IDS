#!/usr/bin/env python
"""
Generate the explanatory / schematic figures used in Chapter 3 (Methodology)
of the ML-IDS report.

These are conceptual diagrams (architectures, mechanisms, pipelines) drawn from
the actual configuration of the implemented models -- they are original figures,
not reproductions of third-party material, so they can be used in the report
without permission issues.

Run:  python build/make_report_figures.py
Out:  ML-IDS/results/fig_*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#5A5A5A"
LGREY = "#D9D9D9"
INK = "#1A1A1A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


def box(ax, x, y, w, h, label, fc="white", ec=GREY, fs=9, weight="normal",
        tc=INK, lw=1.2, radius=0.02):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.008,rounding_size={radius}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, zorder=3, linespacing=1.35)
    return p


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.3, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=lw, color=color,
                                 linestyle=ls, zorder=1,
                                 shrinkA=0, shrinkB=0))


def blank_ax(fig, rect=(0, 0, 1, 1)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


# ----------------------------------------------------------------------------
# 1. Feature-extraction autoencoder architecture (77-64-32-16-32-64-77)
# ----------------------------------------------------------------------------
def fig_ae_architecture():
    fig = plt.figure(figsize=(7.4, 3.9))
    ax = blank_ax(fig)

    dims = [77, 64, 32, 16, 32, 64, 77]
    labels = ["Input\n77", "64", "32", "Latent\n16", "32", "64", "Output\n77"]
    xs = np.linspace(0.07, 0.93, 7)
    # visual height proportional to sqrt of width so 77 vs 16 stays readable
    hs = 0.16 + 0.42 * (np.array(dims) / 77.0) ** 0.75
    cols = [LGREY, SKY, SKY, VERM, SKY, SKY, LGREY]

    for x, h, lab, c, d in zip(xs, hs, labels, cols, dims):
        ax.add_patch(Rectangle((x - 0.035, 0.5 - h / 2), 0.07, h,
                               facecolor=c, edgecolor=GREY, lw=1.0, zorder=2))
        ax.text(x, 0.5, lab, ha="center", va="center", fontsize=8.5,
                color="white" if c in (VERM,) else INK, weight="bold", zorder=3)
    for i in range(6):
        arrow(ax, xs[i] + 0.037, 0.5, xs[i + 1] - 0.037, 0.5, color=GREY, lw=1.0)

    # encoder / decoder brackets
    ax.plot([xs[0] - 0.035, xs[3]], [0.90, 0.90], color=BLUE, lw=1.4)
    ax.text((xs[0] + xs[3]) / 2, 0.925, "ENCODER  (used at inference)",
            ha="center", fontsize=8.5, color=BLUE, weight="bold")
    ax.plot([xs[3], xs[6] + 0.035], [0.90, 0.90], color=GREY, lw=1.4, ls="--")
    ax.text((xs[3] + xs[6]) / 2, 0.925, "DECODER  (training only)",
            ha="center", fontsize=8.5, color=GREY)

    ann = [
        (xs[1], "LeakyReLU\n(slope 0.01)", SKY),
        (xs[3], "linear\n(no activation)", VERM),
        (xs[6], "Sigmoid\n-> [0, 1]", GREY),
    ]
    for x, t, c in ann:
        ax.annotate(t, xy=(x, 0.5 - hs[list(xs).index(x)] / 2 - 0.01),
                    xytext=(x, 0.13), ha="center", fontsize=8, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9))

    ax.text(0.5, 0.02,
            r"Trained on pre-SMOTE traffic to minimise  $L = \frac{1}{n}\sum_i \|x_i - \hat{x}_i\|^2$"
            "   ·   best validation MSE $\\approx 2.9\\times10^{-5}$   ·   16/16 latent dimensions live",
            ha="center", fontsize=8.5, color=INK)
    save(fig, "fig_ae_architecture.png")


# ----------------------------------------------------------------------------
# 2. How Mutual Information works
# ----------------------------------------------------------------------------
def fig_mi_concept():
    fig = plt.figure(figsize=(7.6, 3.1))

    # (a) entropy Venn diagram
    ax1 = fig.add_axes([0.01, 0.05, 0.30, 0.86])
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    ax1.add_patch(Ellipse((0.40, 0.52), 0.60, 0.62, facecolor=BLUE, alpha=0.30,
                          edgecolor=BLUE, lw=1.2))
    ax1.add_patch(Ellipse((0.64, 0.52), 0.60, 0.62, facecolor=ORANGE, alpha=0.30,
                          edgecolor=ORANGE, lw=1.2))
    ax1.text(0.20, 0.52, "H(X|Y)", ha="center", fontsize=8.5)
    ax1.text(0.84, 0.52, "H(Y|X)", ha="center", fontsize=8.5)
    ax1.text(0.52, 0.52, "I(X;Y)", ha="center", fontsize=9, weight="bold")
    ax1.text(0.22, 0.90, "H(X)\nfeature", ha="center", fontsize=8.5, color=BLUE)
    ax1.text(0.82, 0.90, "H(Y)\nlabel", ha="center", fontsize=8.5, color=ORANGE)
    ax1.text(0.5, 0.06, "(a)  MI = shared information", ha="center", fontsize=8.5,
             color=GREY)

    # (b) joint vs product of marginals for a dependent feature
    rng = np.random.default_rng(7)
    ax2 = fig.add_axes([0.37, 0.20, 0.26, 0.68])
    joint = np.array([[0.02, 0.30, 0.02],
                      [0.28, 0.03, 0.02],
                      [0.02, 0.03, 0.28]])
    im = ax2.imshow(joint, cmap="Blues", vmin=0, vmax=0.32)
    ax2.set_xticks([0, 1, 2]); ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(["low", "mid", "high"], fontsize=8)
    ax2.set_yticklabels(["Benign", "DoS", "Bot"], fontsize=8)
    ax2.set_xlabel("feature bin  $x$", fontsize=8.5)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{joint[i, j]:.2f}", ha="center", va="center",
                     fontsize=7.5,
                     color="white" if joint[i, j] > 0.16 else INK)
    ax2.set_title("(b)  observed  $p(x,y)$", fontsize=8.5, color=GREY, pad=6)

    ax3 = fig.add_axes([0.68, 0.20, 0.26, 0.68])
    px = joint.sum(0); py = joint.sum(1)
    indep = np.outer(py, px)
    ax3.imshow(indep, cmap="Blues", vmin=0, vmax=0.32)
    ax3.set_xticks([0, 1, 2]); ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels(["low", "mid", "high"], fontsize=8)
    ax3.set_yticklabels([])
    ax3.set_xlabel("feature bin  $x$", fontsize=8.5)
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f"{indep[i, j]:.2f}", ha="center", va="center",
                     fontsize=7.5,
                     color="white" if indep[i, j] > 0.16 else INK)
    ax3.set_title("(c)  if independent  $p(x)p(y)$", fontsize=8.5, color=GREY, pad=6)

    mi = np.sum(joint * np.log2(joint / indep))
    fig.text(0.66, 0.035,
             r"$I(X;Y)=\sum_{x,y} p(x,y)\,\log_2\frac{p(x,y)}{p(x)p(y)}$"
             f"  =  {mi:.2f} bits  (0 only if (b) equals (c))",
             ha="center", fontsize=8.5)
    save(fig, "fig_mi_concept.png")


# ----------------------------------------------------------------------------
# 3. Hybrid feature-space construction
# ----------------------------------------------------------------------------
def fig_hybrid_feature_space():
    fig = plt.figure(figsize=(7.4, 3.2))
    ax = blank_ax(fig)

    box(ax, 0.02, 0.40, 0.16, 0.22,
        "77 scaled\nflow features\n(per flow)", fc="white", fs=8.5)

    # selection path
    box(ax, 0.28, 0.63, 0.22, 0.22,
        "FEATURE SELECTION\nMutual Information\nrank -> keep top 40",
        fc="#EAF3F9", ec=BLUE, fs=8.3)
    box(ax, 0.28, 0.13, 0.22, 0.22,
        "FEATURE EXTRACTION\nAutoencoder encoder\n77 -> 16 latent",
        fc="#FDF1E3", ec=ORANGE, fs=8.3)

    arrow(ax, 0.18, 0.55, 0.28, 0.72, color=BLUE)
    arrow(ax, 0.18, 0.47, 0.28, 0.26, color=ORANGE)

    box(ax, 0.56, 0.63, 0.13, 0.22, "40\nselected\nfeatures", fc=BLUE,
        ec=BLUE, tc="white", fs=8.5, weight="bold")
    box(ax, 0.56, 0.13, 0.13, 0.22, "16\nlatent\nfeatures", fc=ORANGE,
        ec=ORANGE, tc="white", fs=8.5, weight="bold")
    arrow(ax, 0.50, 0.74, 0.56, 0.74, color=BLUE)
    arrow(ax, 0.50, 0.24, 0.56, 0.24, color=ORANGE)

    ax.text(0.735, 0.49, "+", fontsize=16, ha="center", va="center", color=GREY)
    arrow(ax, 0.69, 0.70, 0.78, 0.56, color=BLUE)
    arrow(ax, 0.69, 0.28, 0.78, 0.42, color=ORANGE)

    box(ax, 0.79, 0.34, 0.19, 0.30,
        "HYBRID VECTOR\n56 features\nper flow", fc="white", ec=INK, fs=9,
        weight="bold", lw=1.6)
    ax.text(0.885, 0.28, "input to all five models", ha="center", fontsize=8,
            color=GREY, style="italic")

    ax.text(0.39, 0.94, "interpretable, human-named statistics", ha="center",
            fontsize=8, color=BLUE, style="italic")
    ax.text(0.39, 0.05, "abstract, non-linear learned structure", ha="center",
            fontsize=8, color=ORANGE, style="italic")
    save(fig, "fig_hybrid_feature_space.png")


# ----------------------------------------------------------------------------
# 4. MLP architecture
# ----------------------------------------------------------------------------
def fig_mlp_architecture():
    fig = plt.figure(figsize=(7.4, 4.8))
    ax = blank_ax(fig)

    layers = [("Input\n56", 8, LGREY), ("Hidden 1\n512", 7, SKY),
              ("Hidden 2\n256", 6, SKY), ("Hidden 3\n128", 5, SKY),
              ("Output\n15", 4, VERM)]
    xs = np.linspace(0.10, 0.90, 5)
    positions = []
    for x, (lab, n, c) in zip(xs, layers):
        ys = np.linspace(0.50, 0.86, n)
        positions.append(ys)
        for y in ys:
            ax.add_patch(Circle((x, y), 0.014, facecolor=c, edgecolor=GREY,
                                lw=0.8, zorder=3))
        ax.text(x, 0.94, lab, ha="center", fontsize=8.5, weight="bold")

    for i in range(4):
        for y1 in positions[i]:
            for y2 in positions[i + 1]:
                ax.plot([xs[i] + 0.014, xs[i + 1] - 0.014], [y1, y2],
                        color=GREY, lw=0.25, alpha=0.35, zorder=1)

    for i, x in enumerate(xs[1:4], start=1):
        rate = [0.3, 0.3, 0.2][i - 1]
        ax.text(x, 0.43, f"ReLU\ndropout {rate}", ha="center", fontsize=7.8,
                color=SKY)
    ax.text(xs[4], 0.43, "softmax\n15 classes", ha="center", fontsize=7.8,
            color=VERM)

    ax.text(0.5, 0.315,
            r"$h^{(l)} = \mathrm{ReLU}\!\left(W^{(l)}h^{(l-1)} + b^{(l)}\right)$"
            r"$,\qquad l = 1,2,3$",
            ha="center", fontsize=10)
    ax.text(0.5, 0.205,
            r"$\hat{y}_k = \dfrac{\exp(z_k)}{\sum_{j=1}^{15}\exp(z_j)}$"
            r"$\qquad\qquad L = -\dfrac{1}{N}\sum_{i=1}^{N} w_{y_i}\,\log \hat{y}_{i,y_i}$",
            ha="center", fontsize=10)
    ax.text(0.5, 0.075,
            "weighted cross-entropy with class weights $w$ from 0.13 (Benign) to 402 (SQL Injection)",
            ha="center", fontsize=8.5)
    ax.text(0.5, 0.015,
            "Adam, lr $5\\times10^{-4}$, weight decay $1\\times10^{-4}$, early stopping patience 20",
            ha="center", fontsize=8, color=GREY)
    save(fig, "fig_mlp_architecture.png")


# ----------------------------------------------------------------------------
# 5. Random Forest schematic
# ----------------------------------------------------------------------------
def _mini_tree(ax, cx, cy, w, h, color):
    """Draw a tiny 3-level decision tree inside a box."""
    lvl = [[(0.5, 1.0)], [(0.25, 0.55), (0.75, 0.55)],
           [(0.12, 0.12), (0.38, 0.12), (0.62, 0.12), (0.88, 0.12)]]
    def T(p):
        return (cx + p[0] * w, cy + p[1] * h)
    for a, bs in ((lvl[0][0], lvl[1]), (lvl[1][0], lvl[2][:2]),
                  (lvl[1][1], lvl[2][2:])):
        for b in bs:
            ax.plot(*zip(T(a), T(b)), color=color, lw=0.9, zorder=2)
    for row in lvl:
        for p in row:
            ax.add_patch(Circle(T(p), 0.010, facecolor=color, edgecolor="none",
                                zorder=3))


def fig_rf_schematic():
    fig = plt.figure(figsize=(7.4, 4.3))
    ax = blank_ax(fig)

    box(ax, 0.02, 0.52, 0.16, 0.26,
        "SMOTE-balanced\ntraining set\n18.4 M flows", fs=8.3)

    x = 0.36
    ys = [0.72, 0.47, 0.22]
    for i, y in enumerate(ys):
        box(ax, x, y, 0.16, 0.21, "", fc="white", ec=GREY)
        _mini_tree(ax, x + 0.025, y + 0.025, 0.11, 0.155, BLUE)
        ax.text(x - 0.015, y + 0.105, f"Tree {i + 1}", ha="right", va="center",
                fontsize=8)
        arrow(ax, 0.185, 0.65, x - 0.055, y + 0.105, color=GREY, lw=0.9)
        arrow(ax, x + 0.166, y + 0.105, 0.755, 0.65, color=LGREY, lw=0.9)
    ax.text(x + 0.08, 0.145, ". . .  300 trees in total", fontsize=8.5,
            color=GREY, ha="center", style="italic")

    box(ax, 0.76, 0.52, 0.22, 0.26,
        "average the votes\n$\\rightarrow$ $p(y=k\\,|\\,x)$\nper-class thresholds\n$\\rightarrow$ predicted class",
        fc="#EAF3F9", ec=BLUE, fs=8.3)

    ax.text(0.5, 0.085,
            r"$\hat{p}(y=k\mid x) = \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}\left[f_t(x)=k\right]$"
            "     ($T=300$, max_depth 35, min_samples_leaf 2, class_weight = balanced)",
            ha="center", fontsize=9)
    ax.text(0.5, 0.005,
            "Trees are grown independently and in parallel — each on its own bootstrap sample with a random feature\n"
            "subset per split — so their errors decorrelate and averaging them cuts variance without adding bias.",
            ha="center", fontsize=7.8, color=GREY)
    save(fig, "fig_rf_schematic.png")


# ----------------------------------------------------------------------------
# 6. XGBoost schematic (additive, residual-driven boosting)
# ----------------------------------------------------------------------------
def fig_xgb_schematic():
    fig = plt.figure(figsize=(7.4, 3.5))
    ax = blank_ax(fig)

    xs = [0.06, 0.30, 0.54]
    for i, x in enumerate(xs):
        box(ax, x, 0.46, 0.16, 0.34, "", fc="white", ec=GREY)
        _mini_tree(ax, x + 0.02, 0.50, 0.12, 0.24, GREEN)
        ax.text(x + 0.08, 0.825, f"$f_{i + 1}(x)$", ha="center", fontsize=9)
        ax.text(x + 0.08, 0.415, ["fits the labels",
                                  "fits residual errors of $f_1$",
                                  "fits residual errors of $f_1+f_2$"][i],
                ha="center", fontsize=7.6, color=GREY)
        if i < 2:
            arrow(ax, x + 0.165, 0.63, x + 0.235, 0.63, color=VERM)
            ax.text(x + 0.20, 0.665, "residual", ha="center", fontsize=7.2,
                    color=VERM)
    ax.text(0.79, 0.63, ". . .", fontsize=14, color=GREY, ha="center")
    box(ax, 0.845, 0.46, 0.14, 0.34,
        "up to 8,000\nrounds\n(early stop\n@50)", fc="#E9F6F1", ec=GREEN, fs=8)

    ax.text(0.5, 0.30,
            r"$\hat{y}^{(M)}(x) = \sum_{m=1}^{M}\eta\, f_m(x),\qquad$"
            r"$f_m \approx -\,\partial L/\partial \hat{y}^{(m-1)}$   (gradient of the log-loss)",
            ha="center", fontsize=9.5)
    ax.text(0.5, 0.19,
            r"$\mathcal{L} = \sum_i w_i\,\ell(y_i,\hat{y}_i)"
            r" + \sum_m \left[\gamma T_m + \frac{1}{2}\lambda\,\omega_m^2\right]$"
            "        (per-row weights $w_i$ = balanced class weights)",
            ha="center", fontsize=9)
    ax.text(0.5, 0.06,
            "Unlike Random Forest, the trees are grown sequentially: each new tree is trained on what the\n"
            "current ensemble still gets wrong, which is why boosting drives bias down but reacts sharply to noise.",
            ha="center", fontsize=8, color=GREY)
    save(fig, "fig_xgb_schematic.png")


# ----------------------------------------------------------------------------
# 7. Benign-trained anomaly autoencoder: mechanism
# ----------------------------------------------------------------------------
def fig_anomaly_ae_schematic():
    fig = plt.figure(figsize=(7.6, 3.2))

    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    dims = [56, 32, 16, 32, 56]
    labs = ["x\n56", "32", "z\n16", "32", "$\\hat{x}$\n56"]
    xs = np.linspace(0.10, 0.86, 5)
    hs = 0.18 + 0.40 * (np.array(dims) / 56.0) ** 0.75
    cols = [LGREY, SKY, VERM, SKY, LGREY]
    for x, h, lab, c in zip(xs, hs, labs, cols):
        ax.add_patch(Rectangle((x - 0.045, 0.55 - h / 2), 0.09, h, facecolor=c,
                               edgecolor=GREY, lw=1.0, zorder=2))
        ax.text(x, 0.55, lab, ha="center", va="center", fontsize=8.5,
                weight="bold", color="white" if c == VERM else INK, zorder=3)
    for i in range(4):
        arrow(ax, xs[i] + 0.047, 0.55, xs[i + 1] - 0.047, 0.55, lw=1.0)
    ax.text(0.48, 0.93, "trained on BENIGN flows only", ha="center", fontsize=9,
            color=BLUE, weight="bold")
    ax.text(0.48, 0.16,
            r"anomaly score  $s(x)=\frac{1}{56}\sum_{j}\left(x_j-\hat{x}_j\right)^2$",
            ha="center", fontsize=9.5)
    ax.text(0.48, 0.05, "flag as anomaly if  $s(x) > \\tau$", ha="center",
            fontsize=9)

    # conceptual score distribution
    ax2 = fig.add_axes([0.63, 0.17, 0.35, 0.70])
    x = np.linspace(0, 10, 600)
    benign = np.exp(-0.5 * ((x - 1.6) / 0.55) ** 2)
    attack = 0.55 * np.exp(-0.5 * ((x - 5.2) / 1.5) ** 2) + \
             0.45 * np.exp(-0.5 * ((x - 2.2) / 0.8) ** 2)
    ax2.fill_between(x, benign, color=BLUE, alpha=0.35, lw=0)
    ax2.plot(x, benign, color=BLUE, lw=1.6, label="Benign")
    ax2.fill_between(x, attack, color=VERM, alpha=0.30, lw=0)
    ax2.plot(x, attack, color=VERM, lw=1.6, label="Attack")
    ax2.axvline(3.0, color=INK, lw=1.4, ls="--")
    ax2.text(3.12, 0.97, "threshold $\\tau$", fontsize=8, color=INK)
    ax2.text(6.2, 0.35, "detected", fontsize=8, color=VERM)
    ax2.text(0.15, 0.55, "missed\n(inside benign\nmanifold)", fontsize=7.5,
             color=GREY, ha="left")
    ax2.set_xlabel("reconstruction error  $s(x)$", fontsize=8.5)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylim(0, 1.15)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.set_title("overlap = the precision limit of the detector", fontsize=8,
                  color=GREY, pad=4)
    save(fig, "fig_anomaly_ae_schematic.png")


# ----------------------------------------------------------------------------
# 8. Isolation Forest mechanism (real sklearn demo)
# ----------------------------------------------------------------------------
def fig_isolation_forest_demo():
    from sklearn.ensemble import IsolationForest
    rng = np.random.default_rng(3)
    normal = rng.normal(0, 1.0, size=(400, 2))
    outliers = rng.uniform(-4.5, 4.5, size=(14, 2))
    X = np.vstack([normal, outliers])

    clf = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=0).fit(X)
    gx, gy = np.meshgrid(np.linspace(-5, 5, 220), np.linspace(-5, 5, 220))
    Z = clf.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)

    fig = plt.figure(figsize=(7.4, 3.2))
    ax1 = fig.add_axes([0.05, 0.12, 0.40, 0.78])
    cs = ax1.contourf(gx, gy, Z, levels=12, cmap="Blues_r", alpha=0.85)
    ax1.contour(gx, gy, Z, levels=[0], colors=[INK], linewidths=1.2,
                linestyles="--")
    ax1.scatter(normal[:, 0], normal[:, 1], s=7, color=INK, alpha=0.5,
                label="normal (benign-like)")
    ax1.scatter(outliers[:, 0], outliers[:, 1], s=34, color=VERM,
                edgecolor="white", lw=0.6, label="anomaly")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("(a) anomaly score surface; dashed line = decision boundary",
                  fontsize=8.5, color=GREY, pad=5)
    ax1.legend(frameon=False, fontsize=7.5, loc="lower left")

    # (b) path-length intuition
    ax2 = fig.add_axes([0.53, 0.12, 0.44, 0.78])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    pts = rng.normal(0.45, 0.10, size=(60, 2)) * [0.5, 0.5] + [0.12, 0.30]
    ax2.scatter(pts[:, 0], pts[:, 1], s=7, color=INK, alpha=0.45)
    ax2.scatter([0.72], [0.78], s=40, color=VERM, edgecolor="white", lw=0.6)
    for xv in (0.30, 0.40, 0.46):
        ax2.plot([xv, xv], [0.22, 0.68], color=BLUE, lw=0.8, alpha=0.8)
    for yv in (0.40, 0.52):
        ax2.plot([0.10, 0.62], [yv, yv], color=BLUE, lw=0.8, alpha=0.8)
    ax2.plot([0.62, 0.62], [0.16, 0.95], color=VERM, lw=1.2)
    ax2.plot([0.55, 0.95], [0.70, 0.70], color=VERM, lw=1.2)
    ax2.text(0.36, 0.10,
             "dense region:\nmany random splits\nneeded to isolate a point\n"
             "$\\Rightarrow$ long path, normal",
             ha="center", fontsize=7.8, color=BLUE)
    ax2.text(0.80, 0.90, "2 splits isolate it\n$\\Rightarrow$ short path, anomaly",
             ha="center", fontsize=7.8, color=VERM)
    ax2.text(0.5, 0.005,
             r"$s(x)=2^{-\,\mathbb{E}[h(x)]/c(n)}$   ($h(x)$ = isolation path length)",
             ha="center", fontsize=8.5)
    ax2.set_title("(b) why anomalies isolate faster", fontsize=8.5, color=GREY)
    save(fig, "fig_isolation_forest_demo.png")


# ----------------------------------------------------------------------------
# 9. Confusion matrix definition
# ----------------------------------------------------------------------------
def fig_confusion_matrix_schematic():
    fig = plt.figure(figsize=(7.4, 3.1))

    ax = fig.add_axes([0.03, 0.05, 0.44, 0.86])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    cells = [((0.28, 0.52), "TP\nattack correctly\nflagged", "#D9EEE4", GREEN),
             ((0.62, 0.52), "FN\nattack missed\n(breach)", "#FBE0D3", VERM),
             ((0.28, 0.16), "FP\nbenign wrongly\nflagged (noise)", "#FBE0D3", VERM),
             ((0.62, 0.16), "TN\nbenign correctly\npassed", "#D9EEE4", GREEN)]
    for (x, y), lab, fc, ec in cells:
        box(ax, x, y, 0.33, 0.33, lab, fc=fc, ec=ec, fs=8)
    ax.text(0.28 + 0.165, 0.895, "predicted\nattack", ha="center", fontsize=8.5,
            weight="bold", linespacing=1.2)
    ax.text(0.62 + 0.165, 0.895, "predicted\nbenign", ha="center", fontsize=8.5,
            weight="bold", linespacing=1.2)
    ax.text(0.24, 0.685, "actual\nattack", ha="right", va="center", fontsize=8.5,
            weight="bold")
    ax.text(0.24, 0.325, "actual\nbenign", ha="right", va="center", fontsize=8.5,
            weight="bold")
    ax.set_title("(a) binary confusion matrix", fontsize=9, color=GREY, pad=14)

    ax2 = fig.add_axes([0.55, 0.09, 0.42, 0.80])
    m = np.array([[0.93, 0.05, 0.02],
                  [0.04, 0.94, 0.02],
                  [0.30, 0.06, 0.64]])
    ax2.imshow(m, cmap="Blues", vmin=0, vmax=1)
    labs = ["Benign", "DoS-Hulk", "Infiltration"]
    ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
    ax2.set_xticklabels(labs, fontsize=7.5, rotation=15, ha="right")
    ax2.set_yticklabels(labs, fontsize=7.5)
    ax2.set_xlabel("Predicted", fontsize=8.5)
    ax2.set_ylabel("Actual", fontsize=8.5)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if m[i, j] > 0.5 else INK)
    ax2.add_patch(Rectangle((-0.5, 1.5), 1, 1, fill=False, edgecolor=VERM,
                            lw=1.8))
    ax2.set_title("(b) row-normalised multi-class form (illustrative)",
                  fontsize=8.5, color=GREY, pad=6)
    fig.text(0.76, -0.13,
             "orange box: mass leaking into the Benign column = the attacks the IDS misses",
             ha="center", fontsize=7.5, color=VERM)
    save(fig, "fig_confusion_matrix_schematic.png")


# ----------------------------------------------------------------------------
# 10. SHAP + alert-generation pipeline
# ----------------------------------------------------------------------------
def fig_shap_alert_pipeline():
    fig = plt.figure(figsize=(7.6, 3.6))
    ax = blank_ax(fig)

    box(ax, 0.01, 0.62, 0.155, 0.24, "flow\n(56 hybrid\nfeatures)", fs=8.2)
    box(ax, 0.20, 0.62, 0.165, 0.24, "classifier\n$\\rightarrow$ class +\nconfidence",
        fc="#EAF3F9", ec=BLUE, fs=8.2)
    box(ax, 0.40, 0.62, 0.165, 0.24, "anomaly AE\n$\\rightarrow$ corroboration",
        fc="#FDF1E3", ec=ORANGE, fs=8.2)
    box(ax, 0.60, 0.62, 0.175, 0.24,
        "SHAP layer\n$\\phi_j$ per feature", fc="#E9F6F1", ec=GREEN, fs=8.2)
    box(ax, 0.82, 0.62, 0.17, 0.24, "alert generator\n+ severity", fc="white",
        ec=INK, fs=8.2, lw=1.6)
    for x1, x2 in ((0.165, 0.20), (0.365, 0.40), (0.565, 0.60), (0.775, 0.82)):
        arrow(ax, x1, 0.74, x2, 0.74)

    ax.text(0.50, 0.525,
            r"$g(x') = \phi_0 + \sum_{j=1}^{56}\phi_j x'_j$   —   additive attribution:"
            " every alert's score is decomposed into per-feature contributions",
            ha="center", fontsize=8.5)

    # example alert card
    box(ax, 0.06, 0.05, 0.40, 0.40, "", fc="#FAFAFA", ec=GREY)
    ax.text(0.08, 0.395, "ALERT  #12,699", fontsize=8.5, weight="bold")
    ax.text(0.08, 0.335, "class:  DDOS attack-HOIC     conf. 99.8%", fontsize=8)
    ax.text(0.08, 0.285, "severity:  HIGH        anomaly AE:  agrees", fontsize=8)
    ax.text(0.08, 0.235, "top SHAP drivers:", fontsize=8, style="italic")
    bars = [("Init Fwd Win Byts", 0.62), ("Latent_3", 0.41), ("Fwd Pkts/s", 0.28)]
    for i, (name, v) in enumerate(bars):
        y = 0.180 - i * 0.043
        ax.text(0.10, y, name, fontsize=7.4, va="center")
        ax.add_patch(Rectangle((0.25, y - 0.010), v * 0.19, 0.020,
                               facecolor=GREEN, edgecolor="none", zorder=5))
        ax.text(0.25 + v * 0.19 + 0.008, y, f"{v:.2f}", fontsize=7,
                va="center", color=GREY, zorder=5)
    ax.text(0.26, 0.475, "(persisted to alerts.csv / sample_alerts.json)",
            fontsize=7.4, color=GREY, ha="center")

    box(ax, 0.53, 0.05, 0.44, 0.40, "", fc="#FAFAFA", ec=GREY)
    ax.text(0.55, 0.395, "WHY THIS MATTERS TO AN ANALYST", fontsize=8.5,
            weight="bold")
    for i, t in enumerate([
        "1.  triage by severity, not by arrival order",
        "2.  the evidence is named, so the alert is checkable",
        "3.  latent drivers reveal behaviour that no single",
        "     hand-crafted statistic exposes (e.g. Infiltration)",
        "4.  a second, independent opinion (AE) accompanies each alert",
    ]):
        ax.text(0.55, 0.335 - i * 0.055, t, fontsize=7.6)
    save(fig, "fig_shap_alert_pipeline.png")


# ----------------------------------------------------------------------------
# 11. Live lab validation topology
# ----------------------------------------------------------------------------
def fig_lab_topology():
    fig = plt.figure(figsize=(7.6, 3.3))
    ax = blank_ax(fig)

    box(ax, 0.02, 0.58, 0.19, 0.28,
        "ATTACKER VM (Kali)\n192.168.64.3\nnmap · hydra · slowloris\n· slowhttptest · hping3",
        fc="#FBE0D3", ec=VERM, fs=7.8)
    box(ax, 0.30, 0.58, 0.19, 0.28,
        "TARGET VM (Ubuntu)\n192.168.64.2\nself-owned; SSH/FTP/HTTP\nservices exposed",
        fc="#EAF3F9", ec=BLUE, fs=7.8)
    arrow(ax, 0.21, 0.72, 0.30, 0.72, color=VERM, lw=1.6)
    ax.text(0.255, 0.755, "attack\ntraffic", ha="center", fontsize=7.4,
            color=VERM)
    ax.add_patch(FancyBboxPatch((0.005, 0.51), 0.50, 0.42,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fill=False, edgecolor=GREY, ls="--", lw=1.0))
    ax.text(0.255, 0.955, "ISOLATED, SELF-OWNED LAB NETWORK (no external hosts)",
            ha="center", fontsize=7.8, color=GREY, weight="bold")

    box(ax, 0.56, 0.58, 0.16, 0.28,
        "tcpdump capture\nbenign_all.pcap\nattack_all.pcap", fs=7.8)
    arrow(ax, 0.49, 0.72, 0.56, 0.72)

    box(ax, 0.79, 0.58, 0.20, 0.28,
        "x86 bridge host\nofficial Java\nCICFlowMeter\n(same extractor as 2018)",
        fc="#E9F6F1", ec=GREEN, fs=7.8)
    arrow(ax, 0.72, 0.72, 0.79, 0.72)

    box(ax, 0.06, 0.12, 0.20, 0.26,
        "flow CSVs\n9,314 benign\n116,776 attack\n(7 attack types)", fs=7.8)
    box(ax, 0.33, 0.12, 0.20, 0.26,
        "label by\nattack_manifest.csv\n(time window +\nsrc/dst)", fs=7.8)
    box(ax, 0.60, 0.12, 0.17, 0.26,
        "score with\n2018 pipeline\n(§4.5 finding 1)", fc="#FBE0D3", ec=VERM,
        fs=7.8)
    box(ax, 0.81, 0.12, 0.18, 0.26,
        "lab-native\nXGBoost detector\n(§4.5 finding 3)", fc="#E9F6F1",
        ec=GREEN, fs=7.8)
    arrow(ax, 0.89, 0.575, 0.89, 0.50, color=GREY)
    ax.plot([0.16, 0.89], [0.50, 0.50], color=GREY, lw=1.2)
    arrow(ax, 0.16, 0.50, 0.16, 0.385, color=GREY)
    arrow(ax, 0.26, 0.25, 0.33, 0.25)
    arrow(ax, 0.53, 0.25, 0.60, 0.25)
    arrow(ax, 0.77, 0.25, 0.81, 0.25)
    save(fig, "fig_lab_topology.png")


if __name__ == "__main__":
    fig_ae_architecture()
    fig_mi_concept()
    fig_hybrid_feature_space()
    fig_mlp_architecture()
    fig_rf_schematic()
    fig_xgb_schematic()
    fig_anomaly_ae_schematic()
    fig_isolation_forest_demo()
    fig_confusion_matrix_schematic()
    fig_shap_alert_pipeline()
    fig_lab_topology()
    print("done")


# ----------------------------------------------------------------------------
# 12. System architecture / pipeline flowchart (Figure 3.1)
# ----------------------------------------------------------------------------
def fig_pipeline_flowchart():
    """The end-to-end pipeline as actually implemented.

    Replaces the early draft flowchart: five models rather than two, SHAP only
    (no LIME), an explicit validation branch, and the alert store + live
    dashboard as the terminal stage.
    """
    fig = plt.figure(figsize=(8.4, 11.6))
    ax = blank_ax(fig)

    DATA, FEAT, MODEL = "#EAF3F9", "#FDF1E3", "#E9F6F1"
    OUT, SPLIT = "#F3F0F8", "#FDECEA"

    def stage(n, x, y):
        ax.text(x, y, n, fontsize=8, color="white", ha="center", va="center",
                zorder=6,
                bbox=dict(boxstyle="circle,pad=0.28", fc=GREY, ec="none"))

    # 1 — data source
    box(ax, 0.10, 0.945, 0.80, 0.040,
        "CSE-CIC-IDS2018  ·  ten flow CSVs  ·  16,233,002 rows × 84 columns",
        fc=DATA, ec=BLUE, fs=9, weight="bold")
    stage("1", 0.065, 0.965)

    # 2 — preprocessing
    box(ax, 0.10, 0.856, 0.80, 0.062,
        "DATA PRE-PROCESSING\n"
        "remove 59 corrupted header rows · drop 6 identifier columns · coerce to numeric,\n"
        "Infinity → NaN · label-encode the 15 classes   →   16,232,943 rows × 77 features",
        fc=DATA, ec=BLUE, fs=8.2)
    stage("2", 0.065, 0.887)
    arrow(ax, 0.5, 0.943, 0.5, 0.920, color=BLUE)

    # 3 — the split, drawn as the leakage boundary
    box(ax, 0.10, 0.788, 0.80, 0.036,
        "STRATIFIED SPLIT  70 / 15 / 15   —   performed BEFORE any statistic is fitted",
        fc=SPLIT, ec=VERM, fs=8.6, weight="bold")
    stage("3", 0.065, 0.806)
    arrow(ax, 0.5, 0.854, 0.5, 0.826, color=BLUE)
    ax.plot([0.06, 0.94], [0.776, 0.776], color=VERM, lw=1.1, ls="--")
    ax.text(0.935, 0.780, "leakage boundary", fontsize=7.2, color=VERM,
            ha="right", style="italic")

    cols = [(0.10, 0.245, "TRAINING\n11,363,060 flows", BLUE),
            (0.375, 0.245, "VALIDATION\n2,434,941 flows", ORANGE),
            (0.655, 0.245, "TEST\n2,434,942 flows", GREEN)]
    for x, w, lab, c in cols:
        box(ax, x, 0.712, w, 0.044, lab, fc="white", ec=c, fs=8.2, weight="bold", lw=1.5)
        arrow(ax, x + w / 2, 0.786, x + w / 2, 0.758, color=c)

    # 4 — transforms fitted on training data only
    ax.add_patch(FancyBboxPatch((0.075, 0.572), 0.85, 0.128,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                fill=False, edgecolor=GREY, ls="--", lw=1.0))
    ax.text(0.5, 0.690, "FITTED ON THE TRAINING SPLIT ONLY, THEN APPLIED UNCHANGED TO ALL THREE",
            fontsize=7.8, color=GREY, ha="center", weight="bold")
    ax.text(0.5, 0.678, "(Mutual Information and the encoder use the PRE-SMOTE training rows, "
            "so no synthetic flow influences them)",
            fontsize=6.9, color=GREY, ha="center", style="italic")
    fits = [(0.095, 0.255, "Median imputer\n+ Min-Max scaler", DATA, BLUE),
            (0.372, 0.255, "Mutual Information\nranking → top 40", FEAT, ORANGE),
            (0.650, 0.255, "Autoencoder encoder\n77 → 16 latent", FEAT, ORANGE)]
    for x, w, lab, fc, ec in fits:
        box(ax, x, 0.594, w, 0.062, lab, fc=fc, ec=ec, fs=8.2)
    arrow(ax, 0.222, 0.710, 0.222, 0.658, color=BLUE)
    stage("4", 0.048, 0.625)

    # 5 — hybrid feature space
    box(ax, 0.235, 0.492, 0.53, 0.046,
        "HYBRID FEATURE SPACE\n40 selected features  +  16 latent dimensions  =  56 per flow",
        fc="white", ec=INK, fs=8.6, weight="bold", lw=1.6)
    for x in (0.222, 0.4995, 0.7775):
        ax.plot([x, x], [0.592, 0.562], color=BLUE if x < 0.3 else ORANGE, lw=1.0)
    ax.plot([0.222, 0.7775], [0.562, 0.562], color=GREY, lw=1.0)
    arrow(ax, 0.5, 0.562, 0.5, 0.540, color=GREY)
    stage("5", 0.19, 0.515)

    # 6 — imbalance handling (training partition only)
    box(ax, 0.10, 0.410, 0.36, 0.052,
        "CLASS-IMBALANCE HANDLING\n(training partition only)\n"
        "capped SMOTE k=3 → 18,412,356 rows\n+ balanced class weights 0.13 … 402",
        fc=SPLIT, ec=VERM, fs=7.8)
    arrow(ax, 0.28, 0.490, 0.28, 0.464, color=VERM)
    stage("6", 0.065, 0.436)

    # 7 — models
    box(ax, 0.10, 0.300, 0.44, 0.070,
        "SUPERVISED CLASSIFIERS\nRandom Forest  ·  XGBoost  ·  Multi-Layer Perceptron\n"
        "(15-class, trained on the balanced training set)",
        fc=MODEL, ec=GREEN, fs=8.2)
    box(ax, 0.56, 0.300, 0.34, 0.070,
        "UNSUPERVISED DETECTORS\nAutoencoder  ·  Isolation Forest\n"
        "(trained on Benign flows only)",
        fc=MODEL, ec=GREEN, fs=8.2)
    arrow(ax, 0.28, 0.408, 0.28, 0.372, color=VERM)
    arrow(ax, 0.72, 0.490, 0.72, 0.372, color=GREY, lw=1.0)
    stage("7", 0.065, 0.335)

    # 8 — validation-set tuning, and the single read of the test split
    box(ax, 0.10, 0.208, 0.36, 0.056,
        "VALIDATION-SET TUNING\nper-class decision thresholds ·\nensemble weight · anomaly threshold",
        fc="white", ec=ORANGE, fs=8.0)
    box(ax, 0.54, 0.208, 0.36, 0.056,
        "HELD-OUT TEST — READ ONCE\nmacro-F1 · per-class F1 + 95% CI ·\nbinary recall / FPR",
        fc="white", ec=GREEN, fs=8.0)
    arrow(ax, 0.4975, 0.712, 0.4975, 0.690, color=ORANGE, lw=0.9)
    ax.plot([0.955, 0.955], [0.734, 0.236], color=GREEN, lw=1.0, ls=":")
    arrow(ax, 0.955, 0.236, 0.902, 0.236, color=GREEN, lw=1.0)
    ax.plot([0.035, 0.035], [0.734, 0.236], color=ORANGE, lw=1.0, ls=":")
    arrow(ax, 0.035, 0.236, 0.098, 0.236, color=ORANGE, lw=1.0)
    arrow(ax, 0.32, 0.298, 0.32, 0.266, color=GREY)
    arrow(ax, 0.70, 0.298, 0.70, 0.266, color=GREY)
    arrow(ax, 0.46, 0.236, 0.54, 0.236, color=GREY)
    ax.text(0.50, 0.246, "tuned\nthresholds", fontsize=6.8, color=GREY, ha="center")
    stage("8", 0.065, 0.246)

    # 9 — explainability and response
    box(ax, 0.10, 0.116, 0.24, 0.048, "SHAP\nper-flow attribution", fc=OUT, ec=PURPLE, fs=8.2)
    box(ax, 0.38, 0.116, 0.26, 0.048,
        "ALERT GENERATOR\nclass · severity · top features", fc=OUT, ec=PURPLE, fs=8.2)
    box(ax, 0.68, 0.116, 0.22, 0.048,
        "ALERT STORE\nalerts.csv · JSON", fc=OUT, ec=PURPLE, fs=8.2)
    arrow(ax, 0.22, 0.206, 0.22, 0.164, color=PURPLE)
    arrow(ax, 0.34, 0.140, 0.38, 0.140, color=PURPLE)
    arrow(ax, 0.64, 0.140, 0.68, 0.140, color=PURPLE)
    stage("9", 0.065, 0.140)

    # 10 — deployment path
    ax.add_patch(FancyBboxPatch((0.075, 0.010), 0.85, 0.072,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                fill=False, edgecolor=SKY, ls="--", lw=1.2))
    ax.text(0.30, 0.070, "LIVE DEPLOYMENT  (§4.6)", fontsize=8, color=SKY,
            ha="center", weight="bold")
    dep = [(0.095, 0.155, "packet capture\n(PCAP)"),
           (0.275, 0.165, "CICFlowMeter\n→ flow CSV"),
           (0.465, 0.185, "same saved\ntransforms"),
           (0.675, 0.235, "XGBoost + Autoencoder\n(+ lab detector)")]
    for x, w, lab in dep:
        box(ax, x, 0.020, w, 0.040, lab, fc="white", ec=SKY, fs=7.6)
    for x1, x2 in ((0.250, 0.275), (0.440, 0.465), (0.650, 0.675)):
        arrow(ax, x1, 0.040, x2, 0.040, color=SKY, lw=1.0)
    ax.annotate("", xy=(0.855, 0.114), xytext=(0.855, 0.084),
                arrowprops=dict(arrowstyle="-|>", color=SKY, lw=1.2))
    ax.text(0.845, 0.096, "live dashboard", fontsize=7.2, color=SKY, ha="right")
    stage("10", 0.042, 0.040)

    save(fig, "fig_pipeline_flowchart.png")

# ============================================================
#  SML Project — Part D: Dimensionality Reduction
# ============================================================

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model  import LogisticRegression


# ── Palette (matches Part C) ───────────────────────────────────
PALETTE = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "accent1": "#58a6ff",   # blue
    "accent2": "#3fb950",   # green
    "accent3": "#f78166",   # red-orange
    "accent4": "#d2a8ff",   # purple
    "accent5": "#ffa657",   # orange
}

EMB_COLORS = {
    "DistilBERT CLS":  PALETTE["accent1"],
    "DistilBERT Mean": PALETTE["accent2"],
    "BERT CLS":        PALETTE["accent3"],
    "BERT Mean":       PALETTE["accent4"],
}


def _style(fig, axes):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["muted"])
        ax.yaxis.label.set_color(PALETTE["muted"])
        ax.title.set_color(PALETTE["text"])
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(PALETTE["border"])
        ax.grid(color=PALETTE["border"], linewidth=0.5,
                linestyle="--", alpha=0.5)


# ═══════════════════════════════════════════════════════════════
# PLOT 1 — SVD: Cumulative Variance Curve + Accuracy vs Components
# ═══════════════════════════════════════════════════════════════

def plot_svd_panel(dim_results, save_path="part_d_svd.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _style(fig, axes)

    # ── Left: explained variance ──
    ax = axes[0]
    curve = dim_results["svd_curve"]
    xs = np.arange(1, len(curve) + 1)
    ax.plot(xs, curve, color=PALETTE["accent1"], linewidth=2, zorder=3)
    ax.fill_between(xs, curve, alpha=0.12, color=PALETTE["accent1"])

    for thresh, nc_key, color in [
        (0.80, "svd_n80", PALETTE["accent5"]),
        (0.90, "svd_n90", PALETTE["accent3"]),
    ]:
        nc = dim_results[nc_key]
        ax.axhline(thresh, color=color, linewidth=1, linestyle="--", alpha=0.8)
        ax.axvline(nc,     color=color, linewidth=1, linestyle="--", alpha=0.8)
        ax.annotate(f"{int(thresh*100)}% @ {nc}",
                    xy=(nc, thresh),
                    xytext=(nc + 8, thresh - 0.06),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_xlim(1, len(curve))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of SVD Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("TF-IDF — TruncatedSVD Variance Curve", fontweight="bold")

    # ── Right: accuracy vs n_components ──
    ax2 = axes[1]
    acc = dim_results["svd_acc"]
    xs2 = sorted(acc.keys())
    ys2 = [acc[x] for x in xs2]

    ax2.plot(xs2, ys2, color=PALETTE["accent2"], linewidth=2,
             marker="o", markersize=7, zorder=3)
    ax2.fill_between(xs2, ys2, min(ys2) - 0.005,
                     alpha=0.12, color=PALETTE["accent2"])

    for x, y in zip(xs2, ys2):
        ax2.annotate(f"{y:.3f}", (x, y),
                     textcoords="offset points", xytext=(0, 9),
                     fontsize=8, color=PALETTE["text"], ha="center")

    ax2.set_xlabel("Number of SVD Components")
    ax2.set_ylabel("5-Fold CV Accuracy (LR)")
    ax2.set_title("Accuracy vs SVD Components (TF-IDF)", fontweight="bold")
    ax2.set_ylim(min(ys2) - 0.02, max(ys2) + 0.04)

    fig.suptitle("TruncatedSVD on TF-IDF",
                 fontsize=14, fontweight="bold",
                 color=PALETTE["text"], y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# PLOT 2 — All PCA Variance Curves (TF-IDF dense + embeddings)
# ═══════════════════════════════════════════════════════════════

def plot_pca_variance_all(dim_results, save_path="part_d_pca_variance.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _style(fig, [ax])

    # TF-IDF PCA
    curve_tf = dim_results["pca_tf_curve"]
    xs_tf = np.arange(1, len(curve_tf) + 1)
    ax.plot(xs_tf, curve_tf, color=PALETTE["accent5"],
            linewidth=2, linestyle="-", label="TF-IDF (PCA)", zorder=4)

    # Embedding PCAs
    for emb_name, color in EMB_COLORS.items():
        info = dim_results["pca_emb"][emb_name]
        curve = info["curve"]
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, color=color, linewidth=1.8,
                linestyle="-", label=emb_name, zorder=3)

    # Reference lines
    for thresh, ls in [(0.80, "--"), (0.90, ":")]:
        ax.axhline(thresh, color=PALETTE["muted"],
                   linewidth=0.9, linestyle=ls, alpha=0.6)
        ax.text(2, thresh + 0.012,
                f"{int(thresh*100)}% variance",
                fontsize=8, color=PALETTE["muted"])

    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("PCA Variance Curves — All Feature Spaces",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.15, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"],
              fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# PLOT 3 — Embedding PCA Accuracy vs Components (multi-line)
# ═══════════════════════════════════════════════════════════════

def plot_emb_pca_accuracy(dim_results, save_path="part_d_emb_accuracy.png"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style(fig, [ax])

    for emb_name, color in EMB_COLORS.items():
        acc_map = dim_results["pca_emb"][emb_name]["acc"]
        xs = sorted(acc_map.keys())
        ys = [acc_map[x] for x in xs]

        ax.plot(xs, ys, color=color, linewidth=2,
                marker="o", markersize=6, label=emb_name, zorder=3)
        ax.fill_between(xs, ys, min(ys) - 0.005,
                        alpha=0.07, color=color)

        # annotate last point
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=7.5, color=color, va="center")

    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("5-Fold CV Accuracy (LR)")
    ax.set_title("Accuracy vs PCA Components — Embeddings",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.15, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"],
              fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# PLOT 4 — Components needed for 80% / 90% variance (bar chart)
# ═══════════════════════════════════════════════════════════════

def plot_compression_summary(dim_results, save_path="part_d_compression.png"):
    labels = ["TF-IDF\n(SVD)", "TF-IDF\n(PCA)",
              "DistilBERT\nCLS", "DistilBERT\nMean",
              "BERT\nCLS", "BERT\nMean"]

    n80 = [
        dim_results["svd_n80"],
        dim_results["pca_tf_n80"],
        dim_results["pca_emb"]["DistilBERT CLS"]["n80"],
        dim_results["pca_emb"]["DistilBERT Mean"]["n80"],
        dim_results["pca_emb"]["BERT CLS"]["n80"],
        dim_results["pca_emb"]["BERT Mean"]["n80"],
    ]
    n90 = [
        dim_results["svd_n90"],
        dim_results["pca_tf_n90"],
        dim_results["pca_emb"]["DistilBERT CLS"]["n90"],
        dim_results["pca_emb"]["DistilBERT Mean"]["n90"],
        dim_results["pca_emb"]["BERT CLS"]["n90"],
        dim_results["pca_emb"]["BERT Mean"]["n90"],
    ]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style(fig, [ax])

    b1 = ax.bar(x - w / 2, n80, width=w, color=PALETTE["accent5"],
                label="80% Variance", edgecolor=PALETTE["border"],
                linewidth=0.6)
    b2 = ax.bar(x + w / 2, n90, width=w, color=PALETTE["accent3"],
                label="90% Variance", edgecolor=PALETTE["border"],
                linewidth=0.6)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                str(int(h)), ha="center", va="bottom",
                fontsize=8.5, color=PALETTE["text"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=PALETTE["text"])
    ax.set_ylabel("Components Required")
    ax.set_title("Components Needed for 80% / 90% Variance",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.15, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"],
              fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN plot dispatcher
# ═══════════════════════════════════════════════════════════════

def plot_part_d(dim_results):
    print("\n  Generating plots...")
    plot_svd_panel(dim_results,         save_path="part_d_svd.png")
    plot_pca_variance_all(dim_results,  save_path="part_d_pca_variance.png")
    plot_emb_pca_accuracy(dim_results,  save_path="part_d_emb_accuracy.png")
    plot_compression_summary(dim_results, save_path="part_d_compression.png")
    print("  All plots saved.")


# ═══════════════════════════════════════════════════════════════
# EVALUATE helper (same as Part C)
# ═══════════════════════════════════════════════════════════════

def evaluate_classifier(name, clf, X, y, cv=5, sparse_ok=True):
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if sparse_ok:
        pipeline = clf
    else:
        pipeline = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("clf", clf),
        ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X, y,
        cv=skf,
        scoring="accuracy",
        n_jobs=1
    )
    return scores.mean(), scores.std()


# ═══════════════════════════════════════════════════════════════
# PART D CORE (unchanged logic)
# ═══════════════════════════════════════════════════════════════

def part_d(X_tfidf, embs, y):

    print("\n" + "═" * 60)
    print("PART D — Dimensionality Reduction")
    print("═" * 60)

    results = {}

    # ── (1) Truncated SVD on TF-IDF ───────────────────────────
    print("\n  TruncatedSVD on TF-IDF…")

    n_svd = 300
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    cum_svd = np.cumsum(svd.explained_variance_ratio_)

    n80_svd = int(np.searchsorted(cum_svd, 0.80)) + 1
    n90_svd = int(np.searchsorted(cum_svd, 0.90)) + 1

    print(f"    80% variance @ {n80_svd} components")
    print(f"    90% variance @ {n90_svd} components")

    results["svd_curve"] = cum_svd
    results["svd_n80"]   = n80_svd
    results["svd_n90"]   = n90_svd

    svd_acc = {}
    for nc in [50, 100, n80_svd, n90_svd, 200, 300]:
        nc = min(nc, n_svd)
        svd_nc = TruncatedSVD(n_components=nc, random_state=42)
        X_r = svd_nc.fit_transform(X_tfidf)
        acc, _ = evaluate_classifier(
            "LR",
            LogisticRegression(max_iter=500, random_state=42),
            X_r, y, sparse_ok=True
        )
        svd_acc[nc] = acc
        print(f"    SVD n={nc:<4d} → acc = {acc:.4f}")

    results["svd_acc"] = svd_acc

    # ── (2) PCA on TF-IDF (dense) ─────────────────────────────
    print("\n  PCA on TF-IDF (dense)…")

    X_dense    = X_tfidf.toarray().astype(np.float32)
    n_pca_tf   = min(300, X_dense.shape[1], X_dense.shape[0])
    pca_tf     = PCA(n_components=n_pca_tf, random_state=42)
    X_pca_tf   = pca_tf.fit_transform(X_dense)
    cum_pca_tf = np.cumsum(pca_tf.explained_variance_ratio_)

    n80_pca_tf = int(np.searchsorted(cum_pca_tf, 0.80)) + 1
    n90_pca_tf = int(np.searchsorted(cum_pca_tf, 0.90)) + 1

    print(f"    80% variance @ {n80_pca_tf} components")
    print(f"    90% variance @ {n90_pca_tf} components")

    results["pca_tf_curve"] = cum_pca_tf
    results["pca_tf_n80"]   = n80_pca_tf
    results["pca_tf_n90"]   = n90_pca_tf

    # ── (3) PCA on embeddings ─────────────────────────────────
    pca_emb_results = {}

    for emb_name, X_emb in [
        ("DistilBERT CLS",  embs["distilbert_cls"]),
        ("DistilBERT Mean", embs["distilbert_mean"]),
        ("BERT CLS",        embs["bert_cls"]),
        ("BERT Mean",       embs["bert_mean"]),
    ]:
        print(f"\n  PCA on {emb_name} (dim={X_emb.shape[1]})…")

        n_pca = min(200, X_emb.shape[1], X_emb.shape[0])
        pca   = PCA(n_components=n_pca, random_state=42)
        X_r   = pca.fit_transform(X_emb.astype(np.float32))
        cum   = np.cumsum(pca.explained_variance_ratio_)

        n80 = int(np.searchsorted(cum, 0.80)) + 1
        n90 = int(np.searchsorted(cum, 0.90)) + 1

        print(f"    80% variance @ {n80} components")
        print(f"    90% variance @ {n90} components")

        acc_map = {}
        for nc in [32, 64, n80, n90, 128]:
            nc   = min(nc, n_pca)
            X_nc = X_r[:, :nc]
            acc, _ = evaluate_classifier(
                "LR",
                LogisticRegression(max_iter=500, random_state=42),
                X_nc, y, sparse_ok=True
            )
            acc_map[nc] = acc
            print(f"    PCA n={nc:<4d} → acc = {acc:.4f}")

        pca_emb_results[emb_name] = {
            "curve": cum,
            "n80":   n80,
            "n90":   n90,
            "acc":   acc_map,
        }

    results["pca_emb"] = pca_emb_results
    return results




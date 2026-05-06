# ============================================================
#  SML Project — Part C: Models
# ============================================================

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm          import LinearSVC, SVC
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.cluster      import KMeans
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics      import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


# ── Palette ────────────────────────────────────────────────────
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

MODEL_COLORS = {
    "Logistic Regression": PALETTE["accent1"],
    "Linear SVM":          PALETTE["accent2"],
    "RBF SVM":             PALETTE["accent3"],
    "KNN (k=5)":           PALETTE["accent4"],
}

FS_ORDER = [
    "BoW (Unigram)",
    "TF-IDF (Bigram)",
    "DistilBERT CLS",
    "DistilBERT Mean-Pool",
    "BERT CLS",
    "BERT Mean-Pool",
]


def _apply_dark_theme(fig, axes_list):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes_list:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["muted"])
        ax.yaxis.label.set_color(PALETTE["muted"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])


# ═══════════════════════════════════════════════════════════════
# PLOT 1 — Grouped Lollipop Chart (Accuracy per Feature Space)
# ═══════════════════════════════════════════════════════════════

def plot_lollipop(df_acc, save_path="part_c_lollipop.png"):
    models = list(MODEL_COLORS.keys())
    n_fs   = len(FS_ORDER)
    n_mod  = len(models)

    fig, ax = plt.subplots(figsize=(13, 6))
    _apply_dark_theme(fig, [ax])

    group_gap  = 1.8
    item_gap   = 0.32
    positions  = []

    for gi, fs in enumerate(FS_ORDER):
        base = gi * (n_mod * item_gap + group_gap)
        for mi, model in enumerate(models):
            x = base + mi * item_gap
            positions.append((fs, model, x))

    for fs, model, x in positions:
        row = df_acc[(df_acc["Feature Space"] == fs) & (df_acc["Model"] == model)]
        if row.empty:
            continue
        score = row["Score"].values[0]
        std   = row["Std"].values[0]
        color = MODEL_COLORS[model]

        ax.vlines(x, 0.5, score, color=color, linewidth=1.6, alpha=0.6, zorder=2)
        ax.plot(x, score, "o", color=color, markersize=8, zorder=3)
        ax.errorbar(x, score, yerr=std, fmt="none",
                    ecolor=color, elinewidth=1.2, capsize=3, alpha=0.7, zorder=4)

    # Group x-tick labels
    group_centers = []
    for gi, fs in enumerate(FS_ORDER):
        xs = [x for (f, m, x) in positions if f == fs]
        group_centers.append(np.mean(xs))

    ax.set_xticks(group_centers)
    ax.set_xticklabels(FS_ORDER, rotation=20, ha="right", fontsize=9,
                       color=PALETTE["text"])
    ax.set_ylabel("5-Fold CV Accuracy", color=PALETTE["muted"])
    ax.set_ylim(0.5, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Supervised Model Accuracy by Feature Space",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], pad=14)
    ax.axhline(0.5, color=PALETTE["border"], linewidth=0.8, linestyle="--")
    ax.grid(axis="y", color=PALETTE["border"], linewidth=0.6, linestyle="--", alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()
    ]
    ax.legend(handles=legend_patches, framealpha=0.15,
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# PLOT 2 — Heatmap (Feature Space × Model)
# ═══════════════════════════════════════════════════════════════

def plot_heatmap(df_acc, save_path="part_c_heatmap.png"):
    pivot = (df_acc.pivot(index="Feature Space", columns="Model", values="Score")
                   .reindex(FS_ORDER))
    pivot = pivot[list(MODEL_COLORS.keys())]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _apply_dark_theme(fig, [ax])

    cmap = LinearSegmentedColormap.from_list(
        "dark_heat", ["#1a1f2e", "#1e3a5f", "#2d6a9f", "#58a6ff", "#b3d9ff"]
    )

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right",
                       fontsize=9, color=PALETTE["text"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color=PALETTE["text"])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                txt_color = PALETTE["bg"] if val > 0.78 else PALETTE["text"]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    cbar.set_label("Accuracy", color=PALETTE["muted"], fontsize=9)
    cbar.outline.set_edgecolor(PALETTE["border"])

    ax.set_title("Accuracy Heatmap — Feature Space × Model",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# PLOT 3 — Radar Chart (per Model across Feature Spaces)
# ═══════════════════════════════════════════════════════════════

def plot_radar(df_acc, save_path="part_c_radar.png"):
    models = list(MODEL_COLORS.keys())
    labels = FS_ORDER
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    ax.spines["polar"].set_color(PALETTE["border"])
    ax.tick_params(colors=PALETTE["muted"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5, color=PALETTE["text"])
    ax.set_ylim(0.5, 1.0)
    yticks = [0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}" for v in yticks],
                       fontsize=7, color=PALETTE["muted"])
    ax.yaxis.grid(color=PALETTE["border"], linestyle="--", linewidth=0.6)
    ax.xaxis.grid(color=PALETTE["border"], linestyle="-", linewidth=0.5)

    for model, color in MODEL_COLORS.items():
        vals = []
        for fs in FS_ORDER:
            row = df_acc[(df_acc["Feature Space"] == fs) & (df_acc["Model"] == model)]
            vals.append(row["Score"].values[0] if not row.empty else 0.5)
        vals += vals[:1]

        ax.plot(angles, vals, color=color, linewidth=2, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.10, zorder=2)
        ax.plot(angles[:-1], vals[:-1], "o", color=color,
                markersize=5, zorder=4)

    legend_patches = [
        mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              bbox_to_anchor=(1.35, 1.15),
              framealpha=0.15, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"],
              fontsize=9)

    ax.set_title("Model Performance Radar\nacross Feature Spaces",
                 fontsize=13, fontweight="bold", color=PALETTE["text"],
                 pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# PLOT 4 — KMeans ARI + Best Supervised Score (side-by-side bars)
# ═══════════════════════════════════════════════════════════════

def plot_kmeans_vs_best(results_df, save_path="part_c_kmeans.png"):
    ari_df  = results_df[results_df["Metric"] == "ARI"].set_index("Feature Space")
    best_df = (results_df[results_df["Metric"] == "Accuracy"]
               .groupby("Feature Space")["Score"].max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _apply_dark_theme(fig, axes)

    y_pos = np.arange(len(FS_ORDER))

    # ── Left: KMeans ARI ──
    ax = axes[0]
    ari_vals = [ari_df.loc[fs, "Score"] if fs in ari_df.index else 0 for fs in FS_ORDER]
    colors = [PALETTE["accent5"] if v >= 0 else PALETTE["accent3"] for v in ari_vals]
    bars = ax.barh(y_pos, ari_vals, color=colors, height=0.55,
                   edgecolor=PALETTE["border"], linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(FS_ORDER, fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Adjusted Rand Index", color=PALETTE["muted"])
    ax.set_title("KMeans Clustering (ARI)", fontsize=12,
                 fontweight="bold", color=PALETTE["text"])
    ax.axvline(0, color=PALETTE["muted"], linewidth=0.8)
    ax.set_xlim(-0.05, max(max(ari_vals) * 1.2, 0.1))
    for bar, val in zip(bars, ari_vals):
        ax.text(max(val + 0.005, 0.005), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.5,
                color=PALETTE["text"], fontweight="bold")

    # ── Right: Best supervised accuracy ──
    ax2 = axes[1]
    best_vals = [best_df.get(fs, 0) for fs in FS_ORDER]
    bar_colors = [PALETTE["accent1"] if "BERT" in fs else PALETTE["accent2"]
                  for fs in FS_ORDER]
    bars2 = ax2.barh(y_pos, best_vals, color=bar_colors, height=0.55,
                     edgecolor=PALETTE["border"], linewidth=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(FS_ORDER, fontsize=9, color=PALETTE["text"])
    ax2.set_xlabel("Best CV Accuracy", color=PALETTE["muted"])
    ax2.set_title("Best Supervised Accuracy per Feature Space",
                  fontsize=12, fontweight="bold", color=PALETTE["text"])
    ax2.set_xlim(0.5, 1.02)
    for bar, val in zip(bars2, best_vals):
        ax2.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=8.5,
                 color=PALETTE["text"], fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=PALETTE["accent1"], label="BERT-based"),
        mpatches.Patch(color=PALETTE["accent2"], label="Sparse (BoW / TF-IDF)"),
    ]
    ax2.legend(handles=legend_patches, framealpha=0.15,
               facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)

    fig.suptitle("Unsupervised vs. Supervised — Feature Space Comparison",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN plot dispatcher
# ═══════════════════════════════════════════════════════════════

def plot_part_c(results_df):
    print("\n  Generating plots...")
    df_acc = results_df[results_df["Metric"] == "Accuracy"].copy()

    plot_lollipop(df_acc,      save_path="part_c_lollipop.png")
    plot_heatmap(df_acc,       save_path="part_c_heatmap.png")
    plot_radar(df_acc,         save_path="part_c_radar.png")
    plot_kmeans_vs_best(results_df, save_path="part_c_kmeans.png")

    print("  All plots saved.")


# ═══════════════════════════════════════════════════════════════
# EVALUATE helpers  (unchanged)
# ═══════════════════════════════════════════════════════════════

def evaluate_classifier(name, clf, X, y, cv=5, sparse_ok=True):
    from sklearn.pipeline import Pipeline

    if sparse_ok:
        pipeline = clf
    else:
        pipeline = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("clf",   clf),
        ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X, y,
        cv=skf,
        scoring="accuracy",
        n_jobs=1)
    return scores.mean(), scores.std()


def evaluate_kmeans(X, y, n_clusters=2, seed=42):
    from sklearn.preprocessing import normalize as _norm

    if sp.issparse(X):
        Xd = _norm(X, norm="l2")
    else:
        Xd = _norm(X.astype(np.float32))

    km = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300
    )
    km.fit(Xd)
    ari = adjusted_rand_score(y, km.labels_)
    return ari


# ═══════════════════════════════════════════════════════════════
# PART C CORE  (unchanged logic)
# ═══════════════════════════════════════════════════════════════

def part_c(X_bow, X_tfidf, embs, y):

    print("\n" + "═" * 60)
    print("PART C — Models")
    print("═" * 60)

    feature_spaces = {
        "BoW (Unigram)"       : X_bow,
        "TF-IDF (Bigram)"     : X_tfidf,
        "DistilBERT CLS"      : embs["distilbert_cls"],
        "DistilBERT Mean-Pool": embs["distilbert_mean"],
        "BERT CLS"            : embs["bert_cls"],
        "BERT Mean-Pool"      : embs["bert_mean"],
    }

    supervised_models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000,
                                                    random_state=42,
                                                    C=1.0), True),
        ("Linear SVM",         LinearSVC(max_iter=2000,
                                          random_state=42, C=1.0), True),
        ("RBF SVM",            SVC(kernel="rbf", C=1.0,
                                    gamma="scale",
                                    random_state=42), False),
        ("KNN (k=5)",          KNeighborsClassifier(n_neighbors=5,
                                                     n_jobs=-1), False),
    ]

    rows = []

    for fs_name, X in feature_spaces.items():
        print(f"\n  Feature: {fs_name}")

        # ── Supervised ─────────────────────────
        for clf_name, clf, sparse_ok in supervised_models:
            mean_acc, std_acc = evaluate_classifier(
                clf_name, clf, X, y, sparse_ok=sparse_ok
            )
            print(f"    {clf_name:<25}  acc = {mean_acc:.4f} ± {std_acc:.4f}")
            rows.append({
                "Feature Space": fs_name,
                "Model": clf_name,
                "Metric": "Accuracy",
                "Score": mean_acc,
                "Std": std_acc,
            })

        # ── KMeans ─────────────────────────────
        ari = evaluate_kmeans(X, y)
        print(f"    {'KMeans (ARI)':<25}  ari = {ari:.4f}")
        rows.append({
            "Feature Space": fs_name,
            "Model": "KMeans",
            "Metric": "ARI",
            "Score": ari,
            "Std": 0.0,
        })

    results_df = pd.DataFrame(rows)
    return results_df




# ============================================================
#  SML Project — Part A: Sparse Features
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

# ── Stopwords ────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set("""
    i me my myself we our ours ourselves you your yours yourself yourselves
    he him his himself she her hers herself it its itself they them their
    theirs themselves what which who whom this that these those am is are was
    were be been being have has had having do does did doing a an the and but
    if or because as until while of at by for with about against between into
    through during before after above below to from up down in out on off over
    under again further then once here there when where why how all both each
    few more most other some such no nor not only own same so than too very
    s t can will just don should now d ll m o re ve y ain aren couldn didn
    doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren
    won wouldn would could br
    """.split())

STOPWORDS.update({"br", "http", "https", "www"})


# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ═══════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    import time

    for attempt in range(3):
        try:
            print("Loading noob123/imdb_review_3000 from HuggingFace…")
            ds = load_dataset("noob123/imdb_review_3000")
            break
        except RuntimeError as e:
            if "client has been closed" in str(e):
                print(f"Retrying ({attempt+1}/3)…")
                time.sleep(2)
                import importlib, httpx
                importlib.reload(httpx)
            else:
                raise

    frames = []
    for split in ds.keys():
        frames.append(pd.DataFrame({
            "review": ds[split]["review"],
            "sentiment": ds[split]["sentiment"],
        }))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    print(f"Loaded — shape: {df.shape}")
    print(df["sentiment"].value_counts().to_string())

    return df


# ═══════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def preprocess(text: str):
    return " ".join(remove_stopwords(tokenize(text)))


# ═══════════════════════════════════════════════════════════════
# 3. FEATURES
# ═══════════════════════════════════════════════════════════════

def build_bow(corpus):
    vec = CountVectorizer(
        ngram_range=(1, 1),
        min_df=3,
        max_df=0.90,
        max_features=5000,
        strip_accents="unicode",
    )
    X = vec.fit_transform(corpus)
    print(f"\nBoW — shape: {X.shape}")
    return X, vec


def build_tfidf(corpus):
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        max_features=10000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vec.fit_transform(corpus)
    print(f"TF-IDF — shape: {X.shape}")
    return X, vec


def run_svd(X):
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd.fit(X)
    return np.cumsum(svd.explained_variance_ratio_)


# ═══════════════════════════════════════════════════════════════
# 4. PLOTS
# ═══════════════════════════════════════════════════════════════
DARK = "#0f1117"
CARD = "#1a1d27"
TEXT = "#e0e0e0"
GRID = "#2a2d3a"
AC   = ["#7c6af7", "#f06292", "#4dd0e1", "#aed581", "#ffb74d"]


def _style(ax, title: str):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)

    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


def make_plots(df, X_bow, bow_vec, X_tfidf, tfidf_vec, cum_var,
               save_path="part_a_sparse_features.png"):

    # ── Statistics ────────────────────────────────────────────
    all_tokens = [t for txt in df["processed"] for t in txt.split()]
    token_freq = Counter(all_tokens)

    raw_len  = df["review"].str.split().str.len()
    proc_len = df["processed"].str.split().str.len()

    def top_tokens(sentiment, n=15):
        toks = [
            t for txt in df[df["sentiment"] == sentiment]["processed"]
            for t in txt.split()
        ]
        return Counter(toks).most_common(n)

    top_pos = top_tokens("positive")
    top_neg = top_tokens("negative")

    tfidf_mean = np.asarray(X_tfidf.mean(axis=0)).flatten()
    top_idx    = tfidf_mean.argsort()[::-1][:20]

    feat_names   = np.array(tfidf_vec.get_feature_names_out())
    top_tf_terms = feat_names[top_idx]
    top_tf_vals  = tfidf_mean[top_idx]

    sp_bow   = 1 - X_bow.nnz   / (X_bow.shape[0]   * X_bow.shape[1])
    sp_tfidf = 1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])

    # ── Layout ────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor(DARK)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

    fig.text(0.5, 0.985,
             "Part A — Sparse Features  |  noob123/imdb_review_3000",
             ha="center", color=TEXT, fontsize=15, fontweight="bold")

    fig.text(0.5, 0.970,
             "columns: review (string) · sentiment (positive / negative)"
             " · 3000 samples · CPU",
             ha="center", color="#888", fontsize=9)

    # ① Class distribution
    ax = fig.add_subplot(gs[0, 0])
    vc = df["sentiment"].value_counts()

    bars = ax.bar(vc.index, vc.values, color=AC[:2],
                  width=0.5, edgecolor=DARK, linewidth=1.5)

    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width()/2, v + 10, str(v),
                ha="center", color=TEXT, fontsize=9, fontweight="bold")

    ax.set_ylim(0, vc.max() * 1.2)
    _style(ax, "① Class Distribution (sentiment)")
    ax.set_ylabel("Count", color=TEXT)

    # ② Token length
    ax = fig.add_subplot(gs[0, 1])

    ax.hist(raw_len, bins=30, alpha=0.65, color=AC[0],
            label="Raw (with HTML)", edgecolor=DARK)

    ax.hist(proc_len, bins=30, alpha=0.65, color=AC[1],
            label="Processed", edgecolor=DARK)

    ax.legend(facecolor=CARD, edgecolor=GRID,
              labelcolor=TEXT, fontsize=8)

    _style(ax, "② Token Length: Raw vs Processed")
    ax.set_xlabel("# Tokens", color=TEXT)
    ax.set_ylabel("Frequency", color=TEXT)

    # ③ Top tokens
    ax = fig.add_subplot(gs[0, 2])
    w15, f15 = zip(*token_freq.most_common(15))

    ax.barh(list(w15)[::-1], list(f15)[::-1],
            color=[AC[i % len(AC)] for i in range(15)][::-1],
            edgecolor=DARK)

    _style(ax, "③ Top 15 Tokens (Overall)")
    ax.set_xlabel("Frequency", color=TEXT)

    # ④ Positive
    ax = fig.add_subplot(gs[1, 0])
    pw, pf = zip(*top_pos)

    ax.barh(list(pw)[::-1], list(pf)[::-1],
            color=AC[0], edgecolor=DARK)

    _style(ax, "④ Top Tokens — Positive Reviews")

    # ⑤ Negative
    ax = fig.add_subplot(gs[1, 1])
    nw, nf = zip(*top_neg)

    ax.barh(list(nw)[::-1], list(nf)[::-1],
            color=AC[1], edgecolor=DARK)

    _style(ax, "⑤ Top Tokens — Negative Reviews")

    # ⑥ Zipf
    ax = fig.add_subplot(gs[1, 2])

    sf = sorted(token_freq.values(), reverse=True)[:300]
    rk = np.arange(1, len(sf) + 1)

    ax.loglog(rk, sf, color=AC[2], linewidth=1.8)
    ax.loglog(rk, sf[0]/rk, "--", color=AC[4], linewidth=1.2)

    _style(ax, "⑥ Zipf's Law")

    # ⑦ TF-IDF
    ax = fig.add_subplot(gs[2, 0])

    ax.barh(top_tf_terms[::-1], top_tf_vals[::-1],
            color=[AC[i % len(AC)] for i in range(20)][::-1],
            edgecolor=DARK)

    _style(ax, "⑦ Top TF-IDF")

    # ⑧ Sparsity
    ax = fig.add_subplot(gs[2, 1])

    sp_lbl = ["BoW", "TF-IDF"]
    sp_val = [sp_bow*100, sp_tfidf*100]

    ax.bar(sp_lbl, sp_val, color=[AC[3], AC[2]])

    _style(ax, "⑧ Sparsity")

    # ⑩ SVD
    ax = fig.add_subplot(gs[3, 0])

    ax.plot(np.arange(1, len(cum_var)+1), cum_var*100,
            color=AC[0], linewidth=2)

    _style(ax, "⑩ SVD Variance")

    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor=DARK)

    print(f"Plot saved → {save_path}")
    plt.close()


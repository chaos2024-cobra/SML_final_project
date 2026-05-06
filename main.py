# ============================================================
#  SML Project — Main Entry Point
#  Runs Parts A → B → C → D in sequence.
#  Each part_*.py contains only helper/library functions.
# ============================================================

import scipy.sparse as sp
import pandas as pd

from part_a import (
    load_data,
    preprocess,
    build_bow,
    build_tfidf,
    run_svd,
    make_plots,
)
from part_b import part_b
from part_c import part_c, plot_part_c
from part_d import part_d, plot_part_d


# ═══════════════════════════════════════════════════════════════
# PART A — Sparse Features
# ═══════════════════════════════════════════════════════════════

def run_part_a(df):
    print("\n" + "═" * 60)
    print("PART A — Sparse Features")
    print("═" * 60)

    print("\nPreprocessing…")
    df["processed"] = df["review"].apply(preprocess)

    print("\nBuilding features…")
    X_bow,   bow_vec   = build_bow(df["processed"])
    X_tfidf, tfidf_vec = build_tfidf(df["processed"])

    print("\nRunning SVD…")
    cum_var = run_svd(X_tfidf)

    # Persist artefacts needed by later parts
    sp.save_npz("X_bow.npz",   X_bow)
    sp.save_npz("X_tfidf.npz", X_tfidf)
    pd.DataFrame({"label": df["label"]}).to_csv("labels.csv", index=False)

    print("\nGenerating plots…")
    make_plots(
        df, X_bow, bow_vec, X_tfidf, tfidf_vec, cum_var,
        save_path="part_a_sparse_features.png",
    )

    print("\n✓ Part A complete.")
    return X_bow, X_tfidf, bow_vec, tfidf_vec, cum_var


# ═══════════════════════════════════════════════════════════════
# PART C wrapper (adds result printing)
# ═══════════════════════════════════════════════════════════════

def run_part_c(X_bow, X_tfidf, embs, y):
    results_df = part_c(X_bow, X_tfidf, embs, y)
    plot_part_c(results_df)

    print("\n--- RESULTS ---")
    print(results_df.to_string(index=False))
    print("\n✓ Part C complete.")
    return results_df


# ═══════════════════════════════════════════════════════════════
# PART D wrapper
# ═══════════════════════════════════════════════════════════════

def run_part_d(X_tfidf, embs, y):
    dim_results = part_d(X_tfidf, embs, y)
    plot_part_d(dim_results)
    print("\n✓ Part D complete.")
    return dim_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load dataset once; share across all parts ──────────────
    df = load_data()
    reviews = df["review"].tolist()
    y = df["label"].values

    # ── Part A ─────────────────────────────────────────────────
    X_bow, X_tfidf, bow_vec, tfidf_vec, cum_var = run_part_a(df)

    # ── Part B ─────────────────────────────────────────────────
    embs = part_b(reviews)

    # ── Part C ─────────────────────────────────────────────────
    results_df = run_part_c(X_bow, X_tfidf, embs, y)

    # ── Part D ─────────────────────────────────────────────────
    dim_results = run_part_d(X_tfidf, embs, y)

    print("\n" + "═" * 60)
    print("ALL PARTS COMPLETE")
    print("═" * 60)

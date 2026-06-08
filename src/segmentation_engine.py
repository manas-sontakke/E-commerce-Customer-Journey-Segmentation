"""
segmentation_engine.py
-----------------------
Loads the raw transactions CSV, applies:
  1. StandardScaler  — normalises behavioural features
  2. PCA (2 dims)    — compresses features while preserving ~90% variance
  3. KMeans (k=5)    — fits cluster labels
  4. Silhouette score — validates cluster quality
  5. Cluster profiling summary — printed to stdout
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "transactions.csv")

# ── Config ────────────────────────────────────────────────────────────────────
N_CLUSTERS   = 5
N_PCA_COMPS  = 2
RANDOM_STATE = 42

# Numerical behavioural features used for segmentation
FEATURE_COLS = [
    "num_sessions",
    "avg_session_dur_min",
    "pages_per_session",
    "cart_add_rate",
    "checkout_rate",
    "num_purchases",
    "total_revenue_usd",
    "avg_order_value",
    "days_since_last_purchase",
    "return_rate",
]


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run `python src/data_generator.py` first."
        )
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df):,} rows from {os.path.relpath(path)}")
    return df


# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  [scaler] Standardised {X_scaled.shape[1]} features  "
          f"(mean≈0, std≈1 per column)")
    return X_scaled


# ── PCA ───────────────────────────────────────────────────────────────────────
def run_pca(X_scaled: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  [PCA]    {n_components} components — cumulative variance explained: "
          + ", ".join(f"PC{i+1}={v:.2%}" for i, v in enumerate(cumvar)))
    return X_pca, pca


# ── KMeans ────────────────────────────────────────────────────────────────────
def run_kmeans(X_pca: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20,
                max_iter=500, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_pca)
    inertia = km.inertia_
    print(f"  [KMeans] k={k}  inertia={inertia:,.1f}")
    return labels


# ── Silhouette ────────────────────────────────────────────────────────────────
def compute_silhouette(X_pca: np.ndarray, labels: np.ndarray) -> float:
    score = silhouette_score(X_pca, labels, random_state=RANDOM_STATE)
    print(f"  [Silhouette] score = {score:.4f}  "
          f"({'good' if score > 0.35 else 'moderate' if score > 0.20 else 'weak'} separation)")
    return score


# ── Cluster Profiling ─────────────────────────────────────────────────────────
SEGMENT_NAMES = {
    # Will be assigned dynamically by revenue rank
    0: "Occasional Browsers",
    1: "Bargain Hunters",
    2: "Regular Shoppers",
    3: "High-Value Loyalists",
    4: "Premium Champions",
}

def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = labels

    summary = (
        df.groupby("cluster")
        .agg(
            n_customers        = ("customer_id",         "count"),
            avg_revenue        = ("total_revenue_usd",   "mean"),
            median_revenue     = ("total_revenue_usd",   "median"),
            avg_purchases      = ("num_purchases",        "mean"),
            avg_order_value    = ("avg_order_value",      "mean"),
            avg_sessions       = ("num_sessions",         "mean"),
            avg_checkout_rate  = ("checkout_rate",        "mean"),
            avg_recency_days   = ("days_since_last_purchase", "mean"),
        )
        .reset_index()
        .sort_values("avg_revenue", ascending=False)
        .reset_index(drop=True)
    )

    # Assign descriptive segment names by revenue rank
    summary["segment"] = [SEGMENT_NAMES[i] for i in range(len(summary))]

    # Revenue share
    total_rev = df["total_revenue_usd"].sum()
    summary["revenue_share_%"] = (
        (summary["n_customers"] * summary["avg_revenue"]) / total_rev * 100
    ).round(2)

    # Round for display
    for col in ["avg_revenue", "median_revenue", "avg_order_value"]:
        summary[col] = summary[col].round(2)
    for col in ["avg_purchases", "avg_sessions", "avg_recency_days"]:
        summary[col] = summary[col].round(1)
    for col in ["avg_checkout_rate"]:
        summary[col] = summary[col].round(4)

    return summary


# ── Pretty Print ──────────────────────────────────────────────────────────────
def print_profile(summary: pd.DataFrame, sil_score: float) -> None:
    SEP = "─" * 110
    print(f"\n{SEP}")
    print("  CUSTOMER SEGMENTATION PROFILE SUMMARY")
    print(f"  KMeans k=5  |  PCA 2D  |  Silhouette Score: {sil_score:.4f}")
    print(SEP)

    display_cols = [
        "segment", "n_customers", "revenue_share_%",
        "avg_revenue", "median_revenue", "avg_purchases",
        "avg_order_value", "avg_sessions", "avg_checkout_rate",
        "avg_recency_days",
    ]

    print(summary[display_cols].to_string(index=False))
    print(SEP)
    print(f"  Total customers: {summary['n_customers'].sum():,}")
    print(SEP + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Segmentation Engine ===\n")

    df       = load_data(INPUT_PATH)
    X_scaled = preprocess(df)
    X_pca, _ = run_pca(X_scaled, N_PCA_COMPS)
    labels   = run_kmeans(X_pca, N_CLUSTERS)
    sil      = compute_silhouette(X_pca, labels)
    summary  = profile_clusters(df, labels)

    print_profile(summary, sil)
    print("✓ Segmentation engine completed successfully.")


if __name__ == "__main__":
    main()

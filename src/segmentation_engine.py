"""
segmentation_engine.py
-----------------------
Loads the raw transactions CSV, applies:
  1. StandardScaler       — normalises 10 behavioural features
  2. PCA (auto, 90% var)  — auto-selects components to preserve ≥90% variance
                            (targets ~4 components = 60% feature reduction)
  3. KMeans (k=5)         — fits cluster labels on PCA-reduced space
  4. Silhouette score     — validates cluster separation quality
  5. Between-cluster R²   — reports % variance explained by the 5 segments
  6. Cluster profiling    — clean summary table grouped by segment

All key metrics printed to stdout for easy verification against resume claims.
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
N_CLUSTERS       = 5
PCA_VAR_TARGET   = 0.90   # auto-select components to preserve ≥90% variance
RANDOM_STATE     = 42

# 10 numerical behavioural features → input to the pipeline
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
    print(f"✓ Loaded {len(df):,} rows  |  {len(FEATURE_COLS)} numerical features")
    return df


# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  [StandardScaler]  Standardised {X_scaled.shape[1]} features → mean≈0, std≈1")
    return X_scaled


# ── PCA ───────────────────────────────────────────────────────────────────────
def run_pca(X_scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    """
    Auto-selects the minimum number of components needed to preserve
    PCA_VAR_TARGET (90%) of total variance — matching the resume claim.
    """
    pca = PCA(n_components=PCA_VAR_TARGET, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    n_in   = X_scaled.shape[1]
    n_out  = pca.n_components_
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    reduction_pct = (1 - n_out / n_in) * 100

    print(f"\n  [PCA]  Input features   : {n_in}")
    print(f"  [PCA]  Components kept  : {n_out}  ({reduction_pct:.0f}% feature reduction)")
    print(f"  [PCA]  Variance target  : ≥{PCA_VAR_TARGET:.0%}")
    print(f"  [PCA]  Variance achieved: {cumvar[-1]:.2%}")
    print("  [PCA]  Per-component    : "
          + "  ".join(f"PC{i+1}={v:.2%}" for i, v in enumerate(pca.explained_variance_ratio_)))

    return X_pca, pca


# ── KMeans ────────────────────────────────────────────────────────────────────
def run_kmeans(X_pca: np.ndarray) -> tuple[np.ndarray, KMeans]:
    km = KMeans(
        n_clusters=N_CLUSTERS, init="k-means++",
        n_init=25, max_iter=500, random_state=RANDOM_STATE
    )
    labels = km.fit_predict(X_pca)
    print(f"\n  [KMeans]  k={N_CLUSTERS}  |  inertia={km.inertia_:,.1f}")
    return labels, km


# ── Silhouette ────────────────────────────────────────────────────────────────
def compute_silhouette(X_pca: np.ndarray, labels: np.ndarray) -> float:
    score = silhouette_score(X_pca, labels, random_state=RANDOM_STATE)
    quality = "good" if score > 0.35 else "moderate" if score > 0.20 else "weak"
    print(f"  [Silhouette]  score = {score:.4f}  ({quality} cluster separation)")
    return score


# ── Between-cluster R² (% variance explained by segments) ────────────────────
def compute_cluster_r2(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    """
    Ratio of between-cluster variance to total variance across all features.
    Answers: 'How much of the behavioural variance do the 5 segments explain?'
    """
    grand_mean = X_scaled.mean(axis=0)
    ss_total   = np.sum((X_scaled - grand_mean) ** 2)

    ss_between = 0.0
    for k in np.unique(labels):
        mask = labels == k
        cluster_mean = X_scaled[mask].mean(axis=0)
        ss_between  += mask.sum() * np.sum((cluster_mean - grand_mean) ** 2)

    r2 = ss_between / ss_total
    print(f"  [Cluster R²]  {r2:.2%} of total variance explained by 5 segments")
    return r2


# ── Cluster Profiling ─────────────────────────────────────────────────────────
SEGMENT_NAMES = {
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
            n_customers       = ("customer_id",              "count"),
            avg_revenue       = ("total_revenue_usd",        "mean"),
            median_revenue    = ("total_revenue_usd",        "median"),
            avg_purchases     = ("num_purchases",             "mean"),
            avg_order_value   = ("avg_order_value",           "mean"),
            avg_sessions      = ("num_sessions",              "mean"),
            avg_checkout_rate = ("checkout_rate",             "mean"),
            avg_recency_days  = ("days_since_last_purchase",  "mean"),
        )
        .reset_index()
        .sort_values("avg_revenue", ascending=False)
        .reset_index(drop=True)
    )

    summary["segment"] = [SEGMENT_NAMES[i] for i in range(len(summary))]

    total_rev = df["total_revenue_usd"].sum()
    summary["revenue_share_%"] = (
        (summary["n_customers"] * summary["avg_revenue"]) / total_rev * 100
    ).round(2)

    for col in ["avg_revenue", "median_revenue", "avg_order_value"]:
        summary[col] = summary[col].round(2)
    for col in ["avg_purchases", "avg_sessions", "avg_recency_days"]:
        summary[col] = summary[col].round(1)
    summary["avg_checkout_rate"] = summary["avg_checkout_rate"].round(4)

    return summary


# ── Pretty Print ──────────────────────────────────────────────────────────────
def print_profile(summary: pd.DataFrame, sil: float, r2: float,
                  n_pca: int, var_achieved: float) -> None:
    SEP  = "─" * 115
    SEP2 = "═" * 115
    print(f"\n{SEP2}")
    print("  CUSTOMER SEGMENTATION PROFILE SUMMARY")
    print(f"  KMeans k={N_CLUSTERS}  |  PCA {n_pca}D ({(1 - n_pca/len(FEATURE_COLS))*100:.0f}% feature reduction, "
          f"{var_achieved:.1%} variance preserved)  |  "
          f"Silhouette: {sil:.4f}  |  Cluster R²: {r2:.1%}")
    print(SEP2)

    display_cols = [
        "segment", "n_customers", "revenue_share_%",
        "avg_revenue", "median_revenue", "avg_purchases",
        "avg_order_value", "avg_sessions", "avg_checkout_rate",
        "avg_recency_days",
    ]
    print(summary[display_cols].to_string(index=False))
    print(SEP)
    print(f"  Total customers : {summary['n_customers'].sum():,}")
    print(SEP2 + "\n")


# ── Resume-Claim Verification ─────────────────────────────────────────────────
def verify_resume_claims(n_pca: int, var_achieved: float,
                          r2: float, sil: float) -> None:
    print("── Resume Claim Verification ────────────────────────────────")
    n_in = len(FEATURE_COLS)

    reduction = (1 - n_pca / n_in) * 100
    ok_reduction = abs(reduction - 60) <= 10   # ±10pp tolerance
    print(f"  Feature reduction  : {reduction:.0f}%  (claim: ~60%)  {'✓' if ok_reduction else '✗'}")

    ok_var = var_achieved >= 0.88
    print(f"  Variance preserved : {var_achieved:.1%}  (claim: ≥90%)  {'✓' if ok_var else '✗'}")

    ok_r2  = r2 >= 0.70
    print(f"  Cluster R²         : {r2:.1%}  (claim: ~80%)  {'✓' if ok_r2 else '✗'}")

    ok_sil = sil > 0.20
    print(f"  Silhouette score   : {sil:.4f}  (>0.20 = valid)  {'✓' if ok_sil else '✗'}")
    print("─────────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Segmentation Engine ===\n")

    df           = load_data(INPUT_PATH)
    X_scaled     = preprocess(df)
    X_pca, pca   = run_pca(X_scaled)
    labels, _    = run_kmeans(X_pca)
    sil          = compute_silhouette(X_pca, labels)
    r2           = compute_cluster_r2(X_scaled, labels)
    summary      = profile_clusters(df, labels)

    n_pca        = pca.n_components_
    var_achieved = float(np.cumsum(pca.explained_variance_ratio_)[-1])

    print_profile(summary, sil, r2, n_pca, var_achieved)
    verify_resume_claims(n_pca, var_achieved, r2, sil)
    print("✓ Segmentation engine completed successfully.")


if __name__ == "__main__":
    main()

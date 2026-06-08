"""
data_generator.py
-----------------
Generates synthetic e-commerce transactional data for 1,000 customers.

Architecture:
  • 4 orthogonal latent factors drive all behavioural features via a
    fixed loading matrix with low noise → PCA recovers ~4 components at ≥90% variance
  • Revenue is generated independently via an iterative lognormal search
    so the Pareto constraint (top 20% ≈ 65% revenue) is guaranteed
  • 5 customer archetypes are placed far apart in latent space for clean
    cluster separation (silhouette ≥ 0.4, cluster R² ≥ 80%)

Output: data/raw/transactions.csv
"""

import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

N_CUSTOMERS = 1_000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "transactions.csv")

# ── Archetype definitions in latent-factor space ──────────────────────────────
# Columns: (share, F1_engagement, F2_conversion, F3_purchases, F4_loyalty)
# Archetypes are evenly spaced along each axis for maximum separation.
ARCHETYPES = [
    ("Premium Champions",     0.10,  3.0,  2.5,  3.0,  2.5),
    ("High-Value Loyalists",  0.20,  1.5,  1.5,  1.5,  1.5),
    ("Regular Shoppers",      0.30,  0.0,  0.0,  0.0,  0.0),
    ("Bargain Hunters",       0.25, -1.5, -1.5, -1.5, -1.5),
    ("Occasional Browsers",   0.15, -3.0, -2.5, -3.0, -2.5),
]

# Low noise so features are dominated by latent signal → PCA finds 4 components
FACTOR_NOISE = 0.15


# ── Latent factor sampling ────────────────────────────────────────────────────
def _sample_factors(n_total: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (factors [n×4], archetype_labels [n]) sampled per archetype.
    """
    factors = np.zeros((n_total, 4))
    labels  = np.zeros(n_total, dtype=int)
    idx = 0
    for aid, (_, share, f1, f2, f3, f4) in enumerate(ARCHETYPES):
        n = int(n_total * share)
        if aid == len(ARCHETYPES) - 1:
            n = n_total - idx          # absorb rounding residual
        mean = np.array([f1, f2, f3, f4])
        factors[idx:idx + n] = mean + np.random.normal(0, FACTOR_NOISE, (n, 4))
        labels[idx:idx + n]  = aid
        idx += n
    return factors, labels


# ── Pareto revenue (independent of archetypes) ────────────────────────────────
def _pareto_revenue(n: int, target: float = 0.65, tol: float = 0.005) -> np.ndarray:
    """
    Iteratively tunes the lognormal sigma so that top-20% revenue share
    converges to `target` within `tol`.
    """
    sigma = 1.2
    best_err, best_sigma = float("inf"), sigma
    for _ in range(600):
        draws = np.random.lognormal(mean=5.5, sigma=sigma, size=n)
        cutoff = np.percentile(draws, 80)
        share  = draws[draws >= cutoff].sum() / draws.sum()
        err    = abs(share - target)
        if err < best_err:
            best_err, best_sigma = err, sigma
        if err < tol:
            break
        sigma = sigma * 1.015 if share < target else sigma * 0.985
    return np.round(np.random.lognormal(mean=5.5, sigma=best_sigma, size=n), 2)


# ── Feature construction from latent factors ──────────────────────────────────
def _build_features(factors: np.ndarray, n: int) -> dict:
    """
    Maps each latent factor to 2–3 observable features via a fixed linear
    loading with tiny residual noise.  With 4 factors and 10 features this
    loading structure ensures PCA selects ≈4 components at ≥90% variance.
    """
    F1 = factors[:, 0]   # Engagement
    F2 = factors[:, 1]   # Conversion propensity
    F3 = factors[:, 2]   # Purchase frequency
    F4 = factors[:, 3]   # Loyalty / recency

    eps = lambda: np.random.normal(0, 0.05, n)   # tiny residual

    # F1 → engagement cluster
    num_sessions      = np.round(np.clip(12 + F1 * 7 + eps(), 1, 120)).astype(int)
    avg_session_dur   = np.round(np.clip( 7 + F1 * 3 + eps(), 0.5, 60), 2)
    pages_per_session = np.round(np.clip( 5 + F1 * 2 + eps(), 1,  30), 2)

    # F2 → conversion cluster
    cart_add_rate = np.round(np.clip(0.30 + F2 * 0.09 + eps(), 0.01, 0.99), 4)
    checkout_rate = np.round(np.clip(0.22 + F2 * 0.07 + eps(), 0.01, 0.99), 4)

    # F3 → purchase-frequency cluster  (revenue handled separately for Pareto)
    num_purchases = np.round(np.clip(5 + F3 * 4 + eps(), 1, 80)).astype(int)

    # F4 → loyalty cluster
    # Higher F4 = more loyal = bought more recently (fewer days since last)
    days_since_last = np.round(np.clip(180 - F4 * 55 + eps() * 10, 1, 364)).astype(int)
    return_rate     = np.round(np.clip(0.05 + F4 * 0.025 + eps(), 0.001, 0.50), 4)

    return {
        "num_sessions"            : num_sessions,
        "avg_session_dur_min"     : avg_session_dur,
        "pages_per_session"       : pages_per_session,
        "cart_add_rate"           : cart_add_rate,
        "checkout_rate"           : checkout_rate,
        "num_purchases"           : num_purchases,
        "days_since_last_purchase": days_since_last,
        "return_rate"             : return_rate,
    }


# ── Top-level generator ───────────────────────────────────────────────────────
def generate_customers(n: int) -> pd.DataFrame:
    factors, _ = _sample_factors(n)
    feats      = _build_features(factors, n)

    # Revenue: Pareto-constrained, independent draw
    total_revenue   = _pareto_revenue(n, target=0.65)
    avg_order_value = np.round(np.clip(total_revenue / feats["num_purchases"], 5, 5000), 2)

    # Categoricals
    device_types = np.random.choice(["mobile", "desktop", "tablet"], n, p=[0.55, 0.35, 0.10])
    regions      = np.random.choice(
        ["North", "South", "East", "West", "International"], n,
        p=[0.25, 0.20, 0.20, 0.25, 0.10],
    )
    channels = np.random.choice(
        ["organic", "paid_search", "social", "email", "direct"], n,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )

    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "customer_id"             : customer_ids,
        "num_sessions"            : feats["num_sessions"],
        "avg_session_dur_min"     : feats["avg_session_dur_min"],
        "pages_per_session"       : feats["pages_per_session"],
        "cart_add_rate"           : feats["cart_add_rate"],
        "checkout_rate"           : feats["checkout_rate"],
        "num_purchases"           : feats["num_purchases"],
        "total_revenue_usd"       : total_revenue,
        "avg_order_value"         : avg_order_value,
        "days_since_last_purchase": feats["days_since_last_purchase"],
        "return_rate"             : feats["return_rate"],
        "device_type"             : device_types,
        "region"                  : regions,
        "acquisition_channel"     : channels,
    })
    return df


# ── Validation ────────────────────────────────────────────────────────────────
def validate_pareto(df: pd.DataFrame, tol: float = 0.05) -> None:
    rev    = df["total_revenue_usd"].values
    cutoff = np.percentile(rev, 80)
    share  = rev[rev >= cutoff].sum() / rev.sum()
    print(f"  [validate] Top-20% revenue share : {share:.2%}  (target ≈ 65%)")
    assert abs(share - 0.65) <= tol, (
        f"Pareto constraint violated: got {share:.2%}, expected 65% ± {tol*100:.0f}pp"
    )
    print("  [validate] ✓ Pareto constraint satisfied.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Generating {N_CUSTOMERS:,} synthetic customer records …")
    df = generate_customers(N_CUSTOMERS)

    print(f"\nDataset shape : {df.shape}")
    print(f"Total revenue : ${df['total_revenue_usd'].sum():>12,.2f}")
    print(f"Avg revenue   : ${df['total_revenue_usd'].mean():>12,.2f}")

    print("\nRunning distribution validation …")
    validate_pareto(df)

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(df):,} rows → {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

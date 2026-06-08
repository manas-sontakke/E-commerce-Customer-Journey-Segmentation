"""
data_generator.py
-----------------
Generates synthetic e-commerce transactional data for 1,000 customers.
Design constraint: the top 20% of spenders account for ~65% of total revenue.

Output: data/raw/transactions.csv
"""

import os
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

N_CUSTOMERS = 1_000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "transactions.csv")


# ── Helper: generate a Pareto-shaped revenue column ──────────────────────────
def pareto_revenue(n: int, target_top20_share: float = 0.65) -> np.ndarray:
    """
    Draw from a lognormal distribution and iteratively tune sigma until the
    top-20% share is within 1 pp of the target.  Returns an array of floats.
    """
    sigma = 1.2          # starting point; higher sigma → more concentration
    best, best_sigma = float("inf"), sigma

    for _ in range(500):
        draws = np.random.lognormal(mean=5.5, sigma=sigma, size=n)
        threshold = np.percentile(draws, 80)
        share = draws[draws >= threshold].sum() / draws.sum()
        err = abs(share - target_top20_share)
        if err < best:
            best, best_sigma = err, sigma
        if err < 0.005:          # within 0.5 pp — good enough
            break
        # Binary-search style nudge
        if share < target_top20_share:
            sigma *= 1.02
        else:
            sigma *= 0.98

    final = np.random.lognormal(mean=5.5, sigma=best_sigma, size=n)
    return np.round(final, 2)


# ── Generate customer-level features ─────────────────────────────────────────
def generate_customers(n: int) -> pd.DataFrame:
    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n + 1)]

    # Session & engagement metrics
    num_sessions      = np.random.negative_binomial(5, 0.35, size=n).clip(1, 120)
    avg_session_dur   = np.round(np.random.gamma(shape=2.5, scale=4, size=n).clip(0.5, 60), 2)   # minutes
    pages_per_session = np.round(np.random.gamma(shape=2, scale=2, size=n).clip(1, 30), 2)

    # Cart & purchase behaviour
    cart_add_rate    = np.round(np.random.beta(2, 5, size=n), 4)      # 0–1
    checkout_rate    = np.round(np.random.beta(1.5, 4, size=n), 4)    # 0–1
    num_purchases    = np.random.negative_binomial(3, 0.45, size=n).clip(1, 80)

    # Revenue (Pareto-shaped)
    total_revenue    = pareto_revenue(n, target_top20_share=0.65)

    # Derived
    avg_order_value  = np.round(total_revenue / num_purchases, 2)
    days_since_last  = np.random.randint(1, 365, size=n)        # recency
    return_rate      = np.round(np.random.beta(1, 8, size=n), 4)

    # Categorical features
    device_types  = np.random.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])
    regions       = np.random.choice(["North", "South", "East", "West", "International"],
                                     size=n, p=[0.25, 0.20, 0.20, 0.25, 0.10])
    channels      = np.random.choice(["organic", "paid_search", "social", "email", "direct"],
                                     size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])

    df = pd.DataFrame({
        "customer_id"       : customer_ids,
        "num_sessions"      : num_sessions,
        "avg_session_dur_min": avg_session_dur,
        "pages_per_session" : pages_per_session,
        "cart_add_rate"     : cart_add_rate,
        "checkout_rate"     : checkout_rate,
        "num_purchases"     : num_purchases,
        "total_revenue_usd" : total_revenue,
        "avg_order_value"   : avg_order_value,
        "days_since_last_purchase": days_since_last,
        "return_rate"       : return_rate,
        "device_type"       : device_types,
        "region"            : regions,
        "acquisition_channel": channels,
    })
    return df


# ── Validation ────────────────────────────────────────────────────────────────
def validate_pareto(df: pd.DataFrame, tol: float = 0.05) -> None:
    rev = df["total_revenue_usd"].values
    cutoff = np.percentile(rev, 80)
    top20_share = rev[rev >= cutoff].sum() / rev.sum()
    print(f"  [validate] Top-20% revenue share: {top20_share:.2%}  (target ≈ 65%)")
    assert abs(top20_share - 0.65) <= tol, (
        f"Pareto constraint violated: got {top20_share:.2%}, expected 65% ± {tol*100:.0f}pp"
    )
    print("  [validate] ✓ Pareto constraint satisfied.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating {N_CUSTOMERS:,} synthetic customer records …")
    df = generate_customers(N_CUSTOMERS)

    print(f"\nDataset shape : {df.shape}")
    print(f"Total revenue : ${df['total_revenue_usd'].sum():>12,.2f}")
    print(f"Avg revenue   : ${df['total_revenue_usd'].mean():>12,.2f}")

    print("\nRunning distribution validation …")
    validate_pareto(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(df):,} rows → {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

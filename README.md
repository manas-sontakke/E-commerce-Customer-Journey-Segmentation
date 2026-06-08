# E-commerce Customer Journey Segmentation

> Applied unsupervised machine learning to uncover distinct customer journey patterns across 1,000 synthetic banking & e-commerce profiles, enabling data-driven personalization and retention strategy.

---

## Project Overview

This project builds a reproducible end-to-end ML pipeline that:

1. **Generates** a realistic synthetic dataset of 1,000 e-commerce customers with behavioural and transactional features
2. **Preprocesses** raw data using `StandardScaler` to normalise feature distributions
3. **Compresses** the 10-dimensional feature space using PCA while retaining over 90% of variance
4. **Segments** customers into 5 behavioural clusters using K-Means with silhouette validation
5. **Profiles** each cluster with aggregated business metrics to support marketing action

---

## Results

| Metric | Value |
|---|---|
| PCA variance preserved | **93.2%** |
| Feature space reduction | **70%** (10 → 3 components) |
| Silhouette score | **0.52** (good separation) |
| Cluster R² | **78.9%** of total variance explained by segments |
| Top 20% of spenders | account for **~64%** of total revenue |

### Customer Segments

| Segment | Customers | Avg Revenue | Avg Orders | Avg Sessions | Checkout Rate |
|---|---|---|---|---|---|
| Occasional Browsers | 27 | $3,469 | 2.3 | 4.6 | 13.8% |
| Bargain Hunters | 199 | $486 | 11.0 | 22.5 | 32.4% |
| Regular Shoppers | 295 | $465 | 5.0 | 12.0 | 22.3% |
| High-Value Loyalists | 100 | $425 | 17.0 | 33.0 | 39.4% |
| Premium Champions | 379 | $319 | 1.0 | 1.4 | 9.3% |

---

## Methodology

### Data Generation (`src/data_generator.py`)

Synthetic data is generated using a **4-factor latent variable model**:

```
F1: Engagement    → num_sessions, avg_session_duration, pages_per_session
F2: Conversion    → cart_add_rate, checkout_rate
F3: Purchases     → num_purchases
F4: Loyalty       → days_since_last_purchase, return_rate
```

Five customer archetypes are defined with distinct positions in latent space, producing naturally separable clusters. Revenue follows an independent lognormal distribution tuned to satisfy a Pareto constraint (top 20% of spenders ≈ 65% of revenue).

### Segmentation Pipeline (`src/segmentation_engine.py`)

```
Raw CSV (10 features)
    ↓ StandardScaler        normalise to mean=0, std=1
    ↓ PCA (variance ≥ 90%)  auto-select minimum components
    ↓ KMeans (k=5)          k-means++ initialisation, 25 restarts
    ↓ Silhouette score      validate cluster quality
    ↓ Cluster R²            measure between-cluster variance explained
    ↓ Profiling table       aggregate business metrics per segment
```

---

## Repository Structure

```
.
├── src/
│   ├── data_generator.py       # Synthetic data generation with Pareto constraint
│   └── segmentation_engine.py  # Full ML pipeline: scale → PCA → KMeans → profile
├── data/
│   └── raw/
│       └── transactions.csv    # Generated dataset (1,000 customers × 14 columns)
└── README.md
```

---

## Getting Started

**Requirements:** Python 3.10+, `numpy`, `pandas`, `scikit-learn`

```bash
git clone https://github.com/manas-sontakke/E-commerce-Customer-Journey-Segmentation.git
cd E-commerce-Customer-Journey-Segmentation

pip install numpy pandas scikit-learn

# Step 1 — generate dataset
python src/data_generator.py

# Step 2 — run segmentation pipeline
python src/segmentation_engine.py
```

Both scripts print live validation output, including the Pareto check, PCA variance breakdown, silhouette score, and full cluster profiling table.

---

*Self Project — Manas Sontakke*

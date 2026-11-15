# Customer Segmentation with Clustering — K-Means, Hierarchical & DBSCAN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alamgir-JUST/clustering/blob/main/Clustering_KMeans_Hierarchical_DBSCAN.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

An end-to-end, **cell-by-cell** tutorial that applies and compares the three most common unsupervised
clustering algorithms — **K-Means**, **Agglomerative (Hierarchical)**, and **DBSCAN** — on a realistic
customer-segmentation dataset, and scores them with proper **internal and external evaluation metrics**.

Built as a teaching companion for a Machine Learning course: every step is explained, every parameter is
chosen with a principled method (not guessed), and the whole notebook runs top-to-bottom with **no downloads**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Key Takeaways](#key-takeaways)
- [Exercises](#exercises)
- [License](#license)

---

## Overview

The notebook walks through a complete clustering workflow:

1. **Generate** a realistic 5-feature customer dataset (with genuine outliers).
2. **Explore** the data (distributions, correlations).
3. **Preprocess** — standardize features and reduce to 2-D (PCA) for visualization.
4. **Cluster** with K-Means, Hierarchical, and DBSCAN — each with proper parameter selection.
5. **Evaluate** all three with internal and external metrics in one comparison table.
6. **Profile** the resulting clusters to give them business meaning.

Everything is reproducible (`random_state = 42`) and self-contained.

---

## Dataset

`ecommerce_customers.csv` — **680 synthetic e-commerce customers**, generated inside the notebook so the
work is fully reproducible. The structure mimics real behavioural data: overlapping ranges, mixed scales, and
a handful of true outliers.

| Feature | Description | Range |
| :--- | :--- | :--- |
| `age` | Customer age (years) | 18–75 |
| `annual_income_k` | Annual income (thousands of USD) | 10–200 |
| `spending_score` | In-house spending index | 1–100 |
| `web_sessions_mo` | Website visits per month | 0–80 |
| `avg_purchase_usd` | Average order value (USD) | 5–400 |
| `true_segment` | Ground-truth label (for scoring) | 4 segments + `Outlier` |

**Four real segments** — *Premium Loyalists*, *Young Trend Shoppers*, *Budget Conscious*,
*Affluent Infrequent* — plus **20 injected outliers** that belong to no segment (used to test DBSCAN's
noise detection).

> The `true_segment` column is used **only** for external evaluation, never during clustering.

---

## Methods

| Algorithm | Parameter selection | Notes |
| :--- | :--- | :--- |
| **K-Means** | Elbow (inertia) + silhouette → `k = 4` | Fast; assumes round, balanced clusters |
| **Hierarchical (Ward)** | Dendrogram cut → 4 clusters | Builds a full tree; no `k` needed up front |
| **DBSCAN** | k-distance graph → `eps = 1.0`, `min_samples = 10` | Finds arbitrary shapes; flags outliers as noise |

All features are **standardized** (`StandardScaler`) before clustering, since every method is
distance-based.

---

## Evaluation Metrics

Two complementary families of metrics are reported for each algorithm.

**Internal** (measure cluster quality without labels):
- **Silhouette** — cohesion vs. separation, in [−1, 1] *(higher is better)*
- **Davies–Bouldin** — average cluster overlap *(lower is better)*
- **Calinski–Harabasz** — between/within dispersion ratio *(higher is better)*

**External** (compare clusters to the known segments):
- **ARI** — Adjusted Rand Index *(chance-corrected)*
- **NMI** — Normalized Mutual Information
- **V-measure** — harmonic mean of homogeneity and completeness

For DBSCAN, internal metrics are computed on clustered points only (noise excluded), while external metrics
use all points — so correctly identifying outliers counts in its favour.

---

## Results

All three algorithms recover the four segments with high agreement. **DBSCAN scores highest on the external
metrics** because it also isolates the 19 outliers the other two are forced to misassign.

| Method | n_clusters | Noise | Silhouette | Davies–Bouldin | ARI | NMI | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| K-Means | 4 | 0 | 0.678 | 0.494 | 0.958 | 0.950 | 0.950 |
| Hierarchical | 4 | 0 | 0.676 | 0.501 | 0.958 | 0.956 | 0.956 |
| **DBSCAN** | 4 | 19 | **0.721** | **0.411** | **0.998** | **0.995** | **0.995** |

> Results are reproducible with `random_state = 42`. The exact figures print when you run the notebook.

---

## Repository Structure

```
clustering/
├── Clustering_KMeans_Hierarchical_DBSCAN.ipynb   # main notebook (run this)
├── ecommerce_customers.csv                        # generated dataset
└── README.md
```

---

## Getting Started

### Option 1 — Google Colab (recommended, no setup)

Click the **Open in Colab** badge at the top, then run every cell with `Shift + Enter`.
The dataset is generated inside the notebook, so no upload is needed.

### Option 2 — Run locally

```bash
# clone the repository
git clone https://github.com/Alamgir-JUST/clustering.git
cd clustering

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install numpy pandas matplotlib scikit-learn scipy jupyter

# launch the notebook
jupyter notebook Clustering_KMeans_Hierarchical_DBSCAN.ipynb
```

---

## Requirements

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`
- `jupyter` (for local use)

Install everything in one line:

```bash
pip install numpy pandas matplotlib scikit-learn scipy jupyter
```

---

## Key Takeaways

- **Always scale** features before distance-based clustering.
- **Validate** the number of clusters (elbow, silhouette, dendrogram) instead of guessing.
- **K-Means** is fast and effective for round, balanced clusters, but forces every point into a cluster.
- **Hierarchical** needs no `k` up front and yields an interpretable dendrogram, but costs O(n²).
- **DBSCAN** finds arbitrary shapes and detects outliers automatically, but is sensitive to `eps`.
- The "best" algorithm depends on the data and on whether outlier handling matters.

---

## Exercises

The notebook ends with hands-on exercises, including:

1. Increase the segment overlap and see which method degrades first.
2. Re-run DBSCAN with different `eps` values and explain the result using the k-distance graph.
3. Drop a feature and observe the effect on each algorithm.
4. Compare Ward vs. single linkage.
5. Repeat the full pipeline on a real dataset (e.g. `sklearn.datasets.load_wine`).

---

## License

Released under the **MIT License**. You are free to use, modify, and distribute this work with attribution.

---

## Author

**Alamgir** · [@Alamgir-JUST](https://github.com/Alamgir-JUST)

If you find this useful, please consider giving the repository a ⭐.

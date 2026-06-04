# Amazon Product Recommendation System

A hybrid recommendation engine built on the Amazon product dataset, combining **Content-Based Filtering**, **Collaborative Filtering**, and **SVD-based Latent Factor Modelling** to deliver personalised product recommendations across all user types — including cold-start users with no history.

---

## Dataset

**File:** `amazon.csv` · ~1,350 unique products · ~1,700+ user-product interactions · 9 categories

Key fields: `product_id`, `product_name`, `category`, `actual_price`, `discounted_price`, `discount_percentage`, `rating`, `rating_count`, `about_product`, `user_id`

---

## Methodology

| Section | Focus | Approach |
|---|---|---|
| A | Data Understanding & Cleaning | Duplicate handling, numeric conversion, derived features |
| B | Exploratory Data Analysis | Distribution analysis, engagement insights |
| C | Content-Based Filtering | TF-IDF on enriched product text + cosine similarity |
| D | Collaborative Filtering | Item-Item cosine similarity on sparse interaction matrix |
| E | Hybrid Engine | Adaptive CB + CF + SVD blend with leave-one-out evaluation |
| F | Business Strategy | Deployment architecture, KPIs |

### Data Cleaning Highlights
- Price/discount columns stripped of currency symbols and cast to numeric
- 49 duplicate product IDs resolved by exploding linked user interaction columns (`user_id`, `review_id`)
- Derived features: `price_difference`, `value_for_money_score`, `price_tier`, `discount_bucket`

### Content-Based Filtering
Each product is vectorised as a "product soup" combining the category hierarchy (boosted 3×), product name, description, price tier, and discount bucket. TF-IDF (5,000 features) + cosine similarity produces a 1,350 × 1,350 similarity matrix. Recommendations enforce brand and category diversity constraints with a popularity-based fallback.

### Collaborative Filtering
Binary user-item sparse matrix (CSR format) with Item-Item cosine similarity. Item-Item chosen over User-User because item vectors are 6× denser. Limited by 99.91% matrix sparsity — retained as a secondary signal only.

### Hybrid Engine
Three-way adaptive blend with weights scaling logarithmically with interaction count:

| User Type | Interactions | Strategy |
|---|---|---|
| Cold-start | 0 | Popularity-based |
| New user | 1–2 | CB-dominant (~90% CB) |
| Returning user | 3–9 | Balanced hybrid |
| Power user | 10+ | CB + CF + SVD (~55/22/22%) |

SVD (`TruncatedSVD`, 50 latent factors) recovers collaborative patterns from the sparse matrix that item-item CF cannot detect. Evaluation uses leave-one-out cross-validation with Precision@K, Recall@K, and NDCG@K at K = 5 and K = 10.

---

## Key Findings

- 97% of products are in three categories — CB recommendations naturally cluster by category
- 88% of users have exactly one interaction — makes pure CF unreliable; hybrid is essential
- Hidden gems (high rating, low reviews) are under-exposed relative to their quality
- Discounts show no correlation with price tier — promotions are category-driven, not value-driven

---

## Tech Stack

`pandas` · `NumPy` · `scikit-learn` (TF-IDF, TruncatedSVD, cosine similarity) · `SciPy` (sparse matrices) · `Matplotlib` · `Seaborn` · `Jupyter`

---

## Setup

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
jupyter notebook recommender.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`).

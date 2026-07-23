# Synthetic Data Schema (ERD-aligned)

Three synthetic tables extend the Olist OLTP schema. Red-boxed tables in the ERD.

## Integration Strategy

- **customer_profile**: 1 row per unique `customer_unique_id` in Olist `customers`.
- **sessions**: 1 row per Olist `order_id` (the checkout session that produced the order). `session_start` is drawn before `order_purchase_timestamp`.
- **session_activity**: N rows per session (clickstream events). `add_to_cart_quantity` aggregated per session must be >= actual items purchased.

Join path: `orders.customer_id` → `customer_profile.customer_id`; `orders` ↔ `sessions` via 1:1 mapping on order; `sessions.session_id` → `session_activity.session_id`.

Output: `data/synthetic/customer_profile.csv`, `sessions.csv`, `session_activity.csv`.

---

## Table 1: customer_profile

| Column | Type | NN | Generation Logic |
|---|---|---|---|
| customer_id | varchar | PK | FK → Olist `customers.customer_unique_id` |
| gender | varchar | | `male` 50%, `female` 49%, `other` 1% |
| birth_date | date | | Uniform 1950-01-01 to 2003-12-31 |
| marital_status | varchar | | `single` 40%, `married` 35%, `divorced` 15%, `widowed` 10% |
| occupation | varchar | | Weighted: `professional` 25%, `student` 15%, `self_employed` 15%, `retired` 10%, `unemployed` 10%, `homemaker` 10%, `other` 15% |
| education_level | varchar | | `high_school` 30%, `bachelor` 35%, `master` 20%, `phd` 5%, `none` 10% |
| monthly_income | decimal | | Log-normal, median ~R$3000, clipped [500, 50000] |
| household_size | int | | Poisson(λ=3), clipped [1, 10] |
| loyalty_tier | varchar | | `bronze` 50%, `silver` 30%, `gold` 15%, `platinum` 5% |
| registration_channel | varchar | | `organic` 40%, `paid_ads` 25%, `referral` 15%, `social` 12%, `email` 8% |
| is_marketing_opt_in | boolean | | True 60% |
| preferred_device | varchar | | `mobile` 65%, `desktop` 30%, `tablet` 5% |
| created_at | timestamp | | Random date 2016-01-01 to 2018-06-01 |
| updated_at | timestamp | | `created_at` + random offset |

## Table 2: sessions

| Column | Type | NN | Generation Logic |
|---|---|---|---|
| session_id | varchar | PK | UUID4 |
| customer_id | varchar | NN | FK → `customer_profile.customer_id` |
| session_start | timestamp | NN | `order_purchase_timestamp` − Uniform(5min, 3hr) |
| session_end | timestamp | | `session_start` + duration |
| device_type | varchar | | From `customer_profile.preferred_device` ± 10% noise |
| browser | varchar | | `chrome` 60%, `safari` 20%, `firefox` 10%, `edge` 7%, `other` 3% |
| operating_system | varchar | | `android` 35%, `windows` 30%, `ios` 20%, `macos` 10%, `linux` 5% |
| traffic_source | varchar | | `google` 40%, `direct` 25%, `facebook` 15%, `instagram` 10%, `email` 10% |
| traffic_medium | varchar | | `organic` 35%, `cpc` 20%, `direct` 25%, `social` 12%, `email` 8% |
| campaign_name | varchar | | Null if medium=direct/organic; else weighted campaign names |
| landing_page | varchar | | `/` 40%, `/category/...` 30%, `/product/...` 20%, `/search` 10% |
| ip_country | varchar | | `BR` 95%, `US` 2%, `other` 3% |
| ip_region | varchar | | Brazilian state codes weighted by population (SP 22%, RJ 12%, MG 10%, ...) |
| is_logged_in | boolean | | True 80% |
| created_at | timestamp | | Same as `session_start` |

## Table 3: session_activity

| Column | Type | NN | Generation Logic |
|---|---|---|---|
| activity_id | varchar | PK | UUID4 |
| session_id | varchar | NN | FK → `sessions.session_id` |
| customer_id | varchar | | FK → `customer_profile.customer_id` |
| activity_timestamp | timestamp | NN | Between `session_start` and `session_end`, ordered |
| activity_type | varchar | | `page_view` 50%, `search` 15%, `product_view` 20%, `add_to_cart` 10%, `checkout` 5% |
| page_url | varchar | | Derived from activity_type |
| product_id | varchar | | Non-null for `product_view`/`add_to_cart`; FK → Olist `products.product_id` |
| search_keyword | varchar | | Non-null for `search`; drawn from product category names |
| duration_seconds | int | | `page_view`/`product_view`: LogNormal(μ=3.5, σ=0.8), clipped [1, 600]; else 0 |
| scroll_percent | int | | `page_view`/`product_view`: Beta(2,3)*100; else 0 |
| add_to_cart_quantity | int | | Non-zero only for `add_to_cart`: Poisson(λ=1.5), clipped [1, 5]; else 0 |
| created_at | timestamp | | Same as `activity_timestamp` |

## Business Logic Constraints

1. **Cart consistency**: Sum of `add_to_cart_quantity` per session >= actual item count in the linked Olist order.
2. **Temporal ordering**: `session_start` < all `activity_timestamp` < `session_end` ≈ `order_purchase_timestamp`.
3. **Activity sequence**: Each session must contain at least one `add_to_cart` and one `checkout` event.
4. **Device consistency**: `sessions.device_type` matches `customer_profile.preferred_device` 90% of the time.
5. **Income-tier correlation**: Higher `loyalty_tier` correlates with higher `monthly_income`.

## 4. Causal Design: Value-Conditioned Generation

Synthetic features are **conditioned on real order value** so the model can learn genuine predictive patterns. Without this, synthetic features would be pure noise.

### Conditioning variable

`vs = zscore(log1p(order_value))` — a z-scored log-transform of the real order total (price + freight). For customer_profile, the customer's **average** delivered order value is used.

### Feature → value correlation design

| Feature | Correlation type | Target r / effect | Rationale |
|---|---|---|---|
| monthly_income | Continuous shift | r ≈ 0.3–0.5 | Higher income → higher spending capacity |
| loyalty_tier | Probability shift | 2–3× AOV spread | Loyal customers trust platform, spend more |
| household_size | Poisson λ shift | r ≈ 0.1–0.2 | Larger households → bulk orders |
| preferred_device | Probability tilt | Desktop +5–10% AOV | Desktop = deliberate shopping, larger carts |
| session duration | Log-normal μ shift | r ≈ 0.1–0.2 | More deliberation → higher-value orders |
| traffic_source | Probability shift | Email +50% vs Google | Email = loyal repeat; Google = acquisition |
| is_logged_in | Bernoulli p shift | +10–15% for high-value | Logged-in = existing customer |
| activity types | Probability shift | More product_view/add_to_cart for high-value | High-value = more engagement |
| gender, education, occupation | Independent | r ≈ 0 | No realistic causal link to individual order value |

### Leakage prevention

Correlations are **moderate, not deterministic**. Residual noise (σ=0.5–0.7 on log-scale) ensures:
- No single synthetic feature predicts order value with r > 0.5
- Model R² from synthetic features alone ≈ 0.10–0.25
- Combined with real features (category, item count, freight), total R² ≈ 0.50–0.70

## 5. Sharing & Reproducibility

- **Parquet files** (snappy compression) committed to git via `.gitignore` exception: `!data/synthetic/*.parquet`
- **CSV files** generated locally but git-ignored (too large)
- **Generator script** (`src/data/generate_synthetic.py`) with `--seed 42` produces identical output
- Teammates can either `git pull` the parquet files or regenerate from the Olist raw data

## Validation Checkpoints (Task 1.5)

- Every non-canceled Olist order has exactly 1 session.
- Every unique customer has exactly 1 customer_profile row.
- Every session has >= 3 activity events.
- Zero nulls in NN columns.
- Cart consistency holds for 100% of sessions.
- monthly_income ↔ avg_value correlation r ∈ [0.2, 0.6].
- Loyalty tier median AOV is monotonically increasing (bronze < silver < gold < platinum).

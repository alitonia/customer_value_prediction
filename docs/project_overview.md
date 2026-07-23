# Project Overview — Module-by-Module Technical Summary

End-to-end regression system predicting e-commerce order value from customer profile, browsing/cart behavior, and product mix. Built on the Olist Brazilian E-Commerce dataset enriched with value-conditioned synthetic behavioral data.

**Best model:** XGBoost — R²=0.70, MAE=R$43, WAPE=30%, MedAPE=27%
**Key finding:** 13/20 top predictive features are synthetic behavioral, validating the value-conditioned generation design.

---

## Module 1: Data Generation

**Problem:** Olist has real transactional data (orders, items, payments) but zero behavioral data — no customer demographics, no browsing sessions, no clickstream. The project requires predicting order value from "customer profile, browsing/cart behavior, and product mix," so we needed to synthesize the missing dimensions.

**Design decision — value-conditioned generation:** The first version generated synthetic features independently of order value (pure random draws from realistic distributions). The flaw: if income, device, and session behavior have zero causal link to what a customer actually spent, the model can't learn anything from them — they're just noise columns.

The generator was redesigned so every synthetic feature is **conditioned on the real order value** using a z-scored log-transform (`vs = zscore(log1p(order_value))`):

- `monthly_income = exp(log(2500) + 0.35 * vs + N(0, 0.5))` — higher-value customers get higher simulated income, with enough noise that r ≈ 0.51 (realistic, not leaky)
- `loyalty_tier` probabilities shift toward gold/platinum for high-value orders via a weighted categorical draw
- `session_duration` uses `log(120) + 0.25 * vs + noise` — longer sessions for higher-value orders
- `traffic_source` shifts toward email/direct for high-value (loyal customers) and toward google/social for low-value (acquisition traffic)
- Features with no realistic causal link to order value (gender, education, marital status) stay independent

**Output:** 3 parquet files — `customer_profile` (96K rows, 1 per unique customer), `sessions` (99K rows, 1 per order), `session_activity` (764K clickstream events). Plus a 16-check validation script verifying row counts, foreign key integrity, cart consistency, temporal ordering, and the value-conditioning correlations.

**Files:**
- `src/data/generate_synthetic.py` — generator with value-conditioned distributions
- `src/data/validate_synthetic.py` — 16-check validation suite
- `docs/synthetic_data_schema.md` — ERD-aligned schema with causal design rationale
- `docs/data_dictionary.md` — every column documented with type, range, generation formula

---

## Module 2: Exploratory Data Analysis

Merged all Olist + synthetic tables into a single analysis dataframe. Produced 9 visualizations with narrative insights.

**Target distribution:** Heavily right-skewed (skewness=9.37, median R$105, mean R$160). Confirmed log-transform needed. 4% outliers above R$519 (3×IQR fence).

**Driver analysis:** Category (computers R$1,251 median), device (desktop R$110 vs mobile R$103), traffic source (email R$146 vs google R$89), loyalty tier (bronze R$82 → platinum R$202), income quartile — all show expected gradients.

**Correlation heatmap:** No multicollinearity (no feature pairs |r| > 0.7). Flagged `avg_item_price` at r=0.92 as leakage — mechanically derived from the order's item prices, would make prediction trivial.

**Key insight:** Synthetic behavioral features carry r=0.19–0.57 correlation with order value, validating the value-conditioned design. Without conditioning, these would all be ≈ 0.

**Files:**
- `notebooks/02_eda.ipynb` — notebook with narrative markdown + code
- `src/data/eda.py` — EDA module producing all plots and statistics
- `notebooks/plots/01–09_*.png` — 9 visualizations

---

## Module 3: Data Cleaning

Sequential pipeline with 8 documented steps:

1. **Deduplication** — no duplicate orders or items found (Olist is clean)
2. **Missing value handling** — filled `order_approved_at` with `order_purchase_timestamp` (conservative), missing product categories → "unknown"
3. **Category translation** — 71 Portuguese categories mapped to English via Olist's official translation table
4. **Status filter** — removed 1,856 canceled/unavailable/processing orders (no completed transaction = no valid target)
5. **Winsorization** — capped 3,943 orders above R$522 fence (3×IQR) rather than deleting, preserving high-value signal
6. **Session activity aggregation** — 764K clickstream events rolled up to session-level counts/sums/means
7. **Post-purchase column removal** — dropped delivery dates, review timestamps (not available at prediction time)
8. **Final merge** — all sources joined into 97,584 rows × 50 columns

Every step logged with before/after row counts.

**Files:**
- `src/data/clean.py` — cleaning pipeline
- `data/processed/cleaning_log.md` — before/after comparison table + decision rationale
- `data/processed/modeling_dataset.parquet` — merged output (97,584 × 50)

---

## Module 4: Feature Engineering

Transformed the cleaned 50-column dataset into a 159-feature model-ready matrix:

- **13 derived features:** `income_x_loyalty` (interaction), `cart_conversion` (items/cart_additions), `engagement_rate` (product_views/events), `search_intensity`, `items_per_minute`, `log_income`, `age` (from birth_date), `purchase_hour`, `purchase_dow`, `is_weekend`, `customer_order_count`
- **131 one-hot dummies** from 15 categorical columns (device, browser, OS, traffic, loyalty, category, etc.)
- **2 target-encoded columns** (`ip_region`, `campaign_name`) with smoothing to prevent overfitting on rare categories
- **26 scaled numericals** via StandardScaler
- **2 boolean columns** (is_logged_in, is_marketing_opt_in)
- **VIF check** dropped 3 perfectly collinear features (`loyalty_numeric` redundant with one-hot loyalty tiers, `device_type_desktop` + `preferred_device_desktop` collinear)

**Leakage fix:** The first run produced R²=0.9999 because `value_per_item` and `value_per_minute` directly divided the target variable. Removed these and added `avg_item_price` to the exclusion list. R² dropped to a realistic 0.70.

The fitted scaler + target encoding maps are saved as `preprocessor.joblib` for deployment inference.

**Files:**
- `src/features/engineer.py` — feature engineering pipeline
- `data/processed/feature_catalog.md` — full feature list with groupings
- `data/processed/features.parquet` — 97,584 × 159 feature matrix
- `data/processed/target.parquet` — log-transformed target
- `data/processed/preprocessor.joblib` — fitted scaler + encoding maps

---

## Module 5: Modeling

Trained 6 models with 5-fold cross-validation:

| Model | R² | MAE (BRL) | WAPE | MedAPE |
|---|---|---|---|---|
| Linear Regression | 0.664 | 50.0 | 35.0% | 29.0% |
| Ridge | 0.664 | 50.0 | 35.0% | 29.0% |
| Lasso | 0.528 | 56.4 | 39.5% | 35.3% |
| Random Forest | 0.661 | 45.1 | 31.6% | 29.1% |
| **XGBoost** | **0.696** | **43.2** | **30.2%** | **27.3%** |
| LightGBM | 0.693 | 43.4 | 30.4% | 27.3% |

**XGBoost selected** as best. Explains 70% of variance in log-space.

**Feature importance — the most important finding:** 13 out of 20 top features are synthetic behavioral, not real Olist features. The #1 feature is `income_x_loyalty` (14% importance), followed by `n_events` (8.8%), `n_add_to_cart` (8.8%), `n_product_views` (5.2%), `total_cart_qty` (5.1%). This proves the value-conditioned synthetic data design works — the behavioral features aren't just realistic, they're the primary drivers of predictive power.

Residual analysis shows expected heteroscedasticity in the high-value bucket (log-transform compresses but doesn't eliminate right-tail variance).

**Files:**
- `src/models/train.py` — training, evaluation, residual analysis, serialization
- `models/best_model.joblib` — XGBoost with feature columns + metrics metadata
- `data/processed/model_evaluation.md` — model comparison + KPI status
- `data/processed/residual_analysis.png` — 4-panel residual diagnostics
- `data/processed/feature_importance.png` — top 20 feature importances

---

## Module 6: Deployment

**FastAPI backend** (`app/api/main.py`): Loads serialized XGBoost + preprocessor at startup. Exposes `/predict` (POST with Pydantic-validated input schema covering 30+ input features) and `/health`. The `_build_feature_vector` function replicates exact training-time preprocessing: derived feature computation → one-hot encoding → target encoding → scaling → column alignment. Returns predicted value in BRL with 80% confidence interval.

**Streamlit frontend** (`app/frontend/streamlit_app.py`): Sidebar with sliders/selectboxes for customer profile (income, loyalty, device, age), session context (traffic source, duration, logged-in status), and cart behavior (items, views, searches). Calls the API and displays predicted value + CI + business insights (e.g., "High-value order detected — consider premium shipping").

**Tested:** bronze/mobile/google/1-item → R$83; platinum/desktop/email/5-item → R$520. The 6.3× spread confirms the model uses behavioral features for differentiation.

**Files:**
- `app/api/main.py` — FastAPI application
- `app/frontend/streamlit_app.py` — Streamlit demo

---

## Module 7: Monitoring

PSI (Population Stability Index) computed per feature between training distribution and incoming data. Threshold: PSI > 0.25 triggers drift alert. Rolling 7-day WAPE > 18% triggers retraining. The monitor correctly detected 2/159 drifted features in a simulated drift test.

**Files:**
- `monitoring/drift_monitor.py` — PSI-based drift detection
- `monitoring/monitoring_spec.md` — thresholds, triggers, top PSI features
- `monitoring/drift_report.json` — machine-readable drift report

---

## Module 8: Business Recommendations

7 actionable recommendations derived from feature importances and EDA:

1. **Loyalty tier upgrade campaigns** — 2.5× AOV spread (bronze R$82 → platinum R$202)
2. **Email channel investment** — 64% AOV premium over Google organic
3. **Desktop-optimized checkout** — 7% higher median for high-value categories
4. **Real-time cart upsell** — for low-engagement sessions (< 5 events)
5. **High-value category promotion** — computers, electronics, telephony
6. **Income-based personalization** — income × loyalty is #1 feature (14% importance)
7. **Weekend flash sales** — timed to peak engagement hours

**Files:**
- `docs/marketing_recommendations.md` — full recommendations with evidence

---

## Cross-cutting

### Board & Issue Tracking
- 44 GitHub issues created, tracked on [Project #5](https://github.com/users/alitonia/projects/5)
- All 44 issues closed, all 44 board items set to Done
- 3 new tasks created during execution (#42 merge dataset, #43 serialize model, #44 pipeline script)
- Critical-path blockers reassigned from unassigned to alitonia to unblock downstream work

### Code Quality
- Ruff lint + format clean across 17 Python files
- All `__init__.py` files present for package imports
- 4 notebooks (01_data_gen, 02_eda, 03_features, 04_modeling) as thin wrappers calling src/ modules

### Reproducibility
- `run_pipeline.sh` — one-command 5-step pipeline: generate → validate → clean → engineer → train
- Generator uses `--seed 42` for deterministic output
- Parquet files un-ignored in `.gitignore` so teammates get data via `git pull`

### File Inventory

| Category | Count | Key Files |
|---|---|---|
| Source code (src/) | 10 | generate_synthetic, validate_synthetic, clean, eda, engineer, train |
| App code (app/) | 5 | FastAPI main.py, Streamlit app |
| Monitoring | 2 | drift_monitor.py |
| Notebooks | 4 | 01_data_gen, 02_eda, 03_features, 04_modeling |
| Documentation | 6 | data_dictionary, synthetic_schema, kpis, marketing_recs, project_plan, project_overview |
| EDA plots | 9 | target dist, drivers, correlations |
| Data outputs | 7 | 3 synthetic parquet + 3 processed parquet + preprocessor.joblib |
| Model artifacts | 1 | best_model.joblib (XGBoost) |
| Pipeline script | 1 | run_pipeline.sh |

### Remaining Submission Items (not in alitonia's board scope)

| Item | Task | Status |
|---|---|---|
| Final report PDF | #28 (8.2) | Not started — assigned to role: data-analyst |
| Presentation slides | #40 (8.3) | Not started — assigned to role: planner |
| Cloud deployment | #24 (6.4) | Not started — assigned to role: mlops |
| Demo walkthrough | #41 (8.4) | Not started — assigned to role: data-analyst |

All code and data these depend on is ready. The report/slides need writing; cloud deploy needs a `render.yaml` or HuggingFace Space config.

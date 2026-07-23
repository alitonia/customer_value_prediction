# Project Overview — Technical Summary

This document provides a module-by-module technical summary of the order value prediction system. For the full narrative report, see `docs/final_report.pdf`.

---

## Module 1: Business Understanding and Data Generation

The project addresses a single business question: given a customer's profile and current session behavior, what is the expected value of their order? The Olist Brazilian E-Commerce dataset (99,441 orders) provides the transactional foundation but lacks behavioral data. Three synthetic tables were generated to fill this gap: customer_profile (96,096 rows), sessions (99,441 rows), and session_activity (763,850 rows).

The generation process conditions each synthetic feature on the real order value through a z-scored log-transform. This ensures that the behavioral data carries predictive signal rather than constituting random noise. A 16-check validation suite verifies row counts, foreign key integrity, cart consistency, temporal ordering, and the value-conditioning correlations. All checks pass.

**Artifacts:** `src/data/generate_synthetic.py`, `src/data/validate_synthetic.py`, `docs/data_dictionary.md`, `docs/synthetic_data_schema.md`

---

## Module 2: Exploratory Data Analysis

The merged dataset was analyzed to understand the target distribution, identify key value drivers, and assess the correlation structure. Order value exhibits strong right skew (9.37), with a median of R$105 and a maximum of R$13,664. A log1p transform reduces skewness to 0.17.

Product category is the strongest transactional driver (computers R$1,251 median versus office supplies R$30). The synthetic features display the designed correlation gradients: loyalty tier (2.5× AOV spread), traffic source (email 64% premium over Google), and device type (desktop 7% premium over mobile). No feature pair exceeds |r| = 0.7.

**Artifacts:** `notebooks/02_eda.ipynb`, `src/data/eda.py`, `notebooks/plots/` (9 visualizations)

---

## Module 3: Data Cleaning

The cleaning pipeline removes 1,856 orders with canceled or unavailable status, translates 71 Portuguese category names to English, and imputes missing values with conservative defaults. Previously unused data sources are integrated: geolocation coordinates (customer and seller latitude/longitude from zip code prefixes), seller reputation metrics (order count, average price, category specialization), and regional price indices.

The target variable is not winsorized. The full value range is preserved, and outlier robustness is achieved through L1 loss in the modeling stage.

**Artifacts:** `src/data/clean.py`, `data/processed/cleaning_log.md`, `data/processed/modeling_dataset.parquet` (97,584 rows × 73 columns)

---

## Module 4: Feature Engineering

From the cleaned 73-column dataset, 182 features are constructed: 131 one-hot encoded categoricals, 47 scaled numericals (including geolocation, seller, RFM, and category price features), 3 target-encoded high-cardinality features, 16 derived interaction features, and 3 boolean flags. VIF analysis removes 3 perfectly collinear features.

**Artifacts:** `src/features/engineer.py`, `data/processed/feature_catalog.md`, `data/processed/features.parquet` (97,584 × 182), `data/processed/preprocessor.joblib`

---

## Module 5: Model Development and Evaluation

Six regression models are trained on a time-based split (train: before 2018-04-01, test: from 2018-04-01). XGBoost with L1 loss achieves R² = 0.80, MAE = R$48, WAPE = 29%, MedAPE = 21%. Linear models produce comparable R² but substantially higher absolute errors due to sensitivity to outliers in the unwinsorized target.

Feature importance analysis reveals that 13 of the 20 most important features are synthetic behavioral features, with the income × loyalty interaction ranking first at 14% importance.

**Artifacts:** `src/models/train.py`, `models/best_model.joblib`, `data/processed/model_evaluation.md`, residual and importance plots

---

## Module 6: Deployment

The trained model and preprocessing pipeline are serialized as a single artifact and served via FastAPI. The `/predict` endpoint accepts 30+ input features and returns the predicted value with an 80% confidence interval. A Streamlit application provides an interactive demo. A standalone Streamlit app (`app.py`) loads the model directly for HuggingFace Spaces deployment. A Dockerfile is provided for containerized deployment.

**Artifacts:** `app/api/main.py`, `app/frontend/streamlit_app.py`, `app.py`, `Dockerfile`

---

## Module 7: Monitoring

Two monitoring mechanisms track model health: PSI-based drift detection per feature (threshold: 0.25) and Evidently AI dashboards generating three interactive HTML reports (data drift, regression performance, combined). Retraining is triggered when any feature PSI exceeds 0.25 or rolling 7-day WAPE exceeds 35%.

**Artifacts:** `monitoring/drift_monitor.py`, `monitoring/evidently_dashboard.py`, `monitoring/evidently_*.html`, `monitoring/monitoring_spec.md`

---

## Module 8: Business Recommendations and Reporting

Five actionable recommendations are derived from the model's feature importances and EDA findings: loyalty tier upgrade programs, email channel investment, desktop experience optimization, real-time cart upsell triggers, and income-tier personalization. The final report and presentation slides are included as PDF deliverables.

**Artifacts:** `docs/marketing_recommendations.md`, `docs/final_report.pdf`, `docs/slides.pdf`

---

## Cross-cutting

- **Reproducibility:** `run_pipeline.sh` executes the full pipeline from data generation through model training
- **Testing:** 43 pytest tests covering synthetic data, cleaning, features, model, and API
- **Code quality:** Ruff lint and format checks pass across all 42 Python files
- **CI:** GitHub Actions workflow runs Ruff on every push to main
- **Board:** 46/46 items complete on [GitHub Projects #5](https://github.com/users/alitonia/projects/5)

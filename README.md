# Predicting Customer Order Value for an E-Commerce Platform

An end-to-end regression system that predicts the monetary value of an e-commerce order prior to checkout. Built on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and enriched with value-conditioned synthetic behavioral data.

**Best model:** XGBoost — R² = 0.79, MAE = R$48, WAPE = 29%, MedAPE = 22%, MAPE = 29%

**Key finding:** 13 of the 20 most important predictive features are synthetic behavioral features, validating the value-conditioned data generation design.

---

## Overview

Given a customer's demographic profile, their current browsing and cart behavior, and the product mix in their cart, this system estimates the expected order value in Brazilian Reais and identifies the factors that drive that value up or down.

The Olist dataset provides 99,441 real orders with transactional detail but no behavioral data — no session duration, clickstream events, customer demographics, or loyalty information. Three synthetic tables (customer profiles, sessions, and clickstream activity) were generated to fill this gap. Each synthetic feature is conditioned on the real order value, ensuring that the behavioral data carries genuine predictive signal.

The complete pipeline — data generation, validation, cleaning, feature engineering, model training, deployment, and monitoring — executes with a single command.

---

## Quick Start

```bash
# Environment setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (generate → validate → clean → engineer → train)
bash run_pipeline.sh

# Prediction API
uvicorn app.api.main:app --reload --port 8000

# Interactive demo (separate terminal)
streamlit run app/frontend/streamlit_app.py

# Tests
python3 -m pytest tests/ -v
```

---

## Repository Structure

```
src/
├── data/               Data generation, validation, cleaning, EDA
├── features/           Feature engineering (182 features, VIF, encoding)
└── models/             Model training (6 models, time-based split, L1 loss)
app/
├── api/                FastAPI prediction service (/predict, /health)
├── frontend/           Streamlit interactive demo
└── app.py              Standalone Streamlit app for HuggingFace Spaces
monitoring/             PSI drift detection + Evidently AI dashboards
tests/                  43 pytest tests (data, features, model, API)
notebooks/              4 notebooks (data gen, EDA, features, modeling)
notebooks/plots/        9 EDA visualizations
data/
├── synthetic/          Generated parquet files (committed)
└── processed/          Cleaned data, feature matrix, preprocessor (committed)
models/                 Serialized XGBoost artifact
docs/                   Report, slides, data dictionary, schema, KPIs
scripts/                Packaging script for deliverables zip
run_pipeline.sh         One-command end-to-end pipeline
Dockerfile              Container image for cloud deployment
```

---

## Data Pipeline

### Synthetic Data

Three tables extend the Olist schema with behavioral data conditioned on real order value:

| Table | Rows | Description |
|---|---|---|
| customer_profile | 96,096 | Demographics, income, loyalty tier, device preference |
| sessions | 99,441 | Device, browser, traffic source, session timing, coupon usage |
| session_activity | 763,850 | Clickstream events (page views, searches, cart additions) |

Correlation design: monthly income r = 0.51 with average order value; loyalty tier produces a 2.5× AOV spread (bronze R$82 to platinum R$202); email traffic yields 64% higher median order value than Google organic.

### Feature Engineering

182 features across five groups: 131 one-hot encoded categoricals, 47 scaled numericals (including geolocation, seller reputation, RFM temporal, and category price statistics), 3 target-encoded high-cardinality features, 16 derived interaction features, and 3 boolean flags.

### Modeling

Time-based train/test split at 2018-04-01 (65,124 train / 32,460 test). XGBoost with L1 loss selected as best model. Linear models included for comparison; tree-based models with L1 loss handle the unwinsorized target distribution more effectively.

---

## Deployment

**FastAPI** exposes `/predict` (POST, returns predicted value + 80% CI) and `/health` (GET). The preprocessing pipeline (scaler, encoders, column alignment) is serialized alongside the model for exact reproduction of training-time transforms.

**Streamlit** provides an interactive demo with sidebar controls for all input features and real-time business insights.

**Docker** image available via the included Dockerfile. **HuggingFace Spaces** deployment supported via the standalone `app.py`.

---

## Monitoring

- **PSI-based drift detection** per feature (threshold: PSI > 0.25)
- **Evidently AI dashboards** — three interactive HTML reports (data drift, regression performance, combined)
- **Retraining triggers:** feature PSI > 0.25 or rolling 7-day WAPE > 35%

---

## Documentation

| Document | Description |
|---|---|
| `docs/final_report.pdf` | Complete project report (Centralized) |
| `docs/slides.pdf` | Presentation slides (PDF format) |
| `docs/slides.pptx` | Presentation slides (Editable PPTX format) |
| `docs/data_dictionary.md` | Every synthetic field: type, range, generation logic |
| `docs/synthetic_data_schema.md` | ERD-aligned schema with causal design rationale |
| `docs/demo_walkthrough.md` | Step-by-step guide for running the demo |

---

## Project Board

Tracked at [GitHub Projects #5](https://github.com/users/alitonia/projects/5). 46/46 items complete.

---

## Releases

Complete deliverables package (92MB zip, 78 files) available at [v1.7.0](https://github.com/alitonia/customer_value_prediction/releases/tag/v1.7.0).

Regenerate locally: `bash scripts/package_deliverables.sh`

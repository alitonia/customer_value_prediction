# Customer Order Value Prediction

End-to-end regression system that predicts the monetary value of an e-commerce order before checkout, built on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) enriched with value-conditioned synthetic behavioral data.

**Business question:** Given a customer's profile and current cart/session behavior, what is the expected value of this order, and which factors drive that value up or down?

**Best model:** XGBoost — R²=0.70, MAE=R$43, WAPE=30%, MedAPE=27%
**Key finding:** 13/20 top predictive features are synthetic behavioral, validating the value-conditioned generation design.

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (generate → validate → clean → engineer → train)
bash run_pipeline.sh

# Start API
uvicorn app.api.main:app --reload --port 8000

# Start demo UI (in another terminal)
streamlit run app/frontend/streamlit_app.py

# Run tests
python3 -m pytest tests/ -v
```

## Repository Structure

```
data/
├── raw/               # Olist CSVs (9 files, git-ignored)
├── synthetic/         # customer_profile, sessions, session_activity (parquet, committed)
└── processed/         # modeling_dataset, features, target, preprocessor (committed)
docs/                  # KPIs, schema, data dictionary, marketing recs, project overview
notebooks/             # 01_data_gen, 02_eda, 03_features, 04_modeling
notebooks/plots/       # 9 EDA visualizations
src/
├── data/              # generate_synthetic, validate_synthetic, clean, eda
├── features/          # engineer (180 features, VIF, encoding, scaling)
└── models/            # train (6 models, time-based split, L1 loss)
app/
├── api/               # FastAPI /predict + /health endpoints
└── frontend/          # Streamlit interactive demo
monitoring/            # PSI-based drift detection + retraining triggers
tests/                 # 43 pytest tests (synthetic, cleaning, features, model, API)
models/                # best_model.joblib (XGBoost artifact)
run_pipeline.sh        # One-command end-to-end pipeline
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy, pyarrow (parquet) |
| ML | scikit-learn, XGBoost (L1 loss), LightGBM (MAE loss) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Monitoring | PSI-based drift detection |
| Viz | matplotlib, seaborn |
| Testing | pytest, httpx (FastAPI TestClient) |
| CI | GitHub Actions (Ruff lint + format) |
| Linting | Ruff |

## Data Pipeline

### Synthetic Data (value-conditioned)
Three tables extend the Olist schema with realistic behavioral data. Features are **conditioned on real order value** so the model learns genuine patterns:

| Table | Rows | Key Correlations |
|---|---|---|
| customer_profile | 96,096 | income r=0.51, loyalty 2.5× AOV spread |
| sessions | 99,441 | desktop +7% AOV, email +64% vs google |
| session_activity | 763,850 | cart qty r=0.57, product views r=0.49 |

### Feature Engineering (V2)
180 features from 6 groups:
- **One-hot encoded** (131): device, browser, OS, traffic, loyalty, category, seller_state
- **Scaled numerical** (40+): income, session metrics, geolocation, seller stats, RFM, product aggregates
- **Target encoded** (3): ip_region, campaign_name, seller_state
- **Derived** (17): income×loyalty, cart_conversion, engagement_rate, distance_per_item, etc.
- **Boolean** (2): is_logged_in, is_marketing_opt_in

V2 additions: haversine distance customer↔seller, urban/rural, regional price index, seller reputation, RFM temporal features, category price aggregates.

### Modeling (V2)
- **Time-based split**: train on orders before 2018-04-01, test on orders after (no temporal leakage)
- **L1 loss**: XGBoost `reg:absoluteerror` + LightGBM `mae` (robust to outliers without winsorization)
- **No winsorization**: raw order value preserved (max R$13,664)

## Deliverables

| Deliverable | Location | Status |
|---|---|---|
| Source code | `src/`, `app/`, `monitoring/` | ✅ |
| Data dictionary | `docs/data_dictionary.md` | ✅ |
| Synthetic data schema | `docs/synthetic_data_schema.md` | ✅ |
| KPI & business value spec | `docs/kpis_and_business_value.md` | ✅ |
| Marketing recommendations | `docs/marketing_recommendations.md` | ✅ |
| Project overview | `docs/project_overview.md` | ✅ |
| EDA notebook + 9 plots | `notebooks/02_eda.ipynb`, `notebooks/plots/` | ✅ |
| FastAPI + Streamlit app | `app/` | ✅ |
| Monitoring | `monitoring/` | ✅ |
| Test suite (43 tests) | `tests/` | ✅ |
| Pipeline script | `run_pipeline.sh` | ✅ |
| Final report (PDF) | `docs/final_report.pdf` | 🔲 See `docs/HANDS_OFF_GUIDE.md` |
| Presentation slides | `docs/slides.pptx` | 🔲 See `docs/HANDS_OFF_GUIDE.md` |
| Cloud deployment | Render / HuggingFace | 🔲 See `docs/HANDS_OFF_GUIDE.md` |

## Project Board

Tracked at [GitHub Projects #5](https://github.com/users/alitonia/projects/5). 40/46 items Done.

## KPI Targets vs Actual

| KPI | Target | V1 Actual | Notes |
|---|---|---|---|
| MAE | < R$25 | R$43 | Right-skewed target; V2 with L1 loss may improve |
| WAPE | < 16% | 30% | No product-level pricing features |
| MedAPE | < 12% | 27% | Synthetic data limitations |

See `docs/HANDS_OFF_GUIDE.md` §7 for KPI revision plan.

# Customer Order Value Prediction

End-to-end regression system that predicts the monetary value of an e-commerce order before checkout, built on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) enriched with synthetic behavioral session data.

**Business question:** Given a customer's profile and current cart/session behavior, what is the expected value of this order, and which factors drive that value up or down?

## Deliverables

| Deliverable | Location |
|---|---|
| Source code | This repository |
| Data dictionary | `docs/synthetic_data_schema.md` |
| KPI & business value spec | `docs/kpis_and_business_value.md` |
| Deployed API + demo | `app/` (FastAPI + Streamlit) |
| Monitoring dashboard | `monitoring/` (Evidently) |
| Final report & slides | `docs/` (PDF + PPTX) |

## Repository Structure

```
data/
├── raw/            # Olist CSVs (9 files, git-ignored)
├── synthetic/      # Generated: customer_profile.csv, sessions.csv, session_activity.csv
└── processed/      # Cleaned & merged data ready for modeling
docs/               # KPIs, schema, data dictionary, project announcement PDF
notebooks/          # EDA, cleaning, feature engineering, modeling notebooks
src/
├── data/           # Data loading & cleaning scripts
├── features/       # Feature engineering scripts
└── models/         # Model training & evaluation scripts
app/
├── api/            # FastAPI backend
└── frontend/       # Streamlit demo
monitoring/         # Evidently drift & performance dashboards
plan/               # Project plan & team roles
```

## Tech Stack

Python 3.10+, pandas, numpy, scikit-learn, XGBoost, LightGBM, FastAPI, Streamlit, Evidently AI, Faker, matplotlib, seaborn.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Olist dataset into `data/raw/` (Kaggle CLI or manual download).

## Project Board

Tracked at [GitHub Projects #5](https://github.com/users/alitonia/projects/5).

## KPI Targets

- MAE < R$25
- WAPE < 16%
- MedAPE < 12%

See `docs/kpis_and_business_value.md` for full specification.

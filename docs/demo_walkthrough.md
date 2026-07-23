# Demo Walkthrough — Recording Guide

This document provides a structured script for a 3–5 minute live or recorded demonstration of the order value prediction system. The demonstration covers pipeline reproducibility, model validation, API functionality, and the interactive demo interface.

---

## Segment 1: Pipeline Reproducibility (60 seconds)

Demonstrate that the entire system can be reproduced from scratch with a single command.

```bash
# Remove generated artifacts to show clean state
rm -rf data/synthetic/*.parquet data/processed/*.parquet models/*.joblib

# Execute full pipeline
bash run_pipeline.sh

# Verify outputs
ls -lh data/synthetic/*.parquet
ls -lh data/processed/*.parquet
ls -lh models/best_model.joblib
```

**Narration:** "The pipeline executes five stages sequentially: synthetic data generation, validation, cleaning and merging, feature engineering, and model training. All outputs — three synthetic parquet files, the cleaned modeling dataset, the 182-feature matrix, and the serialized XGBoost model — are produced deterministically from the Olist raw data and a fixed random seed."

---

## Segment 2: Data Validation (30 seconds)

```bash
python3 -m src.data.validate_synthetic
```

**Narration:** "The validation suite runs 16 checks covering row counts, foreign key integrity, cart consistency, temporal ordering, and the value-conditioning correlations. All 16 checks pass, confirming that the synthetic data meets the design specifications documented in the data dictionary."

---

## Segment 3: Test Suite (30 seconds)

```bash
python3 -m pytest tests/ -v --tb=short
```

**Narration:** "The test suite contains 43 automated tests across five modules: synthetic data structure, cleaning outputs, feature matrix integrity, model loading and prediction, and API endpoint behavior. All tests pass."

---

## Segment 4: API Demonstration (60 seconds)

```bash
# Terminal 1: Start the prediction service
uvicorn app.api.main:app --port 8000

# Terminal 2: Health check
curl -s localhost:8000/health | python3 -m json.tool

# Low-value profile prediction
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_income":1500,"loyalty_tier":"bronze","item_count":1,
       "total_cart_qty":1,"n_product_views":2,"session_duration_min":2,
       "device_type":"mobile","traffic_source":"google"}' | python3 -m json.tool

# High-value profile prediction
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_income":15000,"loyalty_tier":"platinum","item_count":5,
       "total_cart_qty":8,"n_product_views":15,"session_duration_min":20,
       "device_type":"desktop","traffic_source":"email"}' | python3 -m json.tool
```

**Narration:** "The API returns a predicted order value with an 80% confidence interval. The low-value profile — a bronze-tier mobile visitor with a single item — yields approximately R$83. The high-value profile — a platinum-tier desktop customer with five items — yields approximately R$520. The 6.3× spread demonstrates that the model differentiates between customer types based on the input features."

---

## Segment 5: Interactive Demo (60 seconds)

```bash
streamlit run app/frontend/streamlit_app.py
```

**Narration:** "The Streamlit application provides an interactive interface for the prediction model. The sidebar contains controls for customer profile parameters, session context, and cart contents. Adjusting the sliders updates the prediction in real time. The application also surfaces contextual business insights — for example, flagging high-value orders for premium shipping consideration or identifying loyal customers for priority support."

Demonstrate by moving from a low-value configuration to a high-value configuration and observing the prediction change.

---

## Segment 6: Monitoring (30 seconds)

```bash
python3 -m monitoring.drift_monitor
cat monitoring/drift_report.json | python3 -m json.tool
```

**Narration:** "The monitoring module computes the Population Stability Index for each feature, comparing the training distribution against incoming data. Features with PSI exceeding 0.25 trigger drift alerts. The Evidently AI dashboards — three interactive HTML reports — provide visual drift analysis and regression performance tracking. Retraining is triggered when drift or performance degradation exceeds defined thresholds."

---

## Key Talking Points

1. The synthetic behavioral data is value-conditioned — each feature is generated with a designed correlation to the real order value, not drawn from independent random distributions.
2. Thirteen of the twenty most important features identified by XGBoost are synthetic behavioral features, confirming that the simulation design produces data the model can learn from.
3. The time-based train/test split prevents temporal leakage and reflects the actual deployment scenario.
4. L1 loss handles the unwinsorized target distribution without capping high-value orders.
5. The complete system — from raw data to deployed model — reproduces with a single command.

---

## Recording Tools

- Terminal recording: `asciinema rec demo.cast` (upload to asciinema.org)
- Screen recording: OBS Studio or `recordmydesktop`
- GIF conversion: `ffmpeg -i demo.mp4 -vf "fps=10,scale=960:-1" demo.gif`

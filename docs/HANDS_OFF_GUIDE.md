# Hands-Off Guide — Remaining Work

Everything below can be run in a fresh session. The code changes are already committed to the working tree (not yet pushed).

---

## 1. Run the V2 Pipeline

The V2 pipeline incorporates 6 improvements: no winsorization, geolocation features, seller features, RFM temporal features, product aggregates, time-based split + L1 loss.

```bash
cd /mnt/data/customer_value_prediction
source .venv/bin/activate

# Step 1: Clean (adds geo, seller, RFM, product aggregates; removes winsorization)
python3 -m src.data.clean

# Step 2: Feature engineering (180+ features with new columns)
python3 -m src.features.engineer

# Step 3: Train (time-based split at 2018-04-01, XGBoost + LightGBM with L1 loss)
python3 -m src.models.train

# Step 4: Validate
python3 -m src.data.validate_synthetic

# Step 5: Run tests
python3 -m pytest tests/ -v

# Step 6: Format + lint
ruff format src/ app/ monitoring/ tests/
ruff check src/ app/ monitoring/ tests/
```

**If training fails with inf/NaN:** The `reg:absoluteerror` objective in XGBoost can be slow. If it times out, reduce `n_estimators` to 100 in `src/models/train.py` line ~138.

**If tests fail:** The test `test_english_categories` checks for `primary_category` column. The test `test_no_target_leakage` checks that `order_value_winsorized` is NOT in features (it shouldn't be in V2). Update tests if column names changed.

---

## 2. Commit & Push V2 Changes

```bash
git add -A
git commit -m "feat: V2 pipeline — geo/seller/RFM features, no winsorization, time-based split, L1 loss"
git push origin main
```

---

## 3. Final Report Outline (Task 8.2 — PDF)

**File:** `docs/final_report.pdf` (generate from markdown via pandoc or write in LaTeX/Word)

```
1. Executive Summary (1 page)
   - Business question, approach, key result (R², MAE, WAPE)
   - Top 3 findings (synthetic features drive predictions, loyalty tier 2.5× spread, email channel premium)

2. Business Understanding (1 page)
   - Olist dataset overview
   - Why synthetic behavioral data was needed
   - KPI targets vs actual achieved

3. Data Strategy (2 pages)
   - ERD diagram (Olist + 3 synthetic tables)
   - Value-conditioned generation design (the "why" behind the correlations)
   - Data dictionary summary (reference appendix)
   - Validation results (16/16 checks)

4. Exploratory Data Analysis (2 pages)
   - Target distribution (skew, log-transform rationale)
   - Key drivers (category, device, traffic, loyalty, income)
   - Correlation heatmap (no multicollinearity, leakage flag)
   - Include 4-5 key plots from notebooks/plots/

5. Data Cleaning & Feature Engineering (1 page)
   - Cleaning steps summary (before/after table from cleaning_log.md)
   - Feature groups (180 features: one-hot, scaled, target-encoded, derived)
   - V2 additions: geolocation, seller, RFM, product aggregates
   - VIF results

6. Modeling (2 pages)
   - Model comparison table (6 models, all metrics)
   - Time-based split rationale (no temporal leakage)
   - L1 loss rationale (robust to outliers without winsorization)
   - Feature importance plot (13/20 synthetic)
   - Residual analysis plot

7. Deployment & Monitoring (1 page)
   - API architecture (FastAPI + Streamlit)
   - Screenshot of Streamlit demo
   - Monitoring spec (PSI thresholds, retraining triggers)

8. Business Recommendations (1 page)
   - 7 recommendations from marketing_recommendations.md
   - Prioritized by expected impact

9. Limitations & Future Work (0.5 page)
   - Synthetic data limitations
   - KPI gap explanation
   - Sequence modeling for clickstream
   - Real-time feature store

10. References & Appendix
    - Olist dataset citation
    - Tool versions
    - Full feature catalog
```

**To generate PDF from markdown:**
```bash
# Install pandoc if needed
sudo apt install pandoc texlive-xetex
# Convert
pandoc docs/final_report.md -o docs/final_report.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

---

## 4. Presentation Slides Outline (Task 8.3 — PPTX, 10-12 min)

**File:** `docs/slides.pptx` (use python-pptx or Google Slides)

```
Slide 1:  Title — "Predicting Customer Order Value for E-Commerce"
Slide 2:  Business Question — "What drives order value? Can we predict it before checkout?"
Slide 3:  Data Overview — Olist (99K orders) + 3 synthetic tables (ERD diagram)
Slide 4:  Synthetic Data Design — Value-conditioned generation (the key innovation)
Slide 5:  EDA Highlights — Target distribution + top 3 driver plots
Slide 6:  Feature Engineering — 180 features, 4 groups, V2 additions
Slide 7:  Model Results — Comparison table, XGBoost wins
Slide 8:  Feature Importances — 13/20 synthetic (the "aha" moment)
Slide 9:  Live Demo — Streamlit screenshot or live walkthrough
Slide 10: Business Recommendations — Top 3 actionable insights
Slide 11: Limitations & Future Work
Slide 12: Thank You / Q&A
```

**To generate PPTX programmatically:**
```bash
pip install python-pptx
python3 scratch/make_slides.py  # write this script
```

---

## 5. Cloud Deployment (Task 6.4)

**Option A: Render (easiest)**
```yaml
# render.yaml
services:
  - type: web
    name: order-value-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.12
```

**Option B: HuggingFace Spaces (for Streamlit)**
```yaml
# .huggingface.yaml or just create a Space with Streamlit SDK
# app.py = app/frontend/streamlit_app.py
# requirements.txt already exists
```

**Option C: Dockerfile (Task 6.5)**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY models/ models/
COPY data/processed/preprocessor.joblib data/processed/
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. Demo Walkthrough (Task 8.4)

Record a 3-5 minute screen recording showing:
1. `bash run_pipeline.sh` — full pipeline from scratch
2. `uvicorn app.api.main:app` — start API
3. `curl localhost:8000/predict` — test with low and high value profiles
4. `streamlit run app/frontend/streamlit_app.py` — interactive demo
5. Show monitoring: `python3 -m monitoring.drift_monitor`

Tools: `asciinema` for terminal recording, or `OBS Studio` for full screen.

---

## 7. KPI Revision (Task 8.5)

Update `docs/kpis_and_business_value.md`:
- Add a "V2 Actual Results" section alongside original targets
- Explain the gap: right-skewed target, synthetic data limitations, no product-level pricing
- Revise operational alert thresholds: rolling WAPE trigger at 35% (was 18%), PSI threshold stays at 0.25
- Add model limitations section

---

## 8. Board Updates After Completion

After each deliverable is done:
```bash
# Close the issue
gh issue close <number> --comment "Completed: <description>"

# Update board status to Done
gh project item-edit --id <item-id> --field-id PVTSSF_lAHOAyXyjs4Bcbe_zhXEY0U --single-select-option-id 98236657 --project-id PVT_kwHOAyXyjs4Bcbe_
```

Remaining open issues: #24, #28, #40, #41, #45, #46

---

## Quick Reference: File Locations

| Deliverable | Path |
|---|---|
| Synthetic data | `data/synthetic/*.parquet` |
| Cleaned dataset | `data/processed/modeling_dataset.parquet` |
| Feature matrix | `data/processed/features.parquet` |
| Model artifact | `models/best_model.joblib` |
| Preprocessor | `data/processed/preprocessor.joblib` |
| EDA plots | `notebooks/plots/*.png` |
| Data dictionary | `docs/data_dictionary.md` |
| Marketing recs | `docs/marketing_recommendations.md` |
| Project overview | `docs/project_overview.md` |
| Pipeline script | `run_pipeline.sh` |
| Tests | `tests/` (43 tests) |

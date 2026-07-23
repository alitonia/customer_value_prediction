#!/usr/bin/env bash
# End-to-end pipeline orchestration (Task 9.1)
# Runs: generate → validate → clean → engineer → train
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Step 1/5: Generate synthetic data ==="
python3 -m src.data.generate_synthetic --seed 42

echo "=== Step 2/5: Validate synthetic data ==="
python3 -m src.data.validate_synthetic

echo "=== Step 3/5: Clean & merge ==="
python3 -m src.data.clean

echo "=== Step 4/5: Feature engineering ==="
python3 -m src.features.engineer

echo "=== Step 5/5: Train & evaluate models ==="
python3 -m src.models.train

echo ""
echo "=== Pipeline complete ==="
echo "Outputs:"
echo "  data/synthetic/     — customer_profile, sessions, session_activity (parquet)"
echo "  data/processed/     — modeling_dataset, features, target, preprocessor"
echo "  models/             — best_model.joblib"
echo "  notebooks/plots/    — EDA visualizations"
echo ""
echo "To start the API:  uvicorn app.api.main:app --reload --port 8000"
echo "To start the UI:  streamlit run app/frontend/streamlit_app.py"

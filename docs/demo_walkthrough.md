# Demo Walkthrough Script (Task 8.4)

## Recording Plan (3-5 minutes)

### Segment 1: Pipeline Reproducibility (1 min)
```bash
# Show clean state
rm -rf data/synthetic/*.parquet data/processed/*.parquet models/*.joblib

# Run full pipeline
bash run_pipeline.sh

# Show outputs
ls -lh data/synthetic/*.parquet
ls -lh data/processed/*.parquet
ls -lh models/best_model.joblib
```

### Segment 2: Validation (30 sec)
```bash
python3 -m src.data.validate_synthetic
# Show 16/16 PASS
```

### Segment 3: Tests (30 sec)
```bash
python3 -m pytest tests/ -v --tb=short
# Show 43 passed
```

### Segment 4: API Demo (1 min)
```bash
# Terminal 1: Start API
uvicorn app.api.main:app --port 8000

# Terminal 2: Test predictions
curl -s localhost:8000/health | python3 -m json.tool

# Low-value profile
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_income":1500,"loyalty_tier":"bronze","item_count":1,"total_cart_qty":1,"n_product_views":2,"session_duration_min":2,"device_type":"mobile","traffic_source":"google"}' | python3 -m json.tool
# Expected: ~R$83

# High-value profile
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_income":15000,"loyalty_tier":"platinum","item_count":5,"total_cart_qty":8,"n_product_views":15,"session_duration_min":20,"device_type":"desktop","traffic_source":"email"}' | python3 -m json.tool
# Expected: ~R$520
```

### Segment 5: Streamlit Demo (1 min)
```bash
streamlit run app/frontend/streamlit_app.py
# Show interactive sidebar
# Adjust sliders: low profile → high profile
# Show prediction changes in real-time
# Show business insights panel
```

### Segment 6: Monitoring (30 sec)
```bash
python3 -m monitoring.drift_monitor
# Show drift report
cat monitoring/drift_report.json | python3 -m json.tool
```

## Recording Tools
- **Terminal recording:** `asciinema rec demo.cast` → upload to asciinema.org
- **Screen recording:** OBS Studio or `recordmydesktop`
- **GIF from video:** `ffmpeg -i demo.mp4 -vf "fps=10,scale=960:-1" demo.gif`

## Key Talking Points
1. "The synthetic data is value-conditioned — not random noise"
2. "13 out of 20 top features are behavioral — the simulation works"
3. "Time-based split prevents temporal leakage"
4. "L1 loss handles outliers without destroying high-value signal"
5. "One command reproduces everything: bash run_pipeline.sh"

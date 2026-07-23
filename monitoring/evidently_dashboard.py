"""Evidently AI monitoring dashboard (Task 7.2).

Generates interactive HTML reports showing data drift and regression
performance using the Evidently library.

Usage:
    python -m monitoring.evidently_dashboard
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, RegressionPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MONITORING_DIR = PROJECT_ROOT / "monitoring"


def main():
    log.info("Loading data...")
    X = pd.read_parquet(PROCESSED / "features.parquet")
    target = pd.read_parquet(PROCESSED / "target.parquet")
    modeling = pd.read_parquet(PROCESSED / "modeling_dataset.parquet")

    with open(MODELS_DIR / "best_model.joblib", "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]

    y_pred_log = model.predict(X)
    y_pred_brl = np.expm1(y_pred_log)
    y_true_brl = np.expm1(target["target_log"].values)

    dates = modeling["order_purchase_timestamp"]
    split_date = pd.Timestamp("2018-04-01")
    train_mask = (dates < split_date).values
    test_mask = (dates >= split_date).values

    # Top 15 features for drift report
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )
        top_features = imp.head(15).index.tolist()
    else:
        top_features = X.columns[:15].tolist()

    ref_df = X.loc[train_mask, top_features].copy()
    ref_df["target"] = y_true_brl[train_mask]
    ref_df["prediction"] = y_pred_brl[train_mask]

    cur_df = X.loc[test_mask, top_features].copy()
    cur_df["target"] = y_true_brl[test_mask]
    cur_df["prediction"] = y_pred_brl[test_mask]

    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=top_features,
    )

    # --- Report 1: Data Drift ---
    log.info("Generating data drift report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=ref_df[top_features],
        current_data=cur_df[top_features],
        column_mapping=ColumnMapping(numerical_features=top_features),
    )
    drift_report.save(str(MONITORING_DIR / "evidently_drift_report.html"))
    log.info("Saved evidently_drift_report.html")

    # --- Report 2: Regression Performance ---
    log.info("Generating regression performance report...")
    perf_report = Report(metrics=[RegressionPreset()])
    perf_report.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=column_mapping,
    )
    perf_report.save(str(MONITORING_DIR / "evidently_performance_report.html"))
    log.info("Saved evidently_performance_report.html")

    # --- Report 3: Combined dashboard ---
    log.info("Generating combined dashboard...")
    dashboard = Report(metrics=[RegressionPreset(), DataDriftPreset()])
    dashboard.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=column_mapping,
    )
    dashboard.save(str(MONITORING_DIR / "evidently_dashboard.html"))
    log.info("Saved evidently_dashboard.html")

    log.info("=== Evidently reports complete ===")


if __name__ == "__main__":
    main()

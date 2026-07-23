"""Tests for model loading and prediction."""

import numpy as np


class TestModel:
    def test_model_loads(self, model_artifact):
        assert "model" in model_artifact
        assert "feature_columns" in model_artifact
        assert "metrics" in model_artifact

    def test_model_name(self, model_artifact):
        assert model_artifact["model_name"] == "XGBoost"

    def test_feature_columns_match(self, model_artifact, features):
        model_cols = set(model_artifact["feature_columns"])
        data_cols = set(features.columns)
        assert model_cols == data_cols, (
            f"Missing: {model_cols - data_cols}, Extra: {data_cols - model_cols}"
        )

    def test_predict_runs(self, model_artifact, features):
        model = model_artifact["model"]
        X = features.head(10)
        preds = model.predict(X)
        assert len(preds) == 10
        assert all(np.isfinite(preds))

    def test_metrics_in_range(self, model_artifact):
        m = model_artifact["metrics"]
        assert 0.5 < m["R2_log"] < 1.0, f"R² out of range: {m['R2_log']}"
        assert m["MAE_BRL"] < 100, f"MAE too high: {m['MAE_BRL']}"
        assert m["WAPE_%"] < 50, f"WAPE too high: {m['WAPE_%']}"

    def test_predictions_positive_in_original_space(self, model_artifact, features):
        model = model_artifact["model"]
        preds_log = model.predict(features.head(100))
        preds_brl = np.expm1(preds_log)
        assert (preds_brl > 0).all()

"""Tests for feature engineering outputs."""

import numpy as np


class TestFeatureMatrix:
    def test_feature_count(self, features):
        assert features.shape[1] >= 100  # expect 159 but allow some flexibility

    def test_row_count(self, features):
        assert len(features) > 90000

    def test_no_target_leakage(self, features):
        """Target-derived columns must not be in features."""
        leakage = [
            "order_value",
            "order_value_winsorized",
            "freight_value",
            "avg_item_price",
            "value_per_item",
            "value_per_minute",
        ]
        for col in leakage:
            assert col not in features.columns, f"Leakage column found: {col}"

    def test_no_nulls(self, features):
        null_counts = features.isnull().sum()
        assert null_counts.sum() == 0, (
            f"Nulls in: {null_counts[null_counts > 0].to_dict()}"
        )

    def test_no_infinite_values(self, features):
        numeric = features.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_scaler_fitted(self, preprocessor):
        assert "scaler" in preprocessor
        assert hasattr(preprocessor["scaler"], "mean_")

    def test_target_encoding_present(self, preprocessor):
        assert "target_encoding" in preprocessor
        assert len(preprocessor["target_encoding"]) > 0


class TestTarget:
    def test_target_log_skew_low(self, target):
        """Log-transformed target should have low skewness."""
        skew = target["target_log"].skew()
        assert abs(skew) < 1.0, f"Target skew too high: {skew:.2f}"

    def test_target_positive(self, target):
        assert (target["target"] > 0).all()

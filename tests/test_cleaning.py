"""Tests for the cleaning pipeline outputs."""

import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


class TestModelingDataset:
    @staticmethod
    def _load():
        return pd.read_parquet(PROCESSED / "modeling_dataset.parquet")

    def test_row_count(self):
        df = self._load()
        assert len(df) > 90000

    def test_no_canceled_orders(self):
        df = self._load()
        assert "canceled" not in df["order_status"].values

    def test_order_value_positive(self):
        df = self._load()
        assert (df["order_value_winsorized"] > 0).all()

    def test_english_categories(self):
        df = self._load()
        # primary_category contains the English category name
        assert "primary_category" in df.columns
        assert df["primary_category"].notna().all()

    def test_synthetic_columns_present(self):
        df = self._load()
        expected = ["monthly_income", "loyalty_tier", "device_type",
                     "session_duration_min", "total_cart_qty"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

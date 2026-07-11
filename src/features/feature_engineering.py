

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from common import setup_logger, save_object, load_object

logger = setup_logger("feature_engineering")


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.merged_dir = config.paths.MERGED_DIR
        self.model_dir = config.paths.SAVED_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Chỉ lưu danh sách để tham khảo, không encode ở đây
        self.categorical_columns = ["device_type", "referral_channel"]

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        self.duration_threshold = None
        self.pages_threshold = None

    # ==========================================================
    # FIT
    # ==========================================================
    def fit(self, X: pd.DataFrame, y=None):

        logger.info("Fitting Feature Engineer...")

        self.duration_threshold = X["session_duration_seconds"].quantile(0.75)
        self.pages_threshold = X["pages_viewed"].quantile(0.75)

        # Fit encoder (dùng sau trong train.py)
        if all(col in X.columns for col in self.categorical_columns):
            self.encoder.fit(X[self.categorical_columns])

        logger.info("Feature Engineer fitted successfully.")
        return self

    # ==========================================================
    # TRANSFORM (Chỉ tạo feature, KHÔNG encode)
    # ==========================================================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        logger.info("Transforming features...")

        df = X.copy()

        def safe_divide(numerator, denominator):
            return np.where(denominator > 0, numerator / denominator, 0)

        # === Numerical + Interaction Features ===
        df["engagement_score"] = (
            df["session_duration_seconds"] * 0.4 +
            df["pages_viewed"] * 0.6
        )

        df["cart_efficiency"] = safe_divide(df["cart_additions"], df["session_duration_seconds"])
        df["session_intensity"] = safe_divide(df["pages_viewed"], df["session_duration_seconds"])
        df["add_to_cart_ratio"] = safe_divide(df["cart_additions"], df["pages_viewed"])
        df["pages_per_cart"] = safe_divide(df["pages_viewed"], df["cart_additions"])
        df["duration_per_cart"] = safe_divide(df["session_duration_seconds"], df["cart_additions"])
        df["discount_per_page"] = safe_divide(df["discount_amount_pct"], df["pages_viewed"])

        if "order_value" in df.columns:
            df["estimated_discount_value"] = df["order_value"] * df["discount_amount_pct"] / 100
        else:
            df["estimated_discount_value"] = 0
        df["has_high_discount"] = (df["discount_amount_pct"] >= 15).astype(int)
        df["is_high_value_session"] = (
            (df["session_duration_seconds"] > self.duration_threshold) &
            (df["pages_viewed"] > self.pages_threshold)
        ).astype(int)

        # Interaction
        df["duration_x_cart"] = df["session_duration_seconds"] * df["cart_additions"]
        df["pages_x_discount"] = df["pages_viewed"] * df["discount_amount_pct"]
        df["cart_x_discount"] = df["cart_additions"] * df["discount_amount_pct"]
        df["engagement_x_discount"] = df["engagement_score"] * df["discount_amount_pct"]
        df["cart_abandon_score"] = df["pages_viewed"] - df["cart_additions"]

        # Tạo categorical mới (giữ nguyên để train.py xử lý)
        df["delivery_speed"] = pd.cut(
            df["delivery_days"],
            bins=[0, 3, 7, 14, np.inf],
            labels=["fast", "normal", "slow", "very_slow"]
        ).astype(str)

        df["discount_level"] = pd.cut(
            df["discount_amount_pct"],
            bins=[-1, 0, 5, 15, 100],
            labels=["none", "low", "medium", "high"]
        ).astype(str)

        # KHÔNG drop và KHÔNG encode device_type, referral_channel ở đây
        # Các cột này sẽ được giữ nguyên để train.py fit & transform

        # Final cleaning
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        logger.info(f"Feature Engineering completed. Shape: {df.shape}")
        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    # ==========================================================
    # RUN
    # ==========================================================
    def run(self):

        logger.info("=" * 60)
        logger.info("START FEATURE ENGINEERING (No Encoding)")
        logger.info("=" * 60)

        input_path = self.merged_dir / "final_dataset.csv"
        output_path = self.merged_dir / "modeling_dataset.csv"

        logger.info(f"Loading from: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

        df = pd.read_csv(input_path)
        logger.info(f"Original shape: {df.shape}")

        df = self.fit_transform(df)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved modeling_dataset.csv to: {output_path}")

        self.save()

        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETED")
        logger.info("=" * 60)

        return df

    def save(self):
        save_object(self, self.model_dir / "feature_engineer.pkl")
        logger.info("Feature Engineer saved successfully.")

    @classmethod
    def load(cls):
        logger.info("Loading Feature Engineer...")
        return load_object(config.paths.SAVED_MODEL_DIR / "feature_engineer.pkl")


if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.run()
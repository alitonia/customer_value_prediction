import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from config import config
from common import setup_logger

logger = setup_logger("create_final_dataset")


class FinalDatasetCreator:
    def __init__(self):
        self.processed_dir = config.paths.PROCESSED_DIR
        self.merged_dir = config.paths.MERGED_DIR

    def create_final_dataset(self):
        logger.info("Đang tạo final dataset...")

        orders = pd.read_csv(self.processed_dir / config.data.CLEANED_ORDERS)
        sessions = pd.read_csv(
            self.processed_dir / config.data.CLEANED_BEHAVIORAL_SESSIONS
        )

        df = sessions.merge(
            orders[
                [
                    "order_id",
                    "order_value",
                    "payment_value",
                    "customer_city",
                    "customer_state",
                    "order_status",
                    "delivery_days",
                ]
            ],
            on="order_id",
            how="left",
        )

        df["order_value"] = df["order_value"].fillna(0)
        df["payment_value"] = df["payment_value"].fillna(0)
        df["delivery_days"] = df["delivery_days"].fillna(df["delivery_days"].median())

        # Feature engineering
        df["log_order_value"] = np.log1p(df["order_value"])
        df["avg_time_per_page"] = np.where(
            df["pages_viewed"] > 0,
            df["session_duration_seconds"] / df["pages_viewed"],
            0,
        )
        df["cart_per_page"] = np.where(
            df["pages_viewed"] > 0, df["cart_additions"] / df["pages_viewed"], 0
        )

        duration_median = df["session_duration_seconds"].median()
        pages_median = df["pages_viewed"].median()
        df["is_high_engagement"] = (
            (df["session_duration_seconds"] > duration_median)
            & (df["pages_viewed"] > pages_median)
        ).astype(int)

        df["has_coupon"] = df["coupon_applied"].astype(int)

        output_path = self.merged_dir / config.data.FINAL_DATASET
        df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu: {output_path}")

        return df

    def run(self):
        self.create_final_dataset()


if __name__ == "__main__":
    creator = FinalDatasetCreator()
    creator.run()

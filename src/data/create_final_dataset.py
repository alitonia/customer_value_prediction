import pandas as pd
import numpy as np
import os

class FinalDatasetCreator:
    def __init__(self, base_path=r"D:\DATA"):
        self.base_path = base_path
        self.processed_path = os.path.join(base_path, "data", "processed")
        self.merged_path = os.path.join(base_path, "data", "merged")
        os.makedirs(self.merged_path, exist_ok=True)

    def create_final_dataset(self):
        orders = pd.read_csv(os.path.join(self.processed_path, "cleaned_orders.csv"))
        sessions = pd.read_csv(os.path.join(self.processed_path, "cleaned_behavioral_sessions.csv"))

        df = sessions.merge(
            orders[['order_id', 'order_value', 'payment_value', 'customer_city',
                    'customer_state', 'order_status', 'delivery_days']],
            on='order_id', how='left'
        )

        # Xử lý missing
        df['order_value'] = df['order_value'].fillna(0)
        df['payment_value'] = df['payment_value'].fillna(0)
        df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())

        # Feature engineering
        df['log_order_value'] = np.log1p(df['order_value'])

        # Tránh chia cho 0
        df['avg_time_per_page'] = np.where(
            df['pages_viewed'] > 0,
            df['session_duration_seconds'] / df['pages_viewed'],
            0
        )
        df['cart_per_page'] = np.where(
            df['pages_viewed'] > 0,
            df['cart_additions'] / df['pages_viewed'],
            0
        )

        # High engagement flag
        duration_median = df['session_duration_seconds'].median()
        pages_median = df['pages_viewed'].median()
        df['is_high_engagement'] = (
            (df['session_duration_seconds'] > duration_median) &
            (df['pages_viewed'] > pages_median)
        ).astype(int)

        df['has_coupon'] = df['coupon_applied'].astype(int)

        # Lưu
        output_path = os.path.join(self.merged_path, "final_dataset.csv")
        df.to_csv(output_path, index=False)


        return df

    def run(self):
        self.create_final_dataset()


if __name__ == "__main__":
    creator = FinalDatasetCreator(base_path=r"D:\DATA")
    creator.run()
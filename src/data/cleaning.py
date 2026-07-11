import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import config
from common import setup_logger

logger = setup_logger("cleaning")


class DataCleaner:
    def __init__(self):
        self.raw_dir = config.paths.RAW_DIR
        self.synthetic_dir = config.paths.SYNTHETIC_DIR
        self.processed_dir = config.paths.PROCESSED_DIR

    def clean_orders(self):
        logger.info("Đang xử lý orders...")

        orders = pd.read_csv(self.raw_dir / "olist_orders_dataset.csv")
        customers = pd.read_csv(self.raw_dir / "olist_customers_dataset.csv")
        payments = pd.read_csv(self.raw_dir / "olist_order_payments_dataset.csv")
        order_items = pd.read_csv(self.raw_dir / "olist_order_items_dataset.csv")

        df = orders[orders['order_status'] == 'delivered'].copy()

        df = df.merge(
            customers[['customer_id', 'customer_city', 'customer_state']],
            on='customer_id', how='left'
        )

        payment_value = payments.groupby('order_id', as_index=False)['payment_value'].sum()
        df = df.merge(payment_value, on='order_id', how='left')

        order_value = order_items.groupby('order_id', as_index=False)['price'].sum()
        order_value.columns = ['order_id', 'order_value']
        df = df.merge(order_value, on='order_id', how='left')

        for col in ['order_purchase_timestamp', 'order_delivered_customer_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        df = df.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'])
        df['delivery_days'] = (df['order_delivered_customer_date'] -
                               df['order_purchase_timestamp']).dt.days
        df = df[(df['delivery_days'] >= 0) & (df['delivery_days'] <= 60)]

        df['order_value'] = df['order_value'].fillna(0)
        df['payment_value'] = df['payment_value'].fillna(0)

        output_path = self.processed_dir / config.data.CLEANED_ORDERS
        df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu: {output_path.name} ({len(df):,} dòng)")

        return df

    def clean_behavioral_sessions(self):
        logger.info("Đang xử lý behavioral_sessions...")

        df = pd.read_csv(self.synthetic_dir / config.data.BEHAVIORAL_SESSIONS)

        df['order_id'] = df['order_id'].astype(str)
        df['session_id'] = df['session_id'].astype(str)

        df = df[df['cart_additions'] >= 1]
        df = df[(df['session_duration_seconds'] >= 10) & (df['session_duration_seconds'] <= 3600)]
        df = df[(df['pages_viewed'] >= 1) & (df['pages_viewed'] <= 80)]

        output_path = self.processed_dir / config.data.CLEANED_BEHAVIORAL_SESSIONS
        df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu: {output_path.name} ({len(df):,} dòng)")

        return df

    def run(self):
        logger.info("=== BẮT ĐẦU CLEANING DATA ===")
        self.clean_orders()
        self.clean_behavioral_sessions()
        logger.info("=== HOÀN TẤT CLEANING ===")


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()
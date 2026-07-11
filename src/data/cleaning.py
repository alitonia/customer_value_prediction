import pandas as pd
import os

class DataCleaner:
    def __init__(self, base_path=r"D:\DATA"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "data", "raw")
        self.synthetic_path = os.path.join(base_path, "data", "synthetic")
        self.processed_path = os.path.join(base_path, "data", "processed")
        os.makedirs(self.processed_path, exist_ok=True)

    def clean_orders(self):
        print("Đang xử lý orders...")

        orders = pd.read_csv(os.path.join(self.raw_path, "olist_orders_dataset.csv"))
        customers = pd.read_csv(os.path.join(self.raw_path, "olist_customers_dataset.csv"))
        payments = pd.read_csv(os.path.join(self.raw_path, "olist_order_payments_dataset.csv"))
        order_items = pd.read_csv(os.path.join(self.raw_path, "olist_order_items_dataset.csv"))

        # Chỉ giữ order delivered
        df = orders[orders['order_status'] == 'delivered'].copy()

        # Merge customer info
        df = df.merge(
            customers[['customer_id', 'customer_city', 'customer_state']],
            on='customer_id', how='left'
        )

        # Tính tổng payment_value
        payment_value = payments.groupby('order_id', as_index=False)['payment_value'].sum()
        df = df.merge(payment_value, on='order_id', how='left')

        # Tính order_value = tổng price của các item trong đơn
        order_value = order_items.groupby('order_id', as_index=False)['price'].sum()
        order_value.columns = ['order_id', 'order_value']
        df = df.merge(order_value, on='order_id', how='left')

        # Xử lý datetime
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at',
                         'order_delivered_carrier_date', 'order_delivered_customer_date']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        df = df.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'])
        df['delivery_days'] = (df['order_delivered_customer_date'] -
                               df['order_purchase_timestamp']).dt.days
        df = df[(df['delivery_days'] >= 0) & (df['delivery_days'] <= 60)]

        # Fill missing
        df['order_value'] = df['order_value'].fillna(0)
        df['payment_value'] = df['payment_value'].fillna(0)

        output_path = os.path.join(self.processed_path, "cleaned_orders.csv")
        df.to_csv(output_path, index=False)
        print(f"Đã lưu: cleaned_orders.csv ({len(df):,} dòng)")

        return df

    def clean_behavioral_sessions(self):
        print("Đang xử lý behavioral_sessions...")

        df = pd.read_csv(os.path.join(self.synthetic_path, "behavioral_sessions.csv"))

        df['order_id'] = df['order_id'].astype(str)
        df['session_id'] = df['session_id'].astype(str)

        df = df[df['cart_additions'] >= 1]
        df = df[(df['session_duration_seconds'] >= 10) & (df['session_duration_seconds'] <= 3600)]
        df = df[(df['pages_viewed'] >= 1) & (df['pages_viewed'] <= 80)]

        output_path = os.path.join(self.processed_path, "cleaned_behavioral_sessions.csv")
        df.to_csv(output_path, index=False)
        print(f"Đã lưu: cleaned_behavioral_sessions.csv ({len(df):,} dòng)")

        return df

    def run(self):
        print("=== BẮT ĐẦU CLEANING DATA ===\n")
        self.clean_orders()
        self.clean_behavioral_sessions()
        print("\n=== HOÀN TẤT CLEANING ===")


if __name__ == "__main__":
    cleaner = DataCleaner(base_path=r"D:\DATA")
    cleaner.run()
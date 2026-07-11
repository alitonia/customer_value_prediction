"""
Data Cleaning Pipeline - Phiên bản nâng cao
Dự án: Predicting Customer Order Value for an E-Commerce Platform
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataCleaner:
    def __init__(self, base_path: str = r"D:\DATA"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "data", "raw")
        self.synthetic_path = os.path.join(base_path, "data", "synthetic")
        self.processed_path = os.path.join(base_path, "data", "processed")
        os.makedirs(self.processed_path, exist_ok=True)

        # Olist data
        self.orders = None
        self.order_items = None
        self.customers = None
        self.payments = None
        self.products = None

        # Synthetic data
        self.customer_profile = None
        self.sessions = None
        self.session_activity = None

        # Final cleaned data
        self.final_data = None

    # ====================== LOAD DATA ======================
    def load_data(self):
        print("🔄 Đang load dữ liệu...")
        # Olist
        self.orders = pd.read_csv(os.path.join(self.raw_path, "olist_orders_dataset.csv"))
        self.order_items = pd.read_csv(os.path.join(self.raw_path, "olist_order_items_dataset.csv"))
        self.customers = pd.read_csv(os.path.join(self.raw_path, "olist_customers_dataset.csv"))
        self.payments = pd.read_csv(os.path.join(self.raw_path, "olist_order_payments_dataset.csv"))
        self.products = pd.read_csv(os.path.join(self.raw_path, "olist_products_dataset.csv"))

        # Synthetic
        self.customer_profile = pd.read_csv(os.path.join(self.synthetic_path, "customer_profile.csv"))
        self.sessions = pd.read_csv(os.path.join(self.synthetic_path, "sessions.csv"))
        self.session_activity = pd.read_csv(os.path.join(self.synthetic_path, "session_activity.csv"))

        print("✅ Load dữ liệu thành công!")

    # ====================== CLEAN OLIST DATA ======================
    def clean_olist_data(self):
        print("\n🔄 Đang cleaning Olist data...")

        before_orders = len(self.orders)
        # Loại bỏ cancelled và unavailable
        self.orders = self.orders[~self.orders['order_status'].isin(['canceled', 'unavailable'])]
        print(f"   - Loại bỏ cancelled/unavailable orders: {before_orders - len(self.orders)} dòng")

        # Xử lý missing datetime
        self.orders['order_approved_at'] = self.orders['order_approved_at'].fillna(
            self.orders['order_purchase_timestamp']
        )

        # Convert datetime
        dt_cols = ['order_purchase_timestamp', 'order_approved_at',
                   'order_delivered_carrier_date', 'order_delivered_customer_date']
        for col in dt_cols:
            self.orders[col] = pd.to_datetime(self.orders[col], errors='coerce')

        # Clean order_items
        self.order_items = self.order_items.dropna(subset=['order_id', 'product_id', 'price'])

        # Clean customers
        self.customers = self.customers.dropna(subset=['customer_id', 'customer_unique_id'])

        print("✅ Cleaning Olist data hoàn tất!")

    # ====================== CLEAN SYNTHETIC DATA ======================
    def clean_synthetic_data(self):
        print("\n🔄 Đang cleaning Synthetic data...")

        # Xử lý missing values
        self.session_activity['product_id'] = self.session_activity['product_id'].fillna('UNKNOWN')
        self.session_activity['search_keyword'] = self.session_activity['search_keyword'].fillna('No Search')
        self.session_activity['add_to_cart_quantity'] = self.session_activity['add_to_cart_quantity'].fillna(0)

        # Loại bỏ activity có duration bất thường
        before_act = len(self.session_activity)
        self.session_activity = self.session_activity[
            (self.session_activity['duration_seconds'] >= 5) &
            (self.session_activity['duration_seconds'] <= 300)
        ]
        print(f"   - Loại bỏ activity duration bất thường: {before_act - len(self.session_activity)} dòng")

        # Chuẩn hóa traffic_source
        self.sessions['traffic_source'] = self.sessions['traffic_source'].str.lower().str.strip()

        # Xử lý outlier monthly_income
        before_income = len(self.customer_profile)
        q_low = self.customer_profile['monthly_income'].quantile(0.01)
        q_high = self.customer_profile['monthly_income'].quantile(0.99)
        self.customer_profile = self.customer_profile[
            (self.customer_profile['monthly_income'] >= q_low) &
            (self.customer_profile['monthly_income'] <= q_high)
        ]
        print(f"   - Loại bỏ outlier monthly_income: {before_income - len(self.customer_profile)} dòng")

        print("✅ Cleaning Synthetic data hoàn tất!")

    # ====================== MERGE DATA ======================
    def merge_data(self):
        print("\n🔄 Đang merge dữ liệu...")

        # Tính order_value từ order_items
        order_value_df = self.order_items.groupby('order_id').agg({
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        order_value_df['order_value'] = order_value_df['price'] + order_value_df['freight_value']

        # Merge orders + order_value + payments
        orders_clean = self.orders.merge(
            order_value_df[['order_id', 'order_value']], on='order_id', how='left'
        )
        orders_clean = orders_clean.merge(
            self.payments.groupby('order_id')['payment_value'].sum().reset_index(),
            on='order_id', how='left'
        )

        # Merge với customers
        orders_clean = orders_clean.merge(
            self.customers[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']],
            on='customer_id', how='left'
        )

        # Merge với customer_profile (synthetic)
        orders_clean = orders_clean.merge(self.customer_profile, on='customer_id', how='left')

        self.final_data = orders_clean
        print(f"✅ Merge hoàn tất! Tổng số dòng: {len(self.final_data)}")

    # ====================== SAVE DATA ======================
    def save_cleaned_data(self):
        # Lưu bảng chính
        output_file = os.path.join(self.processed_path, "cleaned_orders.csv")
        self.final_data.to_csv(output_file, index=False)
        print(f"\n✅ Đã lưu cleaned data tại: {output_file}")

        # Lưu thêm các bảng synthetic đã clean (nếu cần)
        self.sessions.to_csv(os.path.join(self.processed_path, "cleaned_sessions.csv"), index=False)
        self.session_activity.to_csv(os.path.join(self.processed_path, "cleaned_session_activity.csv"), index=False)
        print("✅ Đã lưu thêm cleaned_sessions.csv và cleaned_session_activity.csv")

    # ====================== RUN PIPELINE ======================
    def run_cleaning_pipeline(self):
        print("🚀 Bắt đầu Data Cleaning Pipeline (Phiên bản nâng cao)...\n")
        self.load_data()
        self.clean_olist_data()
        self.clean_synthetic_data()
        self.merge_data()
        self.save_cleaned_data()
        print("\n🎉 Hoàn thành Data Cleaning Pipeline!")


# ====================== MAIN ======================
if __name__ == "__main__":
    cleaner = DataCleaner(base_path=r"D:\DATA")
    cleaner.run_cleaning_pipeline()
import pandas as pd
import os

class FinalDatasetCreator:
    def __init__(self, base_path=r"D:\DATA"):
        self.base_path = base_path
        self.processed_path = os.path.join(base_path, "data", "processed")
        self.synthetic_path = os.path.join(base_path, "data", "synthetic")
        self.merged_path = os.path.join(base_path, "data", "merged")
        os.makedirs(self.merged_path, exist_ok=True)

    def load_data(self):
        print("🔄 Đang load dữ liệu...")
        self.sessions = pd.read_csv(os.path.join(self.processed_path, "cleaned_sessions.csv"))
        self.activities = pd.read_csv(os.path.join(self.processed_path, "cleaned_session_activity.csv"))
        self.orders = pd.read_csv(os.path.join(self.processed_path, "cleaned_orders.csv"))
        self.customer_profile = pd.read_csv(os.path.join(self.synthetic_path, "customer_profile.csv"))
        print("✅ Load xong!")

    def aggregate_features(self):
        print("🔄 Đang aggregate session features...")
        activity_agg = self.activities.groupby('session_id').agg({
            'activity_id': 'count',
            'duration_seconds': 'sum',
            'add_to_cart_quantity': 'sum',
            'scroll_percent': 'mean',
            'activity_type': lambda x: (x == 'add_to_cart').sum()
        }).rename(columns={
            'activity_id': 'num_activities',
            'duration_seconds': 'total_session_duration',
            'add_to_cart_quantity': 'total_add_to_cart',
            'scroll_percent': 'avg_scroll_percent',
            'activity_type': 'num_add_to_cart'
        })

        df = self.sessions.merge(activity_agg, on='session_id', how='left')
        return df

    def create_final_dataset(self):
        print("🔄 Đang tạo final dataset...")

        df = self.aggregate_features()

        # Ép kiểu
        df['order_id'] = df['order_id'].astype(str)
        self.orders['order_id'] = self.orders['order_id'].astype(str)
        df['customer_id'] = df['customer_id'].astype(str)
        self.customer_profile['customer_id'] = self.customer_profile['customer_id'].astype(str)

        # Merge order_value (chỉ khi có order_id thật)
        order_info = self.orders[['order_id', 'order_value', 'payment_value']].drop_duplicates()
        df = df.merge(order_info, on='order_id', how='left')

        # Merge customer profile (nguồn chính)
        profile_cols = ['customer_id', 'loyalty_tier', 'monthly_income', 'preferred_device',
                        'gender', 'education_level', 'marital_status', 'household_size']
        df = df.merge(self.customer_profile[profile_cols], on='customer_id', how='left')

        # Xử lý NaN
        df['order_value'] = df['order_value'].fillna(0)
        df['payment_value'] = df['payment_value'].fillna(0)
        df['total_add_to_cart'] = df['total_add_to_cart'].fillna(0)
        df['num_add_to_cart'] = df['num_add_to_cart'].fillna(0)
        df['num_activities'] = df['num_activities'].fillna(0)
        df['total_session_duration'] = df['total_session_duration'].fillna(0)
        df['avg_scroll_percent'] = df['avg_scroll_percent'].fillna(df['avg_scroll_percent'].median())

        # Feature đơn giản
        df['has_order'] = (df['order_value'] > 0).astype(int)

        # Lưu
        output_path = os.path.join(self.merged_path, "final_dataset.csv")
        df.to_csv(output_path, index=False)

        print(f"\n✅ Hoàn thành!")
        print(f"   Số dòng: {len(df):,}")
        print(f"   File: {output_path}")

        return df

    def run(self):
        self.load_data()
        self.create_final_dataset()


if __name__ == "__main__":
    creator = FinalDatasetCreator()
    creator.run()
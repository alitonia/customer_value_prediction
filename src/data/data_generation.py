"""
Synthetic Data Generation Module - Phiên bản cải tiến
Dự án: Predicting Customer Order Value for E-Commerce Platform
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os
from collections import defaultdict

class SyntheticDataGenerator:
    def __init__(self,
                 num_customers: int = 5000,
                 num_sessions: int = 35000,
                 base_conversion_rate: float = 0.09,
                 min_activities: int = 5,
                 max_activities: int = 22,
                 random_seed: int = 42,
                 base_path: str = r"D:\DATA"):

        self.num_customers = num_customers
        self.num_sessions = num_sessions
        self.base_conversion_rate = base_conversion_rate
        self.min_activities = min_activities
        self.max_activities = max_activities
        self.random_seed = random_seed
        self.base_path = base_path

        self.fake = Faker('vi_VN')
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.customers = None
        self.sessions = None
        self.activities = None
        self.real_orders = None
        self.customer_order_map = {}

    def random_date(self, start_date, end_date):
        delta = end_date - start_date
        return start_date + timedelta(days=random.randint(0, delta.days))

    # ====================== CUSTOMER PROFILE ======================
    def generate_customer_profile(self):
        print("🔄 Đang sinh Customer Profile...")
        data = []
        for i in range(self.num_customers):
            created = self.fake.date_time_between(start_date='-2y', end_date='now')
            data.append({
                'customer_id': f"CUST_{str(i+1000).zfill(6)}",
                'gender': random.choice(['Male', 'Female']),
                'birth_date': self.fake.date_of_birth(minimum_age=18, maximum_age=65),
                'marital_status': random.choice(['Single', 'Married', 'Divorced']),
                'occupation': self.fake.job(),
                'education_level': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
                'monthly_income': round(np.random.normal(12500000, 5500000), -3),
                'household_size': random.randint(1, 6),
                'loyalty_tier': random.choices(['Bronze', 'Silver', 'Gold', 'Platinum'],
                                             weights=[0.45, 0.30, 0.18, 0.07])[0],
                'registration_channel': random.choice(['Organic', 'Ads', 'Referral', 'Email']),
                'preferred_device': random.choice(['Mobile', 'Desktop', 'Tablet']),
                'created_at': created,
                'updated_at': created + timedelta(days=random.randint(0, 400))
            })
        self.customers = pd.DataFrame(data)
        return self.customers

    # ====================== SESSIONS + LINK ORDER ======================
    def generate_sessions(self):
        print("🔄 Đang sinh Sessions và liên kết Order...")
        data = []
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2026, 7, 1)

        # Load orders thật
        orders_path = os.path.join(self.base_path, "data", "raw", "olist_orders_dataset.csv")
        self.real_orders = pd.read_csv(orders_path)
        self.customer_order_map = self.real_orders.groupby('customer_id')['order_id'].apply(list).to_dict()

        for _, cust in self.customers.iterrows():
            cust_orders = self.customer_order_map.get(cust['customer_id'], [])
            num_sess = random.randint(3, 13)

            for _ in range(num_sess):
                session_start = self.random_date(start_date, end_date)
                duration = random.randint(50, 1650)

                # Xác định có chuyển đổi hay không (sẽ cập nhật sau khi có activities)
                has_converted = False
                order_id = None

                if cust_orders and random.random() < self.base_conversion_rate:
                    has_converted = True
                    order_id = random.choice(cust_orders)

                data.append({
                    'session_id': f"SESS_{self.fake.uuid4()[:8]}",
                    'customer_id': cust['customer_id'],
                    'session_start': session_start,
                    'session_end': session_start + timedelta(seconds=duration),
                    'device_type': cust['preferred_device'],
                    'browser': random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),
                    'operating_system': random.choice(['Windows', 'macOS', 'Android', 'iOS']),
                    'traffic_source': random.choice(['Organic', 'Paid', 'Social', 'Direct', 'Email']),
                    'traffic_medium': random.choice(['cpc', 'organic', 'referral', 'email', 'social']),
                    'landing_page': random.choice(['/home', '/category', '/product', '/search']),
                    'campaign_name': random.choice([None, 'Summer_Sale', 'Black_Friday', 'New_User_Promo'])
                                     if random.random() < 0.28 else None,
                    'ip_country': 'Vietnam',
                    'ip_region': random.choice(['Hanoi', 'Ho Chi Minh City', 'Da Nang', 'Hai Phong', 'Can Tho']),
                    'is_logged_in': random.choice([True, False]),
                    'has_converted': has_converted,
                    'order_id': order_id,
                    'created_at': session_start
                })
        self.sessions = pd.DataFrame(data)
        return self.sessions

    # ====================== SESSION ACTIVITY (Realistic) ======================
    def generate_session_activity(self):
        print("🔄 Đang sinh Session Activity với phân phối thực tế...")
        data = []

        for _, sess in self.sessions.iterrows():
            num_act = random.randint(self.min_activities, self.max_activities)
            current_time = sess['session_start']

            has_add_to_cart = False
            has_checkout = False

            for _ in range(num_act):
                duration = random.randint(8, 135)
                activity_type = random.choices(
                    ['view', 'search', 'add_to_cart', 'remove_from_cart', 'checkout_view'],
                    weights=[0.55, 0.18, 0.15, 0.05, 0.07]
                )[0]

                if activity_type == 'add_to_cart':
                    has_add_to_cart = True
                if activity_type == 'checkout_view':
                    has_checkout = True

                data.append({
                    'activity_id': f"ACT_{self.fake.uuid4()[:8]}",
                    'session_id': sess['session_id'],
                    'customer_id': sess['customer_id'],
                    'activity_timestamp': current_time,
                    'activity_type': activity_type,
                    'page_url': self.fake.uri(),
                    'product_id': f"PROD_{random.randint(1000, 99999)}" if random.random() < 0.72 else None,
                    'search_keyword': self.fake.word() if activity_type == 'search' else None,
                    'duration_seconds': duration,
                    'scroll_percent': random.randint(25, 100),
                    'add_to_cart_quantity': random.randint(1, 4) if activity_type == 'add_to_cart' else 0,
                    'created_at': current_time
                })
                current_time += timedelta(seconds=duration + random.randint(3, 35))

            # Cập nhật has_converted dựa trên hành vi thực tế
            if has_add_to_cart and has_checkout:
                self.sessions.loc[self.sessions['session_id'] == sess['session_id'], 'has_converted'] = True

        self.activities = pd.DataFrame(data)
        return self.activities

    # ====================== THỐNG KÊ ======================
    def print_summary(self):
        print("\n" + "="*60)
        print("📊 THỐNG KÊ TỔNG HỢP SAU KHI GENERATE")
        print("="*60)
        print(f"• Số lượng Customer Profile : {len(self.customers):,}")
        print(f"• Số lượng Sessions         : {len(self.sessions):,}")
        print(f"• Số lượng Activities       : {len(self.activities):,}")

        conv_rate = self.sessions['has_converted'].mean() * 100
        print(f"• Conversion Rate thực tế   : {conv_rate:.2f}%")

        avg_act = self.activities.groupby('session_id').size().mean()
        print(f"• Trung bình activities/session : {avg_act:.1f}")

        add_to_cart_rate = (self.activities['add_to_cart_quantity'] > 0).mean() * 100
        print(f"• Tỷ lệ activity có add_to_cart : {add_to_cart_rate:.2f}%")

        print("="*60 + "\n")

    # ====================== LƯU DỮ LIỆU ======================
    def save_data(self):
        path = os.path.join(self.base_path, "data", "synthetic")
        os.makedirs(path, exist_ok=True)

        self.customers.to_csv(os.path.join(path, "customer_profile.csv"), index=False)
        self.sessions.to_csv(os.path.join(path, "sessions.csv"), index=False)
        self.activities.to_csv(os.path.join(path, "session_activity.csv"), index=False)

        print(f"✅ Đã lưu thành công 3 file CSV vào: {path}")

    # ====================== CHẠY TOÀN BỘ ======================
    def generate_all(self):
        self.generate_customer_profile()
        self.generate_sessions()
        self.generate_session_activity()
        self.print_summary()
        self.save_data()
        print("🎉 Hoàn thành Generate Synthetic Data!\n")
        return self


# ====================== MAIN ======================
if __name__ == "__main__":
    generator = SyntheticDataGenerator(
        num_customers=5000,
        num_sessions=35000,
        base_conversion_rate=0.09,
        min_activities=5,
        max_activities=22,
        base_path=r"D:\DATA"
    )
    generator.generate_all()
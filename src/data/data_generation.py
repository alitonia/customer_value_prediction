

import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime

class BehavioralSessionGenerator:
    def __init__(self, base_path: str = r"D:\DATA"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "data", "raw")
        self.synthetic_path = os.path.join(base_path, "data", "synthetic")
        os.makedirs(self.synthetic_path, exist_ok=True)

    def load_real_orders(self):

        orders = pd.read_csv(os.path.join(self.raw_path, "olist_orders_dataset.csv"))
        order_items = pd.read_csv(os.path.join(self.raw_path, "olist_order_items_dataset.csv"))

        # Chỉ lấy các order đã hoàn thành (delivered)
        valid_orders = orders[orders['order_status'] == 'delivered'][['order_id']].copy()

        # Tính số lượng item thực tế của mỗi order
        item_count = order_items.groupby('order_id').size().reset_index(name='item_count')

        # Merge để có item_count
        self.orders = valid_orders.merge(item_count, on='order_id', how='left')
        self.orders['item_count'] = self.orders['item_count'].fillna(1).astype(int)


    def generate_device_type(self, n: int):
        """Phân bố: Mobile 65%, Desktop 30%, Tablet 5%"""
        return np.random.choice(
            ['mobile', 'desktop', 'tablet'],
            size=n,
            p=[0.65, 0.30, 0.05]
        )

    def generate_referral_channel(self, n: int):
        """Phân bố theo spec"""
        return np.random.choice(
            ['search_organic', 'direct', 'search_paid', 'social', 'email'],
            size=n,
            p=[0.35, 0.25, 0.15, 0.15, 0.10]
        )

    def generate_session_duration(self, pages_viewed: np.ndarray):
        """Duration ≈ pages_viewed * T_page + noise (Log-normal style)"""
        T_page = np.random.normal(35, 10, size=len(pages_viewed))
        duration = pages_viewed * T_page + np.random.normal(0, 40, size=len(pages_viewed))
        duration = np.clip(duration, 10, 3600)  # Giới hạn 10s - 1h
        return np.round(duration).astype(int)

    def generate_pages_viewed(self, n: int):
        """Trung bình ~8 trang, phân bố Poisson + noise"""
        return np.random.poisson(8, size=n) + np.random.randint(0, 4, size=n)

    def generate_cart_additions(self, item_count: np.ndarray):
        """cart_additions >= item_count thật"""
        extra = np.random.randint(0, 4, size=len(item_count))  # thêm 0-3 món
        return item_count + extra

    def generate_coupon_and_discount(self, n: int):
        """22% áp dụng coupon"""
        coupon_applied = np.random.choice([0, 1], size=n, p=[0.78, 0.22])
        discount = np.zeros(n, dtype=int)

        # Nếu có coupon thì chọn mức giảm giá
        mask = coupon_applied == 1
        discount[mask] = np.random.choice(
            [5, 10, 15, 20],
            size=mask.sum(),
            p=[0.40, 0.35, 0.15, 0.10]
        )
        return coupon_applied, discount

    def generate_all(self):

        self.load_real_orders()
        n = len(self.orders)

        # Sinh các cột theo spec
        device_type = self.generate_device_type(n)
        referral_channel = self.generate_referral_channel(n)
        pages_viewed = self.generate_pages_viewed(n)
        session_duration = self.generate_session_duration(pages_viewed)
        cart_additions = self.generate_cart_additions(self.orders['item_count'].values)
        coupon_applied, discount_pct = self.generate_coupon_and_discount(n)

        # Tạo session_id
        session_ids = [str(uuid.uuid4()) for _ in range(n)]

        # Tạo DataFrame
        behavioral_df = pd.DataFrame({
            'order_id': self.orders['order_id'].values,
            'session_id': session_ids,
            'device_type': device_type,
            'referral_channel': referral_channel,
            'session_duration_seconds': session_duration,
            'pages_viewed': pages_viewed,
            'cart_additions': cart_additions,
            'coupon_applied': coupon_applied,
            'discount_amount_pct': discount_pct
        })

        # Lưu file
        output_file = os.path.join(self.synthetic_path, "behavioral_sessions.csv")
        behavioral_df.to_csv(output_file, index=False)


        return behavioral_df


if __name__ == "__main__":
    generator = BehavioralSessionGenerator(base_path=r"D:\DATA")
    generator.generate_all()
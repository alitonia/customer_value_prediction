import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import uuid
from config import config
from common import setup_logger

logger = setup_logger("data_generation")


class BehavioralSessionGenerator:
    def __init__(self):
        self.raw_dir = config.paths.RAW_DIR
        self.synthetic_dir = config.paths.SYNTHETIC_DIR

    def load_real_orders(self):
        orders = pd.read_csv(self.raw_dir / "olist_orders_dataset.csv")
        order_items = pd.read_csv(self.raw_dir / "olist_order_items_dataset.csv")

        valid_orders = orders[orders['order_status'] == 'delivered'][['order_id']].copy()
        item_count = order_items.groupby('order_id').size().reset_index(name='item_count')

        self.orders = valid_orders.merge(item_count, on='order_id', how='left')
        self.orders['item_count'] = self.orders['item_count'].fillna(1).astype(int)

    def generate_all(self):
        logger.info("Đang sinh behavioral sessions...")

        self.load_real_orders()
        n = len(self.orders)

        device_type = np.random.choice(['mobile', 'desktop', 'tablet'], size=n, p=[0.65, 0.30, 0.05])
        referral_channel = np.random.choice(
            ['search_organic', 'direct', 'search_paid', 'social', 'email'],
            size=n, p=[0.35, 0.25, 0.15, 0.15, 0.10]
        )

        pages_viewed = np.random.poisson(8, size=n) + np.random.randint(0, 4, size=n)
        T_page = np.random.normal(35, 10, size=n)
        session_duration = np.clip(pages_viewed * T_page + np.random.normal(0, 40, size=n), 10, 3600).astype(int)

        cart_additions = self.orders['item_count'].values + np.random.randint(0, 4, size=n)
        coupon_applied = np.random.choice([0, 1], size=n, p=[0.78, 0.22])
        discount = np.zeros(n, dtype=int)
        mask = coupon_applied == 1
        discount[mask] = np.random.choice([5, 10, 15, 20], size=mask.sum(), p=[0.40, 0.35, 0.15, 0.10])

        behavioral_df = pd.DataFrame({
            'order_id': self.orders['order_id'].values,
            'session_id': [str(uuid.uuid4()) for _ in range(n)],
            'device_type': device_type,
            'referral_channel': referral_channel,
            'session_duration_seconds': session_duration,
            'pages_viewed': pages_viewed,
            'cart_additions': cart_additions,
            'coupon_applied': coupon_applied,
            'discount_amount_pct': discount
        })

        output_path = self.synthetic_dir / config.data.BEHAVIORAL_SESSIONS
        behavioral_df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu: {output_path}")

        return behavioral_df


if __name__ == "__main__":
    generator = BehavioralSessionGenerator()
    generator.generate_all()
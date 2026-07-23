"""Validation script for synthetic data (Task 1.5).

Runs all checkpoints from docs/synthetic_data_schema.md §Validation and
docs/data_dictionary.md. Prints a PASS/FAIL report.

Usage:
    python -m src.data.validate_synthetic [--synthetic-dir data/synthetic] [--raw-dir data/raw]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_data(synthetic_dir: Path, raw_dir: Path) -> dict:
    profile = pd.read_parquet(synthetic_dir / "customer_profile.parquet")
    sessions = pd.read_parquet(synthetic_dir / "sessions.parquet")
    activity = pd.read_parquet(synthetic_dir / "session_activity.parquet")
    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status"],
    )
    customers = pd.read_csv(
        raw_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_unique_id"],
    )
    items = pd.read_csv(
        raw_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "price", "freight_value"],
    )
    return {
        "profile": profile,
        "sessions": sessions,
        "activity": activity,
        "orders": orders,
        "customers": customers,
        "items": items,
    }


class ValidationResult:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        log.info("  [%s] %s%s", status, name, f" — {detail}" if detail else "")

    @property
    def all_passed(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def summary(self) -> str:
        n_pass = sum(1 for _, p, _ in self.checks if p)
        n_fail = sum(1 for _, p, _ in self.checks if not p)
        lines = [
            f"\n{'=' * 60}",
            f"VALIDATION SUMMARY: {n_pass} passed, {n_fail} failed",
            f"{'=' * 60}",
        ]
        for name, passed, detail in self.checks:
            mark = "✓" if passed else "✗"
            lines.append(
                f"  {mark} {name}" + (f" ({detail})" if detail and not passed else "")
            )
        return "\n".join(lines)


def validate(data: dict) -> ValidationResult:
    r = ValidationResult()
    profile, sessions, activity = data["profile"], data["sessions"], data["activity"]
    orders, customers, items = data["orders"], data["customers"], data["items"]

    # 1. Row counts
    r.check("customer_profile rows > 0", len(profile) > 0, f"{len(profile)} rows")
    r.check("sessions rows > 0", len(sessions) > 0, f"{len(sessions)} rows")
    r.check("session_activity rows > 0", len(activity) > 0, f"{len(activity)} rows")

    # 2. Every unique customer has exactly 1 profile row
    n_unique_cust = customers["customer_unique_id"].nunique()
    r.check(
        "1 profile per unique customer",
        len(profile) == n_unique_cust,
        f"profile={len(profile)}, unique_customers={n_unique_cust}",
    )

    # 3. Every order has exactly 1 session
    r.check(
        "1 session per order",
        len(sessions) == len(orders),
        f"sessions={len(sessions)}, orders={len(orders)}",
    )

    # 4. Every session has >= 3 activity events
    act_per_session = activity.groupby("session_id").size()
    min_act = act_per_session.min()
    r.check("Every session has >= 3 activities", min_act >= 3, f"min={min_act}")

    # 5. Zero nulls in NN columns
    nn_profile = ["customer_id"]
    nn_sessions = ["session_id", "customer_id", "order_id", "session_start"]
    nn_activity = ["activity_id", "session_id", "activity_timestamp"]
    nulls_p = profile[nn_profile].isnull().sum().sum()
    nulls_s = sessions[nn_sessions].isnull().sum().sum()
    nulls_a = activity[nn_activity].isnull().sum().sum()
    r.check(
        "No nulls in NN columns",
        nulls_p + nulls_s + nulls_a == 0,
        f"profile={nulls_p}, sessions={nulls_s}, activity={nulls_a}",
    )

    # 6. Cart consistency: sum(add_to_cart_qty) per session >= actual items
    order_item_count = items.groupby("order_id").size()
    cart_sum = activity.groupby("session_id")["add_to_cart_quantity"].sum()
    sess_order = sessions.set_index("session_id")["order_id"]
    actual = sess_order.map(order_item_count).fillna(1).astype(int)
    aligned_cart = cart_sum.reindex(actual.index).fillna(0)
    violations = (aligned_cart < actual).sum()
    r.check(
        "Cart consistency (cart_qty >= items)",
        violations == 0,
        f"{violations} violations",
    )

    # 7. Each session has >= 1 checkout and >= 1 add_to_cart
    sess_types = activity.groupby("session_id")["activity_type"].apply(set)
    no_checkout = sess_types.apply(lambda s: "checkout" not in s).sum()
    no_cart = sess_types.apply(lambda s: "add_to_cart" not in s).sum()
    r.check(
        "Every session has checkout event", no_checkout == 0, f"{no_checkout} missing"
    )
    r.check("Every session has add_to_cart event", no_cart == 0, f"{no_cart} missing")

    # 8. Temporal ordering: session_start < activity_timestamps < session_end
    sess_times = sessions.set_index("session_id")[["session_start", "session_end"]]
    merged = activity[["session_id", "activity_timestamp"]].merge(
        sess_times, left_on="session_id", right_index=True
    )
    # Normalize to ns for comparison (parquet may store us vs ns)
    act_ns = (
        merged["activity_timestamp"].values.astype("datetime64[ns]").astype("int64")
    )
    start_ns = merged["session_start"].values.astype("datetime64[ns]").astype("int64")
    end_ns = merged["session_end"].values.astype("datetime64[ns]").astype("int64")
    before_start = int((act_ns < start_ns).sum())
    after_end = int((act_ns > end_ns).sum())
    r.check(
        "Activity timestamps within session window",
        before_start + after_end == 0,
        f"before_start={before_start}, after_end={after_end}",
    )

    # 9. Value-conditioning correlations
    ov = (
        items.groupby("order_id")
        .agg(val=("price", "sum"), fr=("freight_value", "sum"))
        .reset_index()
    )
    ov["val"] += ov["fr"]
    ov = ov.merge(orders[["order_id", "customer_id"]], on="order_id")
    ov = ov.merge(customers, on="customer_id")

    cust_avg = ov.groupby("customer_unique_id")["val"].mean()
    cust_merged = profile.merge(
        cust_avg.reset_index().rename(columns={"val": "avg_val"}),
        left_on="customer_id",
        right_on="customer_unique_id",
        how="inner",
    )
    r_income = cust_merged["monthly_income"].corr(cust_merged["avg_val"])
    r.check(
        "monthly_income ↔ avg_value r ∈ [0.2, 0.6]",
        0.2 <= r_income <= 0.6,
        f"r={r_income:.3f}",
    )

    # 10. Loyalty tier monotonic AOV
    tier_med = cust_merged.groupby("loyalty_tier")["avg_val"].median()
    tier_order = ["bronze", "silver", "gold", "platinum"]
    tier_vals = [tier_med.get(t, 0) for t in tier_order]
    monotonic = all(tier_vals[i] <= tier_vals[i + 1] for i in range(3))
    r.check(
        "Loyalty tier AOV monotonically increasing",
        monotonic,
        " → ".join(f"{t}={v:.0f}" for t, v in zip(tier_order, tier_vals)),
    )

    # 11. Device effect: desktop median > mobile median
    sess_val = sessions.merge(ov[["order_id", "val"]], on="order_id")
    dev_med = sess_val.groupby("device_type")["val"].median()
    desktop_gt_mobile = dev_med.get("desktop", 0) > dev_med.get("mobile", 0)
    r.check(
        "Desktop median value > mobile",
        desktop_gt_mobile,
        f"desktop={dev_med.get('desktop', 0):.0f}, mobile={dev_med.get('mobile', 0):.0f}",
    )

    # 12. Value ranges
    income_ok = profile["monthly_income"].between(500, 50000).all()
    r.check("monthly_income in [500, 50000]", income_ok)

    hh_ok = profile["household_size"].between(1, 10).all()
    r.check("household_size in [1, 10]", hh_ok)

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic"
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()

    log.info("Loading data...")
    data = load_data(args.synthetic_dir, args.raw_dir)
    log.info("Running validation checks...")
    result = validate(data)
    print(result.summary())

    if not result.all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Synthetic data generator for customer_profile, sessions, and session_activity.

Generates three tables that extend the Olist OLTP schema with realistic
behavioral data.  Synthetic features are CONDITIONED on real order value so
that the model can learn genuine predictive patterns (see
docs/synthetic_data_schema.md §4 for the causal design rationale).

Outputs both CSV and Parquet (snappy) for each table.

Usage:
    python -m src.data.generate_synthetic [--seed 42] [--output-dir data/synthetic]
"""

import argparse
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Categorical distributions (marginal / base rates)
# ---------------------------------------------------------------------------

GENDER = ["male", "female", "other"]
GENDER_W = [0.50, 0.49, 0.01]

MARITAL = ["single", "married", "divorced", "widowed"]
MARITAL_W = [0.40, 0.35, 0.15, 0.10]

OCCUPATION = [
    "professional",
    "student",
    "self_employed",
    "retired",
    "unemployed",
    "homemaker",
    "other",
]
OCCUPATION_W = [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.15]

EDUCATION = ["high_school", "bachelor", "master", "phd", "none"]
EDUCATION_W = [0.30, 0.35, 0.20, 0.05, 0.10]

LOYALTY = ["bronze", "silver", "gold", "platinum"]
LOYALTY_BASE_W = np.array([0.50, 0.30, 0.15, 0.05])

REG_CHANNEL = ["organic", "paid_ads", "referral", "social", "email"]
REG_CHANNEL_W = [0.40, 0.25, 0.15, 0.12, 0.08]

DEVICE = ["mobile", "desktop", "tablet"]

BROWSER = ["chrome", "safari", "firefox", "edge", "other"]
BROWSER_W = [0.60, 0.20, 0.10, 0.07, 0.03]

OS_LIST = ["android", "windows", "ios", "macos", "linux"]
OS_W = np.array([0.35, 0.30, 0.20, 0.10, 0.05])

TRAFFIC_SOURCE = ["google", "direct", "facebook", "instagram", "email"]
TS_BASE_W = np.array([0.40, 0.25, 0.15, 0.10, 0.10])

TRAFFIC_MEDIUM = ["organic", "cpc", "direct", "social", "email"]
TM_BASE_W = np.array([0.35, 0.20, 0.25, 0.12, 0.08])

CAMPAIGNS = [
    "summer_sale",
    "black_friday",
    "mothers_day",
    "fathers_day",
    "christmas",
    "new_user_10off",
    "free_shipping",
    "flash_deal",
    "loyalty_reward",
    "cart_recovery",
]

LANDING_PAGES = ["/", "/category", "/product", "/search"]
LANDING_W = [0.40, 0.30, 0.20, 0.10]

BR_STATES = [
    "SP",
    "RJ",
    "MG",
    "BA",
    "RS",
    "PR",
    "PE",
    "CE",
    "PA",
    "MA",
    "GO",
    "AM",
    "PB",
    "ES",
    "RN",
    "AL",
    "PI",
    "MT",
    "DF",
    "MS",
    "SE",
    "RO",
    "TO",
    "AC",
    "AP",
    "RR",
    "SC",
]
BR_STATE_W = [
    0.22,
    0.08,
    0.10,
    0.07,
    0.06,
    0.06,
    0.05,
    0.04,
    0.04,
    0.03,
    0.03,
    0.02,
    0.02,
    0.02,
    0.02,
    0.02,
    0.02,
    0.02,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.005,
    0.005,
    0.005,
    0.03,
]

ACTIVITY_TYPES = ["page_view", "search", "product_view", "add_to_cart", "checkout"]
AT_BASE_W = np.array([0.50, 0.15, 0.20, 0.10, 0.05])

IP_COUNTRY = ["BR", "US", "AR", "PT", "other"]
IP_COUNTRY_W = [0.93, 0.02, 0.02, 0.01, 0.02]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _uuid_series(n: int) -> list[str]:
    return [str(uuid.uuid4()) for _ in range(n)]


def _weighted_choice_shifted(
    rng: np.random.Generator,
    categories: list,
    base_w: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Per-row categorical draw with probability shifts.

    shift > 0 amplifies higher-index categories; shift < 0 amplifies lower.
    """
    n = len(shift)
    k = len(categories)
    tier_factor = np.linspace(-1, 1, k)
    weights = base_w[np.newaxis, :] * (
        1 + shift[:, np.newaxis] * tier_factor[np.newaxis, :]
    )
    weights = np.clip(weights, 0.001, None)
    weights /= weights.sum(axis=1, keepdims=True)
    cumsum = np.cumsum(weights, axis=1)
    u = rng.random(n)[:, np.newaxis]
    indices = (u < cumsum).argmax(axis=1)
    return np.array([categories[i] for i in indices])


def _value_zscore(values: np.ndarray) -> np.ndarray:
    """Z-scored log1p transform for conditioning."""
    lv = np.log1p(values)
    med = np.median(lv)
    std = np.std(lv) or 1.0
    return (lv - med) / std


# ---------------------------------------------------------------------------
# Load Olist reference data
# ---------------------------------------------------------------------------


def load_olist(raw_dir: Path) -> dict[str, pd.DataFrame]:
    log.info("Loading Olist reference data from %s", raw_dir)
    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )
    customers = pd.read_csv(
        raw_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_unique_id", "customer_state"],
    )
    items = pd.read_csv(
        raw_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "price", "freight_value"],
    )
    products = pd.read_csv(
        raw_dir / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )
    return {
        "orders": orders,
        "customers": customers,
        "items": items,
        "products": products,
    }


def compute_order_aggregates(olist: dict[str, pd.DataFrame]) -> pd.DataFrame:
    items = olist["items"]
    agg = (
        items.groupby("order_id")
        .agg(
            total_value=("price", "sum"),
            freight=("freight_value", "sum"),
            item_count=("price", "size"),
        )
        .reset_index()
    )
    agg["total_value"] += agg["freight"]
    return agg


def compute_customer_aggregates(
    order_agg: pd.DataFrame,
    orders: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    merged = order_agg.merge(
        orders[["order_id", "customer_id", "order_status"]], on="order_id"
    )
    merged = merged.merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id"
    )
    delivered = merged[merged["order_status"] == "delivered"]
    return (
        delivered.groupby("customer_unique_id")
        .agg(
            avg_value=("total_value", "mean"),
            order_count=("total_value", "size"),
            avg_items=("item_count", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# customer_profile  (conditioned on customer-level avg order value)
# ---------------------------------------------------------------------------


def generate_customer_profile(
    customers: pd.DataFrame,
    cust_agg: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cust = customers.drop_duplicates(subset="customer_unique_id").copy()
    n = len(cust)
    log.info("Generating customer_profile: %d rows", n)

    median_val = cust_agg["avg_value"].median()
    cust = cust.merge(
        cust_agg[["customer_unique_id", "avg_value", "order_count", "avg_items"]],
        on="customer_unique_id",
        how="left",
    )
    cust["avg_value"] = cust["avg_value"].fillna(median_val)
    cust["order_count"] = cust["order_count"].fillna(1).astype(int)
    cust["avg_items"] = cust["avg_items"].fillna(1.0)

    vs = _value_zscore(cust["avg_value"].values)

    profile = pd.DataFrame({"customer_id": cust["customer_unique_id"].values})

    # Independent features
    profile["gender"] = rng.choice(GENDER, size=n, p=GENDER_W)
    profile["birth_date"] = pd.to_datetime(
        rng.integers(
            np.datetime64("1950-01-01").astype("int64"),
            np.datetime64("2003-12-31").astype("int64"),
            size=n,
        ).astype("datetime64[D]")
    )
    profile["marital_status"] = rng.choice(MARITAL, size=n, p=MARITAL_W)
    profile["occupation"] = rng.choice(OCCUPATION, size=n, p=OCCUPATION_W)
    profile["education_level"] = rng.choice(EDUCATION, size=n, p=EDUCATION_W)

    # monthly_income: r ≈ 0.30 with avg order value
    log_income = np.log(2500) + 0.35 * vs + rng.normal(0, 0.50, size=n)
    profile["monthly_income"] = np.clip(np.exp(log_income), 500, 50000).round(2)

    # household_size: mild positive
    lam_hh = np.clip(2.5 + 0.4 * vs, 1.0, 8.0)
    profile["household_size"] = np.clip(rng.poisson(lam=lam_hh), 1, 10)

    # loyalty_tier: shifted by value
    profile["loyalty_tier"] = _weighted_choice_shifted(
        rng, LOYALTY, LOYALTY_BASE_W, 0.6 * vs
    )

    profile["registration_channel"] = rng.choice(REG_CHANNEL, size=n, p=REG_CHANNEL_W)

    # is_marketing_opt_in: slight positive
    profile["is_marketing_opt_in"] = rng.random(n) < (0.55 + 0.10 * _sigmoid(vs))

    # preferred_device: desktop tilt for high-value
    p_desktop = 0.25 + 0.12 * _sigmoid(vs * 0.8)
    p_mobile = 0.70 - 0.12 * _sigmoid(vs * 0.8)
    p_tablet = np.full(n, 0.05)
    dev_w = np.column_stack([p_mobile, p_desktop, p_tablet])
    dev_w /= dev_w.sum(axis=1, keepdims=True)
    cumsum = np.cumsum(dev_w, axis=1)
    u = rng.random(n)[:, np.newaxis]
    profile["preferred_device"] = np.array(DEVICE)[(u < cumsum).argmax(axis=1)]

    created = pd.to_datetime(
        rng.integers(
            np.datetime64("2016-01-01").astype("int64"),
            np.datetime64("2018-06-01").astype("int64"),
            size=n,
        ).astype("datetime64[s]")
    )
    profile["created_at"] = created
    profile["updated_at"] = created + pd.to_timedelta(
        rng.integers(0, 180, size=n), unit="D"
    )

    return profile


# ---------------------------------------------------------------------------
# sessions  (conditioned on order-level value)
# ---------------------------------------------------------------------------


def generate_sessions(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    profile: pd.DataFrame,
    order_agg: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(orders)
    log.info("Generating sessions: %d rows", n)

    cust_map = customers.set_index("customer_id")["customer_unique_id"]
    order_cust = orders["customer_id"].map(cust_map)
    pref_dev = profile.set_index("customer_id")["preferred_device"]
    cust_state = customers.drop_duplicates("customer_unique_id").set_index(
        "customer_unique_id"
    )["customer_state"]

    ov = orders[["order_id"]].merge(
        order_agg[["order_id", "total_value", "item_count"]], on="order_id", how="left"
    )
    total_value = ov["total_value"].fillna(100).values
    item_count = ov["item_count"].fillna(1).values
    vs = _value_zscore(total_value)

    sessions = pd.DataFrame()
    sessions["session_id"] = _uuid_series(n)
    sessions["customer_id"] = order_cust.values
    sessions["order_id"] = orders["order_id"].values

    offset_sec = rng.integers(300, 10800, size=n)
    sessions["session_start"] = orders["order_purchase_timestamp"] - pd.to_timedelta(
        offset_sec, unit="s"
    )

    # duration: conditioned on value + items
    log_dur = (
        np.log(120)
        + 0.25 * vs
        + 0.15 * np.log1p(item_count)
        + rng.normal(0, 0.55, size=n)
    )
    duration = np.clip(np.exp(log_dur), 10, 3600).astype(int)
    sessions["session_end"] = sessions["session_start"] + pd.to_timedelta(
        duration, unit="s"
    )

    # device: 90% match preferred
    pref = order_cust.map(pref_dev).values
    device = pref.copy()
    noise_mask = rng.random(n) < 0.10
    device[noise_mask] = rng.choice(DEVICE, size=noise_mask.sum(), p=[0.65, 0.30, 0.05])
    sessions["device_type"] = device

    sessions["browser"] = rng.choice(BROWSER, size=n, p=BROWSER_W)
    sessions["operating_system"] = _weighted_choice_shifted(
        rng, OS_LIST, OS_W, 0.15 * vs
    )
    sessions["traffic_source"] = _weighted_choice_shifted(
        rng, TRAFFIC_SOURCE, TS_BASE_W, 0.3 * vs
    )
    sessions["traffic_medium"] = _weighted_choice_shifted(
        rng, TRAFFIC_MEDIUM, TM_BASE_W, 0.3 * vs
    )

    has_campaign = ~sessions["traffic_medium"].isin(["direct", "organic"])
    sessions["campaign_name"] = None
    sessions.loc[has_campaign, "campaign_name"] = rng.choice(
        CAMPAIGNS, size=has_campaign.sum()
    )

    sessions["landing_page"] = rng.choice(LANDING_PAGES, size=n, p=LANDING_W)
    sessions["ip_country"] = rng.choice(IP_COUNTRY, size=n, p=IP_COUNTRY_W)
    sessions["ip_region"] = order_cust.map(cust_state).values
    sessions["is_logged_in"] = rng.random(n) < (0.70 + 0.15 * _sigmoid(vs))
    sessions["created_at"] = sessions["session_start"]
    return sessions


# ---------------------------------------------------------------------------
# session_activity  (conditioned on order-level value)
# ---------------------------------------------------------------------------


def generate_session_activity(
    sessions: pd.DataFrame,
    products: pd.DataFrame,
    order_agg: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    log.info("Generating session_activity...")

    product_ids = products["product_id"].dropna().unique()
    categories = products["product_category_name"].dropna().unique()

    sess = sessions.merge(
        order_agg[["order_id", "total_value", "item_count"]], on="order_id", how="left"
    )
    total_value = sess["total_value"].fillna(100).values
    item_count = sess["item_count"].fillna(1).values
    vs = _value_zscore(total_value)

    n_sessions = len(sess)
    lam_act = np.clip(6 + 2 * vs + 1.5 * np.log1p(item_count), 5, 30)
    n_activities_per = np.clip(rng.poisson(lam=lam_act), 5, 30)

    total = int(n_activities_per.sum())
    log.info("  Total activity rows: %d", total)

    act_ids = _uuid_series(total)
    session_ids = np.repeat(sess["session_id"].values, n_activities_per)
    customer_ids = np.repeat(sess["customer_id"].values, n_activities_per)
    order_ids = np.repeat(sess["order_id"].values, n_activities_per)
    vs_rep = np.repeat(vs, n_activities_per)

    # Activity types shifted by value
    activity_types = _weighted_choice_shifted(
        rng, ACTIVITY_TYPES, AT_BASE_W, 0.4 * vs_rep
    )

    # Force checkout last + >= 1 add_to_cart per session
    offsets = np.concatenate([[0], np.cumsum(n_activities_per)[:-1]])
    for i in range(n_sessions):
        s, e = offsets[i], offsets[i] + n_activities_per[i]
        activity_types[e - 1] = "checkout"
        if not np.any(activity_types[s : e - 1] == "add_to_cart"):
            activity_types[s + rng.integers(1, max(2, n_activities_per[i] - 1))] = (
                "add_to_cart"
            )

    # Timestamps (force ns resolution — pandas 2.x may use us)
    session_starts = (
        sess["session_start"].values.astype("datetime64[ns]").astype("int64")
    )
    session_ends = sess["session_end"].values.astype("datetime64[ns]").astype("int64")
    timestamps = np.zeros(total, dtype="int64")
    for i in range(n_sessions):
        s, e = offsets[i], offsets[i] + n_activities_per[i]
        dur = max(session_ends[i] - session_starts[i], int(1e9))
        timestamps[s:e] = session_starts[i] + (np.sort(rng.random(e - s)) * dur).astype(
            "int64"
        )
    activity_timestamps = pd.to_datetime(timestamps, unit="ns")

    # page_url
    url_map = {
        "page_view": ["/", "/category", "/about", "/help"],
        "search": ["/search"],
        "product_view": ["/product"],
        "add_to_cart": ["/cart"],
        "checkout": ["/checkout"],
    }
    page_urls = np.array([rng.choice(url_map[at]) for at in activity_types])

    # product_id
    product_id_col = np.full(total, None, dtype=object)
    needs_product = np.isin(activity_types, ["product_view", "add_to_cart"])
    product_id_col[needs_product] = rng.choice(product_ids, size=needs_product.sum())

    # search_keyword
    search_kw = np.full(total, None, dtype=object)
    is_search = activity_types == "search"
    search_kw[is_search] = rng.choice(categories, size=is_search.sum())

    # duration_seconds
    dur_sec = np.zeros(total, dtype=int)
    view_mask = np.isin(activity_types, ["page_view", "product_view"])
    log_page_dur = (
        3.2 + 0.15 * vs_rep[view_mask] + rng.normal(0, 0.7, size=view_mask.sum())
    )
    dur_sec[view_mask] = np.clip(np.exp(log_page_dur), 1, 600).astype(int)

    # scroll_percent
    scroll = np.zeros(total, dtype=int)
    alpha = 2 + 0.5 * np.clip(vs_rep[view_mask], -2, 2)
    scroll[view_mask] = (rng.beta(alpha, 3, size=view_mask.sum()) * 100).astype(int)

    # add_to_cart_quantity
    cart_qty = np.zeros(total, dtype=int)
    is_cart = activity_types == "add_to_cart"
    cart_qty[is_cart] = np.clip(
        rng.poisson(lam=np.clip(1.2 + 0.3 * vs_rep[is_cart], 1, 5)), 1, 5
    )

    activity_df = pd.DataFrame(
        {
            "activity_id": act_ids,
            "session_id": session_ids,
            "customer_id": customer_ids,
            "order_id": order_ids,
            "activity_timestamp": activity_timestamps,
            "activity_type": activity_types,
            "page_url": page_urls,
            "product_id": product_id_col,
            "search_keyword": search_kw,
            "duration_seconds": dur_sec,
            "scroll_percent": scroll,
            "add_to_cart_quantity": cart_qty,
            "created_at": activity_timestamps,
        }
    )

    # Cart consistency
    log.info("  Enforcing cart consistency...")
    cart_sum = activity_df.groupby("session_id")["add_to_cart_quantity"].sum()
    session_order = sess.set_index("session_id")["order_id"]
    actual_items = (
        session_order.map(order_agg.set_index("order_id")["item_count"])
        .fillna(1)
        .astype(int)
    )
    deficit = actual_items - cart_sum
    for sid in deficit[deficit > 0].index:
        mask = (activity_df["session_id"] == sid) & (
            activity_df["activity_type"] == "add_to_cart"
        )
        idx = (
            activity_df.index[mask][0]
            if mask.any()
            else activity_df.index[activity_df["session_id"] == sid][0]
        )
        if not mask.any():
            activity_df.loc[idx, "activity_type"] = "add_to_cart"
        activity_df.loc[idx, "add_to_cart_quantity"] += int(deficit[sid])

    return activity_df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _write(df: pd.DataFrame, stem: Path) -> None:
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_parquet(stem.with_suffix(".parquet"), index=False, compression="snappy")
    log.info("  Wrote %s (.csv + .parquet)", stem.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(seed: int = 42, output_dir: Path = DEFAULT_OUTPUT) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    olist = load_olist(RAW_DIR)
    order_agg = compute_order_aggregates(olist)
    cust_agg = compute_customer_aggregates(
        order_agg, olist["orders"], olist["customers"]
    )

    del_vals = order_agg.merge(
        olist["orders"][["order_id", "order_status"]], on="order_id"
    ).query("order_status == 'delivered'")["total_value"]
    log.info(
        "Order value (delivered): median=%.0f  mean=%.0f  p95=%.0f",
        del_vals.median(),
        del_vals.mean(),
        del_vals.quantile(0.95),
    )

    profile = generate_customer_profile(olist["customers"], cust_agg, rng)
    _write(profile, output_dir / "customer_profile")

    sessions = generate_sessions(
        olist["orders"], olist["customers"], profile, order_agg, rng
    )
    _write(sessions, output_dir / "sessions")

    activity = generate_session_activity(sessions, olist["products"], order_agg, rng)
    _write(activity, output_dir / "session_activity")

    log.info("Done. Output in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic behavioral data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(seed=args.seed, output_dir=args.output_dir)

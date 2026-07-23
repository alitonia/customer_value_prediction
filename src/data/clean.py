"""Data cleaning and merging pipeline (Tasks 3.1–3.5).

Cleans Olist + synthetic data, merges into a single modeling dataset,
and produces a before/after cleaning log.

V2 changes:
- No winsorization (raw order value preserved)
- Geolocation features (customer/seller lat/lng, distance)
- Seller features (order count, avg price, category count)
- RFM temporal features (recency, frequency, customer age)
- Product-level price aggregates (category avg/std/count)

Usage:
    python -m src.data.clean [--output-dir data/processed]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"

# Major Brazilian cities (urban proxy)
MAJOR_CITIES = {
    "sao paulo",
    "rio de janeiro",
    "belo horizonte",
    "brasilia",
    "salvador",
    "fortaleza",
    "curitiba",
    "manaus",
    "recife",
    "porto alegre",
    "belem",
    "goiania",
    "guarulhos",
    "campinas",
    "sao luis",
    "sao goncalo",
    "maceio",
    "duque de caxias",
    "nova iguacu",
    "teresina",
    "são paulo",
    "belo horizonte",
    "goiânia",
}


def load_raw() -> dict[str, pd.DataFrame]:
    log.info("Loading raw data...")
    return {
        "orders": pd.read_csv(
            RAW_DIR / "olist_orders_dataset.csv",
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "items": pd.read_csv(
            RAW_DIR / "olist_order_items_dataset.csv",
            parse_dates=["shipping_limit_date"],
        ),
        "customers": pd.read_csv(RAW_DIR / "olist_customers_dataset.csv"),
        "products": pd.read_csv(RAW_DIR / "olist_products_dataset.csv"),
        "payments": pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv"),
        "cat_trans": pd.read_csv(RAW_DIR / "product_category_name_translation.csv"),
        "sellers": pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv"),
        "geolocation": pd.read_csv(RAW_DIR / "olist_geolocation_dataset.csv"),
        "profile": pd.read_parquet(SYNTH_DIR / "customer_profile.parquet"),
        "sessions": pd.read_parquet(SYNTH_DIR / "sessions.parquet"),
        "activity": pd.read_parquet(SYNTH_DIR / "session_activity.parquet"),
    }


def _haversine(lat1, lng1, lat2, lng2):
    """Haversine distance in km between two lat/lng pairs."""
    R = 6371
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def clean(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"steps": []}

    def record(step, before, after, detail=""):
        stats["steps"].append(
            {"step": step, "before": before, "after": after, "detail": detail}
        )
        log.info(
            "  [%s] %d → %d %s", step, before, after, f"({detail})" if detail else ""
        )

    orders = raw["orders"].copy()
    items = raw["items"].copy()
    customers = raw["customers"].copy()
    products = raw["products"].copy()
    cat_trans = raw["cat_trans"].copy()
    sellers = raw["sellers"].copy()
    geo = raw["geolocation"].copy()
    profile = raw["profile"].copy()
    sessions = raw["sessions"].copy()
    activity = raw["activity"].copy()

    n0 = len(orders)

    # --- 3.1 Dedup ---
    before = len(orders)
    orders = orders.drop_duplicates(subset="order_id")
    record("3.1a Remove duplicate orders", before, len(orders))

    before = len(items)
    items = items.drop_duplicates(subset=["order_id", "order_item_id"])
    record("3.1b Remove duplicate order items", before, len(items))

    # --- 3.1c Missing values ---
    orders["order_approved_at"] = orders["order_approved_at"].fillna(
        orders["order_purchase_timestamp"]
    )
    products["product_category_name"] = products["product_category_name"].fillna(
        "unknown"
    )

    # --- 3.2 Translate categories ---
    cat_map = cat_trans.set_index("product_category_name")[
        "product_category_name_english"
    ].to_dict()
    products["product_category_name_english"] = (
        products["product_category_name"].map(cat_map).fillna("unknown")
    )

    # --- 3.3a Status filter ---
    before = len(orders)
    orders = orders[orders["order_status"].isin({"delivered", "shipped"})]
    record(
        "3.3a Filter to delivered/shipped",
        before,
        len(orders),
        f"removed {before - len(orders)}",
    )

    # --- Order value (NO winsorization) ---
    order_agg = (
        items.groupby("order_id")
        .agg(
            order_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            item_count=("price", "size"),
        )
        .reset_index()
    )
    order_agg["order_value"] += order_agg["freight_value"]
    order_agg = order_agg[order_agg["order_value"] > 0]
    order_agg = order_agg[order_agg["order_id"].isin(orders["order_id"])]
    record("3.3b Order value (no winsorization)", n0, len(order_agg))

    # --- Geolocation: per-zip median lat/lng ---
    log.info("  Computing geolocation features...")
    geo_agg = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(
            geo_lat=("geolocation_lat", "median"),
            geo_lng=("geolocation_lng", "median"),
            geo_city=("geolocation_city", "first"),
            geo_state=("geolocation_state", "first"),
        )
        .reset_index()
    )

    # Customer geo
    cust_geo = customers[
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_state",
        ]
    ].merge(
        geo_agg,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    cust_geo["customer_urban"] = (
        cust_geo["geo_city"].str.lower().isin(MAJOR_CITIES).astype(int)
    )

    # Seller geo
    seller_geo = sellers.merge(
        geo_agg,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
        suffixes=("", "_geo"),
    )

    # --- Seller aggregates ---
    log.info("  Computing seller features...")
    seller_agg = (
        items.groupby("seller_id")
        .agg(
            seller_order_count=("order_id", "nunique"),
            seller_avg_price=("price", "mean"),
            seller_item_count=("price", "size"),
        )
        .reset_index()
    )
    # Seller category specialization
    items_with_cat = items.merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    seller_cat = (
        items_with_cat.groupby("seller_id")["product_category_name_english"]
        .nunique()
        .reset_index()
    )
    seller_cat.columns = ["seller_id", "seller_n_categories"]
    seller_agg = seller_agg.merge(seller_cat, on="seller_id", how="left")
    seller_agg = seller_agg.merge(
        seller_geo[["seller_id", "seller_state", "geo_lat", "geo_lng"]],
        on="seller_id",
        how="left",
        suffixes=("", "_seller"),
    )
    seller_agg.rename(
        columns={"geo_lat": "seller_lat", "geo_lng": "seller_lng"}, inplace=True
    )

    # Primary seller per order (first item's seller)
    order_seller = items.sort_values("order_item_id").drop_duplicates("order_id")[
        ["order_id", "seller_id"]
    ]

    # --- Product category aggregates ---
    log.info("  Computing product-level aggregates...")
    cat_price = items.merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    cat_agg = (
        cat_price.groupby("product_category_name_english")
        .agg(
            cat_avg_price=("price", "mean"),
            cat_std_price=("price", "std"),
            cat_order_count=("order_id", "nunique"),
        )
        .reset_index()
    )
    cat_agg["cat_std_price"] = cat_agg["cat_std_price"].fillna(0)

    # Primary category per order
    order_cat = (
        items_with_cat.groupby("order_id")
        .agg(
            primary_category=(
                "product_category_name_english",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown",
            ),
            n_categories=("product_category_name_english", "nunique"),
        )
        .reset_index()
    )

    # --- Session activity aggregates ---
    act_agg = (
        activity.groupby("session_id")
        .agg(
            n_events=("activity_id", "size"),
            n_product_views=("activity_type", lambda x: (x == "product_view").sum()),
            n_searches=("activity_type", lambda x: (x == "search").sum()),
            n_add_to_cart=("activity_type", lambda x: (x == "add_to_cart").sum()),
            total_cart_qty=("add_to_cart_quantity", "sum"),
            avg_scroll=("scroll_percent", "mean"),
            avg_page_duration=("duration_seconds", "mean"),
        )
        .reset_index()
    )

    # --- RFM temporal features ---
    log.info("  Computing RFM temporal features...")
    orders_sorted = orders.sort_values("order_purchase_timestamp")
    cust_first = (
        orders_sorted.groupby("customer_id")["order_purchase_timestamp"]
        .min()
        .rename("cust_first_order")
    )
    cust_prev = (
        orders_sorted.groupby("customer_id")["order_purchase_timestamp"]
        .apply(lambda x: x.shift(1))
        .rename("cust_prev_order")
    )
    cust_count = orders_sorted.groupby("customer_id").cumcount() + 1

    rfm = orders[["order_id", "customer_id", "order_purchase_timestamp"]].copy()
    rfm = rfm.merge(cust_first, on="customer_id")
    rfm = rfm.merge(cust_prev, on="customer_id", how="left")
    rfm["customer_age_days"] = (
        rfm["order_purchase_timestamp"] - rfm["cust_first_order"]
    ).dt.days
    rfm["recency_days"] = (
        rfm["order_purchase_timestamp"] - rfm["cust_prev_order"]
    ).dt.days
    rfm["recency_days"] = rfm["recency_days"].fillna(0)  # first order = 0 recency
    rfm["frequency"] = cust_count.values
    rfm["purchase_month"] = rfm["order_purchase_timestamp"].dt.month
    rfm["purchase_quarter"] = rfm["order_purchase_timestamp"].dt.quarter
    rfm["purchase_day_of_month"] = rfm["order_purchase_timestamp"].dt.day

    # --- Regional price index ---
    log.info("  Computing regional price index...")
    region_price = (
        order_agg.merge(orders[["order_id", "customer_id"]], on="order_id")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id")
        .groupby("customer_state")["order_value"]
        .median()
    )
    global_median = order_agg["order_value"].median()
    region_index = (region_price / global_median).to_dict()

    # --- FINAL MERGE ---
    log.info("  [3.5] Building unified modeling dataset...")

    df = orders.merge(order_agg, on="order_id")
    df = df.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_state",
                "customer_zip_code_prefix",
            ]
        ],
        on="customer_id",
    )
    df = df.merge(
        cust_geo[["customer_id", "geo_lat", "geo_lng", "customer_urban"]],
        on="customer_id",
        how="left",
    )
    df = df.merge(order_cat, on="order_id", how="left")
    df = df.merge(
        cat_agg,
        left_on="primary_category",
        right_on="product_category_name_english",
        how="left",
    )
    df = df.merge(order_seller, on="order_id", how="left")
    df = df.merge(seller_agg, on="seller_id", how="left")
    df = df.merge(
        profile,
        left_on="customer_unique_id",
        right_on="customer_id",
        suffixes=("", "_profile"),
    )
    df = df.merge(
        sessions.drop(columns=["customer_id"], errors="ignore"),
        on="order_id",
        suffixes=("", "_session"),
    )
    df = df.merge(act_agg, on="session_id", how="left")
    df = df.merge(
        rfm[
            [
                "order_id",
                "customer_age_days",
                "recency_days",
                "frequency",
                "purchase_month",
                "purchase_quarter",
                "purchase_day_of_month",
            ]
        ],
        on="order_id",
    )

    # Payment info
    pay_agg = (
        raw["payments"]
        .groupby("order_id")
        .agg(
            n_payments=("payment_sequential", "size"),
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
        )
        .reset_index()
    )
    df = df.merge(pay_agg, on="order_id", how="left")

    # Review score
    review_agg = (
        raw["reviews"]
        .groupby("order_id")
        .agg(review_score=("review_score", "mean"))
        .reset_index()
    )
    df = df.merge(review_agg, on="order_id", how="left")

    # Session duration
    df["session_duration_min"] = (
        df["session_end"] - df["session_start"]
    ).dt.total_seconds() / 60.0

    # Regional price index
    df["regional_price_index"] = df["customer_state"].map(region_index).fillna(1.0)

    # Haversine distance customer ↔ seller
    df["cust_seller_distance_km"] = _haversine(
        df["geo_lat"].fillna(-15.78),
        df["geo_lng"].fillna(-47.93),
        df["seller_lat"].fillna(-15.78),
        df["seller_lng"].fillna(-47.93),
    )

    # Drop post-purchase / identifier columns
    drop_cols = [
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
        "customer_zip_code_prefix",
        "geolocation_zip_code_prefix",
        "geo_city",
        "geo_state",
        "cust_first_order",
        "cust_prev_order",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Drop datetime columns except key ones
    for c in list(df.columns):
        if pd.api.types.is_datetime64_any_dtype(df[c]) and c not in (
            "order_purchase_timestamp",
            "session_start",
            "session_end",
            "birth_date",
        ):
            df = df.drop(columns=[c], errors="ignore")

    # Fill remaining nulls
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        log.info("  Filling nulls in: %s", list(cols_with_nulls.index))
        for col in cols_with_nulls.index:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(
                    df[col].mode().iloc[0] if len(df[col].mode()) > 0 else pd.NaT
                )
            else:
                df[col] = df[col].fillna("unknown")

    record("3.5 Final modeling dataset", n0, len(df), f"{len(df.columns)} columns")

    stats["final_rows"] = len(df)
    stats["final_cols"] = len(df.columns)
    stats["initial_rows"] = n0
    stats["retention_pct"] = len(df) / n0 * 100
    stats["winsorization"] = "none (raw order value preserved)"

    return df, stats


def write_cleaning_log(stats: dict, output_dir: Path) -> None:
    lines = [
        "# Cleaning Log — Before/After Comparison (V2)",
        "",
        f"**Initial orders:** {stats['initial_rows']:,}",
        f"**Final modeling rows:** {stats['final_rows']:,}",
        f"**Retention rate:** {stats['retention_pct']:.1f}%",
        f"**Winsorization:** {stats.get('winsorization', 'N/A')}",
        "",
        "## V2 Changes",
        "- Removed winsorization — raw order value preserved for robust loss training",
        "- Added geolocation features (customer/seller lat/lng, haversine distance, urban/rural)",
        "- Added seller features (order count, avg price, category count, state)",
        "- Added RFM temporal features (customer age, recency, frequency, seasonality)",
        "- Added product-level price aggregates (category avg/std price, order count)",
        "- Added regional price index (state median / global median)",
        "",
        "## Step-by-step",
        "",
        "| Step | Before | After | Detail |",
        "|---|---|---|---|",
    ]
    for s in stats["steps"]:
        lines.append(
            f"| {s['step']} | {s['before']:,} | {s['after']:,} | {s['detail']} |"
        )

    (output_dir / "cleaning_log.md").write_text("\n".join(lines))
    log.info("Wrote cleaning_log.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    df, stats = clean(raw)

    df.to_parquet(
        args.output_dir / "modeling_dataset.parquet", index=False, compression="snappy"
    )
    df.to_csv(args.output_dir / "modeling_dataset.csv", index=False)
    log.info("Saved modeling_dataset: %d rows × %d cols", len(df), len(df.columns))

    write_cleaning_log(stats, args.output_dir)
    (args.output_dir / "cleaning_stats.json").write_text(
        json.dumps(stats, indent=2, default=str)
    )
    log.info("Done.")


if __name__ == "__main__":
    main()

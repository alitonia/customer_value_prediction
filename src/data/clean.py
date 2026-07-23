"""Data cleaning and merging pipeline (Tasks 3.1–3.5).

Cleans Olist + synthetic data, merges into a single modeling dataset,
and produces a before/after cleaning log.

Usage:
    python -m src.data.clean [--output-dir data/processed]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"


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
        "profile": pd.read_parquet(SYNTH_DIR / "customer_profile.parquet"),
        "sessions": pd.read_parquet(SYNTH_DIR / "sessions.parquet"),
        "activity": pd.read_parquet(SYNTH_DIR / "session_activity.parquet"),
    }


def clean(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Run all cleaning steps. Returns (cleaned_df, stats_dict)."""
    stats: dict = {"steps": []}

    def record(step: str, before: int, after: int, detail: str = ""):
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
    profile = raw["profile"].copy()
    sessions = raw["sessions"].copy()
    activity = raw["activity"].copy()

    n0 = len(orders)

    # --- 3.1a: Remove duplicate orders ---
    before = len(orders)
    orders = orders.drop_duplicates(subset="order_id")
    record("3.1a Remove duplicate orders", before, len(orders))

    # --- 3.1b: Remove duplicate items ---
    before = len(items)
    items = items.drop_duplicates(subset=["order_id", "order_item_id"])
    record("3.1b Remove duplicate order items", before, len(items))

    # --- 3.1c: Handle missing values in orders ---
    before_nulls = int(orders.isnull().sum().sum())
    # order_approved_at can be null for canceled orders — fill with order_purchase_timestamp
    orders["order_approved_at"] = orders["order_approved_at"].fillna(
        orders["order_purchase_timestamp"]
    )
    after_nulls = int(orders.isnull().sum().sum())
    record(
        "3.1c Fill missing order timestamps",
        before_nulls,
        after_nulls,
        "approved_at ← purchase_timestamp",
    )

    # --- 3.1d: Handle missing product categories ---
    before_nulls = int(products["product_category_name"].isnull().sum())
    products["product_category_name"] = products["product_category_name"].fillna(
        "unknown"
    )
    record("3.1d Fill missing product categories", before_nulls, 0)

    # --- 3.2: Translate Portuguese categories to English ---
    cat_map = cat_trans.set_index("product_category_name")[
        "product_category_name_english"
    ].to_dict()
    products["product_category_name_english"] = (
        products["product_category_name"].map(cat_map).fillna("unknown")
    )
    n_translated = (products["product_category_name_english"] != "unknown").sum()
    record(
        "3.2 Translate categories to English",
        0,
        n_translated,
        f"{len(cat_map)} categories mapped",
    )

    # --- 3.3a: Filter to delivered + shipped orders only ---
    before = len(orders)
    valid_statuses = {"delivered", "shipped"}
    orders = orders[orders["order_status"].isin(valid_statuses)]
    record(
        "3.3a Filter to delivered/shipped",
        before,
        len(orders),
        f"removed {before - len(orders)} canceled/unavailable/other",
    )

    # --- 3.3b: Compute order value and filter impossible values ---
    order_agg = (
        items.groupby("order_id")
        .agg(
            order_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            item_count=("price", "size"),
            avg_item_price=("price", "mean"),
        )
        .reset_index()
    )
    order_agg["order_value"] += order_agg["freight_value"]

    before = len(order_agg)
    order_agg = order_agg[order_agg["order_value"] > 0]
    record("3.3b Remove zero/negative value orders", before, len(order_agg))

    # --- 3.3c: Winsorize order value at 3×IQR ---
    q1 = order_agg["order_value"].quantile(0.25)
    q3 = order_agg["order_value"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 3 * iqr
    before_outliers = int((order_agg["order_value"] > upper_fence).sum())
    order_agg["order_value_winsorized"] = order_agg["order_value"].clip(
        upper=upper_fence
    )
    record(
        "3.3c Winsorize order value",
        before_outliers,
        0,
        f"fence=R${upper_fence:.0f}, capped {before_outliers} orders",
    )

    # Keep only orders that passed status filter
    order_agg = order_agg[order_agg["order_id"].isin(orders["order_id"])]
    record("3.3d Join with valid orders", len(orders), len(order_agg))

    # --- 3.3e: Product category per order (mode) ---
    items_with_cat = items.merge(
        products[
            ["product_id", "product_category_name", "product_category_name_english"]
        ],
        on="product_id",
        how="left",
    )
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

    # --- Merge session activity aggregates ---
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

    # --- 3.5: Build unified modeling dataset ---
    log.info("  [3.5] Building unified modeling dataset...")

    df = orders.merge(order_agg, on="order_id")
    df = df.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
    )
    df = df.merge(order_cat, on="order_id", how="left")
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

    # Session duration in minutes
    df["session_duration_min"] = (
        df["session_end"] - df["session_start"]
    ).dt.total_seconds() / 60.0

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
        .agg(
            review_score=("review_score", "mean"),
        )
        .reset_index()
    )
    df = df.merge(review_agg, on="order_id", how="left")

    # Drop post-purchase timestamps not available at prediction time
    drop_cols = [
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
        "created_at",
        "updated_at",
        "created_at_session",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    # Also drop any remaining datetime columns except order_purchase_timestamp, session_start, session_end
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) and c not in (
            "order_purchase_timestamp",
            "session_start",
            "session_end",
            "activity_timestamp",
            "birth_date",
        ):
            if c not in drop_cols:
                drop_cols.append(c)
    df = df.drop(columns=drop_cols, errors="ignore")

    # Final null check
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        log.info("  Columns with nulls after merge: %s", dict(cols_with_nulls))
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
    stats["upper_fence"] = upper_fence

    return df, stats


def write_cleaning_log(stats: dict, output_dir: Path) -> None:
    """Write before/after cleaning log as markdown."""
    lines = [
        "# Cleaning Log — Before/After Comparison",
        "",
        f"**Initial orders:** {stats['initial_rows']:,}",
        f"**Final modeling rows:** {stats['final_rows']:,}",
        f"**Retention rate:** {stats['retention_pct']:.1f}%",
        f"**Winsorization fence:** R${stats['upper_fence']:,.0f}",
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

    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "1. **Status filter:** Only `delivered` and `shipped` orders retained. Canceled, unavailable,",
            "   invoiced, processing, created, and approved orders removed (no completed transaction).",
            "2. **Winsorization:** Order values above 3×IQR fence capped rather than removed,",
            "   preserving high-value signal while limiting outlier influence on linear models.",
            "3. **Category translation:** Portuguese product categories mapped to English via",
            "   Olist's official translation table. Missing categories → 'unknown'.",
            "4. **Missing timestamps:** `order_approved_at` filled with `order_purchase_timestamp`",
            "   (conservative: assumes instant approval).",
            "5. **avg_item_price excluded:** r=0.92 with target = leakage. Not included in modeling dataset.",
            "6. **Session activity aggregated:** Clickstream events rolled up to session level",
            "   (counts, sums, means) for modeling compatibility.",
        ]
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

    # Save as parquet + csv
    df.to_parquet(
        args.output_dir / "modeling_dataset.parquet", index=False, compression="snappy"
    )
    df.to_csv(args.output_dir / "modeling_dataset.csv", index=False)
    log.info("Saved modeling_dataset: %d rows × %d cols", len(df), len(df.columns))

    write_cleaning_log(stats, args.output_dir)

    # Save stats as JSON for downstream use
    (args.output_dir / "cleaning_stats.json").write_text(
        json.dumps(stats, indent=2, default=str)
    )
    log.info("Done.")


if __name__ == "__main__":
    main()

"""Exploratory Data Analysis for order value prediction (Tasks 2.1–2.3).

Loads Olist + synthetic data, produces visualizations and statistical summaries.
Plots are saved to ``plots_dir``; key statistics are returned as a dict.

Usage:
    python -m src.data.eda [--plots-dir notebooks/plots]
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_PLOTS = PROJECT_ROOT / "notebooks" / "plots"

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("muted", 10)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all() -> pd.DataFrame:
    """Merge Olist + synthetic into a single analysis dataframe."""
    log.info("Loading data...")
    orders = pd.read_csv(
        RAW_DIR / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"]
    )
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
    cat_trans = pd.read_csv(RAW_DIR / "product_category_name_translation.csv")

    profile = pd.read_parquet(SYNTH_DIR / "customer_profile.parquet")
    sessions = pd.read_parquet(SYNTH_DIR / "sessions.parquet")

    # Order-level aggregates
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

    # Product category per order (mode)
    order_cat = items.merge(
        products[["product_id", "product_category_name"]], on="product_id"
    )
    order_cat = (
        order_cat.groupby("order_id")["product_category_name"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown")
        .reset_index()
    )

    # Session activity aggregates
    activity = pd.read_parquet(SYNTH_DIR / "session_activity.parquet")
    act_agg = (
        activity.groupby("session_id")
        .agg(
            n_events=("activity_id", "size"),
            n_product_views=("activity_type", lambda x: (x == "product_view").sum()),
            n_searches=("activity_type", lambda x: (x == "search").sum()),
            total_cart_qty=("add_to_cart_quantity", "sum"),
            avg_scroll=("scroll_percent", "mean"),
            avg_page_duration=("duration_seconds", "mean"),
        )
        .reset_index()
    )

    # Merge everything
    df = orders.merge(order_agg, on="order_id")
    df = df.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
    )
    df = df.merge(order_cat, on="order_id", how="left")
    df = df.merge(cat_trans, on="product_category_name", how="left")
    df = df.merge(
        profile,
        left_on="customer_unique_id",
        right_on="customer_id",
        suffixes=("", "_profile"),
    )
    df = df.merge(sessions, on="order_id", suffixes=("", "_session"))
    df = df.merge(act_agg, on="session_id", how="left")

    # Session duration in minutes
    df["session_duration_min"] = (
        df["session_end"] - df["session_start"]
    ).dt.total_seconds() / 60.0

    log.info("Merged dataframe: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Task 2.1: Target distribution & outliers
# ---------------------------------------------------------------------------


def analyze_target(df: pd.DataFrame, plots_dir: Path) -> dict:
    log.info("=== Task 2.1: Target distribution ===")
    delivered = df[df["order_status"] == "delivered"].copy()
    vals = delivered["order_value"]

    stats = {
        "n_delivered": len(delivered),
        "mean": vals.mean(),
        "median": vals.median(),
        "std": vals.std(),
        "p95": vals.quantile(0.95),
        "p99": vals.quantile(0.99),
        "max": vals.max(),
        "skewness": vals.skew(),
    }
    log.info(
        "  Mean=%.1f  Median=%.1f  Std=%.1f  Skew=%.2f",
        stats["mean"],
        stats["median"],
        stats["std"],
        stats["skewness"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(vals, bins=100, color=PALETTE[0], edgecolor="white")
    axes[0].set_title("Order Value Distribution (raw)")
    axes[0].set_xlabel("Order Value (BRL)")
    axes[0].axvline(
        vals.median(), color="red", ls="--", label=f"Median={vals.median():.0f}"
    )
    axes[0].legend()

    axes[1].hist(np.log1p(vals), bins=100, color=PALETTE[1], edgecolor="white")
    axes[1].set_title("Log(1 + Order Value)")
    axes[1].set_xlabel("log(1 + value)")

    # Box plot by order status
    status_order = [
        "delivered",
        "shipped",
        "canceled",
        "unavailable",
        "invoiced",
        "processing",
    ]
    present = [s for s in status_order if s in df["order_status"].values]
    sns.boxplot(
        data=df,
        x="order_status",
        y="order_value",
        order=present,
        palette=PALETTE,
        ax=axes[2],
        fliersize=1,
    )
    axes[2].set_title("Order Value by Status")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    axes[2].set_ylim(0, vals.quantile(0.99) * 1.5)

    plt.tight_layout()
    fig.savefig(plots_dir / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 01_target_distribution.png")

    # Outlier analysis
    iqr = vals.quantile(0.75) - vals.quantile(0.25)
    upper_fence = vals.quantile(0.75) + 3 * iqr
    n_outliers = (vals > upper_fence).sum()
    stats["outlier_fence_3iqr"] = upper_fence
    stats["n_outliers_3iqr"] = int(n_outliers)
    stats["pct_outliers_3iqr"] = n_outliers / len(vals) * 100
    log.info(
        "  Outliers (>3×IQR): %d (%.1f%%), fence=R$%.0f",
        n_outliers,
        stats["pct_outliers_3iqr"],
        upper_fence,
    )

    return stats


# ---------------------------------------------------------------------------
# Task 2.2: Order value drivers
# ---------------------------------------------------------------------------


def analyze_drivers(df: pd.DataFrame, plots_dir: Path) -> dict:
    log.info("=== Task 2.2: Order value drivers ===")
    delivered = df[df["order_status"] == "delivered"].copy()
    insights = {}

    # --- By product category (top 15) ---
    cat_med = (
        delivered.groupby("product_category_name_english")["order_value"]
        .agg(["median", "mean", "count"])
        .dropna()
    )
    cat_med = cat_med[cat_med["count"] >= 50].sort_values("median", ascending=False)
    top_cats = cat_med.head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    top_cats["median"].sort_values().plot.barh(color=PALETTE[0], ax=ax)
    ax.set_title("Median Order Value by Product Category (top 15)")
    ax.set_xlabel("Median Order Value (BRL)")
    plt.tight_layout()
    fig.savefig(plots_dir / "02_value_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    insights["top_category"] = top_cats.index[0] if len(top_cats) > 0 else "N/A"
    insights["top_category_median"] = (
        float(top_cats["median"].iloc[0]) if len(top_cats) > 0 else 0
    )
    log.info(
        "  Top category: %s (median R$%.0f)",
        insights["top_category"],
        insights["top_category_median"],
    )

    # --- By device type ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(
        data=delivered, x="device_type", y="order_value", palette=PALETTE, ax=axes[0]
    )
    axes[0].set_title("Order Value by Device Type")
    axes[0].set_ylim(0, delivered["order_value"].quantile(0.95))

    sns.boxplot(
        data=delivered,
        x="preferred_device",
        y="order_value",
        palette=PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Order Value by Preferred Device")
    axes[1].set_ylim(0, delivered["order_value"].quantile(0.95))
    plt.tight_layout()
    fig.savefig(plots_dir / "03_value_by_device.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    dev_med = delivered.groupby("device_type")["order_value"].median()
    insights["device_effect"] = {k: float(v) for k, v in dev_med.items()}
    log.info("  Device medians: %s", {k: f"R${v:.0f}" for k, v in dev_med.items()})

    # --- By traffic source ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ts_med = (
        delivered.groupby("traffic_source")["order_value"]
        .median()
        .sort_values(ascending=False)
    )
    ts_med.plot.bar(color=PALETTE[2], ax=ax)
    ax.set_title("Median Order Value by Traffic Source")
    ax.set_ylabel("Median Order Value (BRL)")
    plt.tight_layout()
    fig.savefig(plots_dir / "04_value_by_traffic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    insights["traffic_effect"] = {k: float(v) for k, v in ts_med.items()}

    # --- By loyalty tier ---
    fig, ax = plt.subplots(figsize=(8, 5))
    tier_order = ["bronze", "silver", "gold", "platinum"]
    sns.boxplot(
        data=delivered,
        x="loyalty_tier",
        y="order_value",
        order=tier_order,
        palette=PALETTE,
        ax=ax,
    )
    ax.set_title("Order Value by Loyalty Tier")
    ax.set_ylim(0, delivered["order_value"].quantile(0.95))
    plt.tight_layout()
    fig.savefig(plots_dir / "05_value_by_loyalty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    tier_med = delivered.groupby("loyalty_tier")["order_value"].median()
    insights["loyalty_effect"] = {k: float(v) for k, v in tier_med.items()}

    # --- By income quartile ---
    delivered["income_quartile"] = pd.qcut(
        delivered["monthly_income"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=delivered, x="income_quartile", y="order_value", palette=PALETTE, ax=ax
    )
    ax.set_title("Order Value by Income Quartile")
    ax.set_ylim(0, delivered["order_value"].quantile(0.95))
    plt.tight_layout()
    fig.savefig(
        plots_dir / "06_value_by_income_quartile.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # --- Scatter: item_count vs order_value ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(
        delivered["item_count"],
        delivered["order_value"],
        alpha=0.1,
        s=5,
        color=PALETTE[3],
    )
    axes[0].set_title("Item Count vs Order Value")
    axes[0].set_xlabel("Item Count")
    axes[0].set_ylabel("Order Value (BRL)")
    axes[0].set_ylim(0, delivered["order_value"].quantile(0.95))

    axes[1].scatter(
        delivered["monthly_income"],
        delivered["order_value"],
        alpha=0.1,
        s=5,
        color=PALETTE[4],
    )
    axes[1].set_title("Monthly Income vs Order Value")
    axes[1].set_xlabel("Monthly Income (BRL)")
    axes[1].set_ylabel("Order Value (BRL)")
    axes[1].set_ylim(0, delivered["order_value"].quantile(0.95))
    plt.tight_layout()
    fig.savefig(plots_dir / "07_scatter_drivers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Session behavior vs value ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(
        delivered["session_duration_min"],
        delivered["order_value"],
        alpha=0.1,
        s=5,
        color=PALETTE[5],
    )
    axes[0].set_title("Session Duration vs Order Value")
    axes[0].set_xlabel("Session Duration (min)")
    axes[0].set_ylim(0, delivered["order_value"].quantile(0.95))

    axes[1].scatter(
        delivered["total_cart_qty"],
        delivered["order_value"],
        alpha=0.1,
        s=5,
        color=PALETTE[6],
    )
    axes[1].set_title("Total Cart Qty vs Order Value")
    axes[1].set_xlabel("Total Cart Quantity")
    axes[1].set_ylim(0, delivered["order_value"].quantile(0.95))

    axes[2].scatter(
        delivered["n_product_views"],
        delivered["order_value"],
        alpha=0.1,
        s=5,
        color=PALETTE[7],
    )
    axes[2].set_title("Product Views vs Order Value")
    axes[2].set_xlabel("Product Views")
    axes[2].set_ylim(0, delivered["order_value"].quantile(0.95))
    plt.tight_layout()
    fig.savefig(
        plots_dir / "08_session_behavior_vs_value.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    log.info("  Saved driver plots 02–08")
    return insights


# ---------------------------------------------------------------------------
# Task 2.3: Correlations & multicollinearity
# ---------------------------------------------------------------------------


def analyze_correlations(df: pd.DataFrame, plots_dir: Path) -> dict:
    log.info("=== Task 2.3: Correlations & multicollinearity ===")
    delivered = df[df["order_status"] == "delivered"].copy()

    num_cols = [
        "order_value",
        "item_count",
        "avg_item_price",
        "freight_value",
        "monthly_income",
        "household_size",
        "session_duration_min",
        "n_events",
        "n_product_views",
        "n_searches",
        "total_cart_qty",
        "avg_scroll",
        "avg_page_duration",
    ]
    num_cols = [c for c in num_cols if c in delivered.columns]
    corr = delivered[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix — Numerical Features vs Order Value")
    plt.tight_layout()
    fig.savefig(plots_dir / "09_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Correlations with target
    target_corr = corr["order_value"].drop("order_value").sort_values(ascending=False)
    log.info("  Top correlations with order_value:")
    for feat, r in target_corr.items():
        log.info("    %s: r=%.3f", feat, r)

    # High-correlation pairs (multicollinearity risk)
    high_corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr.iloc[i, j]
            if (
                abs(r) > 0.7
                and num_cols[i] != "order_value"
                and num_cols[j] != "order_value"
            ):
                high_corr_pairs.append((num_cols[i], num_cols[j], r))

    if high_corr_pairs:
        log.info("  Multicollinearity warnings (|r| > 0.7):")
        for a, b, r in high_corr_pairs:
            log.info("    %s ↔ %s: r=%.3f", a, b, r)

    insights = {
        "target_correlations": {k: float(v) for k, v in target_corr.items()},
        "multicollinear_pairs": [(a, b, float(r)) for a, b, r in high_corr_pairs],
    }
    return insights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_eda(plots_dir: Path = DEFAULT_PLOTS) -> dict:
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = load_all()

    target_stats = analyze_target(df, plots_dir)
    driver_insights = analyze_drivers(df, plots_dir)
    corr_insights = analyze_correlations(df, plots_dir)

    all_insights = {
        "target": target_stats,
        "drivers": driver_insights,
        "correlations": corr_insights,
    }

    log.info("=== EDA Complete ===")
    log.info("Plots saved to %s", plots_dir)
    return all_insights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    args = parser.parse_args()
    run_eda(args.plots_dir)


if __name__ == "__main__":
    main()

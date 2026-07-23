"""Feature engineering pipeline (Tasks 4.1–4.4).

Transforms the cleaned modeling dataset into a model-ready feature matrix:
- Derived features (RFM, ratios, interactions)
- Categorical encoding (one-hot for low-cardinality, target encoding for high)
- Numerical scaling (StandardScaler)
- Target log-transform
- VIF-based feature selection

Outputs:
- data/processed/features.parquet  (feature matrix)
- data/processed/target.parquet    (log-transformed target)
- data/processed/feature_catalog.md
- data/processed/preprocessor.joblib (fitted encoders/scalers for deployment)

Usage:
    python -m src.features.engineer [--input data/processed/modeling_dataset.parquet]
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"

# Columns to exclude from features (identifiers, target, post-purchase)
EXCLUDE = {
    "order_id",
    "customer_id",
    "customer_unique_id",
    "session_id",
    "order_status",
    "order_purchase_timestamp",
    "session_start",
    "session_end",
    "order_value",
    "freight_value",
    "product_category_name",  # keep only english version
    "customer_state",  # keep ip_region instead
    "activity_id",
}

# Categorical columns for one-hot encoding (low cardinality)
ONEHOT_COLS = [
    "device_type",
    "preferred_device",
    "browser",
    "operating_system",
    "traffic_source",
    "traffic_medium",
    "landing_page",
    "ip_country",
    "gender",
    "marital_status",
    "education_level",
    "loyalty_tier",
    "registration_channel",
    "payment_type",
    "primary_category",
]

# Categorical columns for target encoding (high cardinality)
TARGET_ENCODE_COLS = ["ip_region", "campaign_name", "seller_state"]

# Numerical columns to scale
NUMERICAL_COLS = [
    "item_count",
    "monthly_income",
    "household_size",
    "session_duration_min",
    "n_events",
    "n_product_views",
    "n_searches",
    "n_add_to_cart",
    "total_cart_qty",
    "avg_scroll",
    "avg_page_duration",
    "n_categories",
    "payment_installments",
    "review_score",
    "n_payments",
    # V2: Geolocation
    "geo_lat",
    "geo_lng",
    "customer_urban",
    "cust_seller_distance_km",
    "regional_price_index",
    # V2: Seller
    "seller_order_count",
    "seller_avg_price",
    "seller_item_count",
    "seller_n_categories",
    "seller_lat",
    "seller_lng",
    # V2: RFM temporal
    "customer_age_days",
    "recency_days",
    "frequency",
    "purchase_month",
    "purchase_quarter",
    "purchase_day_of_month",
    # V2: Product aggregates
    "cat_avg_price",
    "cat_std_price",
    "cat_order_count",
]

# Boolean columns (pass through as 0/1)
BOOL_COLS = ["is_logged_in", "is_marketing_opt_in"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features (Task 4.1)."""
    log.info("=== Task 4.1: Engineering derived features ===")
    feat = df.copy()

    # Drop leakage columns that should never be features
    leakage_cols = [
        "avg_item_price",
        "order_value",
        "freight_value",
    ]
    feat = feat.drop(
        columns=[c for c in leakage_cols if c in feat.columns], errors="ignore"
    )

    # RFM-like: order count per customer (proxy for frequency)
    cust_freq = df.groupby("customer_unique_id")["order_id"].transform("count")
    feat["customer_order_count"] = cust_freq

    # V2: Distance × value interaction (longer distance → higher freight → higher value)
    if "cust_seller_distance_km" in feat.columns and "item_count" in feat.columns:
        feat["distance_per_item"] = feat["cust_seller_distance_km"] / feat[
            "item_count"
        ].clip(lower=1)

    # V2: Seller reputation × loyalty interaction
    if "seller_order_count" in feat.columns and "loyalty_numeric" in feat.columns:
        feat["seller_rep_x_loyalty"] = (
            feat["seller_order_count"] * feat["loyalty_numeric"]
        )

    # V2: Category price × income interaction
    if "cat_avg_price" in feat.columns and "log_income" in feat.columns:
        feat["cat_price_x_income"] = feat["cat_avg_price"] * feat["log_income"]

    # V2: Urban × device interaction
    if "customer_urban" in feat.columns:
        feat["urban_x_desktop"] = feat["customer_urban"] * (
            feat["device_type"] == "desktop"
        ).astype(float)

    # Value per item — REMOVED: leaks target (uses order_value_winsorized)
    # feat["value_per_item"] = feat["order_value_winsorized"] / feat["item_count"].clip(lower=1)

    # Cart conversion rate: items purchased / items added to cart
    feat["cart_conversion"] = feat["item_count"] / feat["total_cart_qty"].clip(lower=1)

    # Engagement rate: product views / total events
    feat["engagement_rate"] = feat["n_product_views"] / feat["n_events"].clip(lower=1)

    # Search intensity: searches / total events
    feat["search_intensity"] = feat["n_searches"] / feat["n_events"].clip(lower=1)

    # Session efficiency: items per minute (NOT value per minute — that leaks target)
    feat["items_per_minute"] = feat["item_count"] / feat["session_duration_min"].clip(
        lower=0.1
    )

    # Income × loyalty interaction (encoded numerically)
    tier_map = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4}
    feat["loyalty_numeric"] = feat["loyalty_tier"].map(tier_map).fillna(1)
    feat["income_x_loyalty"] = feat["monthly_income"] * feat["loyalty_numeric"]

    # Log income (reduce skew)
    feat["log_income"] = np.log1p(feat["monthly_income"])

    # Hour of day and day of week from purchase timestamp
    feat["purchase_hour"] = feat["order_purchase_timestamp"].dt.hour
    feat["purchase_dow"] = feat["order_purchase_timestamp"].dt.dayofweek
    feat["is_weekend"] = (feat["purchase_dow"] >= 5).astype(int)

    # Age from birth_date
    feat["age"] = (pd.Timestamp("2018-01-01") - feat["birth_date"]).dt.days / 365.25
    feat["age"] = feat["age"].clip(15, 70)

    derived = [
        "customer_order_count",
        "cart_conversion",
        "engagement_rate",
        "search_intensity",
        "items_per_minute",
        "loyalty_numeric",
        "income_x_loyalty",
        "log_income",
        "purchase_hour",
        "purchase_dow",
        "is_weekend",
        "age",
        # V2 derived
        "distance_per_item",
        "seller_rep_x_loyalty",
        "cat_price_x_income",
        "urban_x_desktop",
    ]
    log.info("  Created %d derived features", len(derived))
    return feat


def encode_and_scale(
    feat: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, dict]:
    """Encode categoricals and scale numericals (Task 4.2)."""
    log.info("=== Task 4.2: Encoding & scaling ===")
    preprocessor = {}

    # One-hot encoding
    onehot_present = [c for c in ONEHOT_COLS if c in feat.columns]
    dummies = pd.get_dummies(feat[onehot_present], dtype=np.float32)
    # Drop first to avoid multicollinearity
    dummies = dummies.drop(
        columns=[c for c in dummies.columns if c.endswith("_unknown")], errors="ignore"
    )
    preprocessor["onehot_cols"] = onehot_present
    preprocessor["onehot_dummies"] = list(dummies.columns)
    log.info(
        "  One-hot: %d cols → %d dummies", len(onehot_present), len(dummies.columns)
    )

    # Target encoding for high-cardinality categoricals
    te_present = [c for c in TARGET_ENCODE_COLS if c in feat.columns]
    te_cols = {}
    target_vals = target.values.ravel()
    global_mean = float(np.mean(target_vals))
    for col in te_present:
        # Smoothed target encoding
        temp = pd.DataFrame({"cat": feat[col].values, "target": target_vals})
        agg = temp.groupby("cat")["target"].agg(["mean", "count"])
        smoothing = 20
        te = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
            agg["count"] + smoothing
        )
        te_cols[col] = te.to_dict()
        feat[f"{col}_te"] = feat[col].map(te).fillna(global_mean)
    preprocessor["target_encoding"] = te_cols
    preprocessor["target_global_mean"] = float(global_mean)
    log.info("  Target encoded: %s", te_present)

    # Scale numericals
    num_present = [
        c
        for c in NUMERICAL_COLS
        + [
            "customer_order_count",
            "cart_conversion",
            "engagement_rate",
            "search_intensity",
            "items_per_minute",
            "loyalty_numeric",
            "income_x_loyalty",
            "log_income",
            "purchase_hour",
            "purchase_dow",
            "age",
        ]
        if c in feat.columns
    ]

    # Replace inf with NaN before scaling
    feat = feat.replace([np.inf, -np.inf], np.nan)

    scaler = StandardScaler()
    feat[num_present] = scaler.fit_transform(feat[num_present].fillna(0))
    preprocessor["scaler"] = scaler
    preprocessor["numerical_cols"] = num_present
    log.info("  Scaled %d numerical features", len(num_present))

    # Boolean cols
    bool_present = [c for c in BOOL_COLS if c in feat.columns]
    for c in bool_present:
        feat[c] = feat[c].astype(float)

    # Merge one-hot dummies into feat
    feat = pd.concat([feat, dummies], axis=1)

    # Assemble final feature matrix
    keep_cols = (
        num_present
        + bool_present
        + list(dummies.columns)
        + [f"{c}_te" for c in te_present]
    )
    keep_cols = [c for c in keep_cols if c in feat.columns]
    # Add purchase_hour, purchase_dow, is_weekend if not already in num_present
    for c in ["purchase_hour", "purchase_dow", "is_weekend"]:
        if c in feat.columns and c not in keep_cols:
            keep_cols.append(c)

    X = feat[keep_cols].copy()
    log.info("  Final feature matrix: %d rows × %d cols", len(X), len(X.columns))
    return X, preprocessor


def check_vif(X: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    """Check VIF and return columns to drop (Task 4.4)."""
    log.info("=== Task 4.4: VIF check ===")
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        log.warning("  statsmodels not installed — skipping VIF check")
        return []

    # Only check numerical columns (skip one-hot dummies)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Sample for speed
    sample = X[num_cols].sample(min(5000, len(X)), random_state=42).fillna(0)

    to_drop = []
    for iteration in range(3):  # iterative removal
        vifs = []
        cols = [c for c in sample.columns if c not in to_drop]
        if len(cols) < 2:
            break
        X_vif = sample[cols].values
        for i, col in enumerate(cols):
            try:
                vif = variance_inflation_factor(X_vif, i)
            except Exception:
                vif = 0
            vifs.append((col, vif))

        vifs.sort(key=lambda x: -x[1])
        worst = vifs[0]
        if worst[1] > threshold:
            log.info("  Dropping %s (VIF=%.1f)", worst[0], worst[1])
            to_drop.append(worst[0])
        else:
            break

    if not to_drop:
        log.info("  No VIF violations (threshold=%.0f)", threshold)
    return to_drop


def write_feature_catalog(
    X: pd.DataFrame, preprocessor: dict, output_dir: Path
) -> None:
    """Write feature catalog document."""
    lines = [
        "# Feature Catalog",
        "",
        f"**Total features:** {len(X.columns)}",
        f"**Rows:** {len(X):,}",
        "",
        "## Feature Groups",
        "",
    ]

    groups = {
        "Numerical (scaled)": preprocessor.get("numerical_cols", []),
        "Boolean": [c for c in X.columns if c in BOOL_COLS],
        "One-hot encoded": [
            c for c in X.columns if any(c.startswith(p + "_") for p in ONEHOT_COLS)
        ],
        "Target encoded": [c for c in X.columns if c.endswith("_te")],
        "Derived": [
            c
            for c in X.columns
            if c
            in [
                "customer_order_count",
                "cart_conversion",
                "engagement_rate",
                "search_intensity",
                "items_per_minute",
                "loyalty_numeric",
                "income_x_loyalty",
                "log_income",
                "purchase_hour",
                "purchase_dow",
                "is_weekend",
                "age",
            ]
        ],
    }

    for group_name, cols in groups.items():
        present = [c for c in cols if c in X.columns]
        lines.append(f"### {group_name} ({len(present)} features)")
        lines.append("")
        for c in present:
            lines.append(f"- `{c}`")
        lines.append("")

    lines.extend(
        [
            "## Excluded Features",
            "",
            "- `avg_item_price` — r=0.92 with target (leakage: mechanically derived from order contents)",
            "- `order_value` — target variable",
            "- `freight_value` — component of target",
            "- Post-purchase timestamps — not available at prediction time",
            "",
            "## Preprocessing Artifacts",
            "",
            "- `preprocessor.joblib` — fitted StandardScaler + target encoding maps",
            "- Load with `joblib.load('preprocessor.joblib')` for inference",
        ]
    )

    (output_dir / "feature_catalog.md").write_text("\n".join(lines))
    log.info("Wrote feature_catalog.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    log.info("Loading cleaned dataset...")
    df = pd.read_parquet(args.input)
    log.info("  %d rows × %d cols", len(df), len(df.columns))

    # Target: raw order value (no winsorization)
    target = df[["order_value"]].rename(columns={"order_value": "target"})

    # Task 4.3: Log-transform target
    log.info("=== Task 4.3: Log-transform target ===")
    target["target_log"] = np.log1p(target["target"])
    log.info(
        "  Target skew before: %.2f, after: %.2f",
        target["target"].skew(),
        target["target_log"].skew(),
    )

    # Task 4.1: Derived features
    feat = engineer_features(df)

    # Task 4.2: Encode & scale
    X, preprocessor = encode_and_scale(feat, target["target_log"])

    # Task 4.4: VIF check
    vif_drops = check_vif(X)
    if vif_drops:
        X = X.drop(columns=vif_drops)
        log.info("  Final features after VIF: %d", len(X.columns))

    # Save outputs
    X.to_parquet(
        args.output_dir / "features.parquet", index=False, compression="snappy"
    )
    target.to_parquet(
        args.output_dir / "target.parquet", index=False, compression="snappy"
    )

    with open(args.output_dir / "preprocessor.joblib", "wb") as f:
        pickle.dump(preprocessor, f)

    write_feature_catalog(X, preprocessor, args.output_dir)

    log.info("Done. Features: %d × %d", len(X), len(X.columns))


if __name__ == "__main__":
    main()

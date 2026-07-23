"""Standalone Streamlit app for HuggingFace Spaces deployment.

Loads the model directly (no FastAPI dependency) so it runs as a
single-process Streamlit Space.

Usage (local):
    streamlit run app.py

HF Spaces: auto-detected via sdk: streamlit in README frontmatter.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Load model + preprocessor (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    root = Path(__file__).resolve().parent
    with open(root / "models" / "best_model.joblib", "rb") as f:
        artifact = pickle.load(f)
    with open(root / "data" / "processed" / "preprocessor.joblib", "rb") as f:
        preprocessor = pickle.load(f)
    return artifact, preprocessor


artifact, preprocessor = load_model()
model = artifact["model"]
feature_cols = artifact["feature_columns"]
scaler = preprocessor["scaler"]
numerical_cols = preprocessor["numerical_cols"]
target_encoding = preprocessor.get("target_encoding", {})
target_global_mean = preprocessor.get("target_global_mean", 0.0)
onehot_cols = preprocessor.get("onehot_cols", [])

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Order Value Predictor", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Order Value Predictor")
st.markdown(
    "Predict the expected value of an e-commerce order based on customer profile, "
    "session behavior, and cart contents. Powered by **XGBoost** trained on "
    "Olist Brazilian E-Commerce data + synthetic behavioral features."
)
st.markdown(
    f"**Model:** {artifact['model_name']} | "
    f"**R²:** {artifact['metrics']['R2_log']:.3f} | "
    f"**Features:** {len(feature_cols)}"
)

# Sidebar
st.sidebar.header("Customer Profile")
monthly_income = st.sidebar.slider("Monthly Income (BRL)", 500, 50000, 3000, 500)
household_size = st.sidebar.slider("Household Size", 1, 10, 3)
loyalty_tier = st.sidebar.selectbox(
    "Loyalty Tier", ["bronze", "silver", "gold", "platinum"]
)
gender = st.sidebar.selectbox("Gender", ["male", "female", "other"])
education = st.sidebar.selectbox(
    "Education", ["high_school", "bachelor", "master", "phd", "none"]
)
preferred_device = st.sidebar.selectbox(
    "Preferred Device", ["mobile", "desktop", "tablet"]
)
age = st.sidebar.slider("Age", 15, 70, 30)

st.sidebar.header("Session Context")
device_type = st.sidebar.selectbox(
    "Current Device", ["mobile", "desktop", "tablet"], key="dev"
)
traffic_source = st.sidebar.selectbox(
    "Traffic Source", ["google", "direct", "facebook", "instagram", "email"]
)
traffic_medium = st.sidebar.selectbox(
    "Traffic Medium", ["organic", "cpc", "direct", "social", "email"]
)
is_logged_in = st.sidebar.checkbox("Logged In", value=True)
coupon_applied = st.sidebar.checkbox("Coupon Applied", value=False)
discount_pct = (
    st.sidebar.slider("Discount %", 0.0, 25.0, 0.0, 1.0) if coupon_applied else 0.0
)
session_duration = st.sidebar.slider("Session Duration (min)", 0.5, 60.0, 5.0, 0.5)

st.sidebar.header("Cart & Behavior")
item_count = st.sidebar.slider("Items in Cart", 1, 20, 2)
total_cart_qty = st.sidebar.slider("Total Cart Additions", 1, 30, max(2, item_count))
n_product_views = st.sidebar.slider("Product Views", 0, 50, 5)
n_searches = st.sidebar.slider("Searches", 0, 20, 2)
primary_category = st.sidebar.selectbox(
    "Primary Category",
    [
        "electronics",
        "computers",
        "housewares",
        "health_beauty",
        "watches_gifts",
        "sports_leisure",
        "toys",
        "furniture_decor",
        "telephony",
        "auto",
    ],
)

# Predict
if st.button("🔮 Predict Order Value", type="primary", use_container_width=True):
    # Build feature vector
    tier_map = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4}
    row = {
        "monthly_income": monthly_income,
        "household_size": household_size,
        "loyalty_tier": loyalty_tier,
        "gender": gender,
        "education_level": education,
        "preferred_device": preferred_device,
        "age": age,
        "device_type": device_type,
        "traffic_source": traffic_source,
        "traffic_medium": traffic_medium,
        "is_logged_in": float(is_logged_in),
        "coupon_applied": float(coupon_applied),
        "discount_amount_pct": discount_pct,
        "session_duration_min": session_duration,
        "item_count": item_count,
        "total_cart_qty": total_cart_qty,
        "n_product_views": n_product_views,
        "n_searches": n_searches,
        "primary_category": primary_category,
        "browser": "chrome",
        "operating_system": "android",
        "landing_page": "/",
        "ip_country": "BR",
        "ip_region": "SP",
        "marital_status": "single",
        "registration_channel": "organic",
        "is_marketing_opt_in": 1.0,
        "payment_type": "credit_card",
        "payment_installments": 1,
        "review_score": 4.0,
        "n_events": n_product_views + n_searches + 3,
        "n_add_to_cart": max(1, total_cart_qty // 2),
        "avg_scroll": 40,
        "avg_page_duration": 30,
        "purchase_hour": 14,
        "purchase_dow": 2,
        # Derived
        "loyalty_numeric": tier_map.get(loyalty_tier, 1),
        "log_income": np.log1p(monthly_income),
        "customer_order_count": 1,
        "cart_conversion": item_count / max(total_cart_qty, 1),
        "engagement_rate": n_product_views / max(n_product_views + n_searches + 3, 1),
        "search_intensity": n_searches / max(n_product_views + n_searches + 3, 1),
        "items_per_minute": item_count / max(session_duration, 0.1),
        "n_categories": 1,
        "n_payments": 1,
    }
    row["income_x_loyalty"] = monthly_income * row["loyalty_numeric"]

    # Target encoding
    for col, te_map in target_encoding.items():
        row[f"{col}_te"] = te_map.get(row.get(col, ""), target_global_mean)

    df = pd.DataFrame([row])

    # One-hot
    oh_present = [c for c in onehot_cols if c in df.columns]
    if oh_present:
        dummies = pd.get_dummies(df[oh_present], dtype=np.float32)
        df = pd.concat([df, dummies], axis=1)

    # Scale
    num_present = [c for c in numerical_cols if c in df.columns]
    if num_present:
        df[num_present] = scaler.transform(df[num_present].fillna(0))

    # Align columns
    missing = {c: 0.0 for c in feature_cols if c not in df.columns}
    if missing:
        df = pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)
    X = df[feature_cols]

    # Predict
    y_log = float(model.predict(X)[0])
    y_brl = float(np.expm1(y_log))
    residual_std = 0.35
    lower = float(np.expm1(y_log - 1.28 * residual_std))
    upper = float(np.expm1(y_log + 1.28 * residual_std))

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Order Value", f"R$ {y_brl:,.2f}")
    col2.metric("80% Confidence Interval", f"R$ {lower:,.0f} – R$ {upper:,.0f}")
    col3.metric("Model", artifact["model_name"])

    st.success("Prediction generated!")

    st.subheader("💡 Business Insights")
    if y_brl > 200:
        st.info(
            "🎯 **High-value order.** Consider premium shipping or bundle discounts."
        )
    elif y_brl < 50:
        st.warning(
            "📦 **Low-value order.** Consider cart upsell or free shipping threshold."
        )
    if loyalty_tier in ("gold", "platinum"):
        st.info(
            "⭐ **Loyal customer.** Priority support and exclusive offers recommended."
        )
    if coupon_applied:
        st.info(
            f"🏷️ **Coupon applied:** {discount_pct:.0f}% discount factored into prediction."
        )
else:
    st.info("👈 Adjust parameters in the sidebar and click **Predict Order Value**.")

st.markdown("---")
st.markdown(
    "**Project:** Predicting Customer Order Value | "
    "**Course:** Business Analytics | "
    "[GitHub](https://github.com/alitonia/customer_value_prediction)"
)

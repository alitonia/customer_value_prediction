"""Streamlit frontend for order value prediction (Task 6.2).

Interactive demo simulating a real-time checkout order-value estimator.

Usage:
    streamlit run app/frontend/streamlit_app.py
"""

import sys
from pathlib import Path

import requests
import streamlit as st

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Order Value Predictor", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Order Value Predictor")
st.markdown("""
Predict the expected value of an e-commerce order based on customer profile,
session behavior, and cart contents. Powered by **XGBoost** trained on
Olist Brazilian E-Commerce data + synthetic behavioral features.
""")

# Sidebar inputs
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
    "Current Device", ["mobile", "desktop", "tablet"], key="session_device"
)
traffic_source = st.sidebar.selectbox(
    "Traffic Source", ["google", "direct", "facebook", "instagram", "email"]
)
traffic_medium = st.sidebar.selectbox(
    "Traffic Medium", ["organic", "cpc", "direct", "social", "email"]
)
is_logged_in = st.sidebar.checkbox("Logged In", value=True)
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

# Predict button
if st.button(" Predict Order Value", type="primary", use_container_width=True):
    payload = {
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
        "is_logged_in": is_logged_in,
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
        "is_marketing_opt_in": True,
        "payment_type": "credit_card",
        "payment_installments": 1,
        "review_score": 4.0,
        "n_events": n_product_views + n_searches + 3,
        "n_add_to_cart": max(1, total_cart_qty // 2),
        "avg_scroll": 40,
        "avg_page_duration": 30,
        "purchase_hour": 14,
        "purchase_dow": 2,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Order Value", f"R$ {data['predicted_value_brl']:,.2f}")
        col2.metric(
            "Confidence Interval (80%)",
            f"R$ {data['confidence_lower_brl']:,.0f} – R$ {data['confidence_upper_brl']:,.0f}",
        )
        col3.metric("Model", data["model_name"])

        st.success("Prediction generated successfully!")

        # Business insights
        st.subheader("💡 Business Insights")
        if data["predicted_value_brl"] > 200:
            st.info(
                "🎯 **High-value order detected.** Consider offering premium shipping or bundle discounts."
            )
        elif data["predicted_value_brl"] < 50:
            st.warning(
                "📦 **Low-value order.** Consider cart upsell recommendations or free shipping threshold."
            )

        if loyalty_tier in ("gold", "platinum"):
            st.info(
                "⭐ **Loyal customer.** Priority support and exclusive offers recommended."
            )

    except requests.ConnectionError:
        st.error(
            "❌ Cannot connect to API. Is the FastAPI server running? Start it with:\n\n"
            "`uvicorn app.api.main:app --reload --port 8000`"
        )
    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info(
        "👈 Adjust the parameters in the sidebar and click **Predict Order Value**."
    )

# Footer
st.markdown("---")
st.markdown("""
**Project:** Predicting Customer Order Value for an E-Commerce Platform |
**Course:** Business Analytics |
**Model:** XGBoost (R²=0.70, 159 features)
""")

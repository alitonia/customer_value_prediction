"""FastAPI backend for order value prediction (Task 6.1).

Loads the serialized XGBoost model and preprocessor, exposes a /predict endpoint.

Usage:
    uvicorn app.api.main:app --reload --port 8000
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED = PROJECT_ROOT / "data" / "processed"

app = FastAPI(
    title="Order Value Prediction API",
    description="Predicts e-commerce order value from customer profile and session behavior.",
    version="1.0.0",
)

# Load model artifact
with open(MODELS_DIR / "best_model.joblib", "rb") as f:
    _artifact = pickle.load(f)
_model = _artifact["model"]
_feature_cols = _artifact["feature_columns"]

# Load preprocessor
with open(PROCESSED / "preprocessor.joblib", "rb") as f:
    _preprocessor = pickle.load(f)
_scaler = _preprocessor["scaler"]
_numerical_cols = _preprocessor["numerical_cols"]
_target_encoding = _preprocessor.get("target_encoding", {})
_target_global_mean = _preprocessor.get("target_global_mean", 0.0)


class PredictionRequest(BaseModel):
    """Input features for order value prediction."""

    # Customer profile
    monthly_income: float = Field(
        3000, ge=500, le=50000, description="Monthly income in BRL"
    )
    household_size: int = Field(3, ge=1, le=10)
    loyalty_tier: str = Field("bronze", description="bronze|silver|gold|platinum")
    gender: str = Field("male")
    education_level: str = Field("bachelor")
    marital_status: str = Field("single")
    preferred_device: str = Field("mobile", description="mobile|desktop|tablet")
    registration_channel: str = Field("organic")
    is_marketing_opt_in: bool = Field(True)
    age: float = Field(30, ge=15, le=70)

    # Session
    device_type: str = Field("mobile", description="mobile|desktop|tablet")
    browser: str = Field("chrome")
    operating_system: str = Field("android")
    traffic_source: str = Field("google")
    traffic_medium: str = Field("organic")
    landing_page: str = Field("/")
    ip_country: str = Field("BR")
    ip_region: str = Field("SP")
    is_logged_in: bool = Field(True)
    session_duration_min: float = Field(5.0, ge=0.1)

    # Order / cart
    item_count: int = Field(1, ge=1)
    primary_category: str = Field("electronics")
    payment_type: str = Field("credit_card")
    payment_installments: int = Field(1, ge=1)
    review_score: float = Field(4.0, ge=1, le=5)

    # Behavioral (session activity aggregates)
    n_events: int = Field(8, ge=1)
    n_product_views: int = Field(3, ge=0)
    n_searches: int = Field(1, ge=0)
    n_add_to_cart: int = Field(2, ge=0)
    total_cart_qty: int = Field(2, ge=1)
    avg_scroll: float = Field(40, ge=0, le=100)
    avg_page_duration: float = Field(30, ge=0)

    # Temporal
    purchase_hour: int = Field(14, ge=0, le=23)
    purchase_dow: int = Field(2, ge=0, le=6)


class PredictionResponse(BaseModel):
    predicted_value_brl: float
    predicted_value_log: float
    confidence_lower_brl: float
    confidence_upper_brl: float
    model_name: str


def _build_feature_vector(req: PredictionRequest) -> pd.DataFrame:
    """Convert request to a feature dataframe matching training schema."""
    row = req.model_dump()

    # Derived features
    tier_map = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4}
    row["loyalty_numeric"] = tier_map.get(req.loyalty_tier, 1)
    row["income_x_loyalty"] = req.monthly_income * row["loyalty_numeric"]
    row["log_income"] = np.log1p(req.monthly_income)
    row["is_weekend"] = 1 if req.purchase_dow >= 5 else 0
    row["customer_order_count"] = 1  # default for new prediction
    row["cart_conversion"] = req.item_count / max(req.total_cart_qty, 1)
    row["engagement_rate"] = req.n_product_views / max(req.n_events, 1)
    row["search_intensity"] = req.n_searches / max(req.n_events, 1)
    row["items_per_minute"] = req.item_count / max(req.session_duration_min, 0.1)
    row["n_categories"] = 1
    row["n_payments"] = 1

    # Target encoding
    for col, te_map in _target_encoding.items():
        row[f"{col}_te"] = te_map.get(row.get(col, ""), _target_global_mean)

    df = pd.DataFrame([row])

    # One-hot encode categoricals
    onehot_cols = [c for c in _preprocessor.get("onehot_cols", []) if c in df.columns]
    if onehot_cols:
        dummies = pd.get_dummies(df[onehot_cols], dtype=np.float32)
        df = pd.concat([df, dummies], axis=1)

    # Scale numericals
    num_cols = [c for c in _numerical_cols if c in df.columns]
    if num_cols:
        df[num_cols] = _scaler.transform(df[num_cols].fillna(0))

    # Ensure all feature columns present (build missing cols in one concat)
    missing = {col: 0.0 for col in _feature_cols if col not in df.columns}
    if missing:
        df = pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)

    return df[_feature_cols]


@app.get("/health")
def health():
    return {"status": "ok", "model": _artifact["model_name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    X = _build_feature_vector(req)
    y_log = float(_model.predict(X)[0])
    y_brl = float(np.expm1(y_log))

    # Approximate 80% confidence interval (±1.28 * residual_std)
    residual_std = 0.35  # from training residuals
    lower_log = y_log - 1.28 * residual_std
    upper_log = y_log + 1.28 * residual_std

    return PredictionResponse(
        predicted_value_brl=round(y_brl, 2),
        predicted_value_log=round(y_log, 4),
        confidence_lower_brl=round(float(np.expm1(lower_log)), 2),
        confidence_upper_brl=round(float(np.expm1(upper_log)), 2),
        model_name=_artifact["model_name"],
    )

"""Standalone Streamlit app for HuggingFace Spaces deployment.

Loads the model directly (no FastAPI dependency) so it runs as a
single-process Streamlit Space. The estimate updates live as inputs change
(debounced via result caching + Streamlit's rerun coalescing) and supports a
dark (ink + green) / light (ink + gold) theme.

Usage (local):
    streamlit run app.py

HF Spaces: auto-detected via sdk: streamlit in README frontmatter.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
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


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# Prediction (cached — acts as the debounce for live updates)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Estimating order value…", max_entries=512)
def compute_prediction(
    monthly_income,
    household_size,
    loyalty_tier,
    gender,
    education,
    preferred_device,
    age,
    device_type,
    traffic_source,
    traffic_medium,
    is_logged_in,
    coupon_applied,
    discount_pct,
    session_duration,
    item_count,
    total_cart_qty,
    n_product_views,
    n_searches,
    primary_category,
):
    scaler = preprocessor["scaler"]
    numerical_cols = preprocessor["numerical_cols"]
    target_encoding = preprocessor.get("target_encoding", {})
    target_global_mean = preprocessor.get("target_global_mean", 0.0)
    onehot_cols = preprocessor.get("onehot_cols", [])

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

    for col, te_map in target_encoding.items():
        row[f"{col}_te"] = te_map.get(row.get(col, ""), target_global_mean)

    df = pd.DataFrame([row])

    oh_present = [c for c in onehot_cols if c in df.columns]
    if oh_present:
        dummies = pd.get_dummies(df[oh_present], dtype=np.float32)
        df = pd.concat([df, dummies], axis=1)

    # Present ALL numerical columns the scaler saw at fit time; features not
    # derivable from the demo inputs default to 0 (≈ training mean).
    num_frame = pd.DataFrame(0.0, index=df.index, columns=numerical_cols)
    for c in numerical_cols:
        if c in df.columns:
            num_frame[c] = df[c]
    scaled = pd.DataFrame(
        scaler.transform(num_frame.fillna(0.0)),
        index=df.index,
        columns=numerical_cols,
    )
    for c in numerical_cols:
        df[c] = scaled[c]

    missing = {c: 0.0 for c in feature_cols if c not in df.columns}
    if missing:
        df = pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)
    X = df[feature_cols]

    y_log = float(model.predict(X)[0])
    y_brl = float(np.expm1(y_log))
    residual_std = 0.35
    lower = float(np.expm1(y_log - 1.28 * residual_std))
    upper = float(np.expm1(y_log + 1.28 * residual_std))

    explainer = get_explainer()
    shap_values = explainer.shap_values(X)
    base_log = float(explainer.expected_value)

    avg_brl = float(np.expm1(base_log))
    delta_pct = (y_brl - avg_brl) / avg_brl * 100 if avg_brl else None

    ti = int(np.argmax(np.abs(shap_values[0])))
    top_driver = X.columns[ti]
    top_dir = "pushes value up" if shap_values[0][ti] > 0 else "pushes value down"

    order = np.argsort(np.abs(shap_values[0]))[::-1]
    factors = []
    for i in order[:10]:
        sv = float(shap_values[0][i])
        if abs(sv) < 1e-4:
            continue
        factors.append((X.columns[i], sv))

    return {
        "y_brl": y_brl,
        "lower": lower,
        "upper": upper,
        "avg_brl": avg_brl,
        "delta_pct": delta_pct,
        "top_driver": top_driver,
        "top_dir": top_dir,
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# Theme palettes (CSS custom properties)
# ---------------------------------------------------------------------------

LIGHT_VARS = """
    --bg:#FBFAF7; --dot:#E9E4D9;
    --surface:#FFFFFF; --surface2:#F3EFE6; --row-hover:#FBF4E6;
    --text:#201D19; --muted:#5F594E; --faint:#8E877A; --line:#E2DCCF;
    --accent:#C8102E; --accent-hover:#E02640; --accent-deep:#8E0B20;
    --ticker-bg:#171310; --top-rule:#E8A33D; --number:#F5C05A;
    --number-glow:rgba(245,192,90,.28); --ticker-label:#A2977F; --ticker-text:#F7F3EA;
    --pos:#0E8A4C; --neg:#C8102E; --pos-bg:#DCF3E5; --neg-bg:#FBE2E6; --neu-bg:#EFEAE0; --neu:#5F594E;
    --delta-pos:#3FCB7E; --delta-neg:#FF6B7A;
    --sb-bg:#171310; --sb-text:#F7F3EA; --sb-label:#C7BDA9;
    --sb-widget-bg:#241E17; --sb-widget-border:#3D3427; --sb-thumb:#E8A33D;
    --sb-header-border:#352C21; --sb-tick:#8E8471;
"""

DARK_VARS = """
    --bg:#0C100D; --dot:#18211A;
    --surface:#141B16; --surface2:#1B241D; --row-hover:#1D2821;
    --text:#E4ECE6; --muted:#9FB0A4; --faint:#6E7F73; --line:#24302A;
    --accent:#2FB673; --accent-hover:#3ED089; --accent-deep:#1B7A4C;
    --ticker-bg:#111814; --top-rule:#34D399; --number:#4ADE80;
    --number-glow:rgba(74,222,128,.32); --ticker-label:#7C8F82; --ticker-text:#E4ECE6;
    --pos:#4ADE80; --neg:#F87171; --pos-bg:#12291C; --neg-bg:#2B1517; --neu-bg:#1B241D; --neu:#9FB0A4;
    --delta-pos:#4ADE80; --delta-neg:#F87171;
    --sb-bg:#0A0E0B; --sb-text:#E4ECE6; --sb-label:#8FA396;
    --sb-widget-bg:#141B17; --sb-widget-border:#26322A; --sb-thumb:#34D399;
    --sb-header-border:#223027; --sb-tick:#5F7266;
"""

MAIN_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,900&family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    [data-testid="stSidebarDeployButton"],
    .st-emotion-cache-1wbqy5l,
    button[kind="headerNoPadding"] { display: none !important; }

    /* ===== Canvas ===== */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg);
        background-image: radial-gradient(var(--dot) 1px, transparent 1px);
        background-size: 26px 26px;
    }
    .block-container { padding-top: clamp(1rem, 3vw, 2rem); padding-bottom: 0; max-width: 1180px; }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] { background: var(--sb-bg); }
    [data-testid="stSidebar"] > div:first-child { background: var(--sb-bg) !important; }
    [data-testid="stSidebar"] h2 {
        color: var(--sb-text) !important; font-family: 'Fraunces', serif !important;
        font-weight: 700 !important; font-size: 1.02rem !important;
        border-bottom: 1px solid var(--sb-header-border); padding-bottom: .45rem; margin-top: 1.1rem;
    }
    [data-testid="stSidebar"] label {
        color: var(--sb-label) !important; font-family: 'IBM Plex Mono', monospace !important;
        font-size: .7rem !important; text-transform: uppercase; letter-spacing: .09em;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: var(--sb-widget-bg) !important; border-color: var(--sb-widget-border) !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div * { color: var(--sb-text) !important; }
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div { background: var(--sb-widget-border) !important; }
    [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
        background: var(--sb-thumb) !important; border-color: var(--sb-thumb) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="checkbox"] {
        background: var(--sb-widget-bg) !important; border-color: var(--sb-widget-border) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="checkbox"][aria-checked="true"] {
        background: var(--accent) !important; border-color: var(--accent) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] { color: var(--sb-tick) !important; }

    /* ===== Header ===== */
    .ov-overline { font: 700 .68rem 'IBM Plex Mono', monospace; text-transform: uppercase;
                   letter-spacing: .24em; color: var(--accent); margin-bottom: .4rem; }
    h1 { font-family: 'Fraunces', serif !important; font-weight: 900 !important;
         font-size: clamp(1.6rem, 3.4vw, 2.5rem) !important; letter-spacing: -.02em;
         color: var(--text) !important; margin-bottom: .15rem !important; }
    [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

    /* ===== Estimate ticker ===== */
    .ov-quote {
        display: flex; align-items: stretch; flex-wrap: wrap;
        background: var(--ticker-bg); border: 1px solid var(--line); border-top: 3px solid var(--top-rule);
        border-radius: 12px; margin: 1.1rem 0 1.3rem 0; overflow: hidden;
        box-shadow: 0 14px 34px rgba(0,0,0,.22);
        animation: ov-rise .5s cubic-bezier(.22,1,.36,1);
    }
    @keyframes ov-rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
    .ov-quote-main { padding: 1.25rem 1.7rem; flex: 1 1 280px; min-width: 0; }
    .ov-quote-label { font: 700 .66rem 'IBM Plex Mono', monospace; text-transform: uppercase;
        letter-spacing: .18em; color: var(--ticker-label); }
    .ov-quote-value { font: 900 clamp(2.5rem, 6vw, 4.3rem)/1 'Fraunces', serif; color: var(--number);
        letter-spacing: -.02em; font-variant-numeric: tabular-nums; margin: .35rem 0 .3rem 0;
        text-shadow: 0 0 26px var(--number-glow); }
    .ov-quote-ci { font: 500 .84rem 'IBM Plex Sans', sans-serif; color: var(--ticker-label); }
    .ov-quote-divider { width: 1px; background: var(--line); margin: 1.1rem 0; }
    .ov-quote-stat { padding: 1.25rem 1.7rem; display: flex; flex-direction: column;
        justify-content: center; flex: 1 1 190px; min-width: 0; }
    .ov-stat-label { font: 700 .62rem 'IBM Plex Mono', monospace; text-transform: uppercase;
        letter-spacing: .15em; color: var(--ticker-label); }
    .ov-stat-value { font: 600 clamp(1.2rem, 2.4vw, 1.55rem)/1.2 'IBM Plex Mono', monospace;
        color: var(--ticker-text); margin-top: .4rem; font-variant-numeric: tabular-nums; }
    .ov-stat-value.pos { color: var(--delta-pos); }
    .ov-stat-value.neg { color: var(--delta-neg); }
    .ov-stat-driver { font: 600 1.02rem/1.3 'IBM Plex Mono', monospace; color: var(--ticker-text);
        margin-top: .4rem; overflow-wrap: anywhere; }
    .ov-stat-sub { font: 400 .74rem 'IBM Plex Sans', sans-serif; color: var(--ticker-label); margin-top: .22rem; }

    /* ===== Section label ===== */
    .ov-section { display: flex; align-items: center; gap: .8rem;
        font: 700 .72rem 'IBM Plex Mono', monospace; text-transform: uppercase;
        letter-spacing: .2em; color: var(--text); margin: .5rem 0 .65rem 0; }
    .ov-section::before { content: ""; width: 24px; height: 3px; background: var(--accent); }
    .ov-section::after { content: ""; flex: 1; height: 1px; background: var(--line); }

    /* ===== Recommendations ===== */
    .ov-recs { display: flex; flex-direction: column; gap: .55rem; }
    .ov-rec { display: flex; align-items: baseline; gap: .85rem; background: var(--surface);
        border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 7px;
        padding: .62rem .95rem; transition: transform .13s ease, box-shadow .13s ease; }
    .ov-rec:hover { transform: translateX(4px); box-shadow: 0 4px 12px rgba(0,0,0,.12); }
    .ov-rec-tag { font: 700 .58rem 'IBM Plex Mono', monospace; letter-spacing: .13em;
        padding: .22rem .5rem; border-radius: 4px; white-space: nowrap; }
    .tag-pos { background: var(--pos-bg); color: var(--pos); }
    .tag-neg { background: var(--neg-bg); color: var(--neg); }
    .tag-neu { background: var(--neu-bg); color: var(--neu); }
    .ov-rec-text { font: 400 .82rem/1.5 'IBM Plex Sans', sans-serif; color: var(--text); }

    /* ===== Expander (native) ===== */
    [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line);
        border-radius: 10px; color: var(--text); }
    [data-testid="stExpander"] summary { color: var(--text) !important; }
    [data-testid="stExpander"] svg { fill: var(--muted) !important; }

    /* ===== Loading spinner ===== */
    [data-testid="stSpinner"] { color: var(--muted);
        font: 500 .8rem 'IBM Plex Mono', monospace; }

    /* ===== Contribution chart (diverging bars) ===== */
    .wf { display: flex; flex-direction: column; gap: .5rem; background: var(--surface);
        border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.1rem; }
    .wf-row { display: grid; grid-template-columns: minmax(96px, 1.3fr) 2.4fr auto;
        align-items: center; gap: .75rem; }
    .wf-name { font: 600 .74rem 'IBM Plex Mono', monospace; color: var(--text);
        text-align: right; overflow-wrap: anywhere; }
    .wf-track { position: relative; height: 16px; background: var(--surface2); border-radius: 3px; }
    .wf-track::before { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px;
        width: 1px; background: var(--line); }
    .wf-fill { position: absolute; top: 2px; bottom: 2px; border-radius: 2px;
        transition: width .25s ease; }
    .wf-fill.pos { background: var(--pos); }
    .wf-fill.neg { background: var(--neg); }
    .wf-val { font: 600 .74rem 'IBM Plex Mono', monospace; min-width: 54px; text-align: right;
        font-variant-numeric: tabular-nums; }
    .wf-val.pos { color: var(--pos); }
    .wf-val.neg { color: var(--neg); }
    @media (max-width: 560px) {
        .wf-row { grid-template-columns: minmax(70px, 1fr) 2fr auto; gap: .5rem; }
        .wf-name { font-size: .66rem; }
    }

    /* ===== Footer ===== */
    .ov-footer { font: 600 .64rem 'IBM Plex Mono', monospace; text-transform: uppercase;
        letter-spacing: .18em; color: var(--faint); border-top: 2px solid var(--line);
        margin-top: 2.1rem; padding-top: .85rem; }

    /* ===== Responsive ===== */
    @media (max-width: 720px) {
        .ov-quote { flex-direction: column; }
        .ov-quote-divider { width: auto; height: 1px; margin: 0 1.4rem; }
        .ov-quote-main, .ov-quote-stat { padding: 1rem 1.3rem; }
    }

    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
"""

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Order Value Predictor", page_icon="🛒", layout="wide")

dark = st.sidebar.toggle("Dark mode", value=True, help="Ink + green theme")
st.markdown(
    f"<style>:root {{ {DARK_VARS if dark else LIGHT_VARS} }}</style>",
    unsafe_allow_html=True,
)
st.markdown(f"<style>{MAIN_CSS}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="ov-overline">Revenue analytics · Brazilian e-commerce</div>',
    unsafe_allow_html=True,
)
st.title("Order Value Predictor")
st.caption(
    f"{artifact['model_name']} · R² {artifact['metrics']['R2_log']:.2f} on log-target · "
    f"{len(feature_cols)} features · updates live as you adjust inputs"
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Live estimate (no button — recomputes as inputs change; cached = debounced)
# ---------------------------------------------------------------------------

result = compute_prediction(
    monthly_income,
    household_size,
    loyalty_tier,
    gender,
    education,
    preferred_device,
    age,
    device_type,
    traffic_source,
    traffic_medium,
    is_logged_in,
    coupon_applied,
    discount_pct,
    session_duration,
    item_count,
    total_cart_qty,
    n_product_views,
    n_searches,
    primary_category,
)

y_brl = result["y_brl"]
lower = result["lower"]
upper = result["upper"]
avg_brl = result["avg_brl"]
delta_pct = result["delta_pct"]

delta_cls = "pos" if (delta_pct or 0) >= 0 else "neg"
delta_txt = f"{delta_pct:+.0f}%" if delta_pct is not None else "—"
avg_txt = f"avg R$ {avg_brl:,.0f}" if avg_brl else "&nbsp;"

st.markdown(
    f"""
    <div class="ov-quote">
      <div class="ov-quote-main">
        <div class="ov-quote-label">Predicted order value</div>
        <div class="ov-quote-value">R$ {y_brl:,.0f}</div>
        <div class="ov-quote-ci">80% confidence · R$ {lower:,.0f} – R$ {upper:,.0f}</div>
      </div>
      <div class="ov-quote-divider"></div>
      <div class="ov-quote-stat">
        <div class="ov-stat-label">vs. average order</div>
        <div class="ov-stat-value {delta_cls}">{delta_txt}</div>
        <div class="ov-stat-sub">{avg_txt}</div>
      </div>
      <div class="ov-quote-divider"></div>
      <div class="ov-quote-stat">
        <div class="ov-stat-label">strongest driver</div>
        <div class="ov-stat-driver">{result['top_driver']}</div>
        <div class="ov-stat-sub">{result['top_dir']}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ov-section">Contribution breakdown · SHAP</div>',
    unsafe_allow_html=True,
)

factors = result["factors"]
max_abs = max((abs(sv) for _, sv in factors), default=1.0) or 1.0
bars_html = []
for name, sv in factors:
    half = abs(sv) / max_abs * 50
    cls = "pos" if sv > 0 else "neg"
    style = (
        f"left:50%;width:{half:.1f}%"
        if sv > 0
        else f"left:{50 - half:.1f}%;width:{half:.1f}%"
    )
    bars_html.append(
        f'<div class="wf-row">'
        f'<div class="wf-name">{name}</div>'
        f'<div class="wf-track"><div class="wf-fill {cls}" style="{style}"></div></div>'
        f'<div class="wf-val {cls}">{sv:+.3f}</div>'
        f'</div>'
    )
st.markdown(f'<div class="wf">{"".join(bars_html)}</div>', unsafe_allow_html=True)
st.caption(
    "Additive SHAP decomposition in log(1+R$) space — bars to the right raise the "
    "estimate, bars to the left lower it, relative to the average order."
)

recs = []
if y_brl > 200:
    recs.append(("HIGH VALUE", "Offer premium shipping or a bundle discount to protect margin.", "pos"))
elif y_brl < 50:
    recs.append(("LOW VALUE", "Trigger a cart upsell or a free-shipping threshold nudge.", "neg"))
if loyalty_tier in ("gold", "platinum"):
    recs.append(("LOYALTY", "Route to priority support and reserve exclusive offers.", "pos"))
if coupon_applied:
    recs.append(("COUPON", f"The {discount_pct:.0f}% discount is factored into this estimate.", "neu"))
if recs:
    with st.expander("Recommended actions", expanded=True):
        rec_html = "".join(
            f'<div class="ov-rec"><span class="ov-rec-tag tag-{tone}">{tag}</span>'
            f'<span class="ov-rec-text">{txt}</span></div>'
            for tag, txt, tone in recs
        )
        st.markdown(f'<div class="ov-recs">{rec_html}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="ov-footer">Predicting Customer Order Value · Business Analytics · Group 13</div>',
    unsafe_allow_html=True,
)

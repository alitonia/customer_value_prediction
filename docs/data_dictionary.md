# Data Dictionary — Synthetic Behavioral Data

Every synthetically generated field is documented below with column name, data type, unit, valid range, and generation logic.

Source: `src/data/generate_synthetic.py` (seed=42). Output: `data/synthetic/*.parquet`.

---

## Table 1: customer_profile (96,096 rows — 1 per unique customer)

| Column | Type | Unit | Valid Range / Categories | Generation Logic |
|---|---|---|---|---|
| customer_id | varchar | — | 32-char hex | FK → Olist `customers.customer_unique_id` |
| gender | varchar | — | male, female, other | Multinomial [0.50, 0.49, 0.01]. Independent of order value. |
| birth_date | date | — | 1950-01-01 to 2003-12-31 | Uniform integer days. Independent. |
| marital_status | varchar | — | single, married, divorced, widowed | Multinomial [0.40, 0.35, 0.15, 0.10]. Independent. |
| occupation | varchar | — | professional, student, self_employed, retired, unemployed, homemaker, other | Multinomial [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.15]. Independent. |
| education_level | varchar | — | high_school, bachelor, master, phd, none | Multinomial [0.30, 0.35, 0.20, 0.05, 0.10]. Independent. |
| monthly_income | decimal | BRL | [500, 50000] | `exp(log(2500) + 0.35 * vs + N(0, 0.50))`. Conditioned on customer avg order value (vs = z-scored log-value). Target r ≈ 0.3–0.5. |
| household_size | int | persons | [1, 10] | `Poisson(clip(2.5 + 0.4 * vs, 1, 8))`. Mild positive correlation with value. |
| loyalty_tier | varchar | — | bronze, silver, gold, platinum | Base [0.50, 0.30, 0.15, 0.05] shifted by `0.6 * vs`. Higher value → higher tier probability. |
| registration_channel | varchar | — | organic, paid_ads, referral, social, email | Multinomial [0.40, 0.25, 0.15, 0.12, 0.08]. Independent. |
| is_marketing_opt_in | boolean | — | True / False | `Bernoulli(0.55 + 0.10 * sigmoid(vs))`. Slight positive correlation. |
| preferred_device | varchar | — | mobile, desktop, tablet | Base [0.65, 0.30, 0.05] with desktop probability increasing by `0.12 * sigmoid(0.8 * vs)`. |
| created_at | timestamp | — | 2016-01-01 to 2018-06-01 | Uniform random seconds. |
| updated_at | timestamp | — | created_at + [0, 180] days | created_at + Uniform(0, 180) days. |

## Table 2: sessions (99,441 rows — 1 per order)

| Column | Type | Unit | Valid Range / Categories | Generation Logic |
|---|---|---|---|---|
| session_id | varchar | — | UUID4 | Globally unique. |
| customer_id | varchar | — | 32-char hex | FK → `customer_profile.customer_id`. |
| order_id | varchar | — | 32-char hex | FK → Olist `orders.order_id`. 1:1 mapping. |
| session_start | timestamp | — | Before order_purchase_timestamp | `order_purchase_timestamp − Uniform(300, 10800)s`. |
| session_end | timestamp | — | After session_start | `session_start + duration`. |
| device_type | varchar | — | mobile, desktop, tablet | 90% matches `customer_profile.preferred_device`; 10% random noise. |
| browser | varchar | — | chrome, safari, firefox, edge, other | Multinomial [0.60, 0.20, 0.10, 0.07, 0.03]. Independent. |
| operating_system | varchar | — | android, windows, ios, macos, linux | Base [0.35, 0.30, 0.20, 0.10, 0.05] shifted by `0.15 * vs`. iOS/macOS tilt for high-value. |
| traffic_source | varchar | — | google, direct, facebook, instagram, email | Base [0.40, 0.25, 0.15, 0.10, 0.10] shifted by `0.3 * vs`. Email/direct up for high-value. |
| traffic_medium | varchar | — | organic, cpc, direct, social, email | Base [0.35, 0.20, 0.25, 0.12, 0.08] shifted by `0.3 * vs`. |
| campaign_name | varchar | — | 10 campaign names or null | Null if medium ∈ {direct, organic}; else uniform from campaign list. |
| landing_page | varchar | — | /, /category, /product, /search | Multinomial [0.40, 0.30, 0.20, 0.10]. Independent. |
| ip_country | varchar | — | BR, US, AR, PT, other | Multinomial [0.93, 0.02, 0.02, 0.01, 0.02]. Independent. |
| ip_region | varchar | — | 27 Brazilian state codes | Mapped from customer's `customer_state`. |
| is_logged_in | boolean | — | True / False | `Bernoulli(0.70 + 0.15 * sigmoid(vs))`. Higher for high-value orders. |
| coupon_applied | boolean | — | True / False | `Bernoulli(0.35 - 0.10 * sigmoid(0.5 * vs))`. Mildly negatively correlated with value (price-sensitive customers use coupons more). |
| discount_amount_pct | decimal | % | 0 or [5.0, 25.0] | 0 if no coupon; else Uniform(5, 25). Applied discount percentage. |
| created_at | timestamp | — | Same as session_start | Copy of session_start. |

## Table 3: session_activity (763,850 rows — ~8 events per session)

| Column | Type | Unit | Valid Range / Categories | Generation Logic |
|---|---|---|---|---|
| activity_id | varchar | — | UUID4 | Globally unique. |
| session_id | varchar | — | UUID4 | FK → `sessions.session_id`. |
| customer_id | varchar | — | 32-char hex | FK → `customer_profile.customer_id`. |
| order_id | varchar | — | 32-char hex | FK → Olist `orders.order_id`. Denormalized for join convenience. |
| activity_timestamp | timestamp | — | Between session_start and session_end | Sorted uniform fractions within session window. |
| activity_type | varchar | — | page_view, search, product_view, add_to_cart, checkout | Base [0.50, 0.15, 0.20, 0.10, 0.05] shifted by `0.4 * vs`. High-value → more product_view/add_to_cart. Last event forced to checkout; ≥1 add_to_cart per session. |
| page_url | varchar | — | /, /category, /product, /search, /cart, /checkout, /about, /help | Derived from activity_type. |
| product_id | varchar | — | Olist product_id or null | Non-null for product_view/add_to_cart; uniform from Olist products. |
| search_keyword | varchar | — | Olist category name or null | Non-null for search; uniform from product categories. |
| duration_seconds | int | seconds | [0, 600] | 0 for non-view events. For page_view/product_view: `exp(3.2 + 0.15 * vs + N(0, 0.7))`, clipped [1, 600]. |
| scroll_percent | int | % | [0, 100] | 0 for non-view events. For views: `Beta(2 + 0.5 * clip(vs, -2, 2), 3) * 100`. Deeper scroll for high-value. |
| add_to_cart_quantity | int | items | [0, 5] | 0 for non-cart events. For add_to_cart: `Poisson(clip(1.2 + 0.3 * vs, 1, 5))`, clipped [1, 5]. Post-hoc fix ensures sum per session ≥ actual order item count. |
| created_at | timestamp | — | Same as activity_timestamp | Copy of activity_timestamp. |

---

## Olist Base Tables (reference only)

| Table | Rows | Key Columns |
|---|---|---|
| olist_orders_dataset | 99,441 | order_id, customer_id, order_status, order_purchase_timestamp |
| olist_order_items_dataset | 112,650 | order_id, product_id, price, freight_value |
| olist_customers_dataset | 99,441 | customer_id, customer_unique_id, customer_state |
| olist_products_dataset | 32,951 | product_id, product_category_name |
| olist_order_payments_dataset | 103,886 | order_id, payment_type, payment_value |
| olist_order_reviews_dataset | 99,224 | order_id, review_score |
| olist_sellers_dataset | 3,095 | seller_id, seller_state |
| olist_geolocation_dataset | 1,000,163 | geolocation_zip_code_prefix, lat, lng |
| product_category_name_translation | 71 | product_category_name, product_category_name_english |

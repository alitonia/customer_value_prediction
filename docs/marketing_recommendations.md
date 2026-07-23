# Marketing & Promotion Recommendations (Task 8.1)

Based on XGBoost feature importances and EDA insights from the order value prediction model.

## Key Value Drivers (from model)

| Rank | Feature | Importance | Business Interpretation |
|---|---|---|---|
| 1 | income × loyalty interaction | 14.0% | High-income loyal customers are the most valuable segment |
| 2 | Session events count | 8.8% | More engaged sessions → higher order values |
| 3 | Cart additions | 8.8% | Cart-building behavior strongly predicts order value |
| 4 | Product views | 5.2% | Browsing depth correlates with basket size |
| 5 | Total cart quantity | 5.1% | Larger carts → higher order values |
| 6 | Payment installments | 3.5% | Installment buyers tend to purchase higher-value items |
| 7 | Category: telephony | 2.7% | Phone/electronics category drives high AOV |
| 8 | Log income | 2.6% | Income is a direct predictor of spending capacity |
| 9 | Category: electronics | 2.4% | Electronics = high-value category |
| 10 | Loyalty: bronze | 2.3% | Bronze tier = lowest value segment (upgrade opportunity) |

## Actionable Recommendations

### 1. Loyalty Tier Upgrade Campaigns
**Finding:** Bronze → platinum shows 2.5× AOV spread (R$82 → R$202).
**Action:** Target bronze-tier customers with personalized upgrade incentives (e.g., "Spend R$50 more to unlock Silver benefits"). The model shows loyalty tier is a top-10 feature — upgrading customers directly increases predicted order value.

### 2. Email Channel Investment
**Finding:** Email traffic median AOV = R$146 vs Google organic = R$89 (64% premium).
**Action:** Increase email marketing budget. Email subscribers are existing customers with higher purchase intent. Segment email campaigns by predicted order value tier.

### 3. Desktop-Optimized Checkout
**Finding:** Desktop users have 7% higher median order value (R$110 vs R$103).
**Action:** Invest in desktop UX for high-value categories (computers, electronics). Consider desktop-exclusive bundle offers for categories where desktop AOV premium is largest.

### 4. Cart Upsell for Low-Engagement Sessions
**Finding:** Sessions with < 5 events and < 2 cart additions predict low order values.
**Action:** Trigger real-time upsell recommendations ("Customers who bought X also bought Y") when session engagement is below threshold. The model shows cart additions are the #3 feature — increasing cart size directly increases predicted value.

### 5. High-Value Category Promotion
**Finding:** Computers (R$1,251 median), electronics, and telephony are top-value categories.
**Action:** Feature these categories prominently in homepage banners and push notifications. Offer free shipping on orders > R$200 in these categories to increase conversion.

### 6. Income-Based Personalization
**Finding:** Monthly income (r=0.51 with order value) and the income×loyalty interaction (#1 feature) are the strongest predictors.
**Action:** Use predicted income tier (from loyalty + browsing behavior) to personalize product recommendations. Show premium products to high-income segments, value deals to budget segments.

### 7. Weekend Flash Sales
**Finding:** Purchase timing (hour, day-of-week) contributes to predictions.
**Action:** Schedule flash sales during peak engagement hours (identified from session duration patterns). Weekend shoppers show different category preferences — tailor promotions accordingly.

## Model Limitations & Future Work

1. **Synthetic data:** Behavioral features are simulated with value-conditioned correlations. Real clickstream data would likely improve predictions.
2. **KPI targets not met:** MAE=R$43 (target <R$25), WAPE=30% (target <16%). Additional features (product-level pricing, promotion history, competitor pricing) needed.
3. **Temporal features:** Seasonality and trend not captured. Time-series features (rolling averages, month-over-month growth) could improve accuracy.
4. **Customer lifetime value:** Current model predicts single-order value. Extending to CLV prediction would enable longer-term marketing optimization.

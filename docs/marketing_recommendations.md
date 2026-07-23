# Marketing and Promotion Recommendations

The following recommendations are derived from the XGBoost model's feature importance ranking and the exploratory data analysis findings. Each recommendation identifies a specific lever that the business can act upon, supported by quantitative evidence from the model.

---

## 1. Loyalty Tier Upgrade Programs

**Evidence:** The loyalty tier feature exhibits a 2.5× spread in median order value, from R$82 for bronze members to R$202 for platinum members. The income × loyalty interaction is the single most important feature in the model (14% importance), indicating that the combination of spending capacity and platform engagement is the strongest predictor of order value.

**Recommendation:** Implement structured upgrade incentives that encourage bronze-tier customers to reach silver status. Threshold-based rewards — for example, "accumulate R$200 in purchases this quarter to unlock Silver benefits" — directly target the most influential categorical predictor. The expected return is proportional to the R$120 median AOV increase per upgraded customer.

---

## 2. Email Channel Investment

**Evidence:** Email-sourced traffic produces a median order value of R$146, compared to R$89 for Google organic traffic — a 64% premium. Email subscribers are predominantly existing customers with established purchase intent and brand familiarity.

**Recommendation:** Increase email marketing budget allocation, particularly for segmented campaigns targeting high-value customer profiles. The cost per acquisition for email campaigns is typically lower than paid search, and the higher AOV among email-sourced orders improves the return on marketing spend. Priority segments include silver and gold loyalty members who have not made a purchase in the past 30 days.

---

## 3. Desktop Experience Optimization

**Evidence:** Desktop sessions yield a 7% higher median order value (R$110) than mobile sessions (R$103). For high-value categories such as computers and electronics, where comparison shopping and detailed product evaluation are common, the desktop advantage is likely more pronounced.

**Recommendation:** Invest in desktop-optimized checkout flows with enhanced product comparison tools, side-by-side specification views, and persistent cart functionality. These features support the deliberative purchasing behavior associated with higher-value orders.

---

## 4. Real-Time Cart Upsell Triggers

**Evidence:** Cart additions rank third in feature importance (8.8%), and total cart quantity ranks fifth (5.1%). Sessions with low engagement — fewer than five events and a single cart item — represent orders with the lowest predicted values.

**Recommendation:** Deploy real-time "customers also purchased" recommendations when session engagement falls below a defined threshold. The objective is to increase cart depth before the customer proceeds to checkout. The model indicates that each additional cart item contributes meaningfully to predicted order value.

---

## 5. Income-Tier Personalization

**Evidence:** The income × loyalty interaction (14% importance) and log income (2.6% importance) together account for the largest share of predictive power among customer-level features. The model effectively segments customers by their predicted spending capacity.

**Recommendation:** Use the model's predicted value tier as a real-time segmentation variable. High-value segments should receive premium product recommendations and expedited shipping options. Budget segments should receive value-oriented promotions and bundle discounts. This personalization can be implemented at the recommendation engine level without modifying the core prediction model.

---

## 6. High-Value Category Promotion

**Evidence:** Product category is the strongest transactional driver of order value. Computers (R$1,251 median), electronics, and telephony represent the highest-value categories. The category price × income interaction feature further confirms that high-income customers purchasing from expensive categories produce the highest order values.

**Recommendation:** Feature high-value categories prominently in homepage banners, push notifications, and email campaigns. Consider category-specific free shipping thresholds calibrated to the median order value within each category.

---

## 7. Weekend and Seasonal Campaign Timing

**Evidence:** Temporal features (purchase month, quarter, day of month) contribute to the model's predictions, indicating that order value varies with seasonality and purchasing cycles.

**Recommendation:** Align promotional campaigns with peak engagement periods identified in the temporal analysis. Flash sales and limited-time offers should be scheduled during periods when the model predicts higher baseline order values, maximizing the incremental revenue from promotional spend.

---

## Model Limitations Relevant to Marketing Strategy

The recommendations above are constrained by the following limitations:

- The behavioral data is synthetically generated with designed correlations. Real clickstream data may reveal different feature importances and modify the priority ordering of these recommendations.
- The model does not incorporate product-level pricing, which limits its ability to distinguish between high-value and low-value purchases within the same category.
- Ninety-seven percent of customers have a single order, limiting the utility of customer-level behavioral trends for personalization.

These limitations should be considered when allocating budget across the recommended initiatives. Pilot programs with controlled measurement are advisable before full-scale rollout.

# Task 1.1: KPIs and Business Value Specification

This document defines the key performance indicators (KPIs) and the financial value model for the Customer Order Value Prediction system. 

---

## 1. Project Evaluation Metrics (ML KPIs)

To measure the predictive power of our regression models, we will track four primary metrics. Our target values represent the threshold for a production-ready model.

| Metric | Definition | Purpose | Target Threshold |
| :--- | :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\frac{1}{n} \sum |y_i - \hat{y}_i|$ | Measures the average absolute error in the currency unit (Brazilian Real, BRL). Easy to explain to business stakeholders. | **< R$ 25.00** |
| **RMSE** (Root Mean Squared Error) | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ | Penalizes larger prediction errors heavily. Useful for identifying anomalous orders. | **< R$ 40.00** |
| **MAPE** (Mean Absolute Percentage Error) | $\frac{100\%}{n} \sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|$ | Evaluates error relative to the actual order size. Standardizes error scale across different customer segments. | **< 15.0%** |
| **$R^2$** (Coefficient of Determination) | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Quantifies the proportion of order value variance explained by customer behavioral features. | **> 0.70** |

---

## 2. Business Value Formulations

Predicting order value is not just an ML exercise; it directly affects e-commerce margins and operational costs. We model two primary business impacts:

### Impact A: Promotional Campaign Optimization (Margin Erosion vs. Conversion Lift)
E-commerce platforms use predicted order value to decide the value of promotional coupons. 
* **The Naive Approach**: Offering a flat R$ 15 coupon to all users.
* **The ML-driven Approach**: Offering dynamically scaled coupons based on predicted order value ($COUPON = 10\% \times \hat{y}$).

#### Mathematical Modeling of Errors:
1. **Overestimation ($\hat{y} > y$)**: We issue a coupon value that is too high relative to the actual checkout value, causing **Margin Erosion**:
   $$\text{Margin Loss} = \text{Coupon Issued} - \text{Optimal Coupon} = 0.10 \times (\hat{y} - y)$$
2. **Underestimation ($\hat{y} < y$)**: We issue a coupon value that is too low to incentivize the purchase, leading to a **Missed Conversion**:
   $$\text{Opportunity Loss} = P(\text{Conversion} | y) \times y \times \text{Gross Margin \%}$$

> **Objective**: By keeping MAPE < 15%, we minimize the total loss:
> $$\text{Min } \sum \left( \text{Margin Loss} + \text{Opportunity Loss} \right)$$

### Impact B: Logistics and Carrier Allocation (SLA Violations vs. Over-capacity)
In e-commerce, order monetary value correlates strongly with item weight/dimensions. Correctly predicting the basket value allows logistics partners to pre-allocate courier capacity.
* **Over-allocation Cost**: Pre-booking cargo space for a predicted high-value order that turns out to be small:
   $$\text{Over-capacity Loss} = \text{Reserved Volume Rate} - \text{Actual Shipping Rate}$$
* **Under-allocation Cost**: Not reserving enough space for a large package, leading to shipping delays and compensation payouts:
   $$\text{SLA Penalty} = \text{Delay Compensation Fee} + \text{Customer Churn Risk}$$

---

## 3. Financial Target Case Study (ROI Projection)

Based on the raw Olist dataset characteristics:
* **Average Order Value (AOV)**: ~R$ 138.00
* **Total Monthly Orders**: ~10,000 orders
* **Monthly Revenue**: R$ 1,380,000.00
* **Naive Baseline MAPE (using historical AOV mean)**: ~38.0%

### Estimated Financial Value of Improving MAPE to 15.0%:
By reducing our error rate from 38.0% to 15.0% (a **23.0% absolute improvement**):
1. **Promo Spend Efficiency**: Saving R$ 1.50 per order in over-allocated discounts:
   $$\text{Monthly Savings} = 10,000 \times R\$ 1.50 = R\$ 15,000.00$$
2. **Reduced Churn from Logistics Delays**: Preventing SLA penalties on approximately 2% of total orders (200 orders saved) at R$ 50.00 in churn value per customer:
   $$\text{Monthly Retention Value} = 200 \times R\$ 50.00 = R\$ 10,000.00$$

**Total Projected Monthly Financial Return**: **R$ 25,000.00** (approx. **R$ 300,000.00 annually**).

---

## 4. Operational Dashboard Targets

Our model tracking dashboard (Module 7) will fire alerts when the rolling ML KPIs exceed the following operational limits:
* **Critical Alert**: Rolling 7-day MAPE exceeds **18.0%** (triggers immediate retraining).
* **Warning Alert**: Rolling 7-day MAE exceeds **R$ 28.00** (notifies the modeling team).
* **Data Drift Alert**: PSI (Population Stability Index) of predicted values exceeds **0.25** (indicates customer behavior shift).

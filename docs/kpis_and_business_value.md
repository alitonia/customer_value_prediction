# Task 1.1: KPIs and Business Value Specification (Revised V2)

This document defines the key performance indicators (KPIs) and the financial value model for the Customer Order Value Prediction system, addressing right-skewness and commercial realities.

---

## 1. Project Evaluation Metrics (ML KPIs)

To measure the predictive power of our regression models, we track four primary metrics. Our target values represent the threshold for a production-ready model.

| Metric | Definition | Purpose | Target Threshold |
| :--- | :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\frac{1}{n} \sum |y_i - \hat{y}_i|$ | Measures the average absolute error in the currency unit (Brazilian Real, BRL). Easy to explain to business stakeholders. | **< R$ 25.00** |
| **WAPE** (Weighted Absolute Percentage Error) | $\frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$ | Evaluates total absolute error relative to total revenue. Replaces MAPE to avoid artificial inflation from low-value orders. | **< 16.0%** |
| **MedAPE** (Median Absolute Percentage Error) | $\text{median}\left( \left|\frac{y_i - \hat{y}_i}{y_i}\right| \right)$ | Measures the median percentage error. Extremely robust to extreme high-value outliers. | **< 12.0%** |
| **$R^2$** (Coefficient of Determination) | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Quantifies the proportion of order value variance explained by customer behavioral features. | **> 0.70** |

---

## 2. Business Value Formulations

Predicting order value directly affects e-commerce margins and operational costs. We model two primary business impacts:

### Impact A: Dynamic Upsell Coupons (AOV Lift vs. Margin Erosion)
Instead of cannibalizing margins by giving discounts to customers who are already predicted to spend a lot, we use the prediction $\hat{y}$ to set **dynamic upselling thresholds**.
* **Formula**: If a customer's predicted order value is $\hat{y}$, we present them with a personalized offer:
  $$\text{Target Threshold } (T) = \hat{y} \times 1.25$$
  $$\text{Offer: Spend } T \text{ to get } d \text{ off (where } d = 10\% \text{ of } T\text{)}$$
* **Example**: If predicted spend is R$ 120.00, we present an offer: *"Spend R$ 150.00 to get R$ 15.00 off."*
* **Opportunity Loss (Underestimation)**: If we underestimate $\hat{y}$ as R$ 80.00 (actual is R$ 120.00), we offer a threshold of R$ 100.00. The customer easily reaches it without adding new items, resulting in **Margin Cannibalization**:
  $$\text{Cannibalization Loss} = d$$
* **Opportunity Loss (Overestimation)**: If we overestimate $\hat{y}$ as R$ 200.00, we offer a threshold of R$ 250.00. The target is too far out of reach, leading to a **Missed Upsell Conversion**.

---

### Impact B: Transit Insurance & Liability Pool Optimization
In shipping, insurance premiums and courier liability caps are proportional to the declared monetary value of the cargo. 
* **Over-declaration**: Overestimating order value leads to over-allocated insurance premium spend on shipping carriers:
  $$\text{Insurance Waste} = (\hat{y} - y) \times \text{Insurance Premium Rate \%}$$
* **Under-declaration**: Underestimating order value leads to under-insured cargo. If the package is lost or damaged in transit, the platform faces **Uncovered Liability**:
  $$\text{Uncovered Loss} = P(\text{Transit Loss}) \times (y - \hat{y})$$

---

## 3. Financial Target Case Study (ROI Projection)

Based on the raw Olist dataset characteristics:
* **Average Order Value (AOV)**: ~R$ 138.00
* **Total Monthly Orders**: ~10,000 orders
* **Monthly Revenue**: R$ 1,380,000.00
* **Naive Baseline WAPE (using AOV mean)**: ~38.0%

### Projected Monthly Return (Improving WAPE from 38.0% to 16.0%):
1. **Upsell Margin Lift**: By targeting users with accurate thresholds, we convert an estimated 8% of sessions into successful R$ 30.00 upsells (with a 30% gross margin):
   $$\text{Monthly Upsell Profit} = 10,000 \times 0.08 \times R\$ 30.00 \times 0.30 = R\$ 7,200.00$$
2. **Insurance Cost Optimization**: Saving R$ 1.80 per order in insurance overhead and lost transit claims by matching declarations to actual values:
   $$\text{Monthly Risk Savings} = 10,000 \times R\$ 1.80 = R\$ 18,000.00$$

**Total Projected Monthly Financial Return**: **R$ 25,200.00** (~R$ 302,400.00 annually).

---

## 4. Operational Dashboard Targets

Our model tracking dashboard (Module 7) will fire alerts when the rolling ML KPIs exceed the following operational limits:
* **Critical Alert**: Rolling 7-day WAPE exceeds **18.0%** (triggers immediate retraining).
* **Warning Alert**: Rolling 7-day MedAPE exceeds **14.0%** (notifies the modeling team).
* **Data Drift Alert**: PSI (Population Stability Index) of predicted values exceeds **0.25** (indicates customer behavior shift).

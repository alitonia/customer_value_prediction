# Task 1.1: KPIs and Business Value Specification (Revised V3 - Production Ready)

This document defines the key performance indicators (KPIs), model optimization guidelines, and the corrected financial value models for the Customer Order Value Prediction system. It addresses right-skewness and economic constraints.

---

## 1. Project Evaluation Metrics (ML KPIs)

To measure the predictive power of our regression models, we track four primary metrics. Our target values represent the threshold for a production-ready model.

| Metric | Definition | Purpose | Target Threshold |
| :--- | :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\frac{1}{n} \sum |y_i - \hat{y}_i|$ | Measures the average absolute error in the currency unit (Brazilian Real, BRL). Easy to explain to business stakeholders. | **< R$ 25.00** |
| **WAPE** (Weighted Absolute Percentage Error) | $\frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$ | Evaluates total absolute error relative to total revenue. Replaces MAPE to avoid artificial inflation from low-value orders. | **< 16.0%** |
| **MedAPE** (Median Absolute Percentage Error) | $\text{median}\left( \left|\frac{y_i - \hat{y}_i}{y_i}\right| \right)$ | Measures the median percentage error. Extremely robust to extreme high-value outliers. | **< 12.0%** |
| **$R^2$** (Coefficient of Determination) | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Quantifies the proportion of variance explained by behavioral features. **Downgraded to a secondary descriptive metric.** | *Secondary Tracking Only* |

### **The "Why": Loss Function & Metric Optimization Conflict**
* **The Problem**: There is a mathematical conflict between optimizing $R^2$ and WAPE/MAE:
  * Minimizing Mean Squared Error (MSE, which maximizes $R^2$) forces predictions to converge to the **conditional mean** ($E[y\|x]$).
  * Minimizing Mean Absolute Error (MAE, which minimizes WAPE) forces predictions to converge to the **conditional median** ($Median[y\|x]$).
  * Because Olist order values are heavily right-skewed, the conditional mean is significantly higher than the conditional median. 
  * If we train a model using an L2 loss (MSE) to chase a high $R^2$ target, it will systematically over-predict typical transactions, which inflates WAPE and MedAPE. 
* **The Decision**: We prioritize **WAPE** and **MedAPE** for commercial decisions. Consequently, during model training (Module 4), we must utilize **L1 Loss (MAE)** or **Huber Loss** (with a small delta to resist outlier influence). $R^2$ is removed as a hard gate and will be tracked purely for statistical context.

---

## 2. Business Value Formulations

### Impact A: Dynamic Incremental Upsell Coupons (Corrected for ROI)
To prevent margin cannibalization, we use the prediction $\hat{y}$ to set personalized upselling thresholds, but we scale the discount **only against the incremental spend**, not the entire cart value.
* **The Flawed Formulation (V2)**: Setting a target $T = 1.25 \times \hat{y}$ and offering a coupon of $d = 10\% \times T$. 
  * *Why it failed*: If $\hat{y} = R\$ 100$, then $T = R\$ 125$ and $d = R\$ 12.50$. The gross margin earned on the extra R$ 25 spend (at 30% margin) is only R$ 7.50, but we gave away R$ 12.50. This results in a **net loss of -R$ 5.00** per conversion.
* **The Corrected Formulation (V3)**: 
  * **Target Threshold** ($T$):
    $$T = \hat{y} \times 1.25$$
  * **Incremental Discount** ($d$): We calculate the discount as a percentage of the *upsell margin*, ensuring the discount never exceeds the incremental margin gained:
    $$d = (T - \hat{y}) \times \text{Discount Rate \%} \quad (\text{where Discount Rate \%} < \text{Gross Margin \%})$$
  * **Example**: If $\hat{y} = R\$ 100.00$, target threshold $T = R\$ 125.00$ (incremental spend of R$ 25.00). If our gross margin is 30% and we offer a 15% discount rate on the incremental spend:
    $$\text{Discount } d = R\$ 25.00 \times 15\% = R\$ 3.75$$
    $$\text{Incremental Margin Gained} = R\$ 25.00 \times 30\% = R\$ 7.50$$
    $$\text{Net Positive Profit Contribution} = R\$ 7.50 - R\$ 3.75 = +R\$ 3.75$$

---

### Impact B: Transit Insurance & Asymmetric Liability Optimization
In shipping, declarations are governed by carrier contracts and insurance liability pools.
* **The Asymmetric Risk**: 
  * **Over-declaration**: Overestimating value ($\hat{y} > y$) wastes insurance premiums on 100% of shipments:
    $$\text{Premium Waste} = (\hat{y} - y) \times \text{Premium Rate \%}$$
  * **Under-declaration**: Underestimating value ($\hat{y} < y$) leads to uncovered liability *only* if the package is lost/damaged in transit (low probability $P(\text{Transit Loss}) \approx 1\%$):
    $$\text{Expected Uncovered Loss} = P(\text{Transit Loss}) \times (y - \hat{y})$$
* **The "Why"**: 
  * While expected value math suggests under-declaring is cheaper if $P(\text{Loss}) \times (y - \hat{y})$ is small, carrier compliance rules and contract terms strictly prohibit under-declaring (which can void shipping coverage completely). 
  * Therefore, we must implement an **asymmetric loss function** in our evaluation pipeline that penalizes under-estimation ($y > \hat{y}$) twice as heavily as over-estimation ($\hat{y} > y$) to prevent catastrophic un-insured losses while remaining compliant.

---

## 3. Financial Target Case Study (ROI Projection)

Based on the raw Olist dataset characteristics:
* **Average Order Value (AOV)**: ~R$ 138.00
* **Total Monthly Orders**: ~10,000 orders
* **Monthly Revenue**: R$ 1,380,000.00
* **Naive Baseline WAPE (using AOV mean)**: ~38.0%

### Projected Monthly Return (Improving WAPE from 38.0% to 16.0%):
1. **Dynamic Upsell Profit**: By targeting users with accurate thresholds, we convert 8% of sessions into successful R$ 34.50 upsells ($25\% \times AOV$) with a 15% incremental discount rate and a 30% gross margin:
   $$\text{Monthly Upsell Profit} = 10,000 \times 0.08 \times R\$ 34.50 \times (0.30 - 0.15) = R\$ 4,140.00$$
2. **Insurance Compliance Savings**: Eliminating over-declaration waste and preventing voided claims through symmetric compliance bounds (saving R$ 1.20 per order average):
   $$\text{Monthly Compliance Savings} = 10,000 \times R\$ 1.20 = R\$ 12,000.00$$

**Total Projected Monthly Financial Return**: **R$ 16,140.00** (~R$ 193,680.00 annually). This represents a highly realistic, commercially viable profit contribution.

---

## 4. Operational Dashboard Targets

Our model tracking dashboard (Module 7) will fire alerts when the rolling ML KPIs exceed the following operational limits:
* **Critical Alert**: Rolling 7-day WAPE exceeds **18.0%** (triggers immediate retraining).
* **Warning Alert**: Rolling 7-day MedAPE exceeds **14.0%** (notifies the modeling team).
* **Data Drift Alert**: PSI (Population Stability Index) of predicted values exceeds **0.25** (indicates customer behavior shift).

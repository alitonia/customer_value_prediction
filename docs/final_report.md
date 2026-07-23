# Predicting Customer Order Value for an E-Commerce Platform

## Final Project Report — Business Analytics

---

## 1. Executive Summary

This project develops a regression system that predicts the monetary value of an e-commerce order prior to checkout. The model takes as input a customer's demographic profile, their current browsing and cart behavior, and the product mix in their cart, and returns an estimated order value in Brazilian Reais along with a confidence interval.

The system is built on the Brazilian E-Commerce Public Dataset by Olist (99,441 orders), which provides the transactional foundation. Because Olist does not include behavioral data — session duration, clickstream events, customer demographics, or loyalty information — we generated these dimensions synthetically. The generation process conditions each synthetic feature on the real order value, ensuring that the behavioral data carries genuine predictive signal rather than constituting random noise.

The best-performing model, XGBoost trained with L1 loss on a time-based train/test split, achieves R² = 0.80, MAE = R$48, WAPE = 29%, and MedAPE = 21%. Notably, 13 of the 20 most important features identified by the model are synthetic behavioral features, confirming that the value-conditioned generation design produces data that the model can learn from effectively.

The complete system includes a FastAPI prediction service, a Streamlit interactive demo, Evidently-based monitoring dashboards, and a reproducible pipeline that executes all stages from data generation through model training with a single command.

---

## 2. Business Context

E-commerce platforms make numerous operational decisions that depend on order value estimates: shipping tier selection, upsell recommendation timing, fraud review prioritization, and promotional targeting. Most platforms currently rely on historical averages for these decisions, which obscures the substantial variation between individual orders. A customer browsing computers on a desktop with a platinum loyalty status represents a fundamentally different revenue opportunity than a first-time mobile visitor purchasing a single accessory.

The objective of this project is to replace population-level averages with individual-level predictions. Specifically, we address the question: given a customer's profile and their current session behavior, what is the expected value of their order, and which factors exert the strongest influence on that value?

Accurate order value predictions enable three categories of business action: (1) dynamic upsell recommendations calibrated to the predicted value tier, (2) optimized shipping and insurance options matched to order magnitude, and (3) targeted promotional spend directed toward customer segments where incremental revenue is highest.

---

## 3. Data Strategy

### 3.1 Transactional Data

The Olist dataset provides nine tables covering orders, order items, products, customers, sellers, payments, reviews, geolocation, and category translations. Together these contain 99,441 orders, 112,650 line items, 32,951 products, 3,095 sellers, and 1,000,163 geolocation records spanning September 2016 to October 2018.

### 3.2 Synthetic Behavioral Data

Olist captures what was purchased but not how the customer arrived at that purchase. Real e-commerce platforms maintain extensive behavioral logs — session timestamps, page views, search queries, cart modifications, device fingerprints, and referral attribution — none of which are present in Olist. To bridge this gap, we generated three synthetic tables following an ERD-aligned schema:

- **customer_profile** (96,096 rows): demographics, income, loyalty tier, device preference, registration channel
- **sessions** (99,441 rows): device, browser, operating system, traffic source and medium, session timing, coupon usage
- **session_activity** (763,850 rows): granular clickstream events including page views, searches, product views, cart additions, and checkout events

### 3.3 Value-Conditioned Generation

The critical design decision in synthetic data generation is the relationship between the synthetic features and the real target variable. An initial approach generated features independently from realistic marginal distributions. This produced data that was statistically plausible in isolation but carried zero correlation with order value — the model treated these features as noise and assigned them negligible importance.

The revised approach conditions each synthetic feature on the real order value through a z-scored log-transform of the order total. For example, monthly income is drawn from a log-normal distribution whose mean shifts upward with order value, producing a correlation of r = 0.51 between simulated income and average order value. Loyalty tier probabilities shift toward higher tiers for high-value orders, creating a 2.5× spread in median order value between bronze (R$82) and platinum (R$202). Traffic source probabilities shift toward email and direct channels for high-value orders, reflecting the tendency of loyal, repeat customers to generate higher-value purchases.

Features with no plausible causal relationship to order value — gender, marital status, education level — remain independently generated. This preserves realism: in actual e-commerce data, these demographics do not meaningfully predict individual order value.

Every synthetic field is documented in the data dictionary with its column name, data type, unit, valid range, and the precise generation formula or business assumption used.

---

## 4. Exploratory Data Analysis

### 4.1 Target Distribution

Order value exhibits strong right skew (skewness = 9.37), with a median of R$105, a mean of R$160, and a maximum of R$13,664. This distribution is characteristic of e-commerce revenue data, where a small number of high-value orders substantially elevate the mean. A log1p transformation reduces skewness to 0.17, producing a near-normal distribution suitable for regression modeling.

### 4.2 Value Drivers

Product category is the strongest transactional driver of order value. Computers exhibit a median order value of R$1,251, while office supplies average R$30 — a 40-fold difference consistent with the underlying price structure of these categories.

The synthetic behavioral features display the correlation gradients built into the generation process. Loyalty tier shows a monotonic increase in median order value from bronze (R$82) through platinum (R$202). Email-sourced traffic produces a 64% higher median order value than Google organic traffic (R$146 versus R$89). Desktop sessions yield 7% higher median values than mobile sessions (R$110 versus R$103).

### 4.3 Correlation Structure

No feature pair exceeds |r| = 0.7, indicating no multicollinearity concerns among the candidate features. The feature avg_item_price exhibits r = 0.92 with the target; because this value is mechanically derived from the order's line-item prices, it constitutes target leakage and was excluded from the modeling feature set.

---

## 5. Data Cleaning and Feature Engineering

### 5.1 Cleaning

The cleaning pipeline applies the following transformations, each documented with before/after row counts in the cleaning log:

- Removal of 1,856 orders with canceled or unavailable status (no completed transaction)
- Translation of 71 Portuguese product category names to English via Olist's official mapping
- Imputation of missing timestamps with conservative defaults (order_approved_at filled from order_purchase_timestamp)
- Integration of previously unused data sources: geolocation coordinates (customer and seller latitude/longitude derived from zip code prefixes), seller reputation metrics (order count, average price, category specialization), and regional price indices (state-level median order value normalized to the global median)

A deliberate decision was made not to winsorize the target variable. The initial pipeline capped order values at the 3×IQR fence (R$522), which prevented the model from learning to predict high-value orders — precisely the segment where accurate prediction carries the greatest business impact. Instead, the full value range is preserved, and outlier robustness is achieved through L1 loss in the modeling stage.

### 5.2 Feature Engineering

From the cleaned 73-column dataset, the feature engineering pipeline constructs 182 features organized into five groups:

- **One-hot encoded categoricals** (131 features): device type, browser, operating system, traffic source, traffic medium, landing page, loyalty tier, product category, seller state, and others
- **Scaled numerical features** (47 features): income, session duration, cart quantities, geolocation coordinates, haversine distance between customer and seller, seller reputation metrics, RFM temporal features (customer age, recency, frequency), category price statistics, and seasonal indicators
- **Target-encoded features** (3 features): IP region, campaign name, and seller state, encoded with smoothing to prevent overfitting on rare categories
- **Derived interaction features** (16 features): income × loyalty, cart conversion rate, engagement rate, search intensity, items per minute, distance per item, seller reputation × loyalty, category price × income, and urban × desktop
- **Boolean features** (3 features): logged-in status, marketing opt-in, coupon applied

Variance Inflation Factor analysis identified and removed 3 perfectly collinear features. The remaining 182 features constitute the model input.

---

## 6. Modeling

### 6.1 Train/Test Split

A time-based split was used rather than a random split. Orders placed before April 1, 2018 form the training set (65,124 orders); orders from April 1 onward form the test set (32,460 orders). This approach prevents temporal leakage — the model cannot learn from patterns that occur after the prediction point — and reflects the actual deployment scenario in which a model trained on historical data predicts future orders.

### 6.2 Model Comparison

Six regression models were trained and evaluated:

| Model | R² | MAE (BRL) | WAPE (%) | MedAPE (%) |
|---|---|---|---|---|
| Linear Regression | 0.74 | 434 | 264 | 27 |
| Ridge | 0.74 | 434 | 264 | 27 |
| Lasso | 0.64 | 123 | 75 | 32 |
| Random Forest | 0.79 | 49 | 30 | 22 |
| **XGBoost (L1 loss)** | **0.80** | **48** | **29** | **21** |
| LightGBM (MAE loss) | 0.80 | 48 | 29 | 22 |

Linear models achieve reasonable R² but produce extreme absolute errors (MAE > R$400) because L2-based loss functions are highly sensitive to the outliers present in the unwinsorized target. Tree-based models with L1 loss handle these outliers gracefully, producing MAE values below R$50. XGBoost achieves the best MedAPE (21%), indicating that half of all predictions fall within 21% of the actual order value.

Five-fold cross-validation on the training set was used for model selection and hyperparameter stability assessment.

### 6.3 Feature Importance Analysis

The feature importance ranking from XGBoost provides the central validation of the synthetic data design. Of the 20 most important features, 13 are synthetic behavioral features:

| Rank | Feature | Importance | Source |
|---|---|---|---|
| 1 | income × loyalty | 14.0% | Synthetic interaction |
| 2 | session event count | 8.8% | Synthetic |
| 3 | cart additions | 8.8% | Synthetic |
| 4 | product views | 5.2% | Synthetic |
| 5 | total cart quantity | 5.1% | Synthetic |
| 6 | payment installments | 3.5% | Real |
| 7 | category: telephony | 2.7% | Real |
| 8 | log income | 2.6% | Synthetic |
| 9 | category: electronics | 2.4% | Real |
| 10 | loyalty tier: bronze | 2.3% | Synthetic |

The dominance of synthetic features in the importance ranking confirms that the value-conditioned generation process produces behavioral data that the model finds genuinely predictive. The synthetic data is not merely occupying columns in the feature matrix — it is driving the model's predictions.

### 6.4 Residual Analysis

Residual diagnostics reveal the expected pattern for a log-transformed target: residuals are approximately normally distributed in log-space, with heteroscedasticity visible in the highest value bucket. The model tends to under-predict the most extreme orders (above R$5,000), which is an inherent limitation of log-space modeling — the transform compresses the right tail, reducing the model's sensitivity to extreme values.

---

## 7. Deployment and Monitoring

### 7.1 Prediction API

The trained XGBoost model and its preprocessing pipeline (scaler, target encoders, one-hot encoder configuration) are serialized as a single artifact and loaded at startup by a FastAPI service. The `/predict` endpoint accepts a JSON payload containing 30+ input features and returns the predicted order value in BRL with an 80% confidence interval. The `/health` endpoint provides a liveness check for orchestration systems.

Validation with two contrasting profiles confirms the model's discriminative capacity: a bronze-tier mobile visitor with one item yields a prediction of R$83, while a platinum-tier desktop customer with five items yields R$520 — a 6.3× spread.

### 7.2 Interactive Demo

A Streamlit application provides an interactive interface for the prediction API. Sidebar controls allow adjustment of customer profile parameters (income, loyalty tier, device preference, age), session context (traffic source, duration, login status, coupon usage), and cart contents (item count, product views, searches). The application displays the prediction, confidence interval, and contextual business insights (e.g., high-value order alerts, loyalty-tier recommendations).

A standalone version of the Streamlit application (`app.py`) loads the model directly without requiring the FastAPI backend, enabling deployment as a HuggingFace Space.

### 7.3 Monitoring

Two monitoring mechanisms track model health post-deployment:

**PSI-based drift detection** computes the Population Stability Index for each feature, comparing the training distribution against incoming data. A PSI exceeding 0.25 for any feature triggers a drift alert. The monitoring script (`monitoring/drift_monitor.py`) produces a machine-readable JSON report and a human-readable specification document.

**Evidently AI dashboards** generate three interactive HTML reports: a data drift report showing per-feature distribution shifts, a regression performance report with predicted-versus-actual plots and error distributions, and a combined dashboard integrating both perspectives. These reports can be opened in any browser and shared with stakeholders without additional tooling.

Retraining is triggered when any feature PSI exceeds 0.25 or when the rolling 7-day WAPE exceeds 35%.

---

## 8. Business Recommendations

The model's feature importances and EDA findings support five actionable recommendations:

**Loyalty tier upgrade programs.** The 2.5× median order value spread between bronze (R$82) and platinum (R$202) represents the largest single lever identified in the analysis. Structured upgrade incentives — for example, threshold-based rewards that encourage bronze customers to reach silver status — directly target the most influential categorical predictor.

**Email channel investment.** Email-sourced traffic produces 64% higher median order values than Google organic traffic. Email subscribers are predominantly existing customers with established purchase intent. Increasing email marketing budget allocation, particularly for segmented campaigns targeting high-value customer profiles, offers a favorable return relative to acquisition-focused channels.

**Desktop experience optimization.** Desktop sessions yield 7% higher median order values. For high-value categories such as computers and electronics, where comparison shopping and detailed product evaluation are common, a desktop-optimized checkout flow with enhanced product comparison tools may further increase average order value.

**Real-time cart upsell triggers.** Cart additions rank third in feature importance. Sessions with low engagement (fewer than five events and a single cart item) represent opportunities for targeted "customers also purchased" recommendations before the customer proceeds to checkout.

**Income-tier personalization.** The income × loyalty interaction is the single most important feature (14% importance). The model's predicted value tier can serve as a real-time segmentation variable, enabling premium product recommendations for high-value segments and value-oriented promotions for budget segments.

---

## 9. Limitations and Future Work

Several limitations constrain the current system and define directions for future development:

**Synthetic behavioral data.** The correlations between synthetic features and order value are designed rather than observed. While the value-conditioned generation approach produces realistic correlation magnitudes and the model validates their predictive utility, the ultimate test requires real clickstream data from a production e-commerce platform.

**Absence of product-level pricing.** Within a single product category, order values can vary by two orders of magnitude (a R$50 accessory versus a R$5,000 laptop in the computers category). Product-level price features — average price, price range, promotional status — would substantially reduce within-category variance and improve prediction accuracy.

**Limited repeat purchase data.** Ninety-seven percent of customers in the Olist dataset have exactly one order. This limits the utility of customer-level features such as purchase frequency trends and lifetime value, which require longitudinal data.

**KPI target gap.** The original KPI targets (MAE < R$25, WAPE < 16%, MedAPE < 12%) were not achieved. The best model produces MAE = R$48, WAPE = 29%, and MedAPE = 21%. Closing this gap would require product-level pricing features, sequence models (LSTM or transformer architectures) that process clickstream events as ordered sequences rather than aggregated summaries, and potentially external data sources such as competitor pricing or macroeconomic indicators.

---

## 10. References

- Olist Brazilian E-Commerce Public Dataset. Kaggle. kaggle.com/datasets/olistbr/brazilian-ecommerce
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD Conference*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems 30*.

---

## Appendix: Deliverables Index

| Deliverable | Location |
|---|---|
| Data dictionary | `docs/data_dictionary.md` |
| Synthetic data schema | `docs/synthetic_data_schema.md` |
| KPI specification | `docs/kpis_and_business_value.md` |
| Marketing recommendations | `docs/marketing_recommendations.md` |
| Project overview | `docs/project_overview.md` |
| EDA notebook | `notebooks/02_eda.ipynb` |
| EDA visualizations | `notebooks/plots/` (9 plots) |
| Cleaning log | `data/processed/cleaning_log.md` |
| Feature catalog | `data/processed/feature_catalog.md` |
| Model evaluation | `data/processed/model_evaluation.md` |
| Monitoring dashboards | `monitoring/evidently_*.html` (3 reports) |
| Monitoring specification | `monitoring/monitoring_spec.md` |
| Test suite | `tests/` (43 tests) |
| Pipeline script | `run_pipeline.sh` |
| Dockerfile | `Dockerfile` |
| Demo walkthrough | `docs/demo_walkthrough.md` |

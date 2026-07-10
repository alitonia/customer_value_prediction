# Task 1.2: Synthetic Behavioral Data Schema Specification

This document details the schema design, field constraints, and statistical correlations for the synthetic behavioral data extension of the Olist e-commerce dataset.

---

## 1. Core Integration Strategy

Instead of simulating completely independent data, our synthetic behavior dataset will **enrich the existing Olist orders**. We map each transaction (`order_id`) 1-to-1 to a simulated web/app session that represents the checkout session of that purchase.

* **Key Identifier**: `order_id` (foreign key linking to Olist `olist_orders_dataset.csv`).
* **Output Path**: `data/synthetic/behavioral_sessions.csv`

---

## 2. Field Specifications and Generation Constraints

Below is the schema for the synthetic behavioral variables:

| Field Name | Data Type | Units | Valid Range / Categories | Target Distribution / Logic |
| :--- | :--- | :--- | :--- | :--- |
| **`order_id`** | String (Hex) | N/A | Match Olist `order_id` | Unique identifier (32-character hexadecimal). |
| **`session_id`** | String (UUID) | N/A | 36-char standard UUID | Globally unique identifier for the web session. |
| **`device_type`** | Categorical | N/A | `mobile`, `desktop`, `tablet` | Standard e-commerce ratios:<br>- Mobile: **65%**<br>- Desktop: **30%**<br>- Tablet: **5%** |
| **`referral_channel`** | Categorical | N/A | `direct`, `search_organic`, `search_paid`, `social`, `email` | Channel traffic distribution:<br>- `search_organic`: **35%**<br>- `direct`: **25%**<br>- `search_paid`: **15%**<br>- `social`: **15%**<br>- `email`: **10%** |
| **`session_duration_seconds`**| Integer | Seconds | $10 \le x \le 3600$ | Log-Normal distribution:<br>- Mean: **300 seconds (5 min)**<br>- Max capped: **3600 seconds (1 hour)** |
| **`pages_viewed`** | Integer | Count | $1 \le x \le 100$ | Poisson/Log-Normal distribution, strongly correlated with `session_duration_seconds`. Average: **8 pages**. |
| **`cart_additions`** | Integer | Count | $1 \le x \le 20$ | Correlated with `pages_viewed` and Olist actual order item count: $\ge \text{actual\_item\_count}$. |
| **`coupon_applied`** | Boolean | N/A | `0` or `1` | Average coupon application rate: **22%**. |
| **`discount_amount_pct`** | Integer | % | $0, 5, 10, 15, 20$ | Conditional on `coupon_applied == 1`:<br>- 5% discount: **40%** probability<br>- 10% discount: **35%** probability<br>- 15% discount: **15%** probability<br>- 20% discount: **10%** probability |

---

## 3. Statistical Relations & Multi-Variable Constraints (Business Logic)

To ensure the synthetic data behaves like real e-commerce data (and provides predictive signal for modeling), we enforce the following mathematical relationships during generation:

### A. Session Duration & Pages Viewed
More pages viewed must correspond to longer sessions. We model this using a baseline time-per-page constant ($T_{\text{page}}$) with added random noise ($\epsilon$):
$$\text{Duration} = \max(10, \text{pages\_viewed} \times T_{\text{page}} + \epsilon) \quad \text{where } T_{\text{page}} \sim \mathcal{N}(35, 10)$$

### B. Cart Additions & Order Value
To avoid contradictions:
1. `cart_additions` must be greater than or equal to the actual number of items purchased in that order (`order_items`).
2. High-value orders should correlate with a larger number of cart additions (representing shoppers browsing and building larger baskets).

### C. Coupon Impact
If `coupon_applied` is true, the `discount_amount_pct` is drawn from a multinomial distribution of standard discount rates ($5\%, 10\%, 15\%, 20\%$). If false, `discount_amount_pct` is strictly $0$.

---

## 4. Schema Verification Checkpoints (Task 1.5 Targets)

During the data generation validation phase, we will check:
* **Completeness**: Every non-canceled order in Olist must have a corresponding session record.
* **Integrity**: Zero nulls in the output file.
* **Consistency**: `cart_additions >= items_purchased` for all records.

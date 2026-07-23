# Cleaning Log — Before/After Comparison (V2)

**Initial orders:** 99,441
**Final modeling rows:** 97,584
**Retention rate:** 98.1%
**Winsorization:** none (raw order value preserved)

## V2 Changes
- Removed winsorization — raw order value preserved for robust loss training
- Added geolocation features (customer/seller lat/lng, haversine distance, urban/rural)
- Added seller features (order count, avg price, category count, state)
- Added RFM temporal features (customer age, recency, frequency, seasonality)
- Added product-level price aggregates (category avg/std price, order count)
- Added regional price index (state median / global median)

## Step-by-step

| Step | Before | After | Detail |
|---|---|---|---|
| 3.1a Remove duplicate orders | 99,441 | 99,441 |  |
| 3.1b Remove duplicate order items | 112,650 | 112,650 |  |
| 3.3a Filter to delivered/shipped | 99,441 | 97,585 | removed 1856 |
| 3.3b Order value (no winsorization) | 99,441 | 97,584 |  |
| 3.5 Final modeling dataset | 99,441 | 97,584 | 73 columns |
# Cleaning Log — Before/After Comparison

**Initial orders:** 99,441
**Final modeling rows:** 97,584
**Retention rate:** 98.1%
**Winsorization fence:** R$522

## Step-by-step

| Step | Before | After | Detail |
|---|---|---|---|
| 3.1a Remove duplicate orders | 99,441 | 99,441 |  |
| 3.1b Remove duplicate order items | 112,650 | 112,650 |  |
| 3.1c Fill missing order timestamps | 4,908 | 4,748 | approved_at ← purchase_timestamp |
| 3.1d Fill missing product categories | 610 | 0 |  |
| 3.2 Translate categories to English | 0 | 32,328 | 71 categories mapped |
| 3.3a Filter to delivered/shipped | 99,441 | 97,585 | removed 1856 canceled/unavailable/other |
| 3.3b Remove zero/negative value orders | 98,666 | 98,666 |  |
| 3.3c Winsorize order value | 3,943 | 0 | fence=R$522, capped 3943 orders |
| 3.3d Join with valid orders | 97,585 | 97,584 |  |
| 3.5 Final modeling dataset | 99,441 | 97,584 | 50 columns |

## Decisions

1. **Status filter:** Only `delivered` and `shipped` orders retained. Canceled, unavailable,
   invoiced, processing, created, and approved orders removed (no completed transaction).
2. **Winsorization:** Order values above 3×IQR fence capped rather than removed,
   preserving high-value signal while limiting outlier influence on linear models.
3. **Category translation:** Portuguese product categories mapped to English via
   Olist's official translation table. Missing categories → 'unknown'.
4. **Missing timestamps:** `order_approved_at` filled with `order_purchase_timestamp`
   (conservative: assumes instant approval).
5. **avg_item_price excluded:** r=0.92 with target = leakage. Not included in modeling dataset.
6. **Session activity aggregated:** Clickstream events rolled up to session level
   (counts, sums, means) for modeling compatibility.
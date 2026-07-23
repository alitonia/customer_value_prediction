# Model Evaluation Report

**Best model:** XGBoost

## Model Comparison

| Model | R² (log) | MAE (BRL) | RMSE (BRL) | WAPE (%) | MedAPE (%) | CV MAE (log) |
|---|---|---|---|---|---|---|
| XGBoost **← best** | 0.8036 | 47.6 | 123.8 | 28.9 | 21.4 | 0.2647 ± 0.0013 |
| LightGBM | 0.8007 | 48.0 | 123.5 | 29.2 | 21.7 | 0.2663 ± 0.0014 |
| Random Forest | 0.7862 | 49.0 | 124.9 | 29.8 | 22.1 | 0.2741 ± 0.0010 |
| Ridge | 0.7375 | 325.6 | 45996.9 | 197.9 | 26.5 | 0.3165 ± 0.0013 |
| Linear Regression | 0.7374 | 325.4 | 45960.4 | 197.8 | 26.5 | 0.3165 ± 0.0013 |
| Lasso | 0.6369 | 122.9 | 10001.4 | 74.7 | 31.6 | 0.3831 ± 0.0007 |

## KPI Targets

| KPI | Target | Achieved | Status |
|---|---|---|---|
| MAE | < R$25 | R$47.6 | ✗ |
| WAPE | < 16% | 28.9% | ✗ |
| MedAPE | < 12% | 21.4% | ✗ |

## Residual Analysis

![Residual Analysis](residual_analysis.png)

## Feature Importances

![Feature Importances](feature_importance.png)

## Notes

- Target variable is log1p(order_value). Metrics in BRL use expm1 inverse transform.
- Linear models use StandardScaler; tree-based models use raw encoded features.
- CV = 5-fold stratified by target quartiles.
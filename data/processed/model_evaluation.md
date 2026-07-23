# Model Evaluation Report

**Best model:** XGBoost

## Model Comparison

| Model | R² (log) | MAE (BRL) | RMSE (BRL) | WAPE (%) | MedAPE (%) | MAPE (%) | CV MAE (log) |
|---|---|---|---|---|---|---|---|
| XGBoost **← best** | 0.7943 | 48.3 | 122.9 | 29.3 | 21.6 | 29.2 | 0.2701 ± 0.0014 |
| LightGBM | 0.7907 | 48.6 | 123.2 | 29.6 | 21.9 | 29.6 | 0.2723 ± 0.0012 |
| Random Forest | 0.7775 | 49.6 | 125.3 | 30.2 | 22.6 | 30.6 | 0.2797 ± 0.0010 |
| Ridge | 0.7239 | 573.3 | 90070.5 | 348.5 | 27.0 | 35.5 | 0.3236 ± 0.0010 |
| Linear Regression | 0.7238 | 572.9 | 89991.1 | 348.2 | 27.0 | 35.5 | 0.3236 ± 0.0010 |
| Lasso | 0.6214 | 191.5 | 21883.5 | 116.4 | 32.3 | 43.6 | 0.3911 ± 0.0011 |

## KPI Targets

| KPI | Target | Achieved | Status |
|---|---|---|---|
| MAE | < R$25 | R$48.3 | ✗ |
| WAPE | < 16% | 29.3% | ✗ |
| MedAPE | < 12% | 21.6% | ✗ |
| MAPE | < 15% | 29.2% | ✗ |

## Residual Analysis

![Residual Analysis](residual_analysis.png)

## Feature Importances

![Feature Importances](feature_importance.png)

## Notes

- Target variable is log1p(order_value). Metrics in BRL use expm1 inverse transform.
- Linear models use StandardScaler; tree-based models use raw encoded features.
- CV = 5-fold stratified by target quartiles.
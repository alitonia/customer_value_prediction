# Model Evaluation Report

**Best model:** XGBoost

## Model Comparison

| Model | R² (log) | MAE (BRL) | RMSE (BRL) | WAPE (%) | MedAPE (%) | CV MAE (log) |
|---|---|---|---|---|---|---|
| XGBoost **← best** | 0.6956 | 43.2 | 64.7 | 30.2 | 27.3 | 0.3227 ± 0.0020 |
| LightGBM | 0.6934 | 43.4 | 65.2 | 30.4 | 27.3 | 0.3238 ± 0.0018 |
| Ridge | 0.6639 | 50.0 | 116.4 | 35.0 | 29.0 | 0.3414 ± 0.0018 |
| Linear Regression | 0.6639 | 50.0 | 116.4 | 35.0 | 29.0 | 0.3414 ± 0.0018 |
| Random Forest | 0.6613 | 45.1 | 67.7 | 31.6 | 29.1 | 0.3432 ± 0.0023 |
| Lasso | 0.5282 | 56.4 | 99.2 | 39.5 | 35.3 | 0.4119 ± 0.0028 |

## KPI Targets

| KPI | Target | Achieved | Status |
|---|---|---|---|
| MAE | < R$25 | R$43.2 | ✗ |
| WAPE | < 16% | 30.2% | ✗ |
| MedAPE | < 12% | 27.3% | ✗ |

## Residual Analysis

![Residual Analysis](residual_analysis.png)

## Feature Importances

![Feature Importances](feature_importance.png)

## Notes

- Target variable is log1p(order_value). Metrics in BRL use expm1 inverse transform.
- Linear models use StandardScaler; tree-based models use raw encoded features.
- CV = 5-fold stratified by target quartiles.
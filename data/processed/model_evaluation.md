# Model Evaluation Report

**Best model:** XGBoost

## Model Comparison

| Model | R² (log) | MAE (BRL) | RMSE (BRL) | WAPE (%) | MedAPE (%) | CV MAE (log) |
|---|---|---|---|---|---|---|
| XGBoost **← best** | 0.8019 | 47.5 | 120.8 | 28.9 | 21.5 | 0.2651 ± 0.0009 |
| LightGBM | 0.7996 | 47.7 | 120.9 | 29.0 | 21.6 | 0.2666 ± 0.0011 |
| Random Forest | 0.7843 | 48.9 | 122.4 | 29.7 | 22.4 | 0.2751 ± 0.0016 |
| Ridge | 0.7358 | 433.8 | 65432.7 | 263.6 | 26.5 | 0.3175 ± 0.0009 |
| Linear Regression | 0.7357 | 433.5 | 65382.2 | 263.5 | 26.6 | 0.3175 ± 0.0009 |
| Lasso | 0.6353 | 160.1 | 16714.3 | 97.3 | 31.8 | 0.3830 ± 0.0014 |

## KPI Targets

| KPI | Target | Achieved | Status |
|---|---|---|---|
| MAE | < R$25 | R$47.5 | ✗ |
| WAPE | < 16% | 28.9% | ✗ |
| MedAPE | < 12% | 21.5% | ✗ |

## Residual Analysis

![Residual Analysis](residual_analysis.png)

## Feature Importances

![Feature Importances](feature_importance.png)

## Notes

- Target variable is log1p(order_value). Metrics in BRL use expm1 inverse transform.
- Linear models use StandardScaler; tree-based models use raw encoded features.
- CV = 5-fold stratified by target quartiles.
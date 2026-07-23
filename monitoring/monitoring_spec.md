# Monitoring Specification

## Drift Detection
- **Method:** Population Stability Index (PSI) per feature
- **Threshold:** PSI > 0.25 triggers drift alert
- **Features monitored:** 159
- **Current drifted features:** 2

## Performance Monitoring
- **Metric:** Rolling 7-day WAPE
- **Threshold:** WAPE > 18.0% triggers retraining
- **Current status:** ⚠️ RETRAINING TRIGGERED

## Retraining Triggers
1. Any feature PSI > 0.25
2. Rolling 7-day WAPE > 18.0%
3. Manual trigger via pipeline script

## Top PSI Features
| Feature | PSI |
|---|---|
| search_intensity | 4.9399 |
| n_events | 4.3354 |
| purchase_hour | 0.1001 |
| registration_channel_paid_ads | 0.0058 |
| operating_system_windows | 0.0055 |
| marital_status_married | 0.0053 |
| payment_type_credit_card | 0.0052 |
| operating_system_ios | 0.0051 |
| traffic_source_email | 0.0051 |
| n_add_to_cart | 0.0051 |

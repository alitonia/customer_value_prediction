# Predicting Customer Order Value for an E-Commerce Platform

**Course:** Business Intelligence  
**University:** Hanoi University of Science and Technology (HUST)

---

## 1. Introduction

This project focuses on building a system to **predict customer order value** for an e-commerce platform. The project uses real data from the **Olist** dataset (Brazil) combined with synthetic behavioral session data generated according to the course requirements.

The main goal is to build a complete data pipeline and develop machine learning models capable of predicting order value effectively, evaluated using **WAPE** and **MedAPE** metrics as required by the course.

---

## 2. Project Objectives

- Build a complete data engineering pipeline.
- Generate synthetic behavioral session data.
- Perform high-quality feature engineering.
- Train and compare multiple models (LightGBM, XGBoost, CatBoost).
- Optimize the LightGBM model using Optuna.
- Evaluate models using the required metrics (WAPE, MedAPE, MAE).

---

## 3. Key Results

| Model               | MAE   | WAPE    | MedAPE | Note            |
|---------------------|-------|---------|--------|-----------------|
| *LightGBM (Tuned)*  | *8.47*| *6.21%* | *3.95%*| *Best Model*    |
| LightGBM (Base)     | 9.36  | 6.86%   | 4.44%  | -               |
| CatBoost            | 9.65  | 7.07%   | 4.69%  | -               |
| XGBoost             | 10.49 | 7.69%   | 4.42%  | -               |

**Best Model:** `best_lightgbm_tuned.pkl`

---

## 4. Project Structure
customer_value_prediction/
├── src/
│   ├── config.py                      # Centralized configuration
│   ├── common/                        # Shared utilities
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── io.py
│   │   └── timer.py
│   ├── data/
│   │   ├── data_generation.py         # Generate synthetic behavioral data
│   │   ├── cleaning.py                # Clean Olist data
│   │   └── create_final_dataset.py    # Create final merged dataset
│   ├── features/
│   │   └── feature_engineering.py     # Feature engineering
│   └── models/
│       ├── train.py                   # Train & compare models (50 runs)
│       ├── tune_lightgbm.py           # Hyperparameter tuning with Optuna
│       └── evaluate_final_model.py    # Model evaluation + Feature Importance
├── data/
│   ├── raw/                           # Original Olist data
│   ├── synthetic/                     # Generated behavioral sessions
│   ├── processed/                     # Cleaned data
│   └── merged/                        # final_dataset.csv & modeling_dataset.csv
├── models/
│   └── saved_models/                  # Trained models (.pkl)
├── reports/                           # Results, plots, and predictions
├── docs/                              # Documentation (Data Dictionary, etc.)
├── README.md
└── requirements.txt
text---

## 5. Technologies Used

- Python 3.10
- Pandas, NumPy
- Scikit-learn
- LightGBM, XGBoost, CatBoost
- Optuna (for hyperparameter tuning)
- Matplotlib, Seaborn

---

## 6. How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
Step 2: Run the data pipeline
Bash# 1. Generate synthetic behavioral data
python src/data/data_generation.py

# 2. Clean data
python src/data/cleaning.py

# 3. Create final dataset + feature engineering
python src/data/create_final_dataset.py
python src/features/feature_engineering.py
Step 3: Train and evaluate models
Bash# Train and compare 3 models (50 runs)
python src/models/train.py

# Hyperparameter tuning for LightGBM
python src/models/tune_lightgbm.py

# Train final model with best parameters
python src/models/train_final_lightgbm.py

# Evaluate model and generate Feature Importance
python src/models/evaluate_final_model.py

7. Important Outputs
After running the pipeline, the following key files will be generated:

FileDescriptionbest_lightgbm_tuned.pkl
Final best modeltest_predictions.csvActual vs Predicted values on test setfeature_importance.csvFeature importance rankingfeature_importance.pngTop features visualizationtraining_summary_50_runs.csvModel comparison results (50 runs)

MemberMain ResponsibilitiesMember 1Data Engineering + Feature EngineeringMember 2Modeling + Hyperparameter TuningMember 3Report + Presentation

9. Limitations & Future Work

Time-based splitting was not used (due to limited timestamp information).
No deployment or API was implemented.
Potential improvement: Add customer historical features.


10. References

Olist Brazilian E-Commerce Dataset
Optuna Documentation
LightGBM Documentation
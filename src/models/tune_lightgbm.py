import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from common import setup_logger

logger = setup_logger("tune_lightgbm")


def objective(trial):
    # Load data
    df = pd.read_csv(config.paths.MERGED_DIR / config.data.MODELING_DATASET)

    # Drop ID
    df = df.drop(columns=["order_id", "session_id"], errors="ignore")

    # Chuyển object → category
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    X = df.drop(columns=[config.training.TARGET, config.training.ORIGINAL_TARGET], errors="ignore")
    y = df[config.training.TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    pred_log = model.predict(X_val)
    pred = np.expm1(pred_log)
    truth = np.expm1(y_val)

    mae = mean_absolute_error(truth, pred)
    return mae


def main():
    logger.info("Starting LightGBM Hyperparameter Tuning with Optuna...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  MAE: {trial.value:.4f}")
    logger.info("  Best hyperparameters:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Lưu best params
    import json
    with open("best_lightgbm_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)

    logger.info("Best parameters saved to best_lightgbm_params.json")


if __name__ == "__main__":
    main()
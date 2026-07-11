import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from common import setup_logger, mae, wape, medape

logger = setup_logger("train_final_lightgbm")


def main():
    logger.info("Training final LightGBM model with best hyperparameters...")

    # Load best params
    with open("best_lightgbm_params.json", "r") as f:
        best_params = json.load(f)

    # Load data
    df = pd.read_csv(config.paths.MERGED_DIR / config.data.MODELING_DATASET)

    # Drop ID columns
    df = df.drop(columns=["order_id", "session_id"], errors="ignore")

    # Convert object to category
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    X = df.drop(columns=[config.training.TARGET, config.training.ORIGINAL_TARGET], errors="ignore")
    y = df[config.training.TARGET]

    # Train model with best params
    model = LGBMRegressor(
        objective="regression",
        random_state=42,
        **best_params
    )

    model.fit(X, y)

    # Predict on training data (để đánh giá nhanh)
    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    truth = np.expm1(y)

    logger.info("=== Final LightGBM Performance ===")
    logger.info(f"MAE     : {mae(truth, pred):.4f}")
    logger.info(f"WAPE    : {wape(truth, pred):.4f}")
    logger.info(f"MedAPE  : {medape(truth, pred):.4f}")

    # Save model
    model_path = config.paths.SAVED_MODEL_DIR / "best_lightgbm_tuned.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Final tuned model saved to: {model_path}")


if __name__ == "__main__":
    main()
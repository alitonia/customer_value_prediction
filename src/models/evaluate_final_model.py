import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from common import setup_logger, mae, wape, medape

logger = setup_logger("evaluate_final_model")


def main():

    # 1. Load model đã tune
    model_path = config.paths.SAVED_MODEL_DIR / "best_lightgbm_tuned.pkl"
    model = joblib.load(model_path)
    logger.info(f"Đã load model: {model_path}")

    # 2. Load dữ liệu
    df = pd.read_csv(config.paths.MERGED_DIR / config.data.MODELING_DATASET)

    df = df.drop(columns=["order_id", "session_id"], errors="ignore")

    # Chuyển object → category
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    X = df.drop(columns=[config.training.TARGET, config.training.ORIGINAL_TARGET], errors="ignore")
    y = df[config.training.TARGET]   # log_order_value

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)        # Chuyển về order_value
    truth = np.expm1(y_test)

    # 5. Đánh giá
    logger.info(f"MAE     : {mae(truth, pred):.4f}")
    logger.info(f"WAPE    : {wape(truth, pred):.4f}")
    logger.info(f"MedAPE  : {medape(truth, pred):.4f}")

    # 6. Lưu prediction
    result_df = pd.DataFrame({
        "y_true_order_value": truth,
        "y_pred_order_value": pred,
        "residual": truth - pred
    })
    result_path = config.paths.REPORT_DIR / "test_predictions.csv"
    result_df.to_csv(result_path, index=False)
    logger.info(f"\nĐã lưu prediction: {result_path}")

    # 7. Feature Importance
    importance_df = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importance_path = config.paths.REPORT_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Đã lưu feature importance: {importance_path}")

    # 8. Vẽ biểu đồ Feature Importance (Top 15)
    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=importance_df.head(15),
        x="importance",
        y="feature",
        palette="viridis"
    )
    plt.title("Top 15 Feature Importance - Tuned LightGBM", fontsize=14)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    plot_path = config.paths.REPORT_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Đã lưu biểu đồ: {plot_path}")



if __name__ == "__main__":
    main()
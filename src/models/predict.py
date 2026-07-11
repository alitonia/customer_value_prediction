import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import config


class OrderValuePredictor:
    """
    Class dùng để dự đoán Order Value cho dữ liệu mới.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = config.paths.SAVED_MODEL_DIR / "best_lightgbm_tuned.pkl"

        self.model = joblib.load(model_path)
        print(f"✅ Đã load model từ: {model_path}")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu giống như lúc train.
        """
        df = df.copy()

        # Loại bỏ các cột ID (nếu có)
        df = df.drop(columns=["order_id", "session_id"], errors="ignore")

        # Chuyển các cột object thành category
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category")

        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dự đoán order_value cho dữ liệu mới.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu mới (đã qua feature engineering cơ bản)

        Returns:
        --------
        pd.DataFrame
            DataFrame gốc + cột 'predicted_order_value'
        """
        # Tiền xử lý
        X = self.preprocess(df)

        # Dự đoán trên thang log
        pred_log = self.model.predict(X)

        # Chuyển về giá trị thực (order_value)
        predicted_order_value = np.expm1(pred_log)

        # Thêm cột kết quả vào dataframe gốc
        result_df = df.copy()
        result_df["predicted_order_value"] = predicted_order_value

        return result_df


# ====================== Ví dụ sử dụng ======================

if __name__ == "__main__":
    # Ví dụ: Đọc dữ liệu mới từ file CSV
    # new_data = pd.read_csv("data/new_data.csv")

    # Tạo predictor
    predictor = OrderValuePredictor()

    # Giả sử bạn có DataFrame new_data đã được feature engineering
    # result = predictor.predict(new_data)
    # print(result[["predicted_order_value"]].head())

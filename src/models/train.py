"""
Model Training Pipeline

Workflow
--------
Load Dataset
    │
Train/Test Split
    │
Feature Engineering
    │
Train Models
    │
Evaluate Models
    │
Save Best Model
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))

from config import config

from features.feature_engineering import FeatureEngineer

from common import (
    setup_logger,
    timer,
    mae,
    wape,
    medape,
    save_object,
    load_object,
    save_dataframe,
)
logger = setup_logger("train")


class ModelTrainer:
    """
    Train and compare multiple regression models.

    Models
    ------
    - LightGBM
    - XGBoost
    - CatBoost

    Target
    ------
    log_order_value

    Evaluation
    ----------
    MAE
    WAPE
    MedAPE
    """

    def __init__(self):

        # ==========================================
        # Paths
        # ==========================================

        self.processed_dir = config.paths.PROCESSED_DIR

        self.model_dir = config.paths.SAVED_MODEL_DIR

        self.report_dir = config.paths.REPORT_DIR

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==========================================
        # Dataset
        # ==========================================

        self.dataset_path = (
                config.paths.MERGED_DIR /
                config.data.MODELING_DATASET
        )

        # ==========================================
        # Training Config
        # ==========================================

        self.target = config.training.TARGET

        self.original_target = (
            config.training.ORIGINAL_TARGET
        )

        self.random_state = (
            config.training.RANDOM_STATE
        )

        self.test_size = (
            config.training.TEST_SIZE
        )

        # ==========================================
        # Feature Engineering
        # ==========================================

        self.feature_engineer = FeatureEngineer()

        # ==========================================
        # Models
        # ==========================================

        self.models = {

            "LightGBM": LGBMRegressor(

                random_state=self.random_state,

                **config.model.LIGHTGBM_PARAMS

            ),

            "XGBoost": XGBRegressor(
                random_state=self.random_state,
                enable_categorical=True,
                **config.model.XGBOOST_PARAMS

            ),

            "CatBoost": CatBoostRegressor(

                random_seed=self.random_state,

                **config.model.CATBOOST_PARAMS

            ),

        }

    ############################################################
    # Load Dataset
    ############################################################

    def load_data(self) -> pd.DataFrame:

        logger.info(
            "Loading modeling dataset..."
        )

        df = pd.read_csv(
            self.dataset_path
        )

        logger.info(
            f"Dataset shape: {df.shape}"
        )

        return df

    ############################################################
    # Train/Test Split
    ############################################################

    def split_dataset(self, df: pd.DataFrame, random_state: int = None):

        logger.info("Splitting dataset...")

        if random_state is None:
            random_state = self.random_state

        X = df.drop(
            columns=[self.target, self.original_target],
            errors="ignore"
        )
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=random_state,  # ← Dùng seed động
            shuffle=True,
        )

        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test
    ############################################################
    # Feature Engineering
    ############################################################

    def build_features(
            self,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
    ):

        logger.info("Preparing features for training...")


        id_cols = ["order_id", "session_id"]
        drop_cols = [col for col in id_cols if col in X_train.columns]

        if drop_cols:
            X_train = X_train.drop(columns=drop_cols, errors="ignore")
            X_test = X_test.drop(columns=drop_cols, errors="ignore")

        object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

        for col in object_cols:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        logger.info(
            f"Final Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        return X_train, X_test

    ############################################################
    # Evaluate
    ############################################################

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:

        logger.info("Evaluating model...")

        # Prediction on log scale
        pred_log = model.predict(X_test)

        # Convert back to original scale
        pred = np.expm1(pred_log)

        truth = np.expm1(y_test)

        metrics = {

            "MAE": mae(
                truth,
                pred
            ),

            "WAPE": wape(
                truth,
                pred
            ),

            "MedAPE": medape(
                truth,
                pred
            )

        }

        return metrics

    ############################################################
    # Train LightGBM
    ############################################################

    def train_lightgbm(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):

        logger.info(
            "Training LightGBM..."
        )

        model = self.models["LightGBM"]

        model.fit(
            X_train,
            y_train
        )

        metrics = self.evaluate_model(

            model,

            X_test,

            y_test

        )

        logger.info(
            f"LightGBM Metrics: {metrics}"
        )

        return {

            "name": "LightGBM",

            "model": model,

            "metrics": metrics

        }

    ############################################################
    # Train XGBoost
    ############################################################

    def train_xgboost(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):

        logger.info(
            "Training XGBoost..."
        )

        model = self.models["XGBoost"]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        metrics = self.evaluate_model(

            model,

            X_test,

            y_test

        )

        logger.info(
            f"XGBoost Metrics: {metrics}"
        )

        return {

            "name": "XGBoost",

            "model": model,

            "metrics": metrics

        }

    ############################################################
    # Train CatBoost
    ############################################################

    def train_catboost(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):

        logger.info(
            "Training CatBoost..."
        )

        model = self.models["CatBoost"]
        cat_features = X_train.select_dtypes(include=["category"]).columns.tolist()
        X_train_cat = X_train.copy()
        X_test_cat = X_test.copy()
        for col in cat_features:
            X_train_cat[col] = X_train_cat[col].cat.add_categories("Unknown").fillna("Unknown")
            X_test_cat[col] = X_test_cat[col].cat.add_categories("Unknown").fillna("Unknown")

        model.fit(
            X_train_cat,
            y_train,
            cat_features=cat_features,
            eval_set=(X_test_cat, y_test),
            verbose=False
        )

        metrics = self.evaluate_model(
            model,
            X_test_cat,
            y_test
        )

        logger.info(f"CatBoost Metrics: {metrics}")

        return {
            "name": "CatBoost",
            "model": model,
            "metrics": metrics
        }
    ############################################################
    # Compare Models
    ############################################################

    def compare_models(
        self,
        results: list[dict],
    ) -> tuple[dict, pd.DataFrame]:

        logger.info(
            "Comparing models..."
        )

        metrics_df = pd.DataFrame(

            [

                {

                    "Model": result["name"],

                    "MAE": result["metrics"]["MAE"],

                    "WAPE": result["metrics"]["WAPE"],

                    "MedAPE": result["metrics"]["MedAPE"],

                }

                for result in results

            ]

        )

        metrics_df = metrics_df.sort_values(

            by="WAPE",

            ascending=True,

        ).reset_index(drop=True)

        save_dataframe(

            metrics_df,

            self.report_dir /

            "metrics.csv"

        )

        logger.info(
            f"\n{metrics_df}"
        )

        best_model = next(

            result

            for result in results

            if result["name"] == metrics_df.iloc[0]["Model"]

        )

        logger.info(

            f"Best Model: {best_model['name']}"

        )

        return best_model, metrics_df

    ############################################################
    # Save Best Model
    ############################################################

    def save_best_model(
        self,
        best_model: dict,
    ):

        model_path = (

            self.model_dir /

            "best_model.pkl"

        )

        save_object(

            best_model["model"],

            model_path

        )

        logger.info(

            f"Best model saved to: {model_path}"

        )

    ############################################################
    # Run Pipeline
    ############################################################

    @timer
    def run(self, n_runs: int = 10):
        logger.info("=" * 70)
        logger.info(f"MODEL TRAINING PIPELINE - {n_runs} RUNS")
        logger.info("=" * 70)

        all_results = []

        for run_idx in range(n_runs):
            seed = self.random_state + run_idx
            logger.info(f"\n{'=' * 20} RUN {run_idx + 1}/{n_runs} (seed={seed}) {'=' * 20}")

            df = self.load_data()
            X_train, X_test, y_train, y_test = self.split_dataset(df, random_state=seed)
            X_train, X_test = self.build_features(X_train, X_test)

            # Train 3 models
            results = [
                self.train_lightgbm(X_train, y_train, X_test, y_test),
                self.train_xgboost(X_train, y_train, X_test, y_test),
                self.train_catboost(X_train, y_train, X_test, y_test),
            ]

            for res in results:
                # Làm phẳng metrics dictionary
                flat_result = {
                    "name": res["name"],
                    "run": run_idx + 1,
                    "seed": seed,
                    "MAE": res["metrics"]["MAE"],
                    "WAPE": res["metrics"]["WAPE"],
                    "MedAPE": res["metrics"]["MedAPE"]
                }
                all_results.append(flat_result)

        # ====================== TỔNG HỢP KẾT QUẢ ======================
        logger.info("\n" + "=" * 70)
        logger.info("TỔNG HỢP KẾT QUẢ SAU 50 LẦN CHẠY")
        logger.info("=" * 70)

        df_results = pd.DataFrame(all_results)

        summary = df_results.groupby("name").agg({
            "MAE": ["mean", "std"],
            "WAPE": ["mean", "std"],
            "MedAPE": ["mean", "std"]
        }).round(4)

        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        logger.info("\n" + str(summary))

        # Lưu kết quả
        detail_path = self.report_dir / "training_results_50_runs.csv"
        df_results.to_csv(detail_path, index=False)

        summary_path = self.report_dir / "training_summary_50_runs.csv"
        summary.to_csv(summary_path, index=False)

        logger.info(f"\nKết quả chi tiết đã lưu tại: {detail_path}")
        logger.info(f"Bảng tổng hợp đã lưu tại: {summary_path}")

        logger.info("=" * 70)
        logger.info("TRAINING HOÀN TẤT")
        logger.info("=" * 70)

        return summary
############################################################
# Main
############################################################

if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.run()
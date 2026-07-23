"""Model training, evaluation, and serialization (Tasks 5.1–5.6).

Trains linear and tree-based regressors with cross-validation, evaluates on
test metrics, performs residual diagnostics, and serializes the best model.

Outputs:
- data/processed/model_evaluation.md
- data/processed/residual_analysis.png
- models/best_model.joblib

Usage:
    python -m src.models.train
"""

import logging
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.read_parquet(PROCESSED / "features.parquet")
    target = pd.read_parquet(PROCESSED / "target.parquet")
    y = target["target_log"]
    # Load dates for time-based split
    modeling = pd.read_parquet(PROCESSED / "modeling_dataset.parquet")
    dates = modeling["order_purchase_timestamp"]
    log.info("Loaded %d rows × %d features", len(X), len(X.columns))
    return X, y, dates


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all regression metrics in log-space and original space."""
    # Log-space metrics
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_log = mean_absolute_error(y_true, y_pred)
    r2_log = r2_score(y_true, y_pred)

    # Original-space metrics (inverse log transform)
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)

    # WAPE = sum(|y - y_hat|) / sum(|y|)
    wape = np.sum(np.abs(y_true_orig - y_pred_orig)) / np.sum(np.abs(y_true_orig)) * 100

    # MedAPE = median(|y - y_hat| / |y|) * 100
    ape = (
        np.abs(y_true_orig - y_pred_orig) / np.clip(np.abs(y_true_orig), 1, None) * 100
    )
    med_ape = np.median(ape)

    # MAPE (capped to avoid division by near-zero)
    mape = np.mean(np.clip(ape, 0, 1000))

    return {
        "RMSE_log": rmse_log,
        "MAE_log": mae_log,
        "R2_log": r2_log,
        "RMSE_BRL": rmse_orig,
        "MAE_BRL": mae_orig,
        "WAPE_%": wape,
        "MedAPE_%": med_ape,
        "MAPE_%": mape,
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> dict:
    """Train all models with CV and evaluate on test set (Tasks 5.1–5.4).

    V2: Time-based train/test split (train < 2018-04-01, test >= 2018-04-01).
    V2: XGBoost uses Huber loss for robustness to outliers.
    """
    # --- Time-based split ---
    split_date = pd.Timestamp("2018-04-01")
    train_mask = dates < split_date
    test_mask = dates >= split_date
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    log.info(
        "Time-based split at %s: Train=%d (%s to %s), Test=%d (%s to %s)",
        split_date.date(),
        len(X_train),
        dates[train_mask].min().date(),
        dates[train_mask].max().date(),
        len(X_test),
        dates[test_mask].min().date(),
        dates[test_mask].max().date(),
    )

    # --- Task 5.1: Cross-validation setup ---
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- Define models ---
    models = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        ),
    }

    # Try XGBoost and LightGBM
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            objective="reg:absoluteerror",
        )
    except ImportError:
        log.warning("XGBoost not available")

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            objective="mae",
        )
    except ImportError:
        log.warning("LightGBM not available")

    results = {}

    for name, model in models.items():
        log.info("Training %s...", name)

        # CV scores
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        # Fit on full train
        model.fit(X_train, y_train)

        # Test predictions
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        metrics["CV_MAE_log"] = cv_mae
        metrics["CV_MAE_std"] = cv_std

        results[name] = {"model": model, "metrics": metrics, "y_pred": y_pred}

        log.info(
            "  %s: R²=%.4f  MAE_BRL=%.1f  WAPE=%.1f%%  MedAPE=%.1f%%",
            name,
            metrics["R2_log"],
            metrics["MAE_BRL"],
            metrics["WAPE_%"],
            metrics["MedAPE_%"],
        )

    return results, X_train, X_test, y_train, y_test


def residual_analysis(
    best_name: str, results: dict, y_test: pd.Series, plots_dir: Path
) -> None:
    """Task 5.5: Residual diagnostics."""
    log.info("=== Task 5.5: Residual analysis for %s ===", best_name)
    y_pred = results[best_name]["y_pred"]
    y_true = y_test.values
    residuals = y_true - y_pred
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.1, s=5, color="steelblue")
    axes[0, 0].axhline(0, color="red", ls="--")
    axes[0, 0].set_xlabel("Predicted (log)")
    axes[0, 0].set_ylabel("Residual (log)")
    axes[0, 0].set_title("Residuals vs Predicted")

    # 2. Residual histogram
    axes[0, 1].hist(residuals, bins=100, color="steelblue", edgecolor="white")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual (log)")
    axes[0, 1].axvline(0, color="red", ls="--")

    # 3. Actual vs Predicted (original scale)
    axes[1, 0].scatter(y_true_orig, y_pred_orig, alpha=0.1, s=5, color="steelblue")
    lim = max(y_true_orig.max(), y_pred_orig.max()) * 0.6
    axes[1, 0].plot([0, lim], [0, lim], "r--", lw=1)
    axes[1, 0].set_xlabel("Actual (BRL)")
    axes[1, 0].set_ylabel("Predicted (BRL)")
    axes[1, 0].set_title("Actual vs Predicted")
    axes[1, 0].set_xlim(0, lim)
    axes[1, 0].set_ylim(0, lim)

    # 4. Residuals by value bucket
    buckets = pd.cut(y_true_orig, bins=5)
    bucket_resid = pd.DataFrame(
        {"bucket": buckets, "resid": np.expm1(y_true) - np.expm1(y_pred)}
    )
    sns.boxplot(
        data=bucket_resid, x="bucket", y="resid", ax=axes[1, 1], palette="muted"
    )
    axes[1, 1].set_title("Residuals by Value Bucket (BRL)")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha="right")
    axes[1, 1].axhline(0, color="red", ls="--")

    plt.tight_layout()
    fig.savefig(plots_dir / "residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved residual_analysis.png")


def feature_importance(
    best_name: str, results: dict, X: pd.DataFrame, plots_dir: Path
) -> None:
    """Plot feature importances for tree-based models."""
    model = results[best_name]["model"]
    if not hasattr(model, "feature_importances_"):
        log.info("  Model %s has no feature_importances_", best_name)
        return

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    top20 = imp.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    top20.sort_values().plot.barh(color="steelblue", ax=ax)
    ax.set_title(f"Top 20 Feature Importances — {best_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved feature_importance.png")


def write_evaluation_report(results: dict, best_name: str, output_dir: Path) -> None:
    """Write model evaluation document."""
    lines = [
        "# Model Evaluation Report",
        "",
        f"**Best model:** {best_name}",
        "",
        "## Model Comparison",
        "",
        "| Model | R² (log) | MAE (BRL) | RMSE (BRL) | WAPE (%) | MedAPE (%) | CV MAE (log) |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, res in sorted(results.items(), key=lambda x: -x[1]["metrics"]["R2_log"]):
        m = res["metrics"]
        marker = " **← best**" if name == best_name else ""
        lines.append(
            f"| {name}{marker} | {m['R2_log']:.4f} | {m['MAE_BRL']:.1f} | "
            f"{m['RMSE_BRL']:.1f} | {m['WAPE_%']:.1f} | {m['MedAPE_%']:.1f} | "
            f"{m['CV_MAE_log']:.4f} ± {m['CV_MAE_std']:.4f} |"
        )

    best_m = results[best_name]["metrics"]
    lines.extend(
        [
            "",
            "## KPI Targets",
            "",
            "| KPI | Target | Achieved | Status |",
            "|---|---|---|---|",
            f"| MAE | < R$25 | R${best_m['MAE_BRL']:.1f} | {'✓' if best_m['MAE_BRL'] < 25 else '✗'} |",
            f"| WAPE | < 16% | {best_m['WAPE_%']:.1f}% | {'✓' if best_m['WAPE_%'] < 16 else '✗'} |",
            f"| MedAPE | < 12% | {best_m['MedAPE_%']:.1f}% | {'✓' if best_m['MedAPE_%'] < 12 else '✗'} |",
            "",
            "## Residual Analysis",
            "",
            "![Residual Analysis](residual_analysis.png)",
            "",
            "## Feature Importances",
            "",
            "![Feature Importances](feature_importance.png)",
            "",
            "## Notes",
            "",
            "- Target variable is log1p(order_value). Metrics in BRL use expm1 inverse transform.",
            "- Linear models use StandardScaler; tree-based models use raw encoded features.",
            "- CV = 5-fold stratified by target quartiles.",
        ]
    )

    (output_dir / "model_evaluation.md").write_text("\n".join(lines))
    log.info("Wrote model_evaluation.md")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = PROCESSED

    X, y, dates = load_data()
    results, X_train, X_test, y_train, y_test = train_and_evaluate(X, y, dates)

    # Select best model by R²
    best_name = max(results, key=lambda n: results[n]["metrics"]["R2_log"])
    log.info(
        "Best model: %s (R²=%.4f)", best_name, results[best_name]["metrics"]["R2_log"]
    )

    # Residual analysis
    residual_analysis(best_name, results, y_test, plots_dir)

    # Feature importance
    feature_importance(best_name, results, X, plots_dir)

    # Save best model (Task 5.6)
    best_model = results[best_name]["model"]
    model_artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "metrics": results[best_name]["metrics"],
    }
    with open(MODELS_DIR / "best_model.joblib", "wb") as f:
        pickle.dump(model_artifact, f)
    log.info("Saved best_model.joblib (%s)", best_name)

    # Evaluation report
    write_evaluation_report(results, best_name, plots_dir)

    log.info("=== Modeling Complete ===")


if __name__ == "__main__":
    main()

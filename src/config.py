"""
Centralized Configuration - Version ổn định
"""

from dataclasses import dataclass, field
from pathlib import Path


# ==========================================================
# Path Configuration
# ==========================================================

@dataclass(frozen=True, slots=True)
class PathConfig:
    """Quản lý đường dẫn dự án"""

    # Lên 2 cấp từ src/config.py → về thư mục gốc D:\DATA
    ROOT_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def DATA_DIR(self) -> Path:
        return self.ROOT_DIR / "data"

    @property
    def RAW_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def SYNTHETIC_DIR(self) -> Path:
        return self.DATA_DIR / "synthetic"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def MERGED_DIR(self) -> Path:
        return self.DATA_DIR / "merged"

    @property
    def DOCS_DIR(self) -> Path:
        return self.ROOT_DIR / "docs"

    @property
    def MODEL_DIR(self) -> Path:
        return self.ROOT_DIR / "models"

    @property
    def SAVED_MODEL_DIR(self) -> Path:
        return self.MODEL_DIR / "saved_models"

    @property
    def REPORT_DIR(self) -> Path:
        return self.ROOT_DIR / "reports"

    @property
    def LOG_DIR(self) -> Path:
        return self.ROOT_DIR / "logs"

    def create_directories(self) -> None:
        for d in [
            self.DATA_DIR, self.RAW_DIR, self.SYNTHETIC_DIR,
            self.PROCESSED_DIR, self.MERGED_DIR, self.DOCS_DIR,
            self.MODEL_DIR, self.SAVED_MODEL_DIR, self.REPORT_DIR, self.LOG_DIR
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Data Configuration
# ==========================================================

@dataclass(frozen=True, slots=True)
class DataConfig:
    """Tên các file dữ liệu"""

    # Final & Modeling
    FINAL_DATASET: str = "final_dataset.csv"
    MODELING_DATASET: str = "modeling_dataset.csv"

    # Behavioral (Synthetic)
    BEHAVIORAL_SESSIONS: str = "behavioral_sessions.csv"

    # Cleaned
    CLEANED_ORDERS: str = "cleaned_orders.csv"
    CLEANED_BEHAVIORAL_SESSIONS: str = "cleaned_behavioral_sessions.csv"


# ==========================================================
# Model Configuration
# ==========================================================

@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Tham số mặc định cho các model"""

    LIGHTGBM_PARAMS: dict = field(default_factory=lambda: {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })

    XGBOOST_PARAMS: dict = field(default_factory=lambda: {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror"
    })

    CATBOOST_PARAMS: dict = field(default_factory=lambda: {
        "iterations": 600,
        "learning_rate": 0.03,
        "depth": 8,
        "loss_function": "RMSE",
        "verbose": False
    })


# ==========================================================
# Training Configuration
# ==========================================================

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    TARGET: str = "log_order_value"
    ORIGINAL_TARGET: str = "order_value"


# ==========================================================
# Feature Engineering Configuration
# ==========================================================

@dataclass(frozen=True, slots=True)
class FeatureConfig:
    ID_COLUMNS: tuple[str, ...] = ("order_id", "session_id")
    CATEGORICAL_COLUMNS: tuple[str, ...] = ("device_type", "referral_channel", "discount_level")
    HIGH_DISCOUNT_THRESHOLD: int = 15
    HIGH_SESSION_QUANTILE: float = 0.75


# ==========================================================
# Main Config
# ==========================================================

@dataclass(frozen=True, slots=True)
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)


# Khởi tạo config và tạo thư mục
config = Config()
config.paths.create_directories()
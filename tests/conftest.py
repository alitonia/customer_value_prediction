"""Shared fixtures for the test suite."""

import pickle
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture
def synthetic_data():
    return {
        "profile": pd.read_parquet(SYNTH_DIR / "customer_profile.parquet"),
        "sessions": pd.read_parquet(SYNTH_DIR / "sessions.parquet"),
        "activity": pd.read_parquet(SYNTH_DIR / "session_activity.parquet"),
    }


@pytest.fixture
def features():
    return pd.read_parquet(PROCESSED / "features.parquet")


@pytest.fixture
def target():
    return pd.read_parquet(PROCESSED / "target.parquet")


@pytest.fixture
def model_artifact():
    with open(MODELS_DIR / "best_model.joblib", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def preprocessor():
    with open(PROCESSED / "preprocessor.joblib", "rb") as f:
        return pickle.load(f)

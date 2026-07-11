import joblib
import pandas as pd
from pathlib import Path


def save_object(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path):
    return joblib.load(path)


def save_dataframe(df: pd.DataFrame, path, index: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
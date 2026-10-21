"""Data loading, splitting and resampling helpers."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"
DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
# Override with --data on the command line or the CREDITCARD_CSV env variable.
DEFAULT_DATA_PATH = Path(os.environ.get("CREDITCARD_CSV", "data/creditcard.csv"))


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the credit card fraud CSV and validate it."""
    path = Path(path) if path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Download creditcard.csv from {DATASET_URL} and place it at "
            f"data/creditcard.csv, or pass --data /path/to/creditcard.csv."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected a '{TARGET}' column in {path}.")
    return df


def split_features_target(df: pd.DataFrame):
    return df.drop(columns=TARGET), df[TARGET]


def make_splits(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split of the ORIGINAL (imbalanced) data.

    The test set keeps the real fraud prevalence and is never resampled.
    All resampling (undersampling / SMOTE) is applied to the training part only.
    """
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def undersample(X: pd.DataFrame, y: pd.Series, seed: int = 42, ratio: float = 1.0):
    """Randomly undersample the majority class to `ratio` x minority size."""
    minority_idx = y[y == 1].index
    majority_idx = y[y == 0].index
    n_major = min(len(majority_idx), int(len(minority_idx) * ratio))
    keep_major = majority_idx.to_series().sample(n=n_major, random_state=seed).index
    keep = minority_idx.append(keep_major)
    keep = keep.to_series().sample(frac=1.0, random_state=seed).index  # shuffle
    return X.loc[keep], y.loc[keep]

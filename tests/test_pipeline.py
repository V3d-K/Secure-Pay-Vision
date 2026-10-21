"""Smoke tests on a small synthetic dataset shaped like creditcard.csv (no real data needed)."""
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.data_loader import make_splits, undersample
from src.evaluate import compute_metrics
from src.models import build_pipeline, fit_pipeline


@pytest.fixture(scope="module")
def df():
    X, y = make_classification(n_samples=6000, n_features=30, n_informative=10,
                               weights=[0.98, 0.02], flip_y=0, random_state=0)
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    out = pd.DataFrame(X, columns=cols)
    out["Class"] = y
    return out


def test_test_split_keeps_natural_imbalance(df):
    _, _, _, y_test = make_splits(df, 0.2, 1)
    assert abs(y_test.mean() - df["Class"].mean()) < 0.005


def test_undersample_balances_and_keeps_all_fraud(df):
    X, y = df.drop(columns="Class"), df["Class"]
    _, yu = undersample(X, y, seed=1)
    assert yu.mean() == 0.5 and yu.sum() == y.sum()


@pytest.mark.parametrize("model", ["logreg", "rf"])
@pytest.mark.parametrize("strategy", ["none", "class_weight", "undersample"])
def test_fit_and_metrics(df, model, strategy):
    X_train, X_test, y_train, y_test = make_splits(df, 0.2, 1)
    pipe = fit_pipeline(build_pipeline(model, strategy, n_estimators=30), strategy, X_train, y_train)
    m = compute_metrics(y_test, pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1])
    assert m["tp"] + m["fn"] == int(y_test.sum())
    assert 0.5 < m["roc_auc"] <= 1.0

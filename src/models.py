"""Model construction for the imbalance-handling strategies."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import undersample

MODELS = ("logreg", "rf")
STRATEGIES = ("none", "class_weight", "undersample", "smote")

MODEL_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest"}


def _classifier(model: str, class_weight, seed: int, n_estimators: int):
    if model == "logreg":
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=seed)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators, class_weight=class_weight, n_jobs=-1, random_state=seed
        )
    raise ValueError(f"Unknown model '{model}'. Choose from {MODELS}.")


def build_pipeline(model: str, strategy: str, seed: int = 42, n_estimators: int = 200):
    """Return an (unfitted) pipeline for a model + imbalance strategy.

    * none         - plain model, no imbalance handling (baseline)
    * class_weight - class_weight='balanced'
    * undersample  - training data is undersampled in `fit_pipeline`
    * smote        - SMOTE inside an imblearn pipeline (training data only)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")

    class_weight = "balanced" if strategy == "class_weight" else None
    clf = _classifier(model, class_weight, seed, n_estimators)
    steps = []
    if model == "logreg":  # Time/Amount are on a very different scale than V1..V28
        steps.append(("scale", StandardScaler()))

    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'smote' strategy needs imbalanced-learn: pip install imbalanced-learn"
            ) from exc
        steps.append(("smote", SMOTE(random_state=seed)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)

    steps.append(("clf", clf))
    return Pipeline(steps)


def fit_pipeline(pipe, strategy: str, X_train, y_train, seed: int = 42):
    """Fit on training data only (undersampling happens here, never on test data)."""
    if strategy == "undersample":
        X_train, y_train = undersample(X_train, y_train, seed=seed)
    pipe.fit(X_train, y_train)
    return pipe

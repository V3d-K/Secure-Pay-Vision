"""Train and evaluate Logistic Regression and Random Forest under several
imbalance-handling strategies, all on one untouched, naturally imbalanced test set.

Usage:
    python -m src.train --data data/creditcard.csv
    python -m src.train --models rf --strategies class_weight undersample --n-estimators 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data_loader import DEFAULT_DATA_PATH, load_data, make_splits
from .evaluate import (
    compute_metrics,
    majority_baseline_accuracy,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_pr,
    plot_roc,
)
from .models import MODEL_LABELS, MODELS, STRATEGIES, build_pipeline, fit_pipeline

TABLE_COLS = ["run", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fn", "fp"]


def run_experiments(X_train, X_test, y_train, y_test, models, strategies, seed=42, n_estimators=200, verbose=True):
    rows, scores, fitted = [], {}, {}
    for model in models:
        for strategy in strategies:
            label = f"{MODEL_LABELS[model]} | {strategy}"
            if verbose:
                print(f"Training {label} ...", flush=True)
            pipe = build_pipeline(model, strategy, seed=seed, n_estimators=n_estimators)
            fit_pipeline(pipe, strategy, X_train, y_train, seed=seed)
            y_score = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)  # default 0.5 decision threshold
            rows.append({"run": label, "model": model, "strategy": strategy,
                         **compute_metrics(y_test, y_pred, y_score)})
            scores[label] = y_score
            fitted[label] = pipe
    return rows, scores, fitted


def results_to_markdown(df: pd.DataFrame) -> str:
    """Pipe-table markdown without needing the optional `tabulate` package."""
    d = df[TABLE_COLS].copy()
    d["run"] = d["run"].str.replace("|", "\\|", regex=False)  # '|' would break the table
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        d[c] = d[c].map(lambda v: f"{v:.4f}")
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "|".join("---" for _ in d.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to creditcard.csv")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=200, help="Trees in the Random Forest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--model-dir", default="models")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = make_splits(df, args.test_size, args.seed)
    print(f"Train: {len(X_train):,} rows ({int(y_train.sum())} fraud) | "
          f"Test: {len(X_test):,} rows ({int(y_test.sum())} fraud, untouched/imbalanced)")

    rows, scores, fitted = run_experiments(
        X_train, X_test, y_train, y_test, args.models, args.strategies, args.seed, args.n_estimators
    )

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)
    header = (f"Test set: {len(y_test):,} transactions, {int(y_test.sum())} fraud "
              f"({y_test.mean():.4%}). Always-predict-legit accuracy: "
              f"{majority_baseline_accuracy(y_test):.4f}\n\n")
    (out / "metrics.md").write_text(header + results_to_markdown(table) + "\n")
    plot_roc(y_test, scores, out / "roc_curve.png")
    plot_pr(y_test, scores, out / "pr_curve.png")
    plot_confusion_matrices(rows, out / "confusion_matrix.png")
    plot_model_comparison(rows, out / "model_comparison.png")

    best = table.sort_values("pr_auc", ascending=False).iloc[0]
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best["run"]], model_dir / "best_model.joblib")
    (model_dir / "best_model.json").write_text(json.dumps(
        {"run": best["run"], "selected_by": "pr_auc", "pr_auc": float(best["pr_auc"]),
         "recall": float(best["recall"]), "precision": float(best["precision"]),
         "features": list(X_train.columns)}, indent=2))

    print("\n" + header + results_to_markdown(table))
    print(f"\nBest by PR-AUC: {best['run']}  -> saved to {model_dir / 'best_model.joblib'}")
    print(f"Metrics and plots written to {out}/")


if __name__ == "__main__":
    main()

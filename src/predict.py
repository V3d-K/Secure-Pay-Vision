"""Batch-score a CSV of transactions with the saved model.

This is offline batch scoring, not a real-time service.

Usage:
    python -m src.predict --input new_transactions.csv --output results/scored.csv --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV with the training feature columns (Class column optional)")
    p.add_argument("--output", default="results/scored_transactions.csv")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "best_model.joblib")
    features = json.loads((model_dir / "best_model.json").read_text())["features"]

    df = pd.read_csv(args.input)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing feature columns: {missing}")

    df["fraud_probability"] = model.predict_proba(df[features])[:, 1]
    df["predicted_fraud"] = (df["fraud_probability"] >= args.threshold).astype(int)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Scored {len(df):,} rows; flagged {int(df['predicted_fraud'].sum()):,} as fraud -> {args.output}")


if __name__ == "__main__":
    main()

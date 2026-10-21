"""Metrics and plots suited to highly imbalanced classification."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def majority_baseline_accuracy(y_true) -> float:
    """Accuracy of a 'never fraud' model -- the number accuracy must be compared against."""
    return float(1 - np.mean(y_true))


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, scores: dict, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (imbalanced test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, Path(path))


def plot_pr(y_true, scores: dict, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{label} (AP={average_precision_score(y_true, s):.3f})")
    ax.axhline(np.mean(y_true), color="k", ls="--", lw=1, label="Chance (fraud rate)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (imbalanced test set)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, Path(path))


def plot_confusion_matrices(rows: list, path):
    n = len(rows)
    cols = min(4, n)
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        ax.axis("on")
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(r["run"], fontsize=9)
    _save(fig, Path(path))


def plot_model_comparison(rows: list, path):
    labels = [r["run"] for r in rows]
    x = np.arange(len(rows))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(rows)), 5))
    for k, (metric, name) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("pr_auc", "PR-AUC")]):
        ax.bar(x + (k - 1) * w, [r[metric] for r in rows], w, label=name)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.05), title="Fraud-class metrics on the untouched test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, Path(path))

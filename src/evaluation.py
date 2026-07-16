"""Shared classification metrics and report-ready evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


CLASS_NAMES = ["Down", "Flat", "Up"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate JSON-serializable three-class metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    destination: Path,
    title: str,
) -> Path:
    """Write a compact, readable three-class confusion matrix."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    figure, axis = plt.subplots(figsize=(4.2, 3.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(title=title, xlabel="Predicted", ylabel="True")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def representative_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    examples_per_type: int = 3,
) -> pd.DataFrame:
    """Select confident correct and incorrect examples for qualitative analysis."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    records = []
    for example_type, mask in [("correct", y_true == y_pred), ("error", y_true != y_pred)]:
        candidates = np.flatnonzero(mask)
        selected = candidates[np.argsort(confidence[candidates])[::-1][:examples_per_type]]
        for index in selected:
            records.append(
                {
                    "example_type": example_type,
                    "timestamp": str(timestamps[index]),
                    "symbol": str(symbols[index]),
                    "true_class": CLASS_NAMES[int(y_true[index])],
                    "predicted_class": CLASS_NAMES[int(y_pred[index])],
                    "confidence": float(confidence[index]),
                    "p_down": float(probabilities[index, 0]),
                    "p_flat": float(probabilities[index, 1]),
                    "p_up": float(probabilities[index, 2]),
                }
            )
    return pd.DataFrame.from_records(records)

"""Reusable summaries for historical and cross-regime evaluation."""

from __future__ import annotations

import numpy as np

from src.evaluation import classification_metrics


def transfer_pairs() -> list[tuple[str, str]]:
    """Declare the complete low/high train-to-test transfer matrix."""
    return [
        ("low", "low"),
        ("low", "high"),
        ("high", "high"),
        ("high", "low"),
    ]


def evaluate_slices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    regimes: np.ndarray,
    symbols: np.ndarray,
) -> dict:
    """Evaluate aligned predictions overall and by regime and symbol."""
    y_true = np.asarray(y_true)
    regimes = np.asarray(regimes)
    symbols = np.asarray(symbols)
    if y_true.ndim != 1 or regimes.shape != y_true.shape or symbols.shape != y_true.shape:
        raise ValueError("labels, regimes, and symbols must be aligned one-dimensional arrays")
    if not len(y_true):
        raise ValueError("at least one labelled sample is required")
    aligned_predictions = {name: np.asarray(values) for name, values in predictions.items()}
    if not aligned_predictions or any(
        values.shape != y_true.shape for values in aligned_predictions.values()
    ):
        raise ValueError("predictions must contain aligned one-dimensional arrays")

    slices = {"overall": np.ones(len(y_true), dtype=bool)}
    slices.update(
        {f"regime_{name}": regimes == name for name in ("low", "medium", "high")}
    )
    slices.update(
        {f"symbol_{name}": symbols == name for name in sorted(set(symbols.astype(str)))}
    )
    output = {}
    for slice_name, mask in slices.items():
        if not mask.any():
            continue
        output[slice_name] = {
            "sample_count": int(mask.sum()),
            "class_counts": np.bincount(
                y_true[mask].astype(np.int64), minlength=3
            ).astype(int).tolist(),
            "models": {
                name: classification_metrics(y_true[mask], values[mask])
                for name, values in aligned_predictions.items()
            },
        }
    return output


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    """Summarize the two primary metrics across deterministic seed runs."""
    if not seed_results:
        raise ValueError("at least one seed result is required")
    keys = ("accuracy", "macro_f1")
    return {
        key: {
            "mean": float(np.mean([result[key] for result in seed_results])),
            "std": float(np.std([result[key] for result in seed_results], ddof=0)),
        }
        for key in keys
    }

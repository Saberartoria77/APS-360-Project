"""Simple non-sequential benchmark models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import DatasetBundle


@dataclass
class BaselineModels:
    """Fitted state for the reusable benchmark models."""

    majority_class: int
    logistic_regression: LogisticRegression


def fit_baseline_models(x_train: np.ndarray, y_train: np.ndarray) -> BaselineModels:
    """Fit benchmark models using only training features and labels."""
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=3)
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    logistic.fit(x_train[:, -1, :], y_train)
    return BaselineModels(
        majority_class=int(counts.argmax()), logistic_regression=logistic
    )


def predict_baselines(
    models: BaselineModels,
    features: np.ndarray,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Generate benchmark predictions for preprocessed feature windows."""
    return_index = feature_names.index("return_1h")
    standardized_return = features[:, -1, return_index]
    raw_return = (
        standardized_return * scaler_scale[return_index] + scaler_mean[return_index]
    )
    momentum = np.full(len(raw_return), 1, dtype=np.int64)
    momentum[raw_return < -threshold] = 0
    momentum[raw_return > threshold] = 2
    return {
        "majority": np.full(len(features), models.majority_class, dtype=np.int64),
        "momentum": momentum,
        "logistic_regression": models.logistic_regression.predict(
            features[:, -1, :]
        ).astype(np.int64),
    }


def fit_baselines(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    """Fit the benchmarks on a bundle and predict its test split."""
    models = fit_baseline_models(bundle.x_train, bundle.y_train)
    return predict_baselines(
        models,
        bundle.x_test,
        bundle.feature_names,
        bundle.scaler_mean,
        bundle.scaler_scale,
        bundle.threshold,
    )

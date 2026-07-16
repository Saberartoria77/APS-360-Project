"""Simple non-sequential benchmark models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import DatasetBundle


def fit_baselines(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    """Fit momentum and logistic-regression baselines on a prepared bundle."""
    return_index = bundle.feature_names.index("return_1h")
    standardized_return = bundle.x_test[:, -1, return_index]
    raw_return = (
        standardized_return * bundle.scaler_scale[return_index]
        + bundle.scaler_mean[return_index]
    )
    momentum = np.full(len(raw_return), 1, dtype=np.int64)
    momentum[raw_return < -bundle.threshold] = 0
    momentum[raw_return > bundle.threshold] = 2

    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    logistic.fit(bundle.x_train[:, -1, :], bundle.y_train)
    logistic_predictions = logistic.predict(bundle.x_test[:, -1, :]).astype(np.int64)
    return {
        "momentum": momentum,
        "logistic_regression": logistic_predictions,
    }


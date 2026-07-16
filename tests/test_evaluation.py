from pathlib import Path

import numpy as np
import pytest

from src.baselines import fit_baselines
from src.data import DatasetBundle, FEATURE_COLUMNS
from src.evaluation import classification_metrics, representative_predictions, save_confusion_matrix


@pytest.fixture
def bundle() -> DatasetBundle:
    rng = np.random.default_rng(42)
    shapes = {"train": 90, "val": 18, "test": 21}
    arrays = {name: rng.normal(size=(size, 96, len(FEATURE_COLUMNS))).astype(np.float32) for name, size in shapes.items()}
    labels = {name: np.arange(size, dtype=np.int64) % 3 for name, size in shapes.items()}
    base = np.datetime64("2025-01-01T00")
    times = {
        "train": base + np.arange(shapes["train"]).astype("timedelta64[h]"),
        "val": base + 200 + np.arange(shapes["val"]).astype("timedelta64[h]"),
        "test": base + 300 + np.arange(shapes["test"]).astype("timedelta64[h]"),
    }
    symbols = {name: np.array(["BTCUSDT"] * size) for name, size in shapes.items()}
    return DatasetBundle(
        x_train=arrays["train"], y_train=labels["train"],
        x_val=arrays["val"], y_val=labels["val"],
        x_test=arrays["test"], y_test=labels["test"],
        train_times=times["train"], val_times=times["val"], test_times=times["test"],
        train_symbols=symbols["train"], val_symbols=symbols["val"], test_symbols=symbols["test"],
        feature_names=list(FEATURE_COLUMNS),
        scaler_mean=np.zeros(len(FEATURE_COLUMNS)),
        scaler_scale=np.ones(len(FEATURE_COLUMNS)),
        threshold=0.25,
    )


def test_classification_metrics_has_required_fields() -> None:
    result = classification_metrics(np.array([0, 1, 2]), np.array([0, 2, 2]))
    assert set(result) >= {"accuracy", "macro_f1", "per_class_recall", "confusion_matrix"}
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)


def test_baselines_return_one_prediction_per_test_sample(bundle: DatasetBundle) -> None:
    predictions = fit_baselines(bundle)
    assert set(predictions) == {"momentum", "logistic_regression"}
    assert all(len(values) == len(bundle.y_test) for values in predictions.values())
    assert all(set(np.unique(values)).issubset({0, 1, 2}) for values in predictions.values())


def test_confusion_matrix_plot_is_written(tmp_path: Path) -> None:
    destination = tmp_path / "confusion.png"
    save_confusion_matrix(np.array([0, 1, 2]), np.array([0, 2, 2]), destination, "Example")
    assert destination.exists() and destination.stat().st_size > 0


def test_representative_predictions_contains_correct_and_error_examples(bundle: DatasetBundle) -> None:
    predicted = bundle.y_test.copy()
    predicted[0] = (predicted[0] + 1) % 3
    probabilities = np.eye(3)[predicted]
    examples = representative_predictions(
        bundle.y_test, predicted, probabilities, bundle.test_times, bundle.test_symbols
    )
    assert {"correct", "error"}.issubset(set(examples["example_type"]))
    assert set(examples.columns) >= {"timestamp", "symbol", "true_class", "predicted_class", "confidence"}


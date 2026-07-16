import numpy as np
import pytest
import torch

from src.data import DatasetBundle, FEATURE_COLUMNS
from src.models import CNNLSTM
from src.training import TrainConfig, train_model


@pytest.fixture
def tiny_bundle() -> DatasetBundle:
    rng = np.random.default_rng(7)
    counts = {"train": 48, "val": 18, "test": 15}
    x = {
        name: rng.normal(size=(count, 24, len(FEATURE_COLUMNS))).astype(np.float32)
        for name, count in counts.items()
    }
    y = {name: np.arange(count, dtype=np.int64) % 3 for name, count in counts.items()}
    times = {
        name: np.datetime64("2025-01-01") + np.arange(count).astype("timedelta64[h]")
        for name, count in counts.items()
    }
    symbols = {name: np.array(["BTCUSDT"] * count) for name, count in counts.items()}
    return DatasetBundle(
        x_train=x["train"], y_train=y["train"],
        x_val=x["val"], y_val=y["val"],
        x_test=x["test"], y_test=y["test"],
        train_times=times["train"], val_times=times["val"], test_times=times["test"],
        train_symbols=symbols["train"], val_symbols=symbols["val"], test_symbols=symbols["test"],
        feature_names=list(FEATURE_COLUMNS),
        scaler_mean=np.zeros(len(FEATURE_COLUMNS)),
        scaler_scale=np.ones(len(FEATURE_COLUMNS)),
        threshold=0.002,
    )


def test_cnn_lstm_output_shape() -> None:
    model = CNNLSTM(n_features=len(FEATURE_COLUMNS))
    logits = model(torch.randn(8, 96, len(FEATURE_COLUMNS)))
    assert logits.shape == (8, 3)


def test_training_smoke_run(tiny_bundle: DatasetBundle) -> None:
    result = train_model(
        tiny_bundle,
        TrainConfig(epochs=2, batch_size=16, seed=42, device="cpu"),
    )
    assert len(result.history["train_loss"]) == 2
    assert len(result.history["val_loss"]) == 2
    assert result.test_probabilities.shape == (len(tiny_bundle.y_test), 3)
    np.testing.assert_allclose(result.test_probabilities.sum(1), 1, atol=1e-5)
    assert result.parameter_count > 0


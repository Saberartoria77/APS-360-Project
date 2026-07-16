import numpy as np
import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS, DatasetBundle, prepare_datasets


@pytest.fixture
def source_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    index = pd.date_range("2024-01-01", periods=900, freq="h", tz="UTC")
    for symbol, offset in [("BTCUSDT", 0.0), ("ETHUSDT", 4.0)]:
        close = 100 + offset + np.linspace(0, 20, len(index)) + 2 * np.sin(np.arange(len(index)) / 9)
        frame = pd.DataFrame(index=index)
        frame["close"] = close
        for position, column in enumerate(FEATURE_COLUMNS):
            frame[column] = np.sin(np.arange(len(index)) / (position + 3)) + offset + position / 10
        frames[symbol] = frame
    return frames


@pytest.fixture
def bundle(source_frames: dict[str, pd.DataFrame]) -> DatasetBundle:
    return prepare_datasets(source_frames, window=96, train_fraction=0.70, val_fraction=0.15)


def test_split_is_chronological(bundle: DatasetBundle) -> None:
    assert bundle.train_times.max() < bundle.val_times.min()
    assert bundle.val_times.max() < bundle.test_times.min()


def test_windows_do_not_cross_symbols_or_split_boundaries(bundle: DatasetBundle) -> None:
    assert bundle.x_train.shape[1:] == (96, len(FEATURE_COLUMNS))
    assert len(bundle.y_train) == len(bundle.train_times) == len(bundle.train_symbols)
    assert len(bundle.y_val) == len(bundle.val_times) == len(bundle.val_symbols)
    assert len(bundle.y_test) == len(bundle.test_times) == len(bundle.test_symbols)
    assert set(bundle.train_symbols) == {"BTCUSDT", "ETHUSDT"}


def test_scaler_is_fit_on_training_rows_only(
    source_frames: dict[str, pd.DataFrame], bundle: DatasetBundle
) -> None:
    training_rows = pd.concat(
        [frame.loc[frame.index <= bundle.train_times.max(), FEATURE_COLUMNS] for frame in source_frames.values()]
    )
    np.testing.assert_allclose(bundle.scaler_mean, training_rows.mean().to_numpy(), atol=1e-10)
    assert not np.allclose(bundle.x_test.mean(axis=(0, 1)), 0.0, atol=1e-2)


def test_labels_match_frozen_threshold(bundle: DatasetBundle) -> None:
    assert 0.001 <= bundle.threshold <= 0.01
    assert set(np.unique(bundle.y_train)).issubset({0, 1, 2})
    assert bundle.feature_names == FEATURE_COLUMNS


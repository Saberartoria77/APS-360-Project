import numpy as np
import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS, prepare_evaluation_windows


def _clean_frame() -> pd.DataFrame:
    index = pd.date_range("2026-06-20", "2026-08-01 01:00", freq="h", inclusive="left", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["close"] = 100 * np.power(1.01, np.arange(len(index)))
    for position, name in enumerate(FEATURE_COLUMNS):
        frame[name] = position + np.arange(len(index), dtype=float) / 10
    return frame


def test_evaluation_windows_use_frozen_metadata_and_only_requested_targets() -> None:
    frame = _clean_frame()
    feature_names = list(reversed(FEATURE_COLUMNS))
    scaler_mean = np.linspace(30.0, 42.0, len(FEATURE_COLUMNS))
    scaler_scale = np.linspace(4.0, 16.0, len(FEATURE_COLUMNS))

    dataset = prepare_evaluation_windows(
        {"BTCUSDT": frame},
        feature_names=feature_names,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        threshold=0.005,
        window=96,
        target_start=pd.Timestamp("2026-07-01", tz="UTC"),
        target_end=pd.Timestamp("2026-08-01", tz="UTC"),
    )

    assert dataset.times.min() >= np.datetime64("2026-07-01T00")
    assert dataset.times.max() < np.datetime64("2026-08-01T00")
    assert dataset.times.max() == np.datetime64("2026-07-31T23")
    assert dataset.x.shape[1:] == (96, len(FEATURE_COLUMNS))
    assert len(dataset.x) == len(dataset.y) == len(dataset.times) == len(dataset.symbols)
    np.testing.assert_array_equal(dataset.y, np.full(len(dataset.y), 2))
    np.testing.assert_allclose(dataset.x[0, -1, 0], (12 + 26.4 - 30.0) / 4.0)


def test_evaluation_windows_reject_gapped_hourly_index() -> None:
    frame = _clean_frame().drop(pd.Timestamp("2026-06-25 12:00", tz="UTC"))

    with pytest.raises(ValueError, match="hourly contiguous"):
        prepare_evaluation_windows(
            {"BTCUSDT": frame},
            feature_names=list(FEATURE_COLUMNS),
            scaler_mean=np.zeros(len(FEATURE_COLUMNS)),
            scaler_scale=np.ones(len(FEATURE_COLUMNS)),
            threshold=0.005,
            window=96,
            target_start=pd.Timestamp("2026-07-01", tz="UTC"),
            target_end=pd.Timestamp("2026-08-01", tz="UTC"),
        )


@pytest.mark.parametrize(
    ("scaler_mean", "scaler_scale"),
    [
        (np.full(len(FEATURE_COLUMNS), np.nan), np.ones(len(FEATURE_COLUMNS))),
        (np.zeros(len(FEATURE_COLUMNS)), np.array([0.0, *np.ones(len(FEATURE_COLUMNS) - 1)])),
    ],
)
def test_evaluation_windows_reject_invalid_frozen_scaler_metadata(
    scaler_mean: np.ndarray, scaler_scale: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="frozen feature and scaler metadata"):
        prepare_evaluation_windows(
            {"BTCUSDT": _clean_frame()},
            feature_names=list(FEATURE_COLUMNS),
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            threshold=0.005,
            window=96,
            target_start=pd.Timestamp("2026-07-01", tz="UTC"),
            target_end=pd.Timestamp("2026-08-01", tz="UTC"),
        )

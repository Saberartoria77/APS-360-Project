import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS, prepare_evaluation_windows


def test_evaluation_windows_use_frozen_metadata_and_only_requested_targets() -> None:
    index = pd.date_range("2026-06-20", "2026-08-01", freq="h", inclusive="left", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["close"] = 100 * np.exp(np.cumsum(np.full(len(index), 0.0002)))
    for position, name in enumerate(FEATURE_COLUMNS):
        frame[name] = position + np.arange(len(index), dtype=float) / 1000

    dataset = prepare_evaluation_windows(
        {"BTCUSDT": frame},
        feature_names=list(FEATURE_COLUMNS),
        scaler_mean=np.zeros(len(FEATURE_COLUMNS)),
        scaler_scale=np.ones(len(FEATURE_COLUMNS)),
        threshold=0.001,
        window=96,
        target_start=pd.Timestamp("2026-07-01", tz="UTC"),
        target_end=pd.Timestamp("2026-08-01", tz="UTC"),
    )

    assert dataset.times.min() >= np.datetime64("2026-07-01T00")
    assert dataset.times.max() < np.datetime64("2026-08-01T00")
    assert dataset.x.shape[1:] == (96, len(FEATURE_COLUMNS))
    assert len(dataset.x) == len(dataset.y) == len(dataset.times) == len(dataset.symbols)

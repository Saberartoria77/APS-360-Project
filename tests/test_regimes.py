import numpy as np
import pandas as pd
import pytest

from src.data import DatasetBundle, FEATURE_COLUMNS
from src.regimes import (
    assign_bundle_regimes,
    assign_regimes,
    fit_regime_thresholds,
    subset_bundle,
)


def _frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    frames = {}
    for symbol, scale in [("BTCUSDT", 0.002), ("ETHUSDT", 0.006)]:
        returns = scale * (1.0 + np.sin(np.arange(len(index)) / 17.0))
        frames[symbol] = pd.DataFrame({"log_return_1h": returns}, index=index)
    return frames


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(9)
    sizes = {"train": 12, "val": 6, "test": 6}
    base = np.datetime64("2025-01-08T00")
    values = {
        split: rng.normal(size=(size, 96, len(FEATURE_COLUMNS))).astype(np.float32)
        for split, size in sizes.items()
    }
    labels = {split: np.arange(size) % 3 for split, size in sizes.items()}
    times = {
        split: base + offset + np.arange(size).astype("timedelta64[h]")
        for split, size, offset in [("train", 12, 0), ("val", 6, 100), ("test", 6, 200)]
    }
    symbols = {split: np.array(["BTCUSDT", "ETHUSDT"] * (size // 2)) for split, size in sizes.items()}
    return DatasetBundle(
        x_train=values["train"], y_train=labels["train"],
        x_val=values["val"], y_val=labels["val"],
        x_test=values["test"], y_test=labels["test"],
        train_times=times["train"], val_times=times["val"], test_times=times["test"],
        train_symbols=symbols["train"], val_symbols=symbols["val"], test_symbols=symbols["test"],
        feature_names=list(FEATURE_COLUMNS), scaler_mean=np.zeros(len(FEATURE_COLUMNS)),
        scaler_scale=np.ones(len(FEATURE_COLUMNS)), threshold=0.002,
    )


def test_regime_assignment_is_causal_and_thresholds_are_per_symbol() -> None:
    frames = _frames()
    train_end = frames["BTCUSDT"].index[260]
    thresholds = fit_regime_thresholds(frames, train_end=train_end, lookback=24)
    assert set(thresholds.by_symbol) == {"BTCUSDT", "ETHUSDT"}
    assert thresholds.by_symbol["BTCUSDT"] != thresholds.by_symbol["ETHUSDT"]

    times = np.array([frames["BTCUSDT"].index[280].to_datetime64()])
    original = assign_regimes(frames, times, np.array(["BTCUSDT"]), thresholds)
    changed = {name: frame.copy() for name, frame in frames.items()}
    changed["BTCUSDT"].loc[changed["BTCUSDT"].index[300]:, "log_return_1h"] *= 100
    revised = assign_regimes(changed, times, np.array(["BTCUSDT"]), thresholds)
    np.testing.assert_array_equal(original, revised)


def test_subset_bundle_preserves_alignment() -> None:
    bundle = _bundle()
    masks = {
        "train": np.array([True, False] * 6),
        "val": np.array([True, False] * 3),
        "test": np.array([True, False] * 3),
    }
    subset = subset_bundle(bundle, masks)
    np.testing.assert_array_equal(subset.y_test, bundle.y_test[masks["test"]])
    np.testing.assert_array_equal(subset.test_times, bundle.test_times[masks["test"]])
    np.testing.assert_array_equal(subset.test_symbols, bundle.test_symbols[masks["test"]])
    assert subset.feature_names == bundle.feature_names


def test_assign_regimes_rejects_mismatched_times_and_symbols() -> None:
    frames = _frames()
    thresholds = fit_regime_thresholds(frames, train_end=frames["BTCUSDT"].index[260], lookback=24)

    with pytest.raises(ValueError, match="same shape"):
        assign_regimes(
            frames,
            np.array([frames["BTCUSDT"].index[280].to_datetime64()]),
            np.array(["BTCUSDT", "ETHUSDT"]),
            thresholds,
        )


def test_assign_bundle_regimes_returns_aligned_split_labels() -> None:
    frames = _frames()
    bundle = _bundle()
    thresholds = fit_regime_thresholds(frames, train_end=frames["BTCUSDT"].index[260], lookback=24)

    regimes = assign_bundle_regimes(bundle, frames, thresholds)

    assert set(regimes) == {"train", "val", "test"}
    for split in regimes:
        assert regimes[split].shape == getattr(bundle, f"y_{split}").shape
        assert set(regimes[split]).issubset({"low", "medium", "high"})

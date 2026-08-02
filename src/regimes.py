"""Causal volatility regimes for cross-regime evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import DatasetBundle


REGIME_NAMES = np.array(["low", "medium", "high"])


@dataclass(frozen=True)
class RegimeThresholds:
    by_symbol: dict[str, tuple[float, float]]
    lookback: int = 168


def realized_volatility(frame: pd.DataFrame, lookback: int = 168) -> pd.Series:
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    if "log_return_1h" not in frame:
        raise ValueError("frame is missing log_return_1h")
    if (
        frame.index.tz is None
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("frame index must be timezone-aware, unique, and chronological")
    return frame["log_return_1h"].rolling(lookback, min_periods=lookback).std(ddof=0)


def fit_regime_thresholds(
    frames: dict[str, pd.DataFrame], train_end: pd.Timestamp, lookback: int = 168
) -> RegimeThresholds:
    train_end = pd.Timestamp(train_end)
    train_end = (
        train_end.tz_localize("UTC")
        if train_end.tzinfo is None
        else train_end.tz_convert("UTC")
    )
    fitted = {}
    for symbol, frame in frames.items():
        values = realized_volatility(frame, lookback).loc[:train_end].dropna()
        if values.empty:
            raise ValueError(f"{symbol} has no training volatility values")
        low, high = values.quantile([0.33, 0.67]).to_numpy(dtype=float)
        fitted[symbol] = (float(low), float(high))
    return RegimeThresholds(by_symbol=fitted, lookback=lookback)


def assign_regimes(
    frames: dict[str, pd.DataFrame],
    times: np.ndarray,
    symbols: np.ndarray,
    thresholds: RegimeThresholds,
) -> np.ndarray:
    times = np.asarray(times)
    symbols = np.asarray(symbols)
    if times.ndim != 1 or symbols.ndim != 1 or times.shape != symbols.shape:
        raise ValueError("times and symbols must be one-dimensional arrays with the same shape")
    labels = np.empty(len(times), dtype="<U6")
    volatility = {
        symbol: realized_volatility(frame, thresholds.lookback)
        for symbol, frame in frames.items()
    }
    for index, (time, symbol_value) in enumerate(zip(times, symbols)):
        symbol = str(symbol_value)
        if symbol not in volatility or symbol not in thresholds.by_symbol:
            raise ValueError(f"missing regime inputs for {symbol}")
        timestamp = pd.Timestamp(time)
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        value = volatility[symbol].get(timestamp, np.nan)
        if pd.isna(value):
            raise ValueError(f"missing causal volatility for {symbol} at {timestamp}")
        low, high = thresholds.by_symbol[symbol]
        labels[index] = "low" if value <= low else "high" if value > high else "medium"
    return labels


def assign_bundle_regimes(
    bundle: DatasetBundle,
    frames: dict[str, pd.DataFrame],
    thresholds: RegimeThresholds,
) -> dict[str, np.ndarray]:
    return {
        split: assign_regimes(
            frames,
            getattr(bundle, f"{split}_times"),
            getattr(bundle, f"{split}_symbols"),
            thresholds,
        )
        for split in ("train", "val", "test")
    }


def subset_bundle(bundle: DatasetBundle, masks: dict[str, np.ndarray]) -> DatasetBundle:
    values = {}
    for split in ("train", "val", "test"):
        mask = np.asarray(masks[split], dtype=bool)
        labels = getattr(bundle, f"y_{split}")
        if mask.shape != labels.shape or not mask.any():
            raise ValueError(f"{split} mask must select at least one aligned sample")
        values[split] = {
            "x": getattr(bundle, f"x_{split}")[mask],
            "y": labels[mask],
            "times": getattr(bundle, f"{split}_times")[mask],
            "symbols": getattr(bundle, f"{split}_symbols")[mask],
        }
    if len(np.unique(values["train"]["y"])) < 2:
        raise ValueError("training subset must contain at least two classes")
    return DatasetBundle(
        x_train=values["train"]["x"],
        y_train=values["train"]["y"],
        x_val=values["val"]["x"],
        y_val=values["val"]["y"],
        x_test=values["test"]["x"],
        y_test=values["test"]["y"],
        train_times=values["train"]["times"],
        val_times=values["val"]["times"],
        test_times=values["test"]["times"],
        train_symbols=values["train"]["symbols"],
        val_symbols=values["val"]["symbols"],
        test_symbols=values["test"]["symbols"],
        feature_names=list(bundle.feature_names),
        scaler_mean=bundle.scaler_mean.copy(),
        scaler_scale=bundle.scaler_scale.copy(),
        threshold=float(bundle.threshold),
    )

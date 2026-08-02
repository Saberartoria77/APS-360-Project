"""Market-data collection and causal feature engineering."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base_volume",
    "taker_quote_volume",
    "ignore",
]
FEATURE_COLUMNS = [
    "return_1h",
    "log_return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "bb_width",
    "volatility_24h",
    "volume_z_24h",
]


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_times: np.ndarray
    val_times: np.ndarray
    test_times: np.ndarray
    train_symbols: np.ndarray
    val_symbols: np.ndarray
    test_symbols: np.ndarray
    feature_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    threshold: float


@dataclass
class EvaluationDataset:
    """Prospective windows transformed with frozen training metadata."""

    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    symbols: np.ndarray


def prepare_evaluation_windows(
    frames: dict[str, pd.DataFrame],
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    threshold: float,
    window: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> EvaluationDataset:
    """Build labelled prospective windows without fitting any preprocessing."""
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be at least two")
    if not frames:
        raise ValueError("at least one symbol frame is required")

    start = pd.Timestamp(target_start)
    end = pd.Timestamp(target_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start >= end:
        raise ValueError("target_start must be before target_end")

    feature_array = np.asarray(feature_names, dtype=object)
    scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
    if (
        feature_array.ndim != 1
        or not len(feature_array)
        or scaler_mean.ndim != 1
        or scaler_scale.ndim != 1
        or len(feature_array) != len(scaler_mean)
        or scaler_mean.shape != scaler_scale.shape
        or not np.all(np.isfinite(scaler_mean))
        or not np.all(np.isfinite(scaler_scale))
        or np.any(scaler_scale <= 0.0)
    ):
        raise ValueError("frozen feature and scaler metadata are incompatible")

    collected: dict[str, list] = {"x": [], "y": [], "times": [], "symbols": []}
    for symbol, frame in frames.items():
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{symbol} index must be timezone-aware")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol} index must be unique and chronological")
        if not np.all(np.diff(frame.index.asi8) == pd.Timedelta(hours=1).value):
            raise ValueError(f"{symbol} index must be hourly contiguous")

        required = {"close", *feature_names}
        if missing := required.difference(frame.columns):
            raise ValueError(f"{symbol} is missing columns: {sorted(missing)}")

        subset = frame.copy()
        subset.index = subset.index.tz_convert("UTC")
        subset["label"] = make_direction_labels(subset["close"], horizon=1, threshold=threshold)
        subset = subset.dropna(subset=[*feature_names, "label"])
        if not np.all(np.diff(subset.index.asi8) == pd.Timedelta(hours=1).value):
            raise ValueError(f"{symbol} retained rows must be hourly contiguous")
        values = (subset[feature_names].to_numpy(dtype=np.float64) - scaler_mean) / scaler_scale
        labels = subset["label"].to_numpy(dtype=np.int64)
        times = subset.index.to_numpy(dtype="datetime64[ns]")
        for end_position in range(window - 1, len(subset)):
            timestamp = subset.index[end_position]
            if start <= timestamp < end:
                collected["x"].append(values[end_position - window + 1 : end_position + 1])
                collected["y"].append(labels[end_position])
                collected["times"].append(times[end_position])
                collected["symbols"].append(symbol)

    if not collected["x"]:
        raise ValueError("requested target range produced no complete evaluation windows")
    order = np.argsort(np.asarray(collected["times"]))
    return EvaluationDataset(
        x=np.asarray(collected["x"], dtype=np.float32)[order],
        y=np.asarray(collected["y"], dtype=np.int64)[order],
        times=np.asarray(collected["times"])[order],
        symbols=np.asarray(collected["symbols"])[order],
    )


def prepare_datasets(
    frames: dict[str, pd.DataFrame],
    window: int = 96,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> DatasetBundle:
    """Create per-asset windows with chronological splits and training-only scaling."""
    if not frames:
        raise ValueError("at least one symbol frame is required")
    if window < 2:
        raise ValueError("window must be at least two")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")

    for symbol, frame in frames.items():
        required = {"close", *FEATURE_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{symbol} is missing columns: {sorted(missing)}")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol} index must be unique and chronological")

    all_times = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in frames.values()))))
    train_count = int(len(all_times) * train_fraction)
    val_count = int(len(all_times) * val_fraction)
    if min(train_count, val_count, len(all_times) - train_count - val_count) <= window:
        raise ValueError("each chronological split must contain more rows than the window")
    train_end = all_times[train_count - 1]
    val_start = all_times[train_count]
    val_end = all_times[train_count + val_count - 1]
    test_start = all_times[train_count + val_count]

    training_returns = []
    training_feature_rows = []
    for frame in frames.values():
        train = frame.loc[frame.index <= train_end]
        training_returns.append(train["close"].shift(-1).div(train["close"]).sub(1.0).dropna().abs())
        training_feature_rows.append(train.iloc[:-1][FEATURE_COLUMNS])
    threshold = float(np.clip(pd.concat(training_returns).median(), 0.001, 0.01))
    training_features = pd.concat(training_feature_rows).dropna()
    scaler_mean = training_features.mean().to_numpy(dtype=np.float64)
    scaler_scale = training_features.std(ddof=0).to_numpy(dtype=np.float64)
    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)

    split_ranges = {
        "train": (all_times[0], train_end),
        "val": (val_start, val_end),
        "test": (test_start, all_times[-1]),
    }
    collected: dict[str, dict[str, list]] = {
        name: {"x": [], "y": [], "times": [], "symbols": []} for name in split_ranges
    }
    for symbol, frame in frames.items():
        for split_name, (split_start, split_end) in split_ranges.items():
            subset = frame.loc[(frame.index >= split_start) & (frame.index <= split_end)].copy()
            subset["label"] = make_direction_labels(subset["close"], horizon=1, threshold=threshold)
            subset = subset.dropna(subset=[*FEATURE_COLUMNS, "label"])
            feature_values = subset[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            feature_values = (feature_values - scaler_mean) / scaler_scale
            labels = subset["label"].to_numpy(dtype=np.int64)
            times = subset.index.to_numpy()
            for end_position in range(window - 1, len(subset)):
                start_position = end_position - window + 1
                collected[split_name]["x"].append(feature_values[start_position : end_position + 1])
                collected[split_name]["y"].append(labels[end_position])
                collected[split_name]["times"].append(times[end_position])
                collected[split_name]["symbols"].append(symbol)

    arrays = {}
    for split_name, values in collected.items():
        if not values["x"]:
            raise ValueError(f"{split_name} split produced no complete windows")
        order = np.argsort(np.asarray(values["times"]))
        arrays[split_name] = {
            "x": np.asarray(values["x"], dtype=np.float32)[order],
            "y": np.asarray(values["y"], dtype=np.int64)[order],
            "times": np.asarray(values["times"])[order],
            "symbols": np.asarray(values["symbols"])[order],
        }

    return DatasetBundle(
        x_train=arrays["train"]["x"],
        y_train=arrays["train"]["y"],
        x_val=arrays["val"]["x"],
        y_val=arrays["val"]["y"],
        x_test=arrays["test"]["x"],
        y_test=arrays["test"]["y"],
        train_times=arrays["train"]["times"],
        val_times=arrays["val"]["times"],
        test_times=arrays["test"]["times"],
        train_symbols=arrays["train"]["symbols"],
        val_symbols=arrays["val"]["symbols"],
        test_symbols=arrays["test"]["symbols"],
        feature_names=list(FEATURE_COLUMNS),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        threshold=threshold,
    )


def make_direction_labels(
    close: pd.Series, horizon: int = 1, threshold: float = 0.0025
) -> pd.Series:
    """Classify future returns as down=0, flat=1, or up=2."""
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    future_return = close.shift(-horizon).div(close).sub(1.0)
    labels = pd.Series(np.nan, index=close.index, dtype=float)
    labels.loc[future_return < -threshold] = 0.0
    labels.loc[future_return.abs() <= threshold] = 1.0
    labels.loc[future_return > threshold] = 2.0
    return labels


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV plus technical indicators computed without future values."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be unique and chronological")

    result = frame.copy()
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    result["return_1h"] = close.pct_change(fill_method=None)
    result["log_return_1h"] = np.log(close).diff()
    result["high_low_range"] = (result["high"] - result["low"]) / close
    result["close_open_return"] = close.div(result["open"]).sub(1.0)
    result["volume_change"] = volume.pct_change(fill_method=None)

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain.div(avg_loss.replace(0.0, np.nan))
    result["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    result.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    rolling_mean = close.rolling(20, min_periods=20).mean()
    rolling_std = close.rolling(20, min_periods=20).std(ddof=0)
    result["bb_position"] = (close - rolling_mean).div(2.0 * rolling_std.replace(0.0, np.nan))
    result["bb_width"] = (4.0 * rolling_std).div(rolling_mean)
    result["volatility_24h"] = result["return_1h"].rolling(24, min_periods=24).std(ddof=0)
    volume_mean = volume.rolling(24, min_periods=24).mean()
    volume_std = volume.rolling(24, min_periods=24).std(ddof=0)
    result["volume_z_24h"] = (volume - volume_mean).div(volume_std.replace(0.0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def fetch_klines(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1h",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Download a chronological Binance kline range, optionally using a CSV cache."""
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="open_time", parse_dates=["open_time"])
        cached.index = pd.to_datetime(cached.index, utc=True)
        return cached

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start >= end:
        raise ValueError("start must be before end")

    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []
    session = requests.Session()
    while cursor_ms < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cursor_ms,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
        batch = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
                response.raise_for_status()
                batch = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        if batch is None:
            raise RuntimeError(f"Binance request failed for {symbol}") from last_error
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        next_cursor = max(int(row[0]) for row in batch) + 1
        if next_cursor <= cursor_ms:
            raise RuntimeError("Binance pagination did not advance")
        cursor_ms = next_cursor

    if not rows:
        raise ValueError(f"Binance returned no klines for {symbol}")
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("downloaded OHLCV data failed validation")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index_label="open_time")
    return frame

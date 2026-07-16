"""Market-data collection and causal feature engineering."""

from __future__ import annotations

import time
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

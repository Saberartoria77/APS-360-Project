import numpy as np
import pandas as pd
import pytest

from src.data import engineer_features, fetch_klines, make_direction_labels


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=120, freq="h", tz="UTC")
    close = pd.Series(100 + np.linspace(0, 8, len(index)) + np.sin(np.arange(len(index)) / 4), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000 + np.arange(len(index), dtype=float),
        },
        index=index,
    )


def test_features_are_causal(sample_ohlcv: pd.DataFrame) -> None:
    original = engineer_features(sample_ohlcv)
    changed = sample_ohlcv.copy()
    changed.loc[changed.index[-1], "close"] *= 10
    revised = engineer_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_labels_use_exact_future_horizon() -> None:
    close = pd.Series([100.0, 101.0, 103.0, 100.0])
    labels = make_direction_labels(close, horizon=1, threshold=0.015)
    assert labels.iloc[:3].tolist() == [1.0, 2.0, 0.0]
    assert pd.isna(labels.iloc[3])


def test_features_include_expected_columns(sample_ohlcv: pd.DataFrame) -> None:
    features = engineer_features(sample_ohlcv)
    expected = {
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
    }
    assert expected.issubset(features.columns)


def test_fetch_klines_parses_and_sorts_api_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        [1735693200000, "101", "103", "100", "102", "12", 0, "0", 1, "0", "0", "0"],
        [1735689600000, "100", "102", "99", "101", "10", 0, "0", 1, "0", "0", "0"],
    ]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[list]:
            return rows

    class FakeSession:
        def get(self, *args, **kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("src.data.requests.Session", FakeSession)
    frame = fetch_klines(
        "BTCUSDT",
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01 02:00", tz="UTC"),
    )
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert frame["close"].tolist() == [101.0, 102.0]

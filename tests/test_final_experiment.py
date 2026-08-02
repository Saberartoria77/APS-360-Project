import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_final_experiment import _validate_hourly_frame, run_historical_stage
from src.final_evaluation import evaluate_slices, transfer_pairs
from src.persistence import load_frozen_package


def test_transfer_matrix_contains_all_low_high_combinations() -> None:
    assert set(transfer_pairs()) == {
        ("low", "low"),
        ("low", "high"),
        ("high", "high"),
        ("high", "low"),
    }


def test_evaluate_slices_includes_sample_and_class_counts() -> None:
    y_true = np.array([0, 1, 2, 2])
    predictions = {"model": np.array([0, 1, 1, 2])}
    regimes = np.array(["low", "low", "high", "high"])
    symbols = np.array(["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"])

    result = evaluate_slices(y_true, predictions, regimes, symbols)

    assert result["overall"]["sample_count"] == 4
    assert result["overall"]["class_counts"] == [1, 1, 2]
    assert result["regime_high"]["sample_count"] == 2
    assert result["regime_high"]["class_counts"] == [0, 0, 2]
    assert result["symbol_BTCUSDT"]["class_counts"] == [1, 0, 1]


def test_historical_frame_validation_rejects_prospective_rows() -> None:
    index = pd.date_range("2026-06-30T23:00:00Z", periods=2, freq="h")
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=index)

    with pytest.raises(ValueError, match="historical bounds"):
        _validate_hourly_frame("BTCUSDT", frame)


def test_historical_dry_run_writes_valid_results_and_frozen_package(
    tmp_path: Path, monkeypatch
) -> None:
    def forbid_network(*args, **kwargs):
        raise AssertionError("dry run attempted network access")

    monkeypatch.setattr("run_final_experiment.fetch_klines", forbid_network)
    result = run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))

    assert result["configuration"]["prospective_revealed"] is False
    assert result["configuration"]["seeds"] == [42]
    required = [
        "historical_results.json",
        "cross_regime_results.csv",
        "frozen/manifest.json",
        "frozen/cnn_lstm.pt",
        "frozen/baselines.npz",
    ]
    assert all((tmp_path / name).is_file() for name in required)
    assert not (tmp_path / "prospective_results.json").exists()

    manifest = json.loads((tmp_path / "frozen/manifest.json").read_text())
    assert manifest["prospective_start"] == "2026-07-01T00:00:00Z"
    assert manifest["feature_names"] and len(manifest["feature_names"]) == 13
    assert manifest["window"] == 96
    assert manifest["training_device"] == "cpu"
    load_frozen_package(tmp_path / "frozen")

    persisted = json.loads((tmp_path / "historical_results.json").read_text())
    assert persisted == result
    assert persisted["global_evaluation"]["seed_runs"][0]["validation_loss"] >= 0
    assert persisted["global_evaluation"]["slices"]["overall"]["sample_count"] > 0
    assert len(persisted["global_evaluation"]["slices"]["overall"]["class_counts"]) == 3

    with (tmp_path / "cross_regime_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {
        (row["train_regime"], row["test_regime"])
        for row in rows
    } == set(transfer_pairs())
    assert {row["model"] for row in rows} == {"logistic_regression", "cnn_lstm"}
    cnn_rows = [row for row in rows if row["model"] == "cnn_lstm"]
    assert all(row["seed_count"] == "1" for row in cnn_rows)
    assert all(row["accuracy_mean"] and row["accuracy_std"] for row in cnn_rows)
    assert all(row["macro_f1_mean"] and row["macro_f1_std"] for row in cnn_rows)

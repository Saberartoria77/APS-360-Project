import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from run_final_experiment import (
    HISTORICAL_END,
    HISTORICAL_START,
    PROSPECTIVE_CONTEXT_END,
    PROSPECTIVE_CONTEXT_START,
    PROSPECTIVE_ARTIFACT_INDEX,
    PROSPECTIVE_REVEAL_MARKER,
    _begin_prospective_reveal,
    _validate_prospective_frame,
    _validate_hourly_frame,
    main,
    run_historical_stage,
    run_prospective_stage,
)
from src.final_evaluation import (
    aggregate_seed_metrics,
    evaluate_slices,
    paired_transfer_changes,
    transfer_pairs,
)
from src.persistence import load_frozen_package, sha256_file


def test_prospective_stage_refuses_to_run_without_frozen_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete frozen package"):
        run_prospective_stage(tmp_path, dry_run=True)


def test_prospective_validates_frozen_mode_before_market_data(
    tmp_path: Path, monkeypatch
) -> None:
    run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))

    def forbid_data_access(*args, **kwargs):
        raise AssertionError("market data accessed before frozen-package rejection")

    monkeypatch.setattr("run_final_experiment.fetch_klines", forbid_data_access)
    with pytest.raises(ValueError, match="data mode"):
        run_prospective_stage(tmp_path, dry_run=False)


def test_prospective_dry_run_scores_only_july_and_writes_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    def forbid_network(*args, **kwargs):
        raise AssertionError("dry run attempted network access")

    monkeypatch.setattr("run_final_experiment.fetch_klines", forbid_network)
    run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))
    result = run_prospective_stage(tmp_path, dry_run=True)

    assert result["configuration"]["prospective_revealed"] is True
    assert result["configuration"]["data_mode"] == "synthetic"
    assert result["data"]["target_start"] == "2026-07-01T00:00:00Z"
    assert result["data"]["target_end"] == "2026-08-01T00:00:00Z"
    assert result["data"]["source_start"] == PROSPECTIVE_CONTEXT_START
    assert result["data"]["source_end_exclusive"] == PROSPECTIVE_CONTEXT_END
    assert result["data"]["sample_count"] == 31 * 24 * 2
    assert result["data"]["first_scored_timestamp"] == "2026-07-01T00:00:00Z"
    assert result["data"]["last_scored_timestamp"] == "2026-07-31T23:00:00Z"
    assert sum(result["data"]["class_counts"]) == result["data"]["sample_count"]
    required = [
        "prospective_results.json",
        "qualitative_examples.csv",
        "figures/regime_performance.png",
        "figures/prospective_confusion.png",
        "figures/model_regime_diagram.png",
    ]
    assert all((tmp_path / name).is_file() for name in required)
    assert all((tmp_path / name).stat().st_size > 0 for name in required)

    persisted = json.loads((tmp_path / "prospective_results.json").read_text())
    assert persisted == result
    assert persisted["selected_model"] == "cnn_lstm"
    assert set(persisted["slices"]["overall"]["models"]) == {
        "majority",
        "momentum",
        "logistic_regression",
        "cnn_lstm",
    }

    examples = pd.read_csv(tmp_path / "qualitative_examples.csv")
    assert set(examples["regime"]).issubset({"low", "medium", "high"})
    assert set(examples["example_type"]).issubset({"correct", "error"})
    assert examples.groupby(["regime", "example_type"]).size().max() == 1


def test_prospective_dry_run_is_deterministic_for_same_frozen_package(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical"
    run_historical_stage(historical, dry_run=True, epochs=1, seeds=(42,))
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    shutil.copytree(historical / "frozen", first_dir / "frozen")
    shutil.copytree(historical / "frozen", second_dir / "frozen")
    first = run_prospective_stage(first_dir, dry_run=True)
    first_examples = (first_dir / "qualitative_examples.csv").read_bytes()
    second = run_prospective_stage(second_dir, dry_run=True)

    assert first == second
    assert (second_dir / "qualitative_examples.csv").read_bytes() == first_examples


def test_second_dry_prospective_run_is_refused(tmp_path: Path) -> None:
    run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))
    run_prospective_stage(tmp_path, dry_run=True)

    with pytest.raises(ValueError, match="already contains prospective artifacts"):
        run_prospective_stage(tmp_path, dry_run=True)


def test_second_prospective_refusal_precedes_package_load_fetch_and_training(
    tmp_path: Path, monkeypatch
) -> None:
    run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))
    run_prospective_stage(tmp_path, dry_run=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("repeated-run guard executed too late")

    monkeypatch.setattr("run_final_experiment.load_frozen_package", forbidden)
    monkeypatch.setattr("run_final_experiment._prospective_raw_frames", forbidden)
    monkeypatch.setattr("run_final_experiment.train_model", forbidden)
    with pytest.raises(ValueError, match="already contains prospective artifacts"):
        run_prospective_stage(tmp_path, dry_run=True)


def test_prospective_artifact_index_binds_every_published_output(tmp_path: Path) -> None:
    run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))
    run_prospective_stage(tmp_path, dry_run=True)

    index = json.loads((tmp_path / PROSPECTIVE_ARTIFACT_INDEX).read_text())
    assert index["data_mode"] == "synthetic"
    assert set(index["artifacts"]) == {
        "prospective_results.json",
        "qualitative_examples.csv",
        "figures/regime_performance.png",
        "figures/prospective_confusion.png",
        "figures/model_regime_diagram.png",
    }
    for relative_path, binding in index["artifacts"].items():
        artifact = tmp_path / relative_path
        assert binding == {
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }


def test_genuine_reveal_marker_is_exclusive_and_durable(tmp_path: Path) -> None:
    marker = _begin_prospective_reveal(tmp_path, data_mode="genuine")

    assert marker == tmp_path / PROSPECTIVE_REVEAL_MARKER
    assert json.loads(marker.read_text())["data_mode"] == "genuine"
    with pytest.raises(ValueError, match="already been revealed"):
        _begin_prospective_reveal(tmp_path, data_mode="genuine")


def test_genuine_historical_refuses_revealed_output_before_fetch_or_training(
    tmp_path: Path, monkeypatch
) -> None:
    _begin_prospective_reveal(tmp_path, data_mode="genuine")

    def forbidden(*args, **kwargs):
        raise AssertionError("guard ran after data access or training")

    monkeypatch.setattr("run_final_experiment.fetch_klines", forbidden)
    monkeypatch.setattr("run_final_experiment.train_model", forbidden)
    with pytest.raises(ValueError, match="prospective reveal"):
        run_historical_stage(tmp_path, dry_run=False, epochs=1)


def test_genuine_historical_refuses_saved_prospective_artifacts_without_marker(
    tmp_path: Path, monkeypatch
) -> None:
    shutil.copy(
        Path("artifacts/final/prospective_results.json"),
        tmp_path / "prospective_results.json",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("guard ran after data access or training")

    monkeypatch.setattr("run_final_experiment.fetch_klines", forbidden)
    monkeypatch.setattr("run_final_experiment.train_model", forbidden)
    with pytest.raises(ValueError, match="prospective artifacts"):
        run_historical_stage(tmp_path, dry_run=False, epochs=1)


def test_dry_run_refuses_directory_containing_genuine_state_before_training(
    tmp_path: Path, monkeypatch
) -> None:
    shutil.copytree(Path("artifacts/final/frozen"), tmp_path / "frozen")

    def forbidden(*args, **kwargs):
        raise AssertionError("guard ran after data access or training")

    monkeypatch.setattr("run_final_experiment.train_model", forbidden)
    with pytest.raises(ValueError, match="cannot mix synthetic and genuine"):
        run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))


def test_dry_run_refuses_canonical_genuine_output_before_any_work(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("canonical dry-run guard ran too late")

    monkeypatch.setattr("run_final_experiment.train_model", forbidden)
    with pytest.raises(ValueError, match="canonical genuine output"):
        run_historical_stage(
            Path("artifacts/final"), dry_run=True, epochs=1, seeds=(42,)
        )


def test_prospective_stage_never_retrains(tmp_path: Path, monkeypatch) -> None:
    run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))

    def forbid_training(*args, **kwargs):
        raise AssertionError("prospective stage attempted model training")

    monkeypatch.setattr("run_final_experiment.train_model", forbid_training)
    run_prospective_stage(tmp_path, dry_run=True)


@pytest.mark.parametrize("truncate", ["start", "end"])
def test_prospective_frame_requires_exact_913_hour_coverage(truncate: str) -> None:
    index = pd.date_range(
        PROSPECTIVE_CONTEXT_START,
        PROSPECTIVE_CONTEXT_END,
        freq="h",
        inclusive="left",
    )
    valid = pd.DataFrame({"close": np.ones(len(index))}, index=index)
    assert len(valid) == 913
    _validate_prospective_frame("BTCUSDT", valid)

    truncated = valid.iloc[1:] if truncate == "start" else valid.iloc[:-1]
    with pytest.raises(ValueError, match="exact prospective context coverage"):
        _validate_prospective_frame("BTCUSDT", truncated)


@pytest.mark.parametrize("flag", ["--epochs", "--seed", "--tuning"])
def test_prospective_cli_rejects_training_and_tuning_flags(
    flag: str, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run_final_experiment.py", "prospective", flag, "1"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


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


def test_seed_aggregation_includes_per_class_recall_mean_and_population_std() -> None:
    metrics = [
        {"accuracy": 0.4, "macro_f1": 0.3, "per_class_recall": [0.2, 0.4, 0.6]},
        {"accuracy": 0.8, "macro_f1": 0.7, "per_class_recall": [0.4, 0.8, 1.0]},
    ]

    result = aggregate_seed_metrics(metrics)

    assert result["accuracy"] == pytest.approx({"mean": 0.6, "std": 0.2})
    assert result["macro_f1"] == pytest.approx({"mean": 0.5, "std": 0.2})
    assert result["per_class_recall"]["down"] == pytest.approx(
        {"mean": 0.3, "std": 0.1}
    )
    assert result["per_class_recall"]["flat"] == pytest.approx(
        {"mean": 0.6, "std": 0.2}
    )
    assert result["per_class_recall"]["up"] == pytest.approx(
        {"mean": 0.8, "std": 0.2}
    )


def test_transfer_change_is_paired_by_seed_before_mean_and_std() -> None:
    seed_metrics = {
        ("low", "low"): {
            42: {"macro_f1": 0.5, "checkpoint_sha256": "a" * 64},
            43: {"macro_f1": 0.7, "checkpoint_sha256": "b" * 64},
            44: {"macro_f1": 0.9, "checkpoint_sha256": "c" * 64},
        },
        ("low", "high"): {
            42: {"macro_f1": 0.4, "checkpoint_sha256": "a" * 64},
            43: {"macro_f1": 0.9, "checkpoint_sha256": "b" * 64},
            44: {"macro_f1": 0.8, "checkpoint_sha256": "c" * 64},
        },
        ("high", "high"): {
            42: {"macro_f1": 0.8, "checkpoint_sha256": "d" * 64},
            43: {"macro_f1": 0.6, "checkpoint_sha256": "e" * 64},
            44: {"macro_f1": 0.4, "checkpoint_sha256": "f" * 64},
        },
        ("high", "low"): {
            42: {"macro_f1": 0.7, "checkpoint_sha256": "d" * 64},
            43: {"macro_f1": 0.5, "checkpoint_sha256": "e" * 64},
            44: {"macro_f1": 0.6, "checkpoint_sha256": "f" * 64},
        },
    }

    changes = paired_transfer_changes(seed_metrics)
    low = next(row for row in changes if row["train_regime"] == "low")

    assert low["seed_count"] == 3
    assert low["per_seed"] == [
        {
            "seed": 42,
            "checkpoint_sha256": "a" * 64,
            "macro_f1_change": pytest.approx(-0.1),
        },
        {
            "seed": 43,
            "checkpoint_sha256": "b" * 64,
            "macro_f1_change": pytest.approx(0.2),
        },
        {
            "seed": 44,
            "checkpoint_sha256": "c" * 64,
            "macro_f1_change": pytest.approx(-0.1),
        },
    ]
    assert low["macro_f1_change_mean"] == pytest.approx(0.0, abs=1e-12)
    assert low["macro_f1_change_std"] == pytest.approx(np.sqrt(0.02))
    assert [entry["checkpoint_sha256"] for entry in low["per_seed"]] == [
        "a" * 64,
        "b" * 64,
        "c" * 64,
    ]


def test_transfer_change_rejects_mismatched_checkpoint_digests() -> None:
    metrics = {
        ("low", "low"): {42: {"macro_f1": 0.5, "checkpoint_sha256": "a" * 64}},
        ("low", "high"): {42: {"macro_f1": 0.4, "checkpoint_sha256": "b" * 64}},
        ("high", "high"): {42: {"macro_f1": 0.6, "checkpoint_sha256": "c" * 64}},
        ("high", "low"): {42: {"macro_f1": 0.5, "checkpoint_sha256": "c" * 64}},
    }

    with pytest.raises(ValueError, match="same checkpoint digest"):
        paired_transfer_changes(metrics)


def test_cross_regime_cnn_trains_once_per_training_regime_and_seed(
    tmp_path: Path, monkeypatch
) -> None:
    from run_final_experiment import train_model as real_train_model

    calls = []

    def counted_train_model(bundle, config):
        calls.append((len(bundle.y_train), config.seed))
        return real_train_model(bundle, config)

    monkeypatch.setattr("run_final_experiment.train_model", counted_train_model)
    result = run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))

    assert len(calls) == 3  # one global + low-only + high-only
    assert [seed for _, seed in calls] == [42, 42, 42]
    cnn_rows = {
        (row["train_regime"], row["test_regime"]): row
        for row in result["cross_regime_evaluation"]["rows"]
        if row["model"] == "cnn_lstm"
    }
    assert cnn_rows[("low", "low")]["checkpoint_digests"] == cnn_rows[("low", "high")][
        "checkpoint_digests"
    ]
    assert cnn_rows[("high", "high")]["checkpoint_digests"] == cnn_rows[("high", "low")][
        "checkpoint_digests"
    ]
    assert cnn_rows[("low", "low")]["checkpoint_digests"] != cnn_rows[("high", "high")][
        "checkpoint_digests"
    ]


@pytest.mark.parametrize("truncate", ["start", "end"])
def test_genuine_frame_validation_rejects_truncated_coverage(truncate: str) -> None:
    index = pd.date_range(
        pd.Timestamp(HISTORICAL_START, tz="UTC"),
        pd.Timestamp(HISTORICAL_END, tz="UTC"),
        freq="h",
        inclusive="left",
    )
    index = index[1:] if truncate == "start" else index[:-1]
    frame = pd.DataFrame({"close": np.ones(len(index))}, index=index)

    with pytest.raises(ValueError, match="exact genuine coverage"):
        _validate_hourly_frame("BTCUSDT", frame, data_mode="genuine")


def test_historical_dry_run_writes_valid_results_and_frozen_package(
    tmp_path: Path, monkeypatch
) -> None:
    def forbid_network(*args, **kwargs):
        raise AssertionError("dry run attempted network access")

    monkeypatch.setattr("run_final_experiment.fetch_klines", forbid_network)
    result = run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))

    assert result["configuration"]["prospective_revealed"] is False
    assert result["configuration"]["data_mode"] == "synthetic"
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
    assert manifest["data_mode"] == result["configuration"]["data_mode"]
    load_frozen_package(tmp_path / "frozen", expected_data_mode="synthetic")
    with pytest.raises(ValueError, match="data mode"):
        load_frozen_package(tmp_path / "frozen", expected_data_mode="genuine")

    persisted = json.loads((tmp_path / "historical_results.json").read_text())
    assert persisted == result
    assert persisted["global_evaluation"]["seed_runs"][0]["validation_loss"] >= 0
    assert persisted["global_evaluation"]["slices"]["overall"]["sample_count"] > 0
    assert len(persisted["global_evaluation"]["slices"]["overall"]["class_counts"]) == 3
    assert set(
        persisted["global_evaluation"]["slices"]["overall"]["models"][
            "cnn_lstm"
        ]["per_class_recall"]
    ) == {"down", "flat", "up"}
    paired = persisted["cross_regime_evaluation"][
        "paired_cnn_transfer_macro_f1_changes"
    ]
    assert len(paired) == 2
    assert all(row["seed_count"] == 1 for row in paired)
    assert all(row["macro_f1_change_std"] == 0.0 for row in paired)

    with (tmp_path / "cross_regime_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {
        (row["train_regime"], row["test_regime"])
        for row in rows
    } == set(transfer_pairs())
    assert {row["model"] for row in rows} == {"logistic_regression", "cnn_lstm"}
    row_keys = [
        (row["model"], row["train_regime"], row["test_regime"]) for row in rows
    ]
    assert len(row_keys) == len(set(row_keys)) == 8
    cnn_rows = [row for row in rows if row["model"] == "cnn_lstm"]
    assert all(row["seed_count"] == "1" for row in cnn_rows)
    assert all(row["accuracy_mean"] and row["accuracy_std"] for row in cnn_rows)
    assert all(row["macro_f1_mean"] and row["macro_f1_std"] for row in cnn_rows)
    numeric_columns = set(rows[0]) - {
        "model",
        "train_regime",
        "test_regime",
        "checkpoint_digests",
    }
    assert all(np.isfinite(float(row[column])) for row in rows for column in numeric_columns)
    json_rows = {
        (row["model"], row["train_regime"], row["test_regime"]): row
        for row in persisted["cross_regime_evaluation"]["rows"]
    }
    representative = rows[0]
    representative_json = json_rows[
        (
            representative["model"],
            representative["train_regime"],
            representative["test_regime"],
        )
    ]
    assert float(representative["macro_f1_mean"]) == pytest.approx(
        representative_json["macro_f1_mean"]
    )
    assert int(representative["sample_count"]) == representative_json["sample_count"]

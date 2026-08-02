"""Run the guarded historical stage of the final regime experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from run_experiment import SYMBOLS, _synthetic_ohlcv
from src.baselines import fit_baseline_models, predict_baselines
from src.data import (
    FEATURE_COLUMNS,
    DatasetBundle,
    engineer_features,
    fetch_klines,
    prepare_datasets,
    prepare_evaluation_windows,
)
from src.evaluation import (
    classification_metrics,
    representative_predictions_by_regime,
    save_confusion_matrix,
)
from src.final_evaluation import (
    aggregate_seed_metrics,
    evaluate_slices,
    paired_transfer_changes,
    save_model_regime_diagram,
    save_regime_performance_figure,
    transfer_pairs,
)
from src.persistence import (
    create_frozen_manifest,
    load_frozen_package,
    predict_frozen_probabilities,
    save_frozen_package,
    sha256_file,
    state_dict_sha256,
)
from src.regimes import (
    RegimeThresholds,
    assign_regimes,
    fit_regime_thresholds,
    realized_volatility,
    subset_bundle,
)
from src.training import TrainConfig, TrainResult, predict_probabilities, train_model


HISTORICAL_START = "2023-07-01"
HISTORICAL_END = "2026-07-01"
PROSPECTIVE_START = "2026-07-01T00:00:00Z"
PROSPECTIVE_END = "2026-08-01T00:00:00Z"
PROSPECTIVE_CONTEXT_START = "2026-06-24T00:00:00Z"
PROSPECTIVE_CONTEXT_END = "2026-08-01T01:00:00Z"
SEEDS = (42, 43, 44)
SYNTHETIC_HISTORICAL_START = "2024-01-01T00:00:00Z"
SYNTHETIC_HISTORICAL_END = "2024-02-11T16:00:00Z"
SYNTHETIC_HISTORICAL_ROWS = 1000
SYNTHETIC_PROSPECTIVE_ROWS = 1100
PROSPECTIVE_REVEAL_MARKER = ".prospective-reveal.json"
PROSPECTIVE_ARTIFACT_INDEX = "prospective_artifact_index.json"
PROSPECTIVE_ARTIFACTS = (
    "prospective_results.json",
    "qualitative_examples.csv",
    "figures/regime_performance.png",
    "figures/prospective_confusion.png",
    "figures/model_regime_diagram.png",
)
CANONICAL_GENUINE_OUTPUT = (Path(__file__).resolve().parent / "artifacts/final").resolve()


def _json_data_mode(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        decoded = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return "unknown"
    if not isinstance(decoded, dict):
        return "unknown"
    if path.name in {"manifest.json", PROSPECTIVE_ARTIFACT_INDEX, PROSPECTIVE_REVEAL_MARKER}:
        return decoded.get("data_mode", "unknown")
    configuration = decoded.get("configuration")
    if isinstance(configuration, dict):
        return configuration.get("data_mode", "unknown")
    return "unknown"


def _output_data_modes(output_dir: Path) -> set[str]:
    candidates = (
        output_dir / "frozen/manifest.json",
        output_dir / "historical_results.json",
        output_dir / "prospective_results.json",
        output_dir / PROSPECTIVE_ARTIFACT_INDEX,
        output_dir / PROSPECTIVE_REVEAL_MARKER,
    )
    return {mode for path in candidates if (mode := _json_data_mode(path)) is not None}


def _contains_prospective_state(output_dir: Path) -> bool:
    paths = [output_dir / PROSPECTIVE_REVEAL_MARKER, output_dir / PROSPECTIVE_ARTIFACT_INDEX]
    paths.extend(output_dir / relative for relative in PROSPECTIVE_ARTIFACTS)
    return any(path.exists() for path in paths)


def _guard_output_mode(output_dir: Path, *, data_mode: str, stage: str) -> None:
    """Refuse mixed-mode and repeated runs before data access or training."""
    output_dir = Path(output_dir)
    if data_mode == "synthetic" and output_dir.resolve() == CANONICAL_GENUINE_OUTPUT:
        raise ValueError("dry runs cannot use the canonical genuine output directory")
    modes = _output_data_modes(output_dir)
    if modes and modes != {data_mode}:
        raise ValueError("output data mode cannot mix synthetic and genuine state")
    marker = output_dir / PROSPECTIVE_REVEAL_MARKER
    if stage == "historical" and _contains_prospective_state(output_dir):
        if data_mode == "genuine" and marker.exists():
            raise ValueError("genuine historical execution is forbidden after prospective reveal")
        raise ValueError("historical execution is forbidden after prospective artifacts exist")
    if stage == "prospective" and _contains_prospective_state(output_dir):
        raise ValueError("output already contains prospective artifacts or a reveal marker")


def _begin_prospective_reveal(output_dir: Path, *, data_mode: str) -> Path:
    """Durably and exclusively mark genuine data access before it can begin."""
    if data_mode != "genuine":
        raise ValueError("only genuine prospective access uses a reveal marker")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / PROSPECTIVE_REVEAL_MARKER
    payload = json.dumps(
        {
            "schema_version": "prospective-reveal-v1",
            "data_mode": "genuine",
            "status": "revealed",
        },
        indent=2,
    ).encode()
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError("genuine prospective data have already been revealed") from error
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return marker


def _prospective_artifact_index(staging_dir: Path, *, data_mode: str) -> dict:
    bindings = {}
    for relative in PROSPECTIVE_ARTIFACTS:
        path = staging_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"prospective staging artifact is missing or empty: {relative}")
        bindings[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    result = json.loads((staging_dir / "prospective_results.json").read_text())
    if result.get("configuration", {}).get("data_mode") != data_mode:
        raise ValueError("staged prospective result has the wrong data mode")
    examples = pd.read_csv(staging_dir / "qualitative_examples.csv")
    if examples.empty:
        raise ValueError("staged qualitative examples must not be empty")
    return {
        "schema_version": "prospective-artifact-index-v1",
        "data_mode": data_mode,
        "artifacts": bindings,
    }


def _publish_prospective_artifacts(staging_dir: Path, output_dir: Path) -> None:
    for relative in (*PROSPECTIVE_ARTIFACTS, PROSPECTIVE_ARTIFACT_INDEX):
        source = staging_dir / relative
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


def _validate_hourly_frame(
    symbol: str, frame: pd.DataFrame, *, data_mode: str
) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{symbol} index must be timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{symbol} index must be unique and chronological")
    if not np.all(np.diff(frame.index.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError(f"{symbol} index must be hourly contiguous")
    if data_mode == "genuine":
        expected_start = pd.Timestamp(HISTORICAL_START, tz="UTC")
        expected_end = pd.Timestamp(HISTORICAL_END, tz="UTC")
    elif data_mode == "synthetic":
        expected_start = pd.Timestamp(SYNTHETIC_HISTORICAL_START)
        expected_end = pd.Timestamp(SYNTHETIC_HISTORICAL_END)
    else:
        raise ValueError("data_mode must be genuine or synthetic")
    expected_rows = int((expected_end - expected_start) / pd.Timedelta(hours=1))
    if (
        len(frame) != expected_rows
        or frame.index[0] != expected_start
        or frame.index[-1] != expected_end - pd.Timedelta(hours=1)
    ):
        raise ValueError(f"{symbol} must have exact {data_mode} coverage")


def _eligible_regime_mask(
    frames: dict[str, pd.DataFrame], times: np.ndarray, symbols: np.ndarray, lookback: int
) -> np.ndarray:
    volatility = {
        symbol: realized_volatility(frame, lookback) for symbol, frame in frames.items()
    }
    eligible = np.zeros(len(times), dtype=bool)
    for index, (time, symbol_value) in enumerate(zip(times, symbols)):
        symbol = str(symbol_value)
        timestamp = pd.Timestamp(time)
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        eligible[index] = symbol in volatility and pd.notna(
            volatility[symbol].get(timestamp, np.nan)
        )
    return eligible


def _bundle_regimes_with_warmup(
    bundle: DatasetBundle,
    frames: dict[str, pd.DataFrame],
    thresholds: RegimeThresholds,
) -> dict[str, np.ndarray]:
    """Assign causal regimes, explicitly retaining early warm-up samples as unavailable."""
    output = {}
    for split in ("train", "val", "test"):
        times = getattr(bundle, f"{split}_times")
        symbols = getattr(bundle, f"{split}_symbols")
        eligible = _eligible_regime_mask(frames, times, symbols, thresholds.lookback)
        labels = np.full(len(times), "unavailable", dtype="<U11")
        if eligible.any():
            labels[eligible] = assign_regimes(
                frames, times[eligible], symbols[eligible], thresholds
            )
        output[split] = labels
    return output


def _training_config(seed: int, epochs: int, dry_run: bool) -> TrainConfig:
    return TrainConfig(
        epochs=epochs,
        batch_size=128 if dry_run else 256,
        patience=max(2, min(3, epochs)),
        seed=seed,
        device="cpu" if dry_run else None,
    )


def _class_counts(labels: np.ndarray) -> list[int]:
    return np.bincount(labels, minlength=3).astype(int).tolist()


def _merge_global_slices(
    baseline_slices: dict, cnn_slices_by_seed: list[dict]
) -> dict:
    merged = copy.deepcopy(baseline_slices)
    for slice_name, values in merged.items():
        seed_metrics = [
            slices[slice_name]["models"]["cnn_lstm"]
            for slices in cnn_slices_by_seed
        ]
        values["models"]["cnn_lstm"] = aggregate_seed_metrics(seed_metrics)
    return merged


def _cross_regime_row(
    train_regime: str,
    test_regime: str,
    model: str,
    metrics: list[dict],
    labels: np.ndarray,
    checkpoint_digests: dict[int, str] | None = None,
) -> dict:
    counts = _class_counts(labels)
    recalls = np.asarray([result["per_class_recall"] for result in metrics], dtype=float)
    row = {
        "train_regime": train_regime,
        "test_regime": test_regime,
        "model": model,
        "seed_count": len(metrics) if model == "cnn_lstm" else 1,
        "sample_count": int(len(labels)),
        "class_count_down": counts[0],
        "class_count_flat": counts[1],
        "class_count_up": counts[2],
        "accuracy_mean": float(np.mean([result["accuracy"] for result in metrics])),
        "accuracy_std": float(np.std([result["accuracy"] for result in metrics], ddof=0)),
        "macro_f1_mean": float(np.mean([result["macro_f1"] for result in metrics])),
        "macro_f1_std": float(np.std([result["macro_f1"] for result in metrics], ddof=0)),
        "recall_down_mean": float(recalls[:, 0].mean()),
        "recall_down_std": float(recalls[:, 0].std(ddof=0)),
        "recall_flat_mean": float(recalls[:, 1].mean()),
        "recall_flat_std": float(recalls[:, 1].std(ddof=0)),
        "recall_up_mean": float(recalls[:, 2].mean()),
        "recall_up_std": float(recalls[:, 2].std(ddof=0)),
    }
    if model == "cnn_lstm":
        if not checkpoint_digests or len(checkpoint_digests) != len(metrics):
            raise ValueError("CNN transfer rows require one checkpoint digest per seed")
        row["checkpoint_digests"] = ";".join(
            f"{seed}:{checkpoint_digests[seed]}" for seed in sorted(checkpoint_digests)
        )
    else:
        row["checkpoint_digests"] = ""
    return row


def _deterministic_transfer_changes(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["train_regime"], row["test_regime"], row["model"]): row
        for row in rows
    }
    changes = []
    for train_regime, opposite in (("low", "high"), ("high", "low")):
        model = "logistic_regression"
        matching = indexed[(train_regime, train_regime, model)]["macro_f1_mean"]
        transferred = indexed[(train_regime, opposite, model)]["macro_f1_mean"]
        change = float(transferred - matching)
        changes.append(
            {
                "train_regime": train_regime,
                "model": model,
                "matching_test_regime": train_regime,
                "opposite_test_regime": opposite,
                "seed_count": 1,
                "macro_f1_change": change,
                "macro_f1_change_mean": change,
                "macro_f1_change_std": 0.0,
            }
        )
    return changes


def _evaluate_cross_regime(
    bundle: DatasetBundle,
    regimes: dict[str, np.ndarray],
    *,
    seeds: tuple[int, ...],
    epochs: int,
    dry_run: bool,
) -> tuple[list[dict], dict[tuple[str, str], dict[int, dict]]]:
    """Train once per train-regime/seed and reuse that checkpoint across test regimes."""
    cross_rows: list[dict] = []
    cnn_seed_metrics: dict[tuple[str, str], dict[int, dict]] = {}
    for train_regime, opposite_regime in (("low", "high"), ("high", "low")):
        common_masks = {
            "train": regimes["train"] == train_regime,
            "val": regimes["val"] == train_regime,
        }
        training_subset = subset_bundle(
            bundle,
            {**common_masks, "test": regimes["test"] == train_regime},
        )
        pair_baselines = fit_baseline_models(
            training_subset.x_train, training_subset.y_train
        )
        trained: dict[int, tuple[TrainConfig, TrainResult, str]] = {}
        for seed in seeds:
            config = _training_config(seed, epochs, dry_run)
            training_result = train_model(training_subset, config)
            trained[seed] = (
                config,
                training_result,
                state_dict_sha256(training_result.best_state),
            )

        for test_regime in (train_regime, opposite_regime):
            evaluation_subset = subset_bundle(
                bundle,
                {**common_masks, "test": regimes["test"] == test_regime},
            )
            baseline_predictions = predict_baselines(
                pair_baselines,
                evaluation_subset.x_test,
                evaluation_subset.feature_names,
                evaluation_subset.scaler_mean,
                evaluation_subset.scaler_scale,
                evaluation_subset.threshold,
            )
            logistic_metrics = classification_metrics(
                evaluation_subset.y_test,
                baseline_predictions["logistic_regression"],
            )
            cross_rows.append(
                _cross_regime_row(
                    train_regime,
                    test_regime,
                    "logistic_regression",
                    [logistic_metrics],
                    evaluation_subset.y_test,
                )
            )

            pair_seed_metrics: dict[int, dict] = {}
            checkpoint_digests: dict[int, str] = {}
            for seed, (config, training_result, checkpoint_digest) in trained.items():
                probabilities = predict_probabilities(
                    training_result.best_state,
                    evaluation_subset.x_test,
                    batch_size=config.batch_size,
                    device=training_result.device,
                )
                metrics = classification_metrics(
                    evaluation_subset.y_test,
                    probabilities.argmax(axis=1).astype(np.int64),
                )
                metrics["checkpoint_sha256"] = checkpoint_digest
                pair_seed_metrics[seed] = metrics
                checkpoint_digests[seed] = checkpoint_digest
            cnn_seed_metrics[(train_regime, test_regime)] = pair_seed_metrics
            cross_rows.append(
                _cross_regime_row(
                    train_regime,
                    test_regime,
                    "cnn_lstm",
                    list(pair_seed_metrics.values()),
                    evaluation_subset.y_test,
                    checkpoint_digests=checkpoint_digests,
                )
            )
    return cross_rows, cnn_seed_metrics


def run_historical_stage(
    output_dir: Path | str = Path("artifacts/final"),
    dry_run: bool = False,
    epochs: int = 12,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """Train/evaluate historical models and freeze the validation-selected package."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        raise ValueError("seeds must be unique values drawn from 42, 43, and 44")
    if not dry_run and seeds != SEEDS:
        raise ValueError("genuine historical evaluation requires seeds 42, 43, and 44")
    data_mode = "synthetic" if dry_run else "genuine"

    output_dir = Path(output_dir)
    _guard_output_mode(output_dir, data_mode=data_mode, stage="historical")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frames = {}
    for symbol in SYMBOLS:
        if dry_run:
            # One thousand rows keep the fixture fast while placing both low and
            # high volatility observations in every transfer split.
            raw_frames[symbol] = _synthetic_ohlcv(
                symbol, periods=SYNTHETIC_HISTORICAL_ROWS
            )
        else:
            raw_frames[symbol] = fetch_klines(
                symbol,
                pd.Timestamp(HISTORICAL_START, tz="UTC"),
                pd.Timestamp(HISTORICAL_END, tz="UTC"),
                cache_path=Path("data/raw")
                / f"{symbol}_1h_{HISTORICAL_START}_{HISTORICAL_END}.csv",
            )
        _validate_hourly_frame(symbol, raw_frames[symbol], data_mode=data_mode)

    feature_frames = {
        symbol: engineer_features(frame) for symbol, frame in raw_frames.items()
    }
    bundle = prepare_datasets(feature_frames, window=96)
    thresholds = fit_regime_thresholds(
        feature_frames, train_end=pd.Timestamp(bundle.train_times.max()), lookback=168
    )
    regimes = _bundle_regimes_with_warmup(bundle, feature_frames, thresholds)
    if any(not np.isin(regimes[split], ["low", "medium", "high"]).all() for split in ("val", "test")):
        raise ValueError("validation and test samples require complete regime history")

    global_baselines = fit_baseline_models(bundle.x_train, bundle.y_train)
    baseline_predictions = predict_baselines(
        global_baselines,
        bundle.x_test,
        bundle.feature_names,
        bundle.scaler_mean,
        bundle.scaler_scale,
        bundle.threshold,
    )
    baseline_slices = evaluate_slices(
        bundle.y_test, baseline_predictions, regimes["test"], bundle.test_symbols
    )

    global_runs: list[dict] = []
    global_training: list[tuple[TrainConfig, TrainResult]] = []
    cnn_slices_by_seed = []
    for seed in seeds:
        config = _training_config(seed, epochs, dry_run)
        training_result = train_model(bundle, config)
        slices = evaluate_slices(
            bundle.y_test,
            {"cnn_lstm": training_result.test_predictions},
            regimes["test"],
            bundle.test_symbols,
        )
        validation_loss = float(min(training_result.history["val_loss"]))
        global_runs.append(
            {
                "seed": seed,
                "validation_loss": validation_loss,
                "best_epoch": int(training_result.best_epoch),
                "epochs_completed": len(training_result.history["train_loss"]),
                "device": training_result.device,
                "historical_test_metrics": classification_metrics(
                    bundle.y_test, training_result.test_predictions
                ),
                "slices": slices,
            }
        )
        global_training.append((config, training_result))
        cnn_slices_by_seed.append(slices)

    selected_index = min(
        range(len(global_runs)), key=lambda index: global_runs[index]["validation_loss"]
    )
    selected_config, selected_result = global_training[selected_index]
    selected_validation_loss = global_runs[selected_index]["validation_loss"]

    cross_rows, cnn_seed_metrics = _evaluate_cross_regime(
        bundle,
        regimes,
        seeds=seeds,
        epochs=epochs,
        dry_run=dry_run,
    )

    slices = _merge_global_slices(baseline_slices, cnn_slices_by_seed)
    paired_cnn_changes = paired_transfer_changes(cnn_seed_metrics)
    expected_source_start = (
        SYNTHETIC_HISTORICAL_START
        if dry_run
        else f"{HISTORICAL_START}T00:00:00Z"
    )
    expected_source_end = (
        SYNTHETIC_HISTORICAL_END
        if dry_run
        else f"{HISTORICAL_END}T00:00:00Z"
    )
    results = {
        "configuration": {
            "symbols": list(SYMBOLS),
            "development_start": f"{HISTORICAL_START}T00:00:00Z",
            "development_end": f"{HISTORICAL_END}T00:00:00Z",
            "prospective_start": PROSPECTIVE_START,
            "prospective_end": PROSPECTIVE_END,
            "prospective_revealed": False,
            "data_mode": data_mode,
            "window_hours": 96,
            "feature_count": len(FEATURE_COLUMNS),
            "regime_lookback_hours": thresholds.lookback,
            "epochs_requested": int(epochs),
            "seeds": list(seeds),
            "dry_run": bool(dry_run),
        },
        "data": {
            "source_coverage": {
                "start": expected_source_start,
                "end_exclusive": expected_source_end,
                "expected_rows_per_symbol": int(
                    (
                        pd.Timestamp(expected_source_end)
                        - pd.Timestamp(expected_source_start)
                    )
                    / pd.Timedelta(hours=1)
                ),
            },
            "raw_rows": {symbol: int(len(frame)) for symbol, frame in raw_frames.items()},
            "train_samples": int(len(bundle.y_train)),
            "validation_samples": int(len(bundle.y_val)),
            "test_samples": int(len(bundle.y_test)),
            "train_class_counts": _class_counts(bundle.y_train),
            "validation_class_counts": _class_counts(bundle.y_val),
            "test_class_counts": _class_counts(bundle.y_test),
            "regime_counts": {
                split: {
                    name: int(np.sum(regimes[split] == name))
                    for name in ("low", "medium", "high", "unavailable")
                }
                for split in ("train", "val", "test")
            },
        },
        "regime_thresholds": {
            symbol: list(values) for symbol, values in thresholds.by_symbol.items()
        },
        "global_evaluation": {
            "selected_seed": int(selected_config.seed),
            "selection_rule": "minimum_validation_loss",
            "selected_validation_loss": selected_validation_loss,
            "seed_runs": global_runs,
            "slices": slices,
        },
        "cross_regime_evaluation": {
            "rows": cross_rows,
            "cnn_seed_metrics": [
                {
                    "train_regime": train_regime,
                    "test_regime": test_regime,
                    "seeds": [
                        {"seed": seed, "metrics": metrics}
                        for seed, metrics in sorted(
                            cnn_seed_metrics[(train_regime, test_regime)].items()
                        )
                    ],
                }
                for train_regime, test_regime in transfer_pairs()
            ],
            "paired_cnn_transfer_macro_f1_changes": paired_cnn_changes,
            "transfer_macro_f1_changes": [
                *_deterministic_transfer_changes(cross_rows),
                *paired_cnn_changes,
            ],
        },
    }
    (output_dir / "historical_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(cross_rows).to_csv(output_dir / "cross_regime_results.csv", index=False)

    manifest = create_frozen_manifest(
        feature_names=bundle.feature_names,
        window=bundle.x_train.shape[1],
        threshold=bundle.threshold,
        scaler_mean=bundle.scaler_mean.tolist(),
        scaler_scale=bundle.scaler_scale.tolist(),
        regime_thresholds=thresholds.by_symbol,
        config=selected_config,
        device=selected_result.device,
        validation_loss=selected_validation_loss,
        data_mode=data_mode,
        development_start=f"{HISTORICAL_START}T00:00:00Z",
        development_end=f"{HISTORICAL_END}T00:00:00Z",
        prospective_start=PROSPECTIVE_START,
        prospective_end=PROSPECTIVE_END,
    )
    save_frozen_package(
        output_dir / "frozen",
        manifest,
        selected_result.best_state,
        global_baselines,
    )
    return results


def recompute_historical_transfer_stage(
    output_dir: Path | str = Path("artifacts/final"),
    *,
    dry_run: bool = False,
    epochs: int = 12,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """Recompute only digest-bound transfer evidence without touching frozen or July state."""
    output_dir = Path(output_dir)
    historical_path = output_dir / "historical_results.json"
    if not historical_path.is_file():
        raise ValueError("historical results are required for transfer recomputation")
    try:
        historical = json.loads(historical_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("historical results are invalid") from error
    data_mode = "synthetic" if dry_run else "genuine"
    if historical.get("configuration", {}).get("data_mode") != data_mode:
        raise ValueError("historical result data mode does not match recomputation")
    seeds = tuple(int(seed) for seed in seeds)
    if epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("recomputation epochs and seeds are invalid")
    if not dry_run and (epochs != 12 or seeds != SEEDS):
        raise ValueError("genuine transfer recomputation requires the frozen 12-epoch seed protocol")

    raw_frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        if dry_run:
            frame = _synthetic_ohlcv(symbol, periods=SYNTHETIC_HISTORICAL_ROWS)
        else:
            cache_path = (
                Path("data/raw")
                / f"{symbol}_1h_{HISTORICAL_START}_{HISTORICAL_END}.csv"
            )
            if not cache_path.is_file():
                raise ValueError(
                    "historical-only recomputation requires the existing pre-July cache"
                )
            frame = fetch_klines(
                symbol,
                pd.Timestamp(HISTORICAL_START, tz="UTC"),
                pd.Timestamp(HISTORICAL_END, tz="UTC"),
                cache_path=cache_path,
            )
        _validate_hourly_frame(symbol, frame, data_mode=data_mode)
        raw_frames[symbol] = frame

    feature_frames = {
        symbol: engineer_features(frame) for symbol, frame in raw_frames.items()
    }
    bundle = prepare_datasets(feature_frames, window=96)
    thresholds = fit_regime_thresholds(
        feature_frames, train_end=pd.Timestamp(bundle.train_times.max()), lookback=168
    )
    saved_thresholds = historical.get("regime_thresholds")
    if not isinstance(saved_thresholds, dict) or any(
        not np.array_equal(
            np.asarray(saved_thresholds.get(symbol), dtype=float),
            np.asarray(thresholds.by_symbol[symbol], dtype=float),
        )
        for symbol in SYMBOLS
    ):
        raise ValueError("recomputed regime thresholds do not match frozen historical evidence")
    regimes = _bundle_regimes_with_warmup(bundle, feature_frames, thresholds)
    if any(
        not np.isin(regimes[split], ["low", "medium", "high"]).all()
        for split in ("val", "test")
    ):
        raise ValueError("validation and test samples require complete regime history")
    cross_rows, cnn_seed_metrics = _evaluate_cross_regime(
        bundle,
        regimes,
        seeds=seeds,
        epochs=epochs,
        dry_run=dry_run,
    )
    paired_cnn_changes = paired_transfer_changes(cnn_seed_metrics)
    historical["cross_regime_evaluation"] = {
        "rows": cross_rows,
        "cnn_seed_metrics": [
            {
                "train_regime": train_regime,
                "test_regime": test_regime,
                "seeds": [
                    {"seed": seed, "metrics": metrics}
                    for seed, metrics in sorted(
                        cnn_seed_metrics[(train_regime, test_regime)].items()
                    )
                ],
            }
            for train_regime, test_regime in transfer_pairs()
        ],
        "paired_cnn_transfer_macro_f1_changes": paired_cnn_changes,
        "transfer_macro_f1_changes": [
            *_deterministic_transfer_changes(cross_rows),
            *paired_cnn_changes,
        ],
        "recomputation_provenance": {
            "protocol": "same-checkpoint-paired-v2",
            "prospective_data_accessed": False,
            "frozen_global_model_changed": False,
        },
    }
    staging_dir = Path(tempfile.mkdtemp(prefix=".transfer-staging-", dir=output_dir))
    try:
        staged_json = staging_dir / "historical_results.json"
        staged_csv = staging_dir / "cross_regime_results.csv"
        staged_json.write_text(json.dumps(historical, indent=2))
        pd.DataFrame(cross_rows).to_csv(staged_csv, index=False)
        if not staged_json.stat().st_size or not staged_csv.stat().st_size:
            raise ValueError("historical transfer staging failed validation")
        os.replace(staged_json, historical_path)
        os.replace(staged_csv, output_dir / "cross_regime_results.csv")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return historical


def _prospective_raw_frames(dry_run: bool) -> dict[str, pd.DataFrame]:
    """Load the exact causal context, with deterministic synthetic data for tests."""
    context_start = pd.Timestamp(PROSPECTIVE_CONTEXT_START)
    context_end = pd.Timestamp(PROSPECTIVE_CONTEXT_END)
    frames = {}
    for symbol in SYMBOLS:
        if dry_run:
            generated = _synthetic_ohlcv(symbol, periods=SYNTHETIC_PROSPECTIVE_ROWS)
            generated.index = pd.date_range(
                context_end - pd.Timedelta(hours=SYNTHETIC_PROSPECTIVE_ROWS),
                periods=SYNTHETIC_PROSPECTIVE_ROWS,
                freq="h",
                tz="UTC",
            )
            frame = generated.loc[(generated.index >= context_start) & (generated.index < context_end)]
        else:
            frame = fetch_klines(
                symbol,
                context_start,
                context_end,
                cache_path=Path("data/raw")
                / f"{symbol}_1h_2026-06-24_2026-08-01T01.csv",
            )
        _validate_prospective_frame(symbol, frame)
        frames[symbol] = frame
    return frames


def _validate_prospective_frame(symbol: str, frame: pd.DataFrame) -> None:
    """Reject incomplete or future-extended prospective source coverage."""
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{symbol} prospective index must be timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{symbol} prospective index must be unique and chronological")
    if not np.all(np.diff(frame.index.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError(f"{symbol} prospective index must be hourly contiguous")
    start = pd.Timestamp(PROSPECTIVE_CONTEXT_START)
    end = pd.Timestamp(PROSPECTIVE_CONTEXT_END)
    expected_rows = int((end - start) / pd.Timedelta(hours=1))
    if (
        len(frame) != expected_rows
        or frame.index[0] != start
        or frame.index[-1] != end - pd.Timedelta(hours=1)
    ):
        raise ValueError(f"{symbol} must have exact prospective context coverage")


def run_prospective_stage(
    output_dir: Path | str = Path("artifacts/final"),
    dry_run: bool = False,
) -> dict:
    """Reveal and score July targets once using a complete frozen package."""
    output_dir = Path(output_dir)
    data_mode = "synthetic" if dry_run else "genuine"
    _guard_output_mode(output_dir, data_mode=data_mode, stage="prospective")

    # This integrity gate intentionally precedes all market-data construction or access.
    manifest, cnn_state, baselines = load_frozen_package(
        output_dir / "frozen", expected_data_mode=data_mode
    )
    if (
        manifest.schema_version != "frozen-package-v2"
        or manifest.data_mode != data_mode
        or manifest.feature_names != tuple(FEATURE_COLUMNS)
        or manifest.window != 96
        or manifest.prospective_start != PROSPECTIVE_START
        or manifest.prospective_end != PROSPECTIVE_END
    ):
        raise ValueError("frozen manifest does not match the prospective experiment")
    if not dry_run:
        _begin_prospective_reveal(output_dir, data_mode=data_mode)

    raw_frames = _prospective_raw_frames(dry_run)
    feature_frames = {
        symbol: engineer_features(frame) for symbol, frame in raw_frames.items()
    }
    dataset = prepare_evaluation_windows(
        feature_frames,
        feature_names=list(manifest.feature_names),
        scaler_mean=np.asarray(manifest.scaler_mean, dtype=np.float64),
        scaler_scale=np.asarray(manifest.scaler_scale, dtype=np.float64),
        threshold=manifest.threshold,
        window=manifest.window,
        target_start=pd.Timestamp(manifest.prospective_start),
        target_end=pd.Timestamp(manifest.prospective_end),
    )
    target_start = np.datetime64("2026-07-01T00:00:00")
    target_end = np.datetime64("2026-08-01T00:00:00")
    if not np.all((dataset.times >= target_start) & (dataset.times < target_end)):
        raise ValueError("prospective scoring produced targets outside July 2026")

    thresholds = RegimeThresholds(
        by_symbol={
            symbol: tuple(float(value) for value in values)
            for symbol, values in manifest.regime_thresholds.items()
        },
        lookback=manifest.regime_lookback,
    )
    regimes = assign_regimes(
        feature_frames, dataset.times, dataset.symbols, thresholds
    )
    baseline_predictions = predict_baselines(
        baselines,
        dataset.x,
        list(manifest.feature_names),
        np.asarray(manifest.scaler_mean, dtype=np.float64),
        np.asarray(manifest.scaler_scale, dtype=np.float64),
        manifest.threshold,
    )
    probabilities = predict_frozen_probabilities(
        manifest,
        cnn_state,
        list(manifest.feature_names),
        dataset.x,
        device="cpu",
    )
    cnn_predictions = probabilities.argmax(axis=1).astype(np.int64)
    predictions = {**baseline_predictions, "cnn_lstm": cnn_predictions}
    slices = evaluate_slices(dataset.y, predictions, regimes, dataset.symbols)

    result = {
        "configuration": {
            "prospective_revealed": True,
            "data_mode": data_mode,
            "schema_version": manifest.schema_version,
            "symbols": list(SYMBOLS),
            "window_hours": manifest.window,
            "feature_count": len(manifest.feature_names),
            "selected_seed": manifest.selected_seed,
            "selection_rule": "minimum_historical_validation_loss",
            "selected_validation_loss": manifest.validation_loss,
            "inference_device": "cpu",
            "frozen_state": {
                "cnn_sha256": manifest.cnn_sha256,
                "baselines_sha256": manifest.baselines_sha256,
            },
        },
        "data": {
            "source_start": PROSPECTIVE_CONTEXT_START,
            "source_end_exclusive": PROSPECTIVE_CONTEXT_END,
            "raw_rows": {symbol: int(len(frame)) for symbol, frame in raw_frames.items()},
            "target_start": PROSPECTIVE_START,
            "target_end": PROSPECTIVE_END,
            "sample_count": int(len(dataset.y)),
            "class_counts": _class_counts(dataset.y),
            "regime_counts": {
                regime: int(np.sum(regimes == regime))
                for regime in ("low", "medium", "high")
            },
            "symbol_counts": {
                symbol: int(np.sum(dataset.symbols == symbol)) for symbol in SYMBOLS
            },
            "first_scored_timestamp": pd.Timestamp(dataset.times.min(), tz="UTC").isoformat().replace("+00:00", "Z"),
            "last_scored_timestamp": pd.Timestamp(dataset.times.max(), tz="UTC").isoformat().replace("+00:00", "Z"),
        },
        "selected_model": "cnn_lstm",
        "slices": slices,
    }
    examples = representative_predictions_by_regime(
        dataset.y,
        cnn_predictions,
        probabilities,
        dataset.times,
        dataset.symbols,
        regimes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".prospective-staging-", dir=output_dir))
    try:
        (staging_dir / "prospective_results.json").write_text(json.dumps(result, indent=2))
        examples.to_csv(staging_dir / "qualitative_examples.csv", index=False)
        figures = staging_dir / "figures"
        save_regime_performance_figure(slices, figures / "regime_performance.png")
        save_confusion_matrix(
            dataset.y,
            cnn_predictions,
            figures / "prospective_confusion.png",
            "Frozen CNN-LSTM: July 2026",
        )
        save_model_regime_diagram(figures / "model_regime_diagram.png")
        index = _prospective_artifact_index(staging_dir, data_mode=data_mode)
        (staging_dir / PROSPECTIVE_ARTIFACT_INDEX).write_text(json.dumps(index, indent=2))
        _publish_prospective_artifacts(staging_dir, output_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    historical = subparsers.add_parser("historical")
    historical.add_argument("--output-dir", type=Path, default=Path("artifacts/final"))
    historical.add_argument("--dry-run", action="store_true")
    historical.add_argument("--epochs", type=int, default=12)
    prospective = subparsers.add_parser("prospective")
    prospective.add_argument("--output-dir", type=Path, default=Path("artifacts/final"))
    prospective.add_argument("--dry-run", action="store_true")
    recompute = subparsers.add_parser("recompute-transfer")
    recompute.add_argument("--output-dir", type=Path, default=Path("artifacts/final"))
    recompute.add_argument("--dry-run", action="store_true")
    recompute.add_argument("--epochs", type=int, default=12)
    arguments = parser.parse_args()
    if arguments.stage == "historical":
        result = run_historical_stage(
            output_dir=arguments.output_dir,
            dry_run=arguments.dry_run,
            epochs=arguments.epochs,
        )
    elif arguments.stage == "prospective":
        result = run_prospective_stage(
            output_dir=arguments.output_dir,
            dry_run=arguments.dry_run,
        )
    else:
        result = recompute_historical_transfer_stage(
            output_dir=arguments.output_dir,
            dry_run=arguments.dry_run,
            epochs=arguments.epochs,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

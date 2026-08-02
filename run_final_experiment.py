"""Run the guarded historical stage of the final regime experiment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiment import SYMBOLS, _synthetic_ohlcv
from src.baselines import fit_baseline_models, predict_baselines
from src.data import FEATURE_COLUMNS, DatasetBundle, engineer_features, fetch_klines, prepare_datasets
from src.evaluation import classification_metrics
from src.final_evaluation import (
    aggregate_seed_metrics,
    evaluate_slices,
    paired_transfer_changes,
    transfer_pairs,
)
from src.persistence import create_frozen_manifest, save_frozen_package
from src.regimes import (
    RegimeThresholds,
    assign_regimes,
    fit_regime_thresholds,
    realized_volatility,
    subset_bundle,
)
from src.training import TrainConfig, TrainResult, train_model


HISTORICAL_START = "2023-07-01"
HISTORICAL_END = "2026-07-01"
PROSPECTIVE_START = "2026-07-01T00:00:00Z"
PROSPECTIVE_END = "2026-08-01T00:00:00Z"
SEEDS = (42, 43, 44)
SYNTHETIC_HISTORICAL_START = "2024-01-01T00:00:00Z"
SYNTHETIC_HISTORICAL_END = "2024-02-11T16:00:00Z"
SYNTHETIC_HISTORICAL_ROWS = 1000


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
) -> dict:
    counts = _class_counts(labels)
    recalls = np.asarray([result["per_class_recall"] for result in metrics], dtype=float)
    return {
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

    cross_rows = []
    cnn_seed_metrics: dict[tuple[str, str], dict[int, dict]] = {}
    for train_regime, test_regime in transfer_pairs():
        masks = {
            "train": regimes["train"] == train_regime,
            "val": regimes["val"] == train_regime,
            "test": regimes["test"] == test_regime,
        }
        subset = subset_bundle(bundle, masks)
        pair_baselines = fit_baseline_models(subset.x_train, subset.y_train)
        pair_predictions = predict_baselines(
            pair_baselines,
            subset.x_test,
            subset.feature_names,
            subset.scaler_mean,
            subset.scaler_scale,
            subset.threshold,
        )
        logistic_metrics = classification_metrics(
            subset.y_test, pair_predictions["logistic_regression"]
        )
        cross_rows.append(
            _cross_regime_row(
                train_regime,
                test_regime,
                "logistic_regression",
                [logistic_metrics],
                subset.y_test,
            )
        )

        pair_seed_metrics = {}
        for seed in seeds:
            pair_result = train_model(subset, _training_config(seed, epochs, dry_run))
            pair_seed_metrics[seed] = classification_metrics(
                subset.y_test, pair_result.test_predictions
            )
        cnn_seed_metrics[(train_regime, test_regime)] = pair_seed_metrics
        cross_rows.append(
            _cross_regime_row(
                train_regime,
                test_regime,
                "cnn_lstm",
                list(pair_seed_metrics.values()),
                subset.y_test,
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    historical = subparsers.add_parser("historical")
    historical.add_argument("--output-dir", type=Path, default=Path("artifacts/final"))
    historical.add_argument("--dry-run", action="store_true")
    historical.add_argument("--epochs", type=int, default=12)
    arguments = parser.parse_args()
    result = run_historical_stage(
        output_dir=arguments.output_dir,
        dry_run=arguments.dry_run,
        epochs=arguments.epochs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Run the complete APS360 progress-report experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import fit_baselines
from src.data import FEATURE_COLUMNS, engineer_features, fetch_klines, prepare_datasets
from src.evaluation import (
    CLASS_NAMES,
    classification_metrics,
    representative_predictions,
    save_confusion_matrix,
)
from src.training import TrainConfig, train_model


SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def _synthetic_ohlcv(symbol: str, periods: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(13 if symbol == "BTCUSDT" else 29)
    index = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    returns = rng.normal(0.00005, 0.004, periods) + 0.0015 * np.sin(np.arange(periods) / 17)
    close = (40000 if symbol == "BTCUSDT" else 2200) * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.002, 0.0008, periods))
    high = np.maximum(open_price, close) * (1 + spread)
    low = np.minimum(open_price, close) * (1 - spread)
    volume = rng.lognormal(7.5 if symbol == "BTCUSDT" else 9.0, 0.35, periods)
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _plot_data_summary(bundle, frames: dict[str, pd.DataFrame], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for symbol, frame in frames.items():
        normalized = frame["close"] / frame["close"].iloc[0]
        axes[0].plot(frame.index, normalized, linewidth=1.0, label=symbol.replace("USDT", ""))
    axes[0].set(title="Normalized hourly close", ylabel="Close / first close", xlabel="UTC date")
    axes[0].legend(frameon=False)
    counts = np.vstack(
        [np.bincount(labels, minlength=3) for labels in [bundle.y_train, bundle.y_val, bundle.y_test]]
    )
    positions = np.arange(3)
    width = 0.25
    for index, (split, values) in enumerate(zip(["Train", "Validation", "Test"], counts)):
        axes[1].bar(positions + (index - 1) * width, values, width=width, label=split)
    axes[1].set(title="Window-level class distribution", xticks=positions, xticklabels=CLASS_NAMES, ylabel="Samples")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_learning_curves(history: dict[str, list[float]], destination: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_model_diagram(destination: Path) -> None:
    labels = [
        "Input\n96 x 13",
        "Conv1D\n32 filters, k=5",
        "BatchNorm\nReLU + dropout",
        "LSTM\n48 hidden units",
        "Linear\n3 logits",
        "Softmax\nDown / Flat / Up",
    ]
    figure, axis = plt.subplots(figsize=(10.0, 2.0))
    axis.axis("off")
    x_positions = np.linspace(0.08, 0.92, len(labels))
    colors = ["#E8F1FA", "#D5E8D4", "#D5E8D4", "#FFF2CC", "#F8CECC", "#E1D5E7"]
    for index, (x, label, color) in enumerate(zip(x_positions, labels, colors)):
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#555555"},
        )
        if index < len(labels) - 1:
            axis.annotate("", xy=(x_positions[index + 1] - 0.075, 0.5), xytext=(x + 0.075, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.3})
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=3)
    return {name.lower(): int(count) for name, count in zip(CLASS_NAMES, counts)}


def run_experiment(
    output_dir: Path | str = Path("artifacts"),
    dry_run: bool = False,
    epochs: int = 12,
    start: str = "2023-07-01",
    end: str = "2026-07-01",
) -> dict:
    """Collect/cache data, fit all models, and save every report artifact."""
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    metric_dir = output_dir / "metrics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = {}
    for symbol in SYMBOLS:
        if dry_run:
            raw_frames[symbol] = _synthetic_ohlcv(symbol)
        else:
            raw_frames[symbol] = fetch_klines(
                symbol,
                pd.Timestamp(start, tz="UTC"),
                pd.Timestamp(end, tz="UTC"),
                cache_path=Path("data/raw") / f"{symbol}_1h_{start}_{end}.csv",
            )
    feature_frames = {symbol: engineer_features(frame) for symbol, frame in raw_frames.items()}
    bundle = prepare_datasets(feature_frames, window=96)

    baseline_predictions = fit_baselines(bundle)
    training_result = train_model(
        bundle,
        TrainConfig(
            epochs=epochs,
            batch_size=128 if dry_run else 256,
            patience=max(2, min(3, epochs)),
            seed=42,
            device="cpu" if dry_run else None,
        ),
    )
    all_predictions = {**baseline_predictions, "cnn_lstm": training_result.test_predictions}
    model_metrics = {
        name: classification_metrics(bundle.y_test, predictions)
        for name, predictions in all_predictions.items()
    }

    for name, predictions in all_predictions.items():
        display_name = name.replace("_", " ").title()
        save_confusion_matrix(
            bundle.y_test,
            predictions,
            figure_dir / f"confusion_{name}.png",
            f"{display_name} test confusion matrix",
        )
    _plot_data_summary(bundle, raw_frames, figure_dir / "data_summary.png")
    _plot_learning_curves(training_result.history, figure_dir / "learning_curves.png")
    _plot_model_diagram(figure_dir / "model_diagram.png")

    examples = representative_predictions(
        bundle.y_test,
        training_result.test_predictions,
        training_result.test_probabilities,
        bundle.test_times,
        bundle.test_symbols,
    )
    examples.to_csv(metric_dir / "prediction_examples.csv", index=False)
    results = {
        "configuration": {
            "symbols": SYMBOLS,
            "start": str(min(frame.index.min() for frame in raw_frames.values())),
            "end": str(max(frame.index.max() for frame in raw_frames.values())),
            "window_hours": 96,
            "forecast_horizon_hours": 1,
            "feature_count": len(FEATURE_COLUMNS),
            "epochs_requested": int(epochs),
            "dry_run": bool(dry_run),
        },
        "data": {
            "raw_rows": {symbol: int(len(frame)) for symbol, frame in raw_frames.items()},
            "clean_feature_rows": {
                symbol: int(frame[FEATURE_COLUMNS].dropna().shape[0])
                for symbol, frame in feature_frames.items()
            },
            "train_samples": int(len(bundle.y_train)),
            "validation_samples": int(len(bundle.y_val)),
            "test_samples": int(len(bundle.y_test)),
            "train_class_counts": _class_counts(bundle.y_train),
            "validation_class_counts": _class_counts(bundle.y_val),
            "test_class_counts": _class_counts(bundle.y_test),
            "label_threshold": float(bundle.threshold),
        },
        "models": model_metrics,
        "model_parameter_count": int(training_result.parameter_count),
        "training": {
            "best_epoch": int(training_result.best_epoch),
            "epochs_completed": len(training_result.history["train_loss"]),
            "device": training_result.device,
            "history": training_result.history,
        },
    }
    (metric_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--end", default="2026-07-01")
    arguments = parser.parse_args()
    results = run_experiment(
        output_dir=arguments.output_dir,
        dry_run=arguments.dry_run,
        epochs=arguments.epochs,
        start=arguments.start,
        end=arguments.end,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

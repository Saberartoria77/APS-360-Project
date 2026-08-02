"""Reusable summaries for historical and cross-regime evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import classification_metrics


def transfer_pairs() -> list[tuple[str, str]]:
    """Declare the complete low/high train-to-test transfer matrix."""
    return [
        ("low", "low"),
        ("low", "high"),
        ("high", "high"),
        ("high", "low"),
    ]


def paired_transfer_changes(
    seed_metrics: dict[tuple[str, str], dict[int, dict]],
) -> list[dict]:
    """Compute opposite-minus-matching macro-F1 per seed before aggregation."""
    output = []
    for train_regime, opposite_regime in (("low", "high"), ("high", "low")):
        matching = seed_metrics[(train_regime, train_regime)]
        opposite = seed_metrics[(train_regime, opposite_regime)]
        if not matching or set(matching) != set(opposite):
            raise ValueError("matching and opposite transfer results require identical seeds")
        for seed in matching:
            matching_digest = matching[seed].get("checkpoint_sha256")
            opposite_digest = opposite[seed].get("checkpoint_sha256")
            if (
                not isinstance(matching_digest, str)
                or len(matching_digest) != 64
                or matching_digest != opposite_digest
            ):
                raise ValueError(
                    "paired transfer results must use the same checkpoint digest"
                )
        per_seed = [
            {
                "seed": int(seed),
                "checkpoint_sha256": matching[seed]["checkpoint_sha256"],
                "macro_f1_change": float(
                    opposite[seed]["macro_f1"] - matching[seed]["macro_f1"]
                ),
            }
            for seed in sorted(matching)
        ]
        values = np.asarray(
            [entry["macro_f1_change"] for entry in per_seed], dtype=float
        )
        output.append(
            {
                "train_regime": train_regime,
                "model": "cnn_lstm",
                "matching_test_regime": train_regime,
                "opposite_test_regime": opposite_regime,
                "seed_count": len(per_seed),
                "macro_f1_change_mean": float(values.mean()),
                "macro_f1_change_std": float(values.std(ddof=0)),
                "per_seed": per_seed,
            }
        )
    return output


def evaluate_slices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    regimes: np.ndarray,
    symbols: np.ndarray,
) -> dict:
    """Evaluate aligned predictions overall and by regime and symbol."""
    y_true = np.asarray(y_true)
    regimes = np.asarray(regimes)
    symbols = np.asarray(symbols)
    if y_true.ndim != 1 or regimes.shape != y_true.shape or symbols.shape != y_true.shape:
        raise ValueError("labels, regimes, and symbols must be aligned one-dimensional arrays")
    if not len(y_true):
        raise ValueError("at least one labelled sample is required")
    aligned_predictions = {name: np.asarray(values) for name, values in predictions.items()}
    if not aligned_predictions or any(
        values.shape != y_true.shape for values in aligned_predictions.values()
    ):
        raise ValueError("predictions must contain aligned one-dimensional arrays")

    slices = {"overall": np.ones(len(y_true), dtype=bool)}
    slices.update(
        {f"regime_{name}": regimes == name for name in ("low", "medium", "high")}
    )
    slices.update(
        {f"symbol_{name}": symbols == name for name in sorted(set(symbols.astype(str)))}
    )
    output = {}
    for slice_name, mask in slices.items():
        if not mask.any():
            continue
        output[slice_name] = {
            "sample_count": int(mask.sum()),
            "class_counts": np.bincount(
                y_true[mask].astype(np.int64), minlength=3
            ).astype(int).tolist(),
            "models": {
                name: classification_metrics(y_true[mask], values[mask])
                for name, values in aligned_predictions.items()
            },
        }
    return output


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    """Summarize scalar metrics and class recall across deterministic seed runs."""
    if not seed_results:
        raise ValueError("at least one seed result is required")
    keys = ("accuracy", "macro_f1")
    output = {
        key: {
            "mean": float(np.mean([result[key] for result in seed_results])),
            "std": float(np.std([result[key] for result in seed_results], ddof=0)),
        }
        for key in keys
    }
    recalls = np.asarray(
        [result["per_class_recall"] for result in seed_results], dtype=float
    )
    if recalls.shape != (len(seed_results), 3):
        raise ValueError("each seed result must contain three per-class recalls")
    output["per_class_recall"] = {
        name: {
            "mean": float(recalls[:, index].mean()),
            "std": float(recalls[:, index].std(ddof=0)),
        }
        for index, name in enumerate(("down", "flat", "up"))
    }
    return output


def save_regime_performance_figure(slices: dict, destination: Path) -> Path:
    """Plot prospective macro-F1 by regime for every frozen model."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    regime_names = [
        regime for regime in ("low", "medium", "high") if f"regime_{regime}" in slices
    ]
    if not regime_names:
        raise ValueError("at least one regime slice is required for the figure")
    models = list(slices[f"regime_{regime_names[0]}"]["models"])
    positions = np.arange(len(regime_names), dtype=float)
    width = 0.8 / len(models)
    figure, axis = plt.subplots(figsize=(6.5, 3.4))
    for model_index, model in enumerate(models):
        values = [
            slices[f"regime_{regime}"]["models"][model]["macro_f1"]
            for regime in regime_names
        ]
        offset = (model_index - (len(models) - 1) / 2) * width
        axis.bar(positions + offset, values, width=width, label=model.replace("_", " ").title())
    axis.set(
        title="Prospective performance across volatility regimes",
        xlabel="Frozen volatility regime",
        ylabel="Macro-F1",
        xticks=positions,
        xticklabels=[name.title() for name in regime_names],
        ylim=(0.0, 1.0),
    )
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_model_regime_diagram(destination: Path) -> Path:
    """Render the frozen preprocessing, CNN-LSTM, and regime-analysis flow."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    boxes = [
        (
            0.075,
            0.50,
            "Hourly BTC / ETH\ncausal inputs + unscaled log returns",
            "#E8F1FA",
        ),
        (0.255, 0.73, "Frozen scaling\n96 x 13 window", "#D5E8D4"),
        (0.445, 0.73, "Conv1D + LSTM\n3-class logits", "#FFF2CC"),
        (0.625, 0.73, "Down / Flat / Up\nnext-hour forecast", "#F8CECC"),
        (
            0.325,
            0.25,
            "168 h realized volatility\n+ frozen train-only thresholds",
            "#E1D5E7",
        ),
        (0.625, 0.25, "Frozen volatility regime\nLow / Medium / High", "#E1D5E7"),
        (0.91, 0.50, "Slice metrics +\nqualitative examples", "#DAE8FC"),
    ]
    figure, axis = plt.subplots(figsize=(12.0, 2.4))
    axis.axis("off")
    for x, y, label, color in boxes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#555555"},
        )
    arrows = [
        ((0.14, 0.55), (0.20, 0.68)),
        ((0.14, 0.45), (0.235, 0.29)),
        ((0.32, 0.73), (0.38, 0.73)),
        ((0.51, 0.73), (0.56, 0.73)),
        ((0.70, 0.69), (0.84, 0.56)),
        ((0.43, 0.25), (0.52, 0.25)),
        ((0.72, 0.29), (0.84, 0.44)),
    ]
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"},
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination

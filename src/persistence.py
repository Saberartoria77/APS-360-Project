"""Serialization for a frozen CNN-LSTM model package and its metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle

import torch

from src.baselines import BaselineModels


@dataclass(frozen=True)
class FrozenManifest:
    """All preprocessing, regime, and date metadata needed for reuse."""

    feature_names: list[str]
    window: int
    threshold: float
    scaler_mean: list[float]
    scaler_scale: list[float]
    regime_thresholds: dict[str, list[float]]
    regime_lookback: int
    selected_seed: int
    validation_loss: float
    development_start: str
    development_end: str
    prospective_start: str
    prospective_end: str


def save_frozen_package(
    destination: Path,
    manifest: FrozenManifest,
    cnn_state: dict[str, torch.Tensor],
    baselines: BaselineModels,
) -> None:
    """Write every reusable model artifact to a single package directory."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    torch.save(cnn_state, destination / "cnn_lstm.pt")
    with (destination / "baselines.pkl").open("wb") as handle:
        pickle.dump(baselines, handle)


def load_frozen_package(
    source: Path,
) -> tuple[FrozenManifest, dict[str, torch.Tensor], BaselineModels]:
    """Load a complete package, rejecting incompatible or incomplete artifacts."""
    source = Path(source)
    required = [
        source / "manifest.json",
        source / "cnn_lstm.pt",
        source / "baselines.pkl",
    ]
    if not all(path.is_file() for path in required):
        raise ValueError("incomplete frozen package")
    manifest = FrozenManifest(**json.loads(required[0].read_text()))
    if (
        manifest.window < 2
        or len(manifest.feature_names) != len(manifest.scaler_mean)
        or len(manifest.feature_names) != len(manifest.scaler_scale)
    ):
        raise ValueError("incompatible frozen manifest")
    state = torch.load(required[1], map_location="cpu", weights_only=True)
    with required[2].open("rb") as handle:
        baselines = pickle.load(handle)
    if not isinstance(baselines, BaselineModels):
        raise ValueError("invalid frozen baseline model")
    return manifest, state, baselines

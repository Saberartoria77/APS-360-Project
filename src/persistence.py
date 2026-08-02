"""Strict serialization and safe inference for frozen experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from src.baselines import BaselineModels
from src.data import FEATURE_COLUMNS
from src.models import CNNLSTM
from src.training import predict_probabilities


_MODEL_DEFAULTS = {
    "model_n_features": 13,
    "model_conv_channels": 32,
    "model_hidden_size": 48,
    "model_num_classes": 3,
}
_TRAINING_DEFAULTS = {
    "training_epochs": 12,
    "training_batch_size": 256,
    "training_learning_rate": 0.001,
    "training_patience": 3,
    "training_device": None,
    "training_version": "cnn-lstm-v1",
}
_EXACT_DATES = {
    "development_start": "2023-07-01T00:00:00Z",
    "development_end": "2026-07-01T00:00:00Z",
    "prospective_start": "2026-07-01T00:00:00Z",
    "prospective_end": "2026-08-01T00:00:00Z",
}
_EXPECTED_SYMBOLS = {"BTCUSDT", "ETHUSDT"}
_BASELINE_FIELDS = {
    "majority_class",
    "logistic_classes",
    "logistic_coef",
    "logistic_intercept",
    "logistic_n_features_in",
}


@dataclass(frozen=True)
class FrozenManifest:
    """The complete, fixed schema needed to reuse this experiment exactly."""

    feature_names: tuple[str, ...]
    window: int
    threshold: float
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    regime_thresholds: dict[str, tuple[float, ...]]
    regime_lookback: int
    selected_seed: int
    validation_loss: float
    development_start: str
    development_end: str
    prospective_start: str
    prospective_end: str
    model_n_features: int = 13
    model_conv_channels: int = 32
    model_hidden_size: int = 48
    model_num_classes: int = 3
    training_epochs: int = 12
    training_batch_size: int = 256
    training_learning_rate: float = 0.001
    training_patience: int = 3
    training_device: str | None = None
    training_version: str = "cnn-lstm-v1"

    def __post_init__(self) -> None:
        """Defensively copy mutable inputs supplied by a caller."""
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "scaler_mean", tuple(self.scaler_mean))
        object.__setattr__(self, "scaler_scale", tuple(self.scaler_scale))
        object.__setattr__(
            self,
            "regime_thresholds",
            {
                symbol: tuple(thresholds)
                for symbol, thresholds in self.regime_thresholds.items()
            },
        )


_MANIFEST_FIELDS = {field.name for field in fields(FrozenManifest)}


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _validate_manifest(manifest: FrozenManifest) -> None:
    """Reject any metadata that deviates from the frozen experiment contract."""
    if not isinstance(manifest, FrozenManifest):
        raise ValueError("incompatible frozen manifest")
    if manifest.feature_names != tuple(FEATURE_COLUMNS) or manifest.window != 96:
        raise ValueError("incompatible frozen manifest")

    means = np.asarray(manifest.scaler_mean, dtype=np.float64)
    scales = np.asarray(manifest.scaler_scale, dtype=np.float64)
    if (
        means.shape != (len(FEATURE_COLUMNS),)
        or scales.shape != (len(FEATURE_COLUMNS),)
        or not np.all(np.isfinite(means))
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not _is_finite_number(manifest.threshold)
        or manifest.threshold < 0.0
    ):
        raise ValueError("incompatible frozen manifest")

    if manifest.regime_lookback != 168 or set(manifest.regime_thresholds) != _EXPECTED_SYMBOLS:
        raise ValueError("incompatible frozen manifest")
    for thresholds in manifest.regime_thresholds.values():
        values = np.asarray(thresholds, dtype=np.float64)
        if (
            values.shape != (2,)
            or not np.all(np.isfinite(values))
            or values[0] > values[1]
        ):
            raise ValueError("incompatible frozen manifest")

    if manifest.selected_seed not in {42, 43, 44} or not _is_finite_number(
        manifest.validation_loss
    ):
        raise ValueError("incompatible frozen manifest")
    if any(getattr(manifest, name) != value for name, value in _EXACT_DATES.items()):
        raise ValueError("incompatible frozen manifest")
    if any(getattr(manifest, name) != value for name, value in _MODEL_DEFAULTS.items()):
        raise ValueError("incompatible frozen manifest")
    if any(getattr(manifest, name) != value for name, value in _TRAINING_DEFAULTS.items()):
        raise ValueError("incompatible frozen manifest")


def _baseline_arrays(
    baselines: BaselineModels, manifest: FrozenManifest
) -> dict[str, np.ndarray]:
    """Extract a fitted logistic baseline into finite numeric-only arrays."""
    if not isinstance(baselines, BaselineModels):
        raise ValueError("incompatible frozen baselines")
    model = baselines.logistic_regression
    if not isinstance(model, LogisticRegression):
        raise ValueError("incompatible frozen baselines")
    try:
        majority_class = int(baselines.majority_class)
        classes = np.asarray(model.classes_, dtype=np.int64)
        coef = np.asarray(model.coef_, dtype=np.float64)
        intercept = np.asarray(model.intercept_, dtype=np.float64)
        n_features_in = int(model.n_features_in_)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("incompatible frozen baselines") from error
    expected_features = len(manifest.feature_names)
    if (
        majority_class not in {0, 1, 2}
        or not np.array_equal(classes, np.array([0, 1, 2], dtype=np.int64))
        or coef.shape != (3, expected_features)
        or intercept.shape != (3,)
        or n_features_in != expected_features
        or not np.all(np.isfinite(coef))
        or not np.all(np.isfinite(intercept))
    ):
        raise ValueError("incompatible frozen baselines")
    return {
        "majority_class": np.array([majority_class], dtype=np.int64),
        "logistic_classes": classes,
        "logistic_coef": coef,
        "logistic_intercept": intercept,
        "logistic_n_features_in": np.array([n_features_in], dtype=np.int64),
    }


def _restore_baselines(
    baseline_path: Path, manifest: FrozenManifest
) -> BaselineModels:
    """Rebuild the fitted estimator required for sklearn's public predict API."""
    try:
        with np.load(baseline_path, allow_pickle=False) as stored:
            if set(stored.files) != _BASELINE_FIELDS:
                raise ValueError("incompatible frozen baselines")
            arrays = {name: stored[name] for name in stored.files}
        majority = arrays["majority_class"]
        classes = arrays["logistic_classes"]
        coef = arrays["logistic_coef"]
        intercept = arrays["logistic_intercept"]
        n_features = arrays["logistic_n_features_in"]
        if (
            majority.shape != (1,)
            or n_features.shape != (1,)
            or not np.issubdtype(majority.dtype, np.integer)
            or not np.issubdtype(classes.dtype, np.integer)
            or not np.issubdtype(n_features.dtype, np.integer)
        ):
            raise ValueError("incompatible frozen baselines")
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs"
        )
        model.classes_ = np.asarray(classes, dtype=np.int64)
        model.coef_ = np.asarray(coef, dtype=np.float64)
        model.intercept_ = np.asarray(intercept, dtype=np.float64)
        model.n_features_in_ = int(n_features[0])
        model.n_iter_ = np.ones(1, dtype=np.int32)
        return _validated_restored_baselines(
            BaselineModels(majority_class=int(majority[0]), logistic_regression=model),
            manifest,
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError("incompatible frozen baselines") from error


def _validated_restored_baselines(
    baselines: BaselineModels, manifest: FrozenManifest
) -> BaselineModels:
    _baseline_arrays(baselines, manifest)
    return baselines


def _load_manifest(manifest_path: Path) -> FrozenManifest:
    try:
        decoded: Any = json.loads(manifest_path.read_text())
        if not isinstance(decoded, dict) or set(decoded) != _MANIFEST_FIELDS:
            raise ValueError("incompatible frozen manifest")
        manifest = FrozenManifest(**decoded)
        _validate_manifest(manifest)
        return manifest
    except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("incompatible frozen manifest") from error


def _load_validated_state(
    state_path: Path, manifest: FrozenManifest
) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary")
        model = CNNLSTM(
            n_features=manifest.model_n_features,
            conv_channels=manifest.model_conv_channels,
            hidden_size=manifest.model_hidden_size,
            num_classes=manifest.model_num_classes,
        )
        model.load_state_dict(state, strict=True)
        return state
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("incompatible frozen CNN-LSTM state") from error


def save_frozen_package(
    destination: Path,
    manifest: FrozenManifest,
    cnn_state: dict[str, torch.Tensor],
    baselines: BaselineModels,
) -> None:
    """Write the complete frozen package with no executable serialized objects."""
    _validate_manifest(manifest)
    _load_validated_state_from_memory(cnn_state, manifest)
    baseline_arrays = _baseline_arrays(baselines, manifest)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    torch.save(cnn_state, destination / "cnn_lstm.pt")
    np.savez(destination / "baselines.npz", **baseline_arrays)


def _load_validated_state_from_memory(
    state: dict[str, torch.Tensor], manifest: FrozenManifest
) -> None:
    try:
        model = CNNLSTM(
            n_features=manifest.model_n_features,
            conv_channels=manifest.model_conv_channels,
            hidden_size=manifest.model_hidden_size,
            num_classes=manifest.model_num_classes,
        )
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("incompatible frozen CNN-LSTM state") from error


def load_frozen_package(
    source: Path,
) -> tuple[FrozenManifest, dict[str, torch.Tensor], BaselineModels]:
    """Load a complete, compatible frozen package without unpickling input."""
    source = Path(source)
    required = [
        source / "manifest.json",
        source / "cnn_lstm.pt",
        source / "baselines.npz",
    ]
    if not all(path.is_file() for path in required):
        raise ValueError("incomplete frozen package")
    manifest = _load_manifest(required[0])
    state = _load_validated_state(required[1], manifest)
    baselines = _restore_baselines(required[2], manifest)
    return manifest, state, baselines


def predict_frozen_probabilities(
    manifest: FrozenManifest,
    state: dict[str, torch.Tensor],
    feature_names: list[str] | tuple[str, ...],
    features: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Safely predict only when caller data matches the frozen experiment schema."""
    _validate_manifest(manifest)
    if tuple(feature_names) != manifest.feature_names:
        raise ValueError("feature order does not match the frozen manifest")
    array = np.asarray(features)
    expected_shape = (manifest.window, len(manifest.feature_names))
    if array.ndim != 3 or len(array) == 0 or array.shape[1:] != expected_shape:
        raise ValueError("features must be a non-empty [samples, 96, 13] array")
    _load_validated_state_from_memory(state, manifest)
    return predict_probabilities(state, array, batch_size=batch_size, device=device)

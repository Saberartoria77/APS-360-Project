"""Strict serialization and safe inference for frozen experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import torch

from src.baselines import BaselineModels
from src.data import FEATURE_COLUMNS
from src.models import CNNLSTM
from src.training import TrainConfig, predict_probabilities


_MODEL_DEFAULTS = {
    "model_n_features": 13,
    "model_conv_channels": 32,
    "model_hidden_size": 48,
    "model_num_classes": 3,
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
_REQUIRED_LIBRARY_VERSION_KEYS = {
    "python",
    "numpy",
    "pandas",
    "scikit-learn",
    "pytorch",
}


def current_library_versions() -> dict[str, str]:
    """Capture runtime provenance for later reproducibility diagnosis."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "pytorch": torch.__version__,
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
    data_mode: str
    cnn_sha256: str | None = None
    baselines_sha256: str | None = None
    model_n_features: int = 13
    model_conv_channels: int = 32
    model_hidden_size: int = 48
    model_num_classes: int = 3
    schema_version: str = "frozen-package-v2"
    training_epochs: int = 12
    training_batch_size: int = 256
    training_learning_rate: float = 0.001
    training_patience: int = 3
    training_seed: int = 42
    training_device: str | None = None
    training_version: str = "cnn-lstm-v1"
    library_versions: dict[str, str] = field(default_factory=current_library_versions)

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
        object.__setattr__(self, "library_versions", dict(self.library_versions))


_MANIFEST_FIELDS = {field.name for field in fields(FrozenManifest)}


def create_frozen_manifest(
    *,
    feature_names: list[str] | tuple[str, ...],
    window: int,
    threshold: float,
    scaler_mean: list[float] | tuple[float, ...],
    scaler_scale: list[float] | tuple[float, ...],
    regime_thresholds: dict[str, list[float] | tuple[float, ...]],
    config: TrainConfig,
    device: str | None,
    validation_loss: float,
    data_mode: str,
    development_start: str = "2023-07-01T00:00:00Z",
    development_end: str = "2026-07-01T00:00:00Z",
    prospective_start: str = "2026-07-01T00:00:00Z",
    prospective_end: str = "2026-08-01T00:00:00Z",
) -> FrozenManifest:
    """Create a frozen manifest directly from the actual model training run."""
    return FrozenManifest(
        feature_names=tuple(feature_names),
        window=window,
        threshold=threshold,
        scaler_mean=tuple(scaler_mean),
        scaler_scale=tuple(scaler_scale),
        regime_thresholds={symbol: tuple(values) for symbol, values in regime_thresholds.items()},
        regime_lookback=168,
        selected_seed=config.seed,
        validation_loss=validation_loss,
        development_start=development_start,
        development_end=development_end,
        prospective_start=prospective_start,
        prospective_end=prospective_end,
        data_mode=data_mode,
        training_epochs=config.epochs,
        training_batch_size=config.batch_size,
        training_learning_rate=config.learning_rate,
        training_patience=config.patience,
        training_seed=config.seed,
        training_device=device,
    )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_manifest(
    manifest: FrozenManifest, *, require_digests: bool = True
) -> None:
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

    if (
        manifest.selected_seed not in {42, 43, 44}
        or not _is_finite_number(manifest.validation_loss)
        or manifest.validation_loss < 0.0
    ):
        raise ValueError("incompatible frozen manifest")
    if any(getattr(manifest, name) != value for name, value in _EXACT_DATES.items()):
        raise ValueError("incompatible frozen manifest")
    if manifest.data_mode not in {"genuine", "synthetic"}:
        raise ValueError("incompatible frozen manifest")
    digests = (manifest.cnn_sha256, manifest.baselines_sha256)
    if require_digests:
        if not all(_is_sha256(value) for value in digests):
            raise ValueError("incompatible frozen manifest")
    elif any(value is not None and not _is_sha256(value) for value in digests):
        raise ValueError("incompatible frozen manifest")
    if any(getattr(manifest, name) != value for name, value in _MODEL_DEFAULTS.items()):
        raise ValueError("incompatible frozen manifest")
    if (
        not all(
            _is_positive_integer(getattr(manifest, name))
            for name in ("training_epochs", "training_batch_size", "training_patience")
        )
        or not _is_finite_number(manifest.training_learning_rate)
        or manifest.training_learning_rate <= 0.0
        or manifest.training_seed != manifest.selected_seed
        or (
            manifest.training_device is not None
            and not _is_nonempty_string(manifest.training_device)
        )
        or not _is_nonempty_string(manifest.training_version)
        or not _is_nonempty_string(manifest.schema_version)
    ):
        raise ValueError("incompatible frozen manifest")
    if (
        not isinstance(manifest.library_versions, dict)
        or not _REQUIRED_LIBRARY_VERSION_KEYS.issubset(manifest.library_versions)
        or not all(
            _is_nonempty_string(name) and _is_nonempty_string(version)
            for name, version in manifest.library_versions.items()
        )
    ):
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
) -> FrozenManifest:
    """Write the complete frozen package with no executable serialized objects."""
    _validate_manifest(manifest, require_digests=False)
    _load_validated_state_from_memory(cnn_state, manifest)
    baseline_arrays = _baseline_arrays(baselines, manifest)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "cnn_lstm.pt"
    baseline_path = destination / "baselines.npz"
    manifest_path = destination / "manifest.json"
    state_staging = destination / ".cnn_lstm.pt.tmp"
    baseline_staging = destination / ".baselines.npz.tmp"
    manifest_staging = destination / ".manifest.json.tmp"
    try:
        torch.save(cnn_state, state_staging)
        with baseline_staging.open("wb") as stream:
            np.savez(stream, **baseline_arrays)
            stream.flush()
            os.fsync(stream.fileno())
        saved_manifest = replace(
            manifest,
            cnn_sha256=sha256_file(state_staging),
            baselines_sha256=sha256_file(baseline_staging),
        )
        _validate_manifest(saved_manifest)
        os.replace(state_staging, state_path)
        os.replace(baseline_staging, baseline_path)
        with manifest_staging.open("w") as stream:
            stream.write(json.dumps(asdict(saved_manifest), indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest_staging, manifest_path)
        return saved_manifest
    finally:
        for staging_path in (state_staging, baseline_staging, manifest_staging):
            staging_path.unlink(missing_ok=True)


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a canonical order."""
    if not isinstance(state, dict) or not state:
        raise ValueError("state must be a non-empty tensor dictionary")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state must be a non-empty tensor dictionary")
        contiguous = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
    expected_data_mode: str | None = None,
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
    if expected_data_mode is not None:
        if expected_data_mode not in {"genuine", "synthetic"}:
            raise ValueError("expected data mode must be genuine or synthetic")
        if manifest.data_mode != expected_data_mode:
            raise ValueError("frozen package data mode does not match expectation")
    expected_digests = (manifest.cnn_sha256, manifest.baselines_sha256)
    actual_digests = (sha256_file(required[1]), sha256_file(required[2]))
    if not all(
        hmac.compare_digest(str(expected), actual)
        for expected, actual in zip(expected_digests, actual_digests)
    ):
        raise ValueError("frozen package binary digest mismatch")
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

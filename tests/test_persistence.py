import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.baselines import fit_baseline_models
from src.data import FEATURE_COLUMNS
from src.models import CNNLSTM
from src.persistence import (
    FrozenManifest,
    create_frozen_manifest,
    current_library_versions,
    load_frozen_package,
    predict_frozen_probabilities,
    save_frozen_package,
)
from src.training import TrainConfig


def _manifest(data_mode: str = "genuine") -> FrozenManifest:
    return create_frozen_manifest(
        feature_names=list(FEATURE_COLUMNS),
        window=96,
        threshold=0.002,
        scaler_mean=[0.0] * len(FEATURE_COLUMNS),
        scaler_scale=[1.0] * len(FEATURE_COLUMNS),
        regime_thresholds={
            "BTCUSDT": [0.001, 0.003],
            "ETHUSDT": [0.0015, 0.0035],
        },
        config=TrainConfig(seed=42),
        device="cpu",
        validation_loss=1.0,
        data_mode=data_mode,
    )


def _artifacts() -> tuple[np.ndarray, dict[str, torch.Tensor], object]:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(12, 96, len(FEATURE_COLUMNS))).astype(np.float32)
    labels = np.arange(len(features)) % 3
    model = CNNLSTM(n_features=len(FEATURE_COLUMNS))
    state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    return features, state, fit_baseline_models(features, labels)


def _save_valid_package(
    tmp_path, data_mode: str = "genuine"
) -> tuple[FrozenManifest, np.ndarray, dict[str, torch.Tensor], object]:
    manifest = _manifest(data_mode=data_mode)
    features, state, baselines = _artifacts()
    saved_manifest = save_frozen_package(tmp_path, manifest, state, baselines)
    return saved_manifest, features, state, baselines


def test_frozen_package_round_trip_preserves_safe_predictions_and_baselines(tmp_path) -> None:
    manifest, features, state, baselines = _save_valid_package(tmp_path)

    loaded_manifest, loaded_state, loaded_baselines = load_frozen_package(tmp_path)

    assert loaded_manifest == manifest
    assert len(loaded_manifest.cnn_sha256) == 64
    assert len(loaded_manifest.baselines_sha256) == 64
    assert loaded_manifest.library_versions == manifest.library_versions
    assert set(current_library_versions()).issubset(loaded_manifest.library_versions)
    assert loaded_baselines.majority_class == baselines.majority_class
    assert (tmp_path / "baselines.npz").is_file()
    assert not (tmp_path / "baselines.pkl").exists()
    with np.load(tmp_path / "baselines.npz", allow_pickle=False) as stored:
        assert all(stored[name].dtype != object for name in stored.files)
    before = predict_frozen_probabilities(
        manifest, state, list(FEATURE_COLUMNS), features, batch_size=4, device="cpu"
    )
    after = predict_frozen_probabilities(
        loaded_manifest,
        loaded_state,
        list(FEATURE_COLUMNS),
        features,
        batch_size=4,
        device="cpu",
    )
    np.testing.assert_allclose(before, after, atol=1e-7)
    np.testing.assert_array_equal(
        loaded_baselines.logistic_regression.predict(features[:, -1, :]),
        baselines.logistic_regression.predict(features[:, -1, :]),
    )


def test_manifest_factory_records_actual_non_default_training_run(tmp_path) -> None:
    config = TrainConfig(
        epochs=7, batch_size=128, learning_rate=0.005, patience=2, seed=43, device="cuda"
    )
    manifest = create_frozen_manifest(
        feature_names=list(FEATURE_COLUMNS),
        window=96,
        threshold=0.002,
        scaler_mean=[0.0] * len(FEATURE_COLUMNS),
        scaler_scale=[1.0] * len(FEATURE_COLUMNS),
        regime_thresholds={
            "BTCUSDT": [0.001, 0.003],
            "ETHUSDT": [0.0015, 0.0035],
        },
        config=config,
        device="mps",
        validation_loss=0.4,
        data_mode="genuine",
    )
    _, state, baselines = _artifacts()
    save_frozen_package(tmp_path, manifest, state, baselines)
    loaded_manifest, _, _ = load_frozen_package(tmp_path)

    assert loaded_manifest.selected_seed == loaded_manifest.training_seed == 43
    assert loaded_manifest.training_epochs == 7
    assert loaded_manifest.training_batch_size == 128
    assert loaded_manifest.training_learning_rate == 0.005
    assert loaded_manifest.training_patience == 2
    assert loaded_manifest.training_device == "mps"


def test_load_rejects_synthetic_package_when_genuine_mode_is_expected(tmp_path) -> None:
    _save_valid_package(tmp_path, data_mode="synthetic")

    with pytest.raises(ValueError, match="data mode"):
        load_frozen_package(tmp_path, expected_data_mode="genuine")


def test_safe_inference_rejects_reordered_feature_names(tmp_path) -> None:
    manifest, features, state, _ = _save_valid_package(tmp_path)

    with pytest.raises(ValueError, match="feature order"):
        predict_frozen_probabilities(
            manifest, state, list(reversed(FEATURE_COLUMNS)), features, device="cpu"
        )


def test_safe_inference_rejects_wrong_temporal_length(tmp_path) -> None:
    manifest, features, state, _ = _save_valid_package(tmp_path)

    with pytest.raises(ValueError, match=r"\[samples, 96, 13\]"):
        predict_frozen_probabilities(
            manifest, state, list(FEATURE_COLUMNS), features[:, :-1, :], device="cpu"
        )


def test_load_rejects_missing_frozen_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="incomplete frozen package"):
        load_frozen_package(tmp_path)


def test_load_rejects_state_shape_mismatch(tmp_path) -> None:
    _, _, state, _ = _save_valid_package(tmp_path)
    state["head.1.weight"] = torch.zeros((2, 48))
    torch.save(state, tmp_path / "cnn_lstm.pt")

    with pytest.raises(ValueError, match="digest"):
        load_frozen_package(tmp_path)


@pytest.mark.parametrize("filename", ["cnn_lstm.pt", "baselines.npz"])
def test_load_rejects_frozen_binary_tampering_before_deserialization(
    tmp_path, filename: str, monkeypatch
) -> None:
    _save_valid_package(tmp_path)
    path = tmp_path / filename
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)

    def forbid_deserialization(*args, **kwargs):
        raise AssertionError("tampered package reached deserialization")

    monkeypatch.setattr("src.persistence.torch.load", forbid_deserialization)
    monkeypatch.setattr("src.persistence.np.load", forbid_deserialization)
    with pytest.raises(ValueError, match="digest"):
        load_frozen_package(tmp_path)


def test_committed_genuine_frozen_package_loads_from_a_clean_clone() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest, state, baselines = load_frozen_package(
        root / "artifacts/final/frozen", expected_data_mode="genuine"
    )

    assert manifest.cnn_sha256 == "670265c722336bfd4c1fb4d0ac035ed4fad9a657427f3d26733f96f433ffced6"
    assert manifest.baselines_sha256 == "8cb39d6b8ab30c7482ed83ebc0d4840f52c6bff5591f2089cd29ba2dcf59f84f"
    assert state
    assert baselines.majority_class in {0, 1, 2}


@pytest.mark.parametrize(
    "updates",
    [
        {"feature_names": list(reversed(FEATURE_COLUMNS))},
        {"scaler_mean": [float("nan")] * len(FEATURE_COLUMNS)},
        {"scaler_scale": [0.0] * len(FEATURE_COLUMNS)},
        {"threshold": -0.001},
        {"regime_thresholds": []},
        {"regime_thresholds": {"BTCUSDT": [0.003, 0.001], "ETHUSDT": [0.001, 0.003]}},
        {"development_end": "2026-06-30T23:00:00Z"},
        {"model_hidden_size": 47},
        {"training_seed": 43},
        {"validation_loss": -0.1},
        {"data_mode": "dry_run"},
        {"library_versions": {"python": ""}},
    ],
)
def test_load_rejects_invalid_schema_metadata(tmp_path, updates: dict) -> None:
    _save_valid_package(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw.update(copy.deepcopy(updates))
    manifest_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="incompatible frozen manifest"):
        load_frozen_package(tmp_path)


def test_load_rejects_manifest_missing_required_key(tmp_path) -> None:
    _save_valid_package(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw.pop("library_versions")
    manifest_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="incompatible frozen manifest"):
        load_frozen_package(tmp_path)


def test_load_rejects_baseline_feature_mismatch(tmp_path) -> None:
    _save_valid_package(tmp_path)
    baseline_path = tmp_path / "baselines.npz"
    with np.load(baseline_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["logistic_n_features_in"] = np.array([12], dtype=np.int64)
    np.savez(baseline_path, **arrays)

    with pytest.raises(ValueError, match="digest"):
        load_frozen_package(tmp_path)

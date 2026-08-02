import numpy as np
import pytest
import torch

from src.baselines import fit_baseline_models
from src.models import CNNLSTM
from src.persistence import FrozenManifest, load_frozen_package, save_frozen_package
from src.training import predict_probabilities


def test_frozen_package_round_trip_preserves_predictions(tmp_path) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 24, 13)).astype(np.float32)
    y = np.arange(12) % 3
    model = CNNLSTM(n_features=13)
    state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    baselines = fit_baseline_models(x, y)
    manifest = FrozenManifest(
        feature_names=[f"f{i}" for i in range(13)],
        window=24,
        threshold=0.002,
        scaler_mean=[0.0] * 13,
        scaler_scale=[1.0] * 13,
        regime_thresholds={"BTCUSDT": [0.001, 0.003]},
        regime_lookback=168,
        selected_seed=42,
        validation_loss=1.0,
        development_start="2023-07-01T00:00:00Z",
        development_end="2026-07-01T00:00:00Z",
        prospective_start="2026-07-01T00:00:00Z",
        prospective_end="2026-08-01T00:00:00Z",
    )

    save_frozen_package(tmp_path, manifest, state, baselines)
    loaded_manifest, loaded_state, loaded_baselines = load_frozen_package(tmp_path)

    assert loaded_manifest == manifest
    assert loaded_baselines.majority_class == baselines.majority_class
    before = predict_probabilities(state, x, batch_size=4, device="cpu")
    after = predict_probabilities(loaded_state, x, batch_size=4, device="cpu")
    np.testing.assert_allclose(before, after, atol=1e-7)


def test_load_rejects_missing_frozen_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="incomplete frozen package"):
        load_frozen_package(tmp_path)


def test_load_rejects_manifest_with_incompatible_scaler_scale(tmp_path) -> None:
    x = np.zeros((3, 24, 1), dtype=np.float32)
    manifest = FrozenManifest(
        feature_names=["return_1h"],
        window=24,
        threshold=0.002,
        scaler_mean=[0.0],
        scaler_scale=[1.0, 1.0],
        regime_thresholds={"BTCUSDT": [0.001, 0.003]},
        regime_lookback=168,
        selected_seed=42,
        validation_loss=1.0,
        development_start="2023-07-01T00:00:00Z",
        development_end="2026-07-01T00:00:00Z",
        prospective_start="2026-07-01T00:00:00Z",
        prospective_end="2026-08-01T00:00:00Z",
    )
    model = CNNLSTM(n_features=1)
    state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    baselines = fit_baseline_models(x, np.array([0, 1, 2]))
    save_frozen_package(tmp_path, manifest, state, baselines)

    with pytest.raises(ValueError, match="incompatible frozen manifest"):
        load_frozen_package(tmp_path)

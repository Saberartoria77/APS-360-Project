# Final Regime Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and report a leakage-safe historical regime stress test plus a one-time prospective evaluation on July 2026 BTC/ETH data.

**Architecture:** Add focused modules for causal regime labels, frozen preprocessing/inference, and model persistence while preserving the existing `DatasetBundle`, CNN-LSTM, and progress-report pipeline. A new command-line runner separates historical training/freezing from prospective data access, writes machine-readable artifacts, and supplies every number and figure used by a four-page LaTeX final report.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn, PyTorch, Matplotlib, Seaborn, pytest, LaTeX/Tectonic.

## Global Constraints

- Forecast exactly one hour ahead into down, flat, or up classes.
- Use BTCUSDT and ETHUSDT, 96-hour windows, and the existing 13 causal features.
- Use historical data from `2023-07-01T00:00:00Z` through `2026-06-30T23:00:00Z` for development.
- Score prospective forecast origins only from `2026-07-01T00:00:00Z` through `2026-07-31T23:00:00Z`; download the `2026-08-01T00:00:00Z` close only to resolve the final next-hour label.
- Do not fetch or inspect July data until a complete frozen manifest exists.
- Fit feature scaling, direction threshold, and regime thresholds on the historical training split only.
- Define regimes from trailing 168-hour log-return volatility and per-symbol training quantiles at 0.33 and 0.67.
- Use CNN-LSTM seeds `42`, `43`, and `44`; select the prospective seed using validation loss only.
- Use macro-F1 as the primary metric; also report accuracy and down/flat/up recall.
- Never tune after prospective results are revealed.
- Do not add prediction horizons, neural architectures, trading simulations, or a dashboard before submission.
- Preserve all 25 existing tests.
- The final report main text must be exactly four pages or fewer; references are unlimited.

## File Map

- Create `src/regimes.py`: causal realized volatility, threshold fitting, regime assignment, and aligned bundle subsetting.
- Modify `src/baselines.py`: training-derived majority baseline and reusable fitted baseline models.
- Modify `src/data.py`: frozen preprocessing for prospective target windows.
- Create `src/persistence.py`: frozen CNN-LSTM/baseline package and strict metadata validation.
- Modify `src/training.py`: reusable probability inference from a saved state dictionary.
- Create `src/final_evaluation.py`: slice metrics, seed aggregation, transfer matrix, qualitative selection, and report figures.
- Create `run_final_experiment.py`: separate `historical` and `prospective` commands with integrity gates.
- Create `tests/test_regimes.py`, `tests/test_prospective.py`, `tests/test_persistence.py`, and `tests/test_final_experiment.py`.
- Create `final_report.tex` and `final_report.pdf`; modify `refs.bib`, `README.md`, `.github/workflows/latex-compile.yml`, and `tests/test_report.py`.

---

### Task 1: Causal Regime Labels and Aligned Subsets

**Files:**
- Create: `src/regimes.py`
- Create: `tests/test_regimes.py`

**Interfaces:**
- Consumes: `src.data.DatasetBundle`; feature frames containing `log_return_1h` with UTC chronological indexes.
- Produces: `RegimeThresholds`, `fit_regime_thresholds(...)`, `assign_regimes(...)`, `assign_bundle_regimes(...)`, and `subset_bundle(...)`.

- [ ] **Step 1: Write failing causality, per-symbol, and alignment tests**

```python
# tests/test_regimes.py
import numpy as np
import pandas as pd

from src.data import DatasetBundle, FEATURE_COLUMNS
from src.regimes import (
    assign_bundle_regimes,
    assign_regimes,
    fit_regime_thresholds,
    subset_bundle,
)


def _frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=360, freq="h", tz="UTC")
    frames = {}
    for symbol, scale in [("BTCUSDT", 0.002), ("ETHUSDT", 0.006)]:
        returns = scale * (1.0 + np.sin(np.arange(len(index)) / 17.0))
        frames[symbol] = pd.DataFrame({"log_return_1h": returns}, index=index)
    return frames


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(9)
    sizes = {"train": 12, "val": 6, "test": 6}
    base = np.datetime64("2025-01-08T00")
    values = {
        split: rng.normal(size=(size, 96, len(FEATURE_COLUMNS))).astype(np.float32)
        for split, size in sizes.items()
    }
    labels = {split: np.arange(size) % 3 for split, size in sizes.items()}
    times = {
        split: base + offset + np.arange(size).astype("timedelta64[h]")
        for split, size, offset in [("train", 12, 0), ("val", 6, 100), ("test", 6, 200)]
    }
    symbols = {split: np.array(["BTCUSDT", "ETHUSDT"] * (size // 2)) for split, size in sizes.items()}
    return DatasetBundle(
        x_train=values["train"], y_train=labels["train"],
        x_val=values["val"], y_val=labels["val"],
        x_test=values["test"], y_test=labels["test"],
        train_times=times["train"], val_times=times["val"], test_times=times["test"],
        train_symbols=symbols["train"], val_symbols=symbols["val"], test_symbols=symbols["test"],
        feature_names=list(FEATURE_COLUMNS), scaler_mean=np.zeros(len(FEATURE_COLUMNS)),
        scaler_scale=np.ones(len(FEATURE_COLUMNS)), threshold=0.002,
    )


def test_regime_assignment_is_causal_and_thresholds_are_per_symbol() -> None:
    frames = _frames()
    train_end = frames["BTCUSDT"].index[260]
    thresholds = fit_regime_thresholds(frames, train_end=train_end, lookback=24)
    assert set(thresholds.by_symbol) == {"BTCUSDT", "ETHUSDT"}
    assert thresholds.by_symbol["BTCUSDT"] != thresholds.by_symbol["ETHUSDT"]

    times = np.array([frames["BTCUSDT"].index[280].to_datetime64()])
    original = assign_regimes(frames, times, np.array(["BTCUSDT"]), thresholds)
    changed = {name: frame.copy() for name, frame in frames.items()}
    changed["BTCUSDT"].loc[changed["BTCUSDT"].index[300]:, "log_return_1h"] *= 100
    revised = assign_regimes(changed, times, np.array(["BTCUSDT"]), thresholds)
    np.testing.assert_array_equal(original, revised)


def test_subset_bundle_preserves_alignment() -> None:
    bundle = _bundle()
    masks = {
        "train": np.array([True, False] * 6),
        "val": np.array([True, False] * 3),
        "test": np.array([True, False] * 3),
    }
    subset = subset_bundle(bundle, masks)
    np.testing.assert_array_equal(subset.y_test, bundle.y_test[masks["test"]])
    np.testing.assert_array_equal(subset.test_times, bundle.test_times[masks["test"]])
    np.testing.assert_array_equal(subset.test_symbols, bundle.test_symbols[masks["test"]])
    assert subset.feature_names == bundle.feature_names
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_regimes.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.regimes'`.

- [ ] **Step 3: Implement causal regime APIs and bundle subsetting**

```python
# src/regimes.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data import DatasetBundle

REGIME_NAMES = np.array(["low", "medium", "high"])


@dataclass(frozen=True)
class RegimeThresholds:
    by_symbol: dict[str, tuple[float, float]]
    lookback: int = 168


def realized_volatility(frame: pd.DataFrame, lookback: int = 168) -> pd.Series:
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    if "log_return_1h" not in frame:
        raise ValueError("frame is missing log_return_1h")
    if frame.index.tz is None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be timezone-aware, unique, and chronological")
    return frame["log_return_1h"].rolling(lookback, min_periods=lookback).std(ddof=0)


def fit_regime_thresholds(
    frames: dict[str, pd.DataFrame], train_end: pd.Timestamp, lookback: int = 168
) -> RegimeThresholds:
    train_end = pd.Timestamp(train_end)
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    fitted = {}
    for symbol, frame in frames.items():
        values = realized_volatility(frame, lookback).loc[:train_end].dropna()
        if values.empty:
            raise ValueError(f"{symbol} has no training volatility values")
        low, high = values.quantile([0.33, 0.67]).to_numpy(dtype=float)
        fitted[symbol] = (float(low), float(high))
    return RegimeThresholds(by_symbol=fitted, lookback=lookback)


def assign_regimes(
    frames: dict[str, pd.DataFrame],
    times: np.ndarray,
    symbols: np.ndarray,
    thresholds: RegimeThresholds,
) -> np.ndarray:
    labels = np.empty(len(times), dtype="<U6")
    volatility = {
        symbol: realized_volatility(frame, thresholds.lookback) for symbol, frame in frames.items()
    }
    for index, (time, symbol_value) in enumerate(zip(times, symbols)):
        symbol = str(symbol_value)
        if symbol not in volatility or symbol not in thresholds.by_symbol:
            raise ValueError(f"missing regime inputs for {symbol}")
        timestamp = pd.Timestamp(time, tz="UTC") if pd.Timestamp(time).tzinfo is None else pd.Timestamp(time).tz_convert("UTC")
        value = volatility[symbol].get(timestamp, np.nan)
        if pd.isna(value):
            raise ValueError(f"missing causal volatility for {symbol} at {timestamp}")
        low, high = thresholds.by_symbol[symbol]
        labels[index] = "low" if value <= low else "high" if value > high else "medium"
    return labels


def assign_bundle_regimes(
    bundle: DatasetBundle,
    frames: dict[str, pd.DataFrame],
    thresholds: RegimeThresholds,
) -> dict[str, np.ndarray]:
    return {
        split: assign_regimes(
            frames,
            getattr(bundle, f"{split}_times"),
            getattr(bundle, f"{split}_symbols"),
            thresholds,
        )
        for split in ("train", "val", "test")
    }


def subset_bundle(bundle: DatasetBundle, masks: dict[str, np.ndarray]) -> DatasetBundle:
    values = {}
    for split in ("train", "val", "test"):
        mask = np.asarray(masks[split], dtype=bool)
        labels = getattr(bundle, f"y_{split}")
        if mask.shape != labels.shape or not mask.any():
            raise ValueError(f"{split} mask must select at least one aligned sample")
        values[split] = {
            "x": getattr(bundle, f"x_{split}")[mask],
            "y": labels[mask],
            "times": getattr(bundle, f"{split}_times")[mask],
            "symbols": getattr(bundle, f"{split}_symbols")[mask],
        }
    if len(np.unique(values["train"]["y"])) < 2:
        raise ValueError("training subset must contain at least two classes")
    return DatasetBundle(
        x_train=values["train"]["x"], y_train=values["train"]["y"],
        x_val=values["val"]["x"], y_val=values["val"]["y"],
        x_test=values["test"]["x"], y_test=values["test"]["y"],
        train_times=values["train"]["times"], val_times=values["val"]["times"],
        test_times=values["test"]["times"], train_symbols=values["train"]["symbols"],
        val_symbols=values["val"]["symbols"], test_symbols=values["test"]["symbols"],
        feature_names=list(bundle.feature_names), scaler_mean=bundle.scaler_mean.copy(),
        scaler_scale=bundle.scaler_scale.copy(), threshold=float(bundle.threshold),
    )
```

- [ ] **Step 4: Run regime and existing sequence tests**

Run: `.venv/bin/python -m pytest tests/test_regimes.py tests/test_sequences.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the regime component**

```bash
git add src/regimes.py tests/test_regimes.py
git commit -m "feat: add causal volatility regimes"
```

---

### Task 2: Reusable Majority, Momentum, and Logistic Baselines

**Files:**
- Modify: `src/baselines.py:1-35`
- Modify: `tests/test_evaluation.py:1-48`

**Interfaces:**
- Consumes: training arrays and frozen preprocessing metadata from `DatasetBundle`.
- Produces: `BaselineModels`, `fit_baseline_models(...)`, `predict_baselines(...)`, and a backward-compatible `fit_baselines(...)` wrapper.

- [ ] **Step 1: Extend baseline tests with the training-majority requirement**

```python
# append to tests/test_evaluation.py
from src.baselines import fit_baseline_models, predict_baselines


def test_majority_baseline_uses_training_labels(bundle: DatasetBundle) -> None:
    bundle.y_train[:] = 2
    bundle.y_train[0] = 0
    bundle.y_train[1] = 1
    models = fit_baseline_models(bundle.x_train, bundle.y_train)
    predictions = predict_baselines(
        models,
        bundle.x_test,
        bundle.feature_names,
        bundle.scaler_mean,
        bundle.scaler_scale,
        bundle.threshold,
    )
    assert set(predictions) == {"majority", "momentum", "logistic_regression"}
    assert np.all(predictions["majority"] == 2)
```

Also change the existing assertion in `test_baselines_return_one_prediction_per_test_sample` to expect `{"majority", "momentum", "logistic_regression"}`.

- [ ] **Step 2: Run the focused test and verify import failure**

Run: `.venv/bin/python -m pytest tests/test_evaluation.py::test_majority_baseline_uses_training_labels -v`

Expected: FAIL because `fit_baseline_models` does not exist.

- [ ] **Step 3: Refactor baselines into fit and predict stages**

```python
# replace src/baselines.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import DatasetBundle


@dataclass
class BaselineModels:
    majority_class: int
    logistic_regression: LogisticRegression


def fit_baseline_models(x_train: np.ndarray, y_train: np.ndarray) -> BaselineModels:
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=3)
    logistic = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs"
    )
    logistic.fit(x_train[:, -1, :], y_train)
    return BaselineModels(majority_class=int(counts.argmax()), logistic_regression=logistic)


def predict_baselines(
    models: BaselineModels,
    features: np.ndarray,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    return_index = feature_names.index("return_1h")
    standardized_return = features[:, -1, return_index]
    raw_return = standardized_return * scaler_scale[return_index] + scaler_mean[return_index]
    momentum = np.full(len(raw_return), 1, dtype=np.int64)
    momentum[raw_return < -threshold] = 0
    momentum[raw_return > threshold] = 2
    return {
        "majority": np.full(len(features), models.majority_class, dtype=np.int64),
        "momentum": momentum,
        "logistic_regression": models.logistic_regression.predict(features[:, -1, :]).astype(np.int64),
    }


def fit_baselines(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    models = fit_baseline_models(bundle.x_train, bundle.y_train)
    return predict_baselines(
        models,
        bundle.x_test,
        bundle.feature_names,
        bundle.scaler_mean,
        bundle.scaler_scale,
        bundle.threshold,
    )
```

- [ ] **Step 4: Update the progress experiment's expected model set and run tests**

Change `tests/test_experiment.py:23` to expect `{"majority", "momentum", "logistic_regression", "cnn_lstm"}` and add `"figures/confusion_majority.png"` to its `required` list. Run:

`.venv/bin/python -m pytest tests/test_evaluation.py tests/test_experiment.py -v`

Expected: all tests pass and the dry run writes `confusion_majority.png` in addition to existing artifacts.

- [ ] **Step 5: Commit reusable baselines**

```bash
git add src/baselines.py tests/test_evaluation.py tests/test_experiment.py
git commit -m "feat: add training-majority baseline"
```

---

### Task 3: Frozen Prospective Window Construction

**Files:**
- Modify: `src/data.py:46-165`
- Create: `tests/test_prospective.py`

**Interfaces:**
- Consumes: causal feature frames, frozen scaler arrays, direction threshold, feature order, window length, and UTC target bounds.
- Produces: `EvaluationDataset` and `prepare_evaluation_windows(...)` with no split fitting or preprocessing fitting.

- [ ] **Step 1: Write failing frozen-window and July-boundary tests**

```python
# tests/test_prospective.py
import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS, prepare_evaluation_windows


def test_evaluation_windows_use_frozen_metadata_and_only_requested_targets() -> None:
    index = pd.date_range("2026-06-20", "2026-08-01", freq="h", inclusive="left", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["close"] = 100 * np.exp(np.cumsum(np.full(len(index), 0.0002)))
    for position, name in enumerate(FEATURE_COLUMNS):
        frame[name] = position + np.arange(len(index), dtype=float) / 1000
    dataset = prepare_evaluation_windows(
        {"BTCUSDT": frame},
        feature_names=list(FEATURE_COLUMNS),
        scaler_mean=np.zeros(len(FEATURE_COLUMNS)),
        scaler_scale=np.ones(len(FEATURE_COLUMNS)),
        threshold=0.001,
        window=96,
        target_start=pd.Timestamp("2026-07-01", tz="UTC"),
        target_end=pd.Timestamp("2026-08-01", tz="UTC"),
    )
    assert dataset.times.min() >= np.datetime64("2026-07-01T00")
    assert dataset.times.max() < np.datetime64("2026-08-01T00")
    assert dataset.x.shape[1:] == (96, len(FEATURE_COLUMNS))
    assert len(dataset.x) == len(dataset.y) == len(dataset.times) == len(dataset.symbols)
```

- [ ] **Step 2: Run the test and verify missing API failure**

Run: `.venv/bin/python -m pytest tests/test_prospective.py -v`

Expected: FAIL because `prepare_evaluation_windows` does not exist.

- [ ] **Step 3: Add the frozen evaluation dataset builder**

```python
# add near DatasetBundle in src/data.py
@dataclass
class EvaluationDataset:
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    symbols: np.ndarray


def prepare_evaluation_windows(
    frames: dict[str, pd.DataFrame],
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    threshold: float,
    window: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> EvaluationDataset:
    start = pd.Timestamp(target_start)
    end = pd.Timestamp(target_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start >= end:
        raise ValueError("target_start must be before target_end")
    scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
    if len(feature_names) != len(scaler_mean) or scaler_mean.shape != scaler_scale.shape:
        raise ValueError("frozen feature and scaler metadata are incompatible")

    collected = {"x": [], "y": [], "times": [], "symbols": []}
    for symbol, frame in frames.items():
        required = {"close", *feature_names}
        if missing := required.difference(frame.columns):
            raise ValueError(f"{symbol} is missing columns: {sorted(missing)}")
        subset = frame.copy()
        subset["label"] = make_direction_labels(subset["close"], horizon=1, threshold=threshold)
        subset = subset.dropna(subset=[*feature_names, "label"])
        values = (subset[feature_names].to_numpy(dtype=np.float64) - scaler_mean) / scaler_scale
        labels = subset["label"].to_numpy(dtype=np.int64)
        times = subset.index.to_numpy()
        for end_position in range(window - 1, len(subset)):
            timestamp = subset.index[end_position]
            if start <= timestamp < end:
                collected["x"].append(values[end_position - window + 1 : end_position + 1])
                collected["y"].append(labels[end_position])
                collected["times"].append(times[end_position])
                collected["symbols"].append(symbol)
    if not collected["x"]:
        raise ValueError("requested target range produced no complete evaluation windows")
    order = np.argsort(np.asarray(collected["times"]))
    return EvaluationDataset(
        x=np.asarray(collected["x"], dtype=np.float32)[order],
        y=np.asarray(collected["y"], dtype=np.int64)[order],
        times=np.asarray(collected["times"])[order],
        symbols=np.asarray(collected["symbols"])[order],
    )
```

- [ ] **Step 4: Run data, sequence, and prospective tests**

Run: `.venv/bin/python -m pytest tests/test_data.py tests/test_sequences.py tests/test_prospective.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the frozen window builder**

```bash
git add src/data.py tests/test_prospective.py
git commit -m "feat: build windows with frozen preprocessing"
```

---

### Task 4: Frozen Model Persistence and Reusable Inference

**Files:**
- Modify: `src/training.py:28-151`
- Create: `src/persistence.py`
- Create: `tests/test_persistence.py`

**Interfaces:**
- Consumes: `TrainResult.best_state`, `BaselineModels`, frozen feature/scaler/threshold/regime metadata.
- Produces: `predict_probabilities(...)`, `FrozenManifest`, `save_frozen_package(...)`, and `load_frozen_package(...)`.

- [ ] **Step 1: Write failing round-trip and compatibility tests**

```python
# tests/test_persistence.py
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
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    baselines = fit_baseline_models(x, y)
    manifest = FrozenManifest(
        feature_names=[f"f{i}" for i in range(13)], window=24, threshold=0.002,
        scaler_mean=[0.0] * 13, scaler_scale=[1.0] * 13,
        regime_thresholds={"BTCUSDT": [0.001, 0.003]}, regime_lookback=168,
        selected_seed=42, validation_loss=1.0,
        development_start="2023-07-01T00:00:00Z", development_end="2026-07-01T00:00:00Z",
        prospective_start="2026-07-01T00:00:00Z", prospective_end="2026-08-01T00:00:00Z",
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
```

- [ ] **Step 2: Run persistence tests and verify missing module failure**

Run: `.venv/bin/python -m pytest tests/test_persistence.py -v`

Expected: collection fails because `src.persistence` does not exist.

- [ ] **Step 3: Add state-dictionary probability inference**

```python
# append to src/training.py
def predict_probabilities(
    state: dict[str, torch.Tensor],
    features: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    if features.ndim != 3 or len(features) == 0:
        raise ValueError("features must be a non-empty [samples, time, features] array")
    selected_device = _choose_device(device)
    model = CNNLSTM(n_features=features.shape[-1]).to(selected_device)
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size)
    probabilities = []
    with torch.inference_mode():
        for (batch,) in loader:
            probabilities.append(torch.softmax(model(batch.to(selected_device)), dim=1).cpu().numpy())
    return np.concatenate(probabilities)
```

- [ ] **Step 4: Implement strict frozen-package serialization**

```python
# src/persistence.py
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import torch

from src.baselines import BaselineModels


@dataclass(frozen=True)
class FrozenManifest:
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
    destination: Path, manifest: FrozenManifest,
    cnn_state: dict[str, torch.Tensor], baselines: BaselineModels,
) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    torch.save(cnn_state, destination / "cnn_lstm.pt")
    with (destination / "baselines.pkl").open("wb") as handle:
        pickle.dump(baselines, handle)


def load_frozen_package(
    source: Path,
) -> tuple[FrozenManifest, dict[str, torch.Tensor], BaselineModels]:
    source = Path(source)
    required = [source / "manifest.json", source / "cnn_lstm.pt", source / "baselines.pkl"]
    if not all(path.is_file() for path in required):
        raise ValueError("incomplete frozen package")
    manifest = FrozenManifest(**json.loads(required[0].read_text()))
    if manifest.window < 2 or len(manifest.feature_names) != len(manifest.scaler_mean):
        raise ValueError("incompatible frozen manifest")
    state = torch.load(required[1], map_location="cpu", weights_only=True)
    with required[2].open("rb") as handle:
        baselines = pickle.load(handle)
    if not isinstance(baselines, BaselineModels):
        raise ValueError("invalid frozen baseline model")
    return manifest, state, baselines
```

- [ ] **Step 5: Run model and persistence tests**

Run: `.venv/bin/python -m pytest tests/test_model.py tests/test_persistence.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit frozen persistence**

```bash
git add src/training.py src/persistence.py tests/test_persistence.py
git commit -m "feat: freeze models and preprocessing metadata"
```

---

### Task 5: Historical Seed and Cross-Regime Evaluation

**Files:**
- Create: `src/final_evaluation.py`
- Create: `run_final_experiment.py`
- Create: `tests/test_final_experiment.py`

**Interfaces:**
- Consumes: Task 1 regimes/subsets, Task 2 baseline fit/predict, Task 4 persistence, and existing `train_model`/`classification_metrics`.
- Produces: `evaluate_slices(...)`, `transfer_pairs()`, `run_historical_stage(...)`, `artifacts/final/historical_results.json`, `cross_regime_results.csv`, and a complete frozen package.

- [ ] **Step 1: Write failing transfer-matrix and historical dry-run tests**

```python
# tests/test_final_experiment.py
import json

from run_final_experiment import run_historical_stage
from src.final_evaluation import transfer_pairs


def test_transfer_matrix_contains_all_low_high_combinations() -> None:
    assert set(transfer_pairs()) == {
        ("low", "low"), ("low", "high"),
        ("high", "high"), ("high", "low"),
    }


def test_historical_dry_run_writes_results_and_frozen_manifest(tmp_path) -> None:
    result = run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))
    assert result["configuration"]["prospective_revealed"] is False
    required = [
        "historical_results.json", "cross_regime_results.csv",
        "frozen/manifest.json", "frozen/cnn_lstm.pt", "frozen/baselines.pkl",
    ]
    assert all((tmp_path / name).is_file() for name in required)
    manifest = json.loads((tmp_path / "frozen/manifest.json").read_text())
    assert manifest["prospective_start"] == "2026-07-01T00:00:00Z"
```

- [ ] **Step 2: Run focused tests and verify missing imports**

Run: `.venv/bin/python -m pytest tests/test_final_experiment.py -v`

Expected: collection fails because `run_final_experiment` and `src.final_evaluation` do not exist.

- [ ] **Step 3: Implement reusable result slicing and transfer declarations**

```python
# src/final_evaluation.py
from __future__ import annotations

import numpy as np
from src.evaluation import classification_metrics


def transfer_pairs() -> list[tuple[str, str]]:
    return [("low", "low"), ("low", "high"), ("high", "high"), ("high", "low")]


def evaluate_slices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    regimes: np.ndarray,
    symbols: np.ndarray,
) -> dict:
    slices = {"overall": np.ones(len(y_true), dtype=bool)}
    slices.update({f"regime_{name}": regimes == name for name in ("low", "medium", "high")})
    slices.update({f"symbol_{name}": symbols == name for name in sorted(set(symbols))})
    output = {}
    for slice_name, mask in slices.items():
        if not mask.any():
            continue
        output[slice_name] = {
            "sample_count": int(mask.sum()),
            "class_counts": np.bincount(y_true[mask], minlength=3).astype(int).tolist(),
            "models": {
                name: classification_metrics(y_true[mask], values[mask])
                for name, values in predictions.items()
            },
        }
    return output


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    keys = ["accuracy", "macro_f1"]
    return {
        key: {
            "mean": float(np.mean([result[key] for result in seed_results])),
            "std": float(np.std([result[key] for result in seed_results], ddof=0)),
        }
        for key in keys
    }
```

- [ ] **Step 4: Implement `run_historical_stage` and CLI without prospective access**

In `run_final_experiment.py`, implement these exact public interfaces:

```python
HISTORICAL_START = "2023-07-01"
HISTORICAL_END = "2026-07-01"
PROSPECTIVE_START = "2026-07-01T00:00:00Z"
PROSPECTIVE_END = "2026-08-01T00:00:00Z"
SEEDS = (42, 43, 44)


def run_historical_stage(
    output_dir: Path | str = Path("artifacts/final"),
    dry_run: bool = False,
    epochs: int = 12,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """Train/evaluate historical models and freeze the validation-selected package."""
```

The body must execute in this order:

1. Load only historical frames, using `_synthetic_ohlcv` for dry runs and `fetch_klines` ending at `HISTORICAL_END` otherwise.
2. Engineer features and create the existing historical `DatasetBundle`.
3. Fit 168-hour per-symbol regime thresholds using `bundle.train_times.max()`.
4. Assign regimes to train, validation, and test samples.
5. Fit global baselines and evaluate overall/regime/symbol slices.
6. Train one global CNN-LSTM per supplied seed; store each `min(result.history["val_loss"])`, historical metrics, and state.
7. Select the frozen seed with the smallest validation loss only.
8. For each pair from `transfer_pairs()`, train logistic regression and CNN-LSTM on matching train/validation masks and evaluate the requested test mask. With three genuine seeds this creates mean/std results; dry run uses the supplied seed tuple.
9. Save `historical_results.json` and `cross_regime_results.csv`.
10. Save the selected global state, global baseline models, preprocessing metadata, and regime thresholds through `save_frozen_package`.

Add CLI parsing with required subcommands:

```python
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="stage", required=True)
historical = subparsers.add_parser("historical")
historical.add_argument("--output-dir", type=Path, default=Path("artifacts/final"))
historical.add_argument("--dry-run", action="store_true")
historical.add_argument("--epochs", type=int, default=12)
```

The historical stage must not contain `PROSPECTIVE_START` in any `fetch_klines` call and must write `"prospective_revealed": false`.

- [ ] **Step 5: Run dry-run and regression tests**

Run: `.venv/bin/python -m pytest tests/test_final_experiment.py tests/test_experiment.py -v`

Expected: all tests pass; the dry run completes on CPU without network access.

- [ ] **Step 6: Commit historical evaluation**

```bash
git add src/final_evaluation.py run_final_experiment.py tests/test_final_experiment.py
git commit -m "feat: add historical cross-regime evaluation"
```

---

### Task 6: Guarded Prospective Evaluation and Report Figures

**Files:**
- Modify: `src/evaluation.py:60-91`
- Modify: `src/final_evaluation.py`
- Modify: `run_final_experiment.py`
- Modify: `tests/test_final_experiment.py`

**Interfaces:**
- Consumes: complete frozen package and raw context ending at `2026-08-01T00:00:00Z`.
- Produces: `run_prospective_stage(...)`, regime-aware qualitative examples, `prospective_results.json`, `qualitative_examples.csv`, and report-ready PNG figures.

- [ ] **Step 1: Add failing prospective integrity and artifact tests**

```python
# append to tests/test_final_experiment.py
import pytest
from run_final_experiment import run_prospective_stage


def test_prospective_stage_refuses_to_run_without_frozen_manifest(tmp_path) -> None:
    with pytest.raises(ValueError, match="incomplete frozen package"):
        run_prospective_stage(tmp_path, dry_run=True)


def test_prospective_dry_run_scores_only_july_and_writes_artifacts(tmp_path) -> None:
    run_historical_stage(tmp_path, dry_run=True, epochs=1, seeds=(42,))
    result = run_prospective_stage(tmp_path, dry_run=True)
    assert result["configuration"]["prospective_revealed"] is True
    assert result["data"]["target_start"] == "2026-07-01T00:00:00Z"
    assert result["data"]["target_end"] == "2026-08-01T00:00:00Z"
    required = [
        "prospective_results.json", "qualitative_examples.csv",
        "figures/regime_performance.png", "figures/prospective_confusion.png",
        "figures/model_regime_diagram.png",
    ]
    assert all((tmp_path / name).is_file() for name in required)
```

- [ ] **Step 2: Run tests and verify the missing prospective API failure**

Run: `.venv/bin/python -m pytest tests/test_final_experiment.py -v`

Expected: FAIL because `run_prospective_stage` is not defined.

- [ ] **Step 3: Add deterministic regime-aware qualitative selection**

Add to `src/evaluation.py`:

```python
def representative_predictions_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    regimes: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for regime in ("low", "medium", "high"):
        mask = regimes == regime
        if not mask.any():
            continue
        selected = representative_predictions(
            y_true[mask], y_pred[mask], probabilities[mask],
            timestamps[mask], symbols[mask], examples_per_type=1,
        )
        selected.insert(0, "regime", regime)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
```

- [ ] **Step 4: Implement guarded prospective scoring**

In `run_final_experiment.py`, add:

```python
def run_prospective_stage(
    output_dir: Path | str = Path("artifacts/final"),
    dry_run: bool = False,
) -> dict:
    """Reveal and score July targets once using a complete frozen package."""
```

The implementation must:

1. Call `load_frozen_package(output_dir / "frozen")` before loading any market data.
2. Verify the manifest contains the exact prospective start/end constants and 13-feature schema.
3. For genuine data, fetch enough causal context beginning `2026-06-24T00:00:00Z` and ending exactly `2026-08-01T01:00:00Z`; the extra hour supplies the close needed to label the July 31 23:00 forecast origin. For dry runs, generate at least 1,100 deterministic hourly rows and shift the index to cover the same period.
4. Engineer features, then call `prepare_evaluation_windows` with only frozen scaler/threshold/window metadata.
5. Assert every scored timestamp is `>= PROSPECTIVE_START` and `< PROSPECTIVE_END`.
6. Assign regimes with only frozen per-symbol thresholds.
7. Predict with frozen baselines and `predict_probabilities`; never call `train_model`.
8. Save slice metrics, target bounds, sample counts, and `"prospective_revealed": true`.
9. Save deterministic qualitative examples by regime.
10. Generate three figures from saved results: compact regime macro-F1 comparison, selected-model prospective confusion matrix, and a model-plus-regime data-flow diagram.

Add a `prospective` CLI subcommand with `--output-dir` and `--dry-run`. Do not provide any tuning flags.

- [ ] **Step 5: Run the prospective dry run and full test suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all existing and new tests pass.

- [ ] **Step 6: Commit prospective scoring and figures**

```bash
git add src/evaluation.py src/final_evaluation.py run_final_experiment.py tests/test_final_experiment.py
git commit -m "feat: score frozen models on prospective data"
```

---

### Task 7: Genuine Runs, Four-Page Final Report, and Reproduction Docs

**Files:**
- Create: `final_report.tex`
- Create: `final_report.pdf`
- Modify: `refs.bib`
- Modify: `README.md`
- Modify: `.github/workflows/latex-compile.yml`
- Modify: `tests/test_report.py`
- Generate: `artifacts/final/**`

**Interfaces:**
- Consumes: genuine historical and prospective artifacts produced by Tasks 5–6 and the course rubric at `/Users/bernie/Downloads/final_report.pdf`.
- Produces: the submission-ready PDF and repository reproduction instructions.

- [ ] **Step 1: Run and freeze the genuine historical experiment before accessing July data**

Run:

```bash
.venv/bin/python run_final_experiment.py historical --output-dir artifacts/final --epochs 12
```

Expected: command exits zero; `artifacts/final/frozen/manifest.json`, `historical_results.json`, and `cross_regime_results.csv` exist; the manifest records seeds 42–44 selection evidence and the prospective interval; no prospective artifact exists yet.

- [ ] **Step 2: Commit the frozen protocol evidence before the reveal**

Stage only small, report-critical metadata and results; do not stage ignored raw data or large model checkpoints:

```bash
git add artifacts/final/historical_results.json artifacts/final/cross_regime_results.csv artifacts/final/frozen/manifest.json
git commit -m "exp: freeze historical regime protocol"
```

- [ ] **Step 3: Run the one-time genuine July prospective evaluation**

Run:

```bash
.venv/bin/python run_final_experiment.py prospective --output-dir artifacts/final
```

Expected: command exits zero; all scored timestamps lie in July 2026; `prospective_results.json`, `qualitative_examples.csv`, and three final figures exist. Do not change the model or preprocessing after this command.

- [ ] **Step 4: Write failing final-report integrity tests**

Add tests to `tests/test_report.py` that assert:

```python
def test_final_report_has_every_rubric_section() -> None:
    text = Path("final_report.tex").read_text()
    for heading in [
        "Introduction", "Related Work", "Data Processing", "Architecture",
        "Baseline Models", "Quantitative Results", "Qualitative Results",
        "New Data", "Discussion", "Ethical Considerations",
    ]:
        assert heading in text


def test_final_report_uses_genuine_saved_results() -> None:
    historical = json.loads(Path("artifacts/final/historical_results.json").read_text())
    prospective = json.loads(Path("artifacts/final/prospective_results.json").read_text())
    assert historical["configuration"]["dry_run"] is False
    assert prospective["configuration"]["dry_run"] is False
    assert prospective["configuration"]["prospective_revealed"] is True
    text = Path("final_report.tex").read_text()
    assert f'{prospective["slices"]["overall"]["models"]["cnn_lstm"]["macro_f1"]:.3f}' in text


def test_final_report_has_four_main_pages_plus_references() -> None:
    reader = PdfReader("final_report.pdf")
    assert "Discussion" in (reader.pages[3].extract_text() or "")
    assert "References" in (reader.pages[4].extract_text() or "")
```

Run: `.venv/bin/python -m pytest tests/test_report.py -v`

Expected: FAIL because `final_report.tex` and `final_report.pdf` do not exist.

- [ ] **Step 5: Write the report from saved artifacts only**

Create `final_report.tex` with the course template and numeric citations. Fit these sections into exactly four main-text pages:

- Page 1: introduction, five concise related works, and `model_regime_diagram.png`.
- Page 2: Binance data, causal features/splits/regimes, CNN-LSTM architecture/training, and all three baselines.
- Page 3: one compact table for historical/transfer/prospective macro-F1 and accuracy, plus `regime_performance.png`.
- Page 4: qualitative examples, new-data interpretation, discussion, limitations, ethical considerations, and future work.

Use `\clearpage` before `\bibliography{refs}`. Report unfavorable results honestly, state that July data was revealed only after freezing, and never claim profitability.

- [ ] **Step 6: Ensure at least five related works are cited and update README/CI**

Use the five existing bibliography entries in `refs.bib`, correcting bibliographic metadata if verification shows an error. Update `README.md` with:

````markdown
## Final evaluation

```bash
python run_final_experiment.py historical --output-dir artifacts/final --epochs 12
python run_final_experiment.py prospective --output-dir artifacts/final
```

The prospective command requires the frozen package created by the historical command and scores only July 2026 targets. Do not rerun it as a tuning loop.
````

Update `.github/workflows/latex-compile.yml` to trigger on `final_report.tex`, compile `final_report.tex`, upload `final-report-pdf`, and point to `final_report.pdf`.

- [ ] **Step 7: Compile and verify the report**

Run:

```bash
tectonic final_report.tex
.venv/bin/python -m pytest tests/test_report.py -v
```

Expected: compilation succeeds; all report tests pass; PDF contains four main-text pages followed by references.

- [ ] **Step 8: Commit genuine artifacts and final report**

```bash
git add final_report.tex final_report.pdf refs.bib README.md .github/workflows/latex-compile.yml tests/test_report.py artifacts/final
git commit -m "docs: add final cross-regime report"
```

If ignored checkpoint files appear in `git status`, leave them untracked/ignored; the reproducible manifest and result artifacts are the committed evidence.

---

### Task 8: Final Verification and Submission Audit

**Files:**
- Verify all changed files; modify only if a verification failure identifies a concrete defect.

**Interfaces:**
- Consumes: completed repository and final PDF.
- Produces: evidence that code, artifacts, report claims, page count, and repository state are submission-ready.

- [ ] **Step 1: Run the entire automated suite**

Run: `.venv/bin/python -m pytest -v`

Expected: all existing and new tests pass with zero failures.

- [ ] **Step 2: Run a clean dry-run reproduction outside genuine artifacts**

Run:

```bash
dry_dir=$(mktemp -d /tmp/aps360-final-dry.XXXXXX)
.venv/bin/python run_final_experiment.py historical --output-dir "$dry_dir" --dry-run --epochs 1
.venv/bin/python run_final_experiment.py prospective --output-dir "$dry_dir" --dry-run
```

Expected: both commands exit zero and produce the required historical, frozen, prospective, qualitative, and figure artifacts without network access.

- [ ] **Step 3: Recompile and inspect PDF metadata**

Run:

```bash
tectonic final_report.tex
pdfinfo final_report.pdf
```

Expected: compilation succeeds; PDF page count equals four main-text pages plus reference pages; no fifth main-text page exists.

- [ ] **Step 4: Render every PDF page and inspect visually**

Run:

```bash
render_dir=$(mktemp -d /tmp/aps360-final-report.XXXXXX)
pdftoppm -png -r 150 final_report.pdf "$render_dir/page"
```

Inspect every rendered page for clipped text, overlapping elements, unreadable figures, broken citations, empty sections, and incorrect page transitions. Expected: zero visual defects.

- [ ] **Step 5: Cross-check metrics and repository state**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

text = Path("final_report.tex").read_text()
for name in ["historical_results.json", "prospective_results.json"]:
    data = json.loads((Path("artifacts/final") / name).read_text())
    assert data["configuration"]["dry_run"] is False
assert "not trading advice" in text.lower()
print("genuine artifacts and responsible-use statement verified")
PY
git status --short
git log --oneline --decorate -12
```

Expected: verification message prints; only intentional files are changed; commits show design, plan, implementation, frozen protocol, and final report history.

- [ ] **Step 6: Submit early and preserve the evidence**

Upload `final_report.pdf` to Quercus no later than August 5 if possible. Confirm the Quercus-recorded timestamp and download the submitted copy once to ensure it opens. Do not make post-reveal model changes; any correction after this point must be limited to factual writing, formatting, or code defects and must be documented.

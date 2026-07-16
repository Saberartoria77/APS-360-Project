# APS360 Progress Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a reproducible Binance-to-CNN-LSTM pipeline that produces genuine metrics, figures, a Colab-ready notebook, and a three-page APS360 progress-report draft.

**Architecture:** Focused Python modules will separate causal data preparation, sequence construction, models, evaluation, and experiment orchestration. A thin notebook will install dependencies and call the same tested modules used locally. The LaTeX report will read saved JSON/CSV/PNG artifacts rather than contain invented or manually transcribed results.

**Tech Stack:** Python 3.10+, pandas, NumPy, requests, scikit-learn, PyTorch, matplotlib, seaborn, pytest, Jupyter, LaTeX with `APS360.sty`.

## Global Constraints

- Use hourly `BTCUSDT` and `ETHUSDT` Binance public klines for approximately three years.
- Use present and past observations only for features; fit thresholds and standardization on training data only.
- Use 96-hour model input windows and chronological train/validation/test splits.
- Evaluate momentum, logistic regression, and one compact CNN-LSTM on identical held-out samples.
- Save accuracy, macro-F1, per-class recall, confusion matrices, learning curves, parameter count, and representative predictions.
- Keep progress-report main text at no more than three pages; references must begin afterward.
- Do not make profitability claims from classification accuracy, fabricate results, include credentials, or conceal assistance.

---

### Task 1: Causal Market-Data Pipeline

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data.py`
- Create: `tests/test_data.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Binance `/api/v3/klines` JSON arrays or cached CSV files.
- Produces: `fetch_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "1h") -> pd.DataFrame`, `engineer_features(frame: pd.DataFrame) -> pd.DataFrame`, and `make_direction_labels(close: pd.Series, horizon: int, threshold: float) -> pd.Series`.

- [ ] **Step 1: Add dependencies and failing causal-feature tests**

```text
numpy>=2.0,<3
pandas>=2.2,<3
requests>=2.32,<3
scikit-learn>=1.5,<2
torch>=2.3,<3
matplotlib>=3.9,<4
seaborn>=0.13,<1
pytest>=8,<9
jupyter>=1,<2
```

```python
def test_features_are_causal(sample_ohlcv):
    original = engineer_features(sample_ohlcv)
    changed = sample_ohlcv.copy()
    changed.loc[changed.index[-1], "close"] *= 10
    revised = engineer_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])

def test_labels_use_exact_future_horizon():
    close = pd.Series([100.0, 101.0, 103.0, 102.0])
    labels = make_direction_labels(close, horizon=1, threshold=0.015)
    assert labels.iloc[:3].tolist() == [1, 2, 0]
    assert pd.isna(labels.iloc[3])
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m pytest tests/test_data.py -v`

Expected: FAIL because `src.data` does not exist.

- [ ] **Step 3: Implement paginated collection, validation, indicators, and labels**

```python
FEATURE_COLUMNS = [
    "return_1h", "log_return_1h", "high_low_range", "close_open_return",
    "volume_change", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width", "volatility_24h", "volume_z_24h",
]

def make_direction_labels(close, horizon=1, threshold=0.0025):
    future_return = close.shift(-horizon).div(close).sub(1.0)
    labels = pd.Series(np.nan, index=close.index)
    labels.loc[future_return < -threshold] = 0
    labels.loc[future_return.abs() <= threshold] = 1
    labels.loc[future_return > threshold] = 2
    return labels
```

The implementation must parse millisecond timestamps as UTC, cast OHLCV to floats, reject duplicate/nonmonotonic timestamps, calculate RSI with Wilder-style exponentially weighted gains/losses, MACD with 12/26 EMAs and a 9-period signal, and Bollinger terms with a 20-hour rolling window. API requests must use `limit=1000`, advance from the last returned open time, retry three times with exponential backoff, and stop at `end`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_data.py -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Ignore generated bulk artifacts and commit**

```gitignore
.venv/
__pycache__/
.pytest_cache/
data/raw/
data/processed/
artifacts/checkpoints/
*.pyc
```

```bash
git add requirements.txt src/__init__.py src/data.py tests/test_data.py .gitignore
git commit -m "feat: add causal Binance data pipeline"
```

### Task 2: Chronological Sequences and Leakage-Safe Splits

**Files:**
- Modify: `src/data.py`
- Create: `tests/test_sequences.py`

**Interfaces:**
- Consumes: engineered frames containing `FEATURE_COLUMNS`, `label`, and UTC timestamp index.
- Produces: `prepare_datasets(frames: dict[str, pd.DataFrame], window: int = 96, train_fraction: float = 0.70, val_fraction: float = 0.15) -> DatasetBundle`, with NumPy arrays `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test`, timestamps, symbols, feature names, scaler mean/scale, and threshold.

- [ ] **Step 1: Write split, scaling, and window-alignment tests**

```python
def test_split_is_chronological(bundle):
    assert bundle.train_times.max() < bundle.val_times.min()
    assert bundle.val_times.max() < bundle.test_times.min()

def test_scaler_is_fit_on_training_rows(bundle):
    np.testing.assert_allclose(bundle.x_train.reshape(-1, bundle.x_train.shape[-1]).mean(0), 0, atol=1e-6)
    assert not np.allclose(bundle.x_test.reshape(-1, bundle.x_test.shape[-1]).mean(0), 0, atol=1e-2)

def test_window_label_alignment(bundle, source_frame):
    first_time = bundle.train_times[0]
    expected = source_frame.loc[first_time, "label"]
    assert bundle.y_train[0] == expected
    assert bundle.x_train.shape[1] == 96
```

- [ ] **Step 2: Run tests and verify interface failure**

Run: `python -m pytest tests/test_sequences.py -v`

Expected: FAIL because `prepare_datasets` and `DatasetBundle` are undefined.

- [ ] **Step 3: Implement global chronological boundaries and training-only scaling**

```python
@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_times: np.ndarray
    val_times: np.ndarray
    test_times: np.ndarray
    train_symbols: np.ndarray
    val_symbols: np.ndarray
    test_symbols: np.ndarray
    feature_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    threshold: float
```

Select boundaries from sorted unique timestamps. Calculate the flat threshold as the training-set median absolute one-hour future return, clipped to `[0.001, 0.01]`, then rebuild all labels with that frozen value. Fit mean and standard deviation using feature rows whose timestamps are in the training period only. Create windows separately per symbol so a window never crosses assets.

- [ ] **Step 4: Run sequence and full data tests**

Run: `python -m pytest tests/test_data.py tests/test_sequences.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_sequences.py
git commit -m "feat: add chronological sequence preparation"
```

### Task 3: Baselines and Shared Evaluation

**Files:**
- Create: `src/evaluation.py`
- Create: `src/baselines.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `DatasetBundle` arrays and predictions ordered `[down, flat, up]`.
- Produces: `fit_baselines(bundle: DatasetBundle) -> dict[str, np.ndarray]`, `classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict`, `save_confusion_matrix(...) -> Path`, and `representative_predictions(...) -> pd.DataFrame`.

- [ ] **Step 1: Write failing metric and baseline tests**

```python
def test_classification_metrics_has_required_fields():
    result = classification_metrics(np.array([0, 1, 2]), np.array([0, 2, 2]))
    assert set(result) >= {"accuracy", "macro_f1", "per_class_recall", "confusion_matrix"}
    assert result["accuracy"] == pytest.approx(2 / 3)

def test_baselines_return_one_prediction_per_test_sample(bundle):
    predictions = fit_baselines(bundle)
    assert set(predictions) == {"momentum", "logistic_regression"}
    assert all(len(values) == len(bundle.y_test) for values in predictions.values())
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m pytest tests/test_evaluation.py -v`

Expected: FAIL because evaluation and baseline modules do not exist.

- [ ] **Step 3: Implement deterministic baselines and shared metrics**

```python
def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
```

Momentum uses the last standardized `return_1h` feature sign with a standardized equivalent of the frozen threshold. Logistic regression uses the final timestep feature vector, `class_weight="balanced"`, `max_iter=1000`, and `random_state=42`.

- [ ] **Step 4: Run evaluation tests**

Run: `python -m pytest tests/test_evaluation.py -v`

Expected: all tests PASS and plots are created in a temporary directory.

- [ ] **Step 5: Commit**

```bash
git add src/baselines.py src/evaluation.py tests/test_evaluation.py
git commit -m "feat: add baselines and evaluation artifacts"
```

### Task 4: Compact CNN-LSTM and Training Loop

**Files:**
- Create: `src/models.py`
- Create: `src/training.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `DatasetBundle` float arrays shaped `[samples, 96, features]`.
- Produces: `CNNLSTM(n_features: int, conv_channels: int = 32, hidden_size: int = 48, num_classes: int = 3)`, `train_model(bundle: DatasetBundle, config: TrainConfig) -> TrainResult`, probabilities, predictions, best state, parameter count, and epoch history.

- [ ] **Step 1: Write failing tensor-shape and training-smoke tests**

```python
def test_cnn_lstm_output_shape():
    model = CNNLSTM(n_features=13)
    logits = model(torch.randn(8, 96, 13))
    assert logits.shape == (8, 3)

def test_training_smoke_run(tiny_bundle):
    result = train_model(tiny_bundle, TrainConfig(epochs=2, batch_size=16, seed=42))
    assert len(result.history["train_loss"]) == 2
    assert result.test_probabilities.shape == (len(tiny_bundle.y_test), 3)
    np.testing.assert_allclose(result.test_probabilities.sum(1), 1, atol=1e-5)
```

- [ ] **Step 2: Run tests and verify model failure**

Run: `python -m pytest tests/test_model.py -v`

Expected: FAIL because `CNNLSTM`, `TrainConfig`, and `train_model` are undefined.

- [ ] **Step 3: Implement the compact architecture and deterministic loop**

```python
class CNNLSTM(nn.Module):
    def __init__(self, n_features, conv_channels=32, hidden_size=48, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            nn.Dropout(0.15),
        )
        self.lstm = nn.LSTM(conv_channels, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])
```

Use Adam with learning rate `1e-3`, gradient clipping at `1.0`, optional balanced training-class weights, validation-loss checkpoint selection, fixed seeds, and MPS/CUDA/CPU auto-detection. The initial experiment uses at most 12 epochs and patience 3.

- [ ] **Step 4: Run all model tests**

Run: `python -m pytest tests/test_model.py -v`

Expected: both shape and two-epoch smoke tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models.py src/training.py tests/test_model.py
git commit -m "feat: add reproducible CNN-LSTM training"
```

### Task 5: End-to-End Experiment and Colab Notebook

**Files:**
- Create: `run_experiment.py`
- Create: `notebooks/progress_report.ipynb`
- Create: `tests/test_experiment.py`
- Create: `artifacts/figures/.gitkeep`
- Create: `artifacts/metrics/.gitkeep`

**Interfaces:**
- Consumes: Tasks 1-4 APIs and optional cached data.
- Produces: `artifacts/metrics/results.json`, `artifacts/metrics/prediction_examples.csv`, `artifacts/figures/data_summary.png`, three confusion matrices, `learning_curves.png`, and `model_diagram.png`.

- [ ] **Step 1: Write a failing dry-run artifact test**

```python
def test_experiment_dry_run_writes_required_artifacts(tmp_path):
    result = run_experiment(output_dir=tmp_path, dry_run=True, epochs=1)
    required = [
        "metrics/results.json", "metrics/prediction_examples.csv",
        "figures/data_summary.png", "figures/confusion_momentum.png",
        "figures/confusion_logistic_regression.png", "figures/confusion_cnn_lstm.png",
        "figures/learning_curves.png", "figures/model_diagram.png",
    ]
    assert all((tmp_path / name).exists() for name in required)
    assert result["configuration"]["dry_run"] is True
```

- [ ] **Step 2: Run the dry-run test and verify failure**

Run: `python -m pytest tests/test_experiment.py -v`

Expected: FAIL because `run_experiment` is undefined.

- [ ] **Step 3: Implement orchestration and notebook**

```python
def run_experiment(output_dir=Path("artifacts"), dry_run=False, epochs=12):
    """Collect/cache data, prepare samples, fit all models, save every report artifact, and return results."""
```

The notebook must contain: an academic-use note; a Colab dependency-install cell; repository clone/import setup; configuration; data collection and statistics; baseline execution; CNN-LSTM training; evaluation; artifact display; and a final checklist asking the student to verify metrics and revise report prose. It must call `run_experiment` rather than duplicate implementation logic.

- [ ] **Step 4: Verify dry run, then execute the genuine experiment**

Run: `python -m pytest tests/test_experiment.py -v`

Expected: PASS.

Run: `python run_experiment.py --start 2023-07-01 --end 2026-07-01 --epochs 12`

Expected: exit code 0; `results.json` reports nonzero train/validation/test counts, all three models, and a positive CNN-LSTM parameter count.

- [ ] **Step 5: Audit artifact consistency and commit**

Run: `python -m json.tool artifacts/metrics/results.json >/dev/null && test -s artifacts/metrics/prediction_examples.csv`

Expected: exit code 0.

```bash
git add run_experiment.py notebooks/progress_report.ipynb tests/test_experiment.py artifacts/figures artifacts/metrics
git commit -m "feat: run progress report experiment"
```

### Task 6: Three-Page LaTeX Report and Reproducibility Documentation

**Files:**
- Create: `progress_report.tex`
- Create: `progress_refs.bib`
- Modify: `README.md`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: generated metrics/CSV/PNG artifacts and supplied `APS360.sty`.
- Produces: `progress_report.pdf` with at most three pages of main text plus references and a README with local and Colab reproduction paths.

- [ ] **Step 1: Write failing report-integrity tests**

```python
def test_report_has_required_sections():
    text = Path("progress_report.tex").read_text()
    for heading in ["Brief Project Description", "Data Processing", "Baseline Model", "Primary Model"]:
        assert heading in text

def test_report_numbers_come_from_saved_results():
    text = Path("progress_report.tex").read_text()
    results = json.loads(Path("artifacts/metrics/results.json").read_text())
    for model in ["momentum", "logistic_regression", "cnn_lstm"]:
        assert f'{results["models"][model]["accuracy"]:.3f}' in text
```

- [ ] **Step 2: Run report tests and verify missing-source failure**

Run: `python -m pytest tests/test_report.py -v`

Expected: FAIL because `progress_report.tex` is not present.

- [ ] **Step 3: Draft the report from verified artifacts**

```latex
\section{Brief Project Description}
\section{Notable Contribution}
\subsection{Data Processing}
\subsection{Baseline Model}
\subsection{Primary Model}
\clearpage
\bibliographystyle{plainnat}
\bibliography{progress_refs}
```

The draft must state input `[96 hours x 13 features]`, output `[down, flat, up]`, exact date range and sample/class counts, causal preprocessing, held-out data plan, baseline definitions, model layer sizes and parameter count, genuine test metrics, selected prediction behavior, challenges, limitations, and feasibility argument. It must label performance as a classification feasibility result and not trading advice. Use the stable Colab URL `https://colab.research.google.com/github/Saberartoria77/APS-360-Project/blob/main/notebooks/progress_report.ipynb`; the student must confirm it opens after the notebook is pushed.

- [ ] **Step 4: Compile, render, and inspect page count**

Run: `latexmk -pdf -interaction=nonstopmode -halt-on-error progress_report.tex`

Expected: exit code 0.

Run: `pdfinfo progress_report.pdf | rg '^Pages:'`

Expected: total pages may exceed three only because references begin on page 4; main content ends by page 3.

- [ ] **Step 5: Run all tests and verify the report visually**

Run: `python -m pytest -v`

Expected: all tests PASS.

Render each report page to PNG and inspect for clipped figures, unreadable labels, blank pages, and references intruding into the three main pages. Correct layout defects in `progress_report.tex` and rerun compilation/tests.

- [ ] **Step 6: Document reproduction and commit**

README instructions must include a local virtual-environment setup, `python run_experiment.py` command, notebook/Colab path, artifact inventory, data-source disclosure, and the requirement that submitted claims be personally verified.

```bash
git add progress_report.tex progress_report.pdf progress_refs.bib README.md tests/test_report.py
git commit -m "docs: add verified progress report draft"
```

### Task 7: Final Submission Audit

**Files:**
- Modify only files with verified audit failures.

**Interfaces:**
- Consumes: the complete repository.
- Produces: a clean, reproducible handoff with no false completion claims.

- [ ] **Step 1: Run fresh verification**

Run: `python -m pytest -v`

Expected: all tests PASS with zero failures.

- [ ] **Step 2: Check repository hygiene**

Run: `git status --short && git ls-files | rg '(^data/|checkpoint|\.env$)' || true`

Expected: only intentional small summary artifacts are tracked; no credentials, bulk raw data, or checkpoints appear.

- [ ] **Step 3: Cross-check every report number**

Run: `python -m pytest tests/test_report.py -v`

Expected: all report-integrity tests PASS.

- [ ] **Step 4: Record the student verification checklist**

The handoff must ask the student to restart and run the Colab notebook, confirm the stable Colab URL opens, inspect saved examples and metrics, revise wording into claims they understand, comply with course disclosure rules, and submit before the Quercus deadline.

- [ ] **Step 5: Commit audit fixes, if any**

```bash
git add -u
git commit -m "chore: complete submission audit"
```

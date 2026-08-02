# Cross-Regime Generalization in Cryptocurrency Direction Prediction

An APS360 project studying the feasibility and robustness of next-hour BTC/ETH direction classification with a hybrid CNN-LSTM. The final evaluation measures historical volatility-regime transfer and a frozen one-time prospective test on July 2026.

## Final evaluation

The genuine July evaluation is complete and sealed. Do **not** run the genuine
`historical` or `prospective` commands against `artifacts/final/` again. The durable
`.prospective-reveal.json` marker makes a second reveal fail before data access, and
`prospective_artifact_index.json` binds the saved JSON, CSV, and figures by SHA-256.

For a network-free smoke test, use a new output directory each time:

```bash
python run_final_experiment.py historical --output-dir /tmp/aps360-final-dry --dry-run --epochs 1
python run_final_experiment.py prospective --output-dir /tmp/aps360-final-dry --dry-run
```

The prospective command requires the frozen package created by the historical command and scores only July 2026 forecast origins. Synthetic and genuine state cannot share an output directory. The maintenance-only `recompute-transfer` stage reads pre-July caches and can replace only the historical transfer JSON/CSV; it never loads prospective data or saves frozen weights.

The genuine frozen CNN-LSTM reached 0.409 macro-F1 and 0.489 accuracy on 1,488 July targets. Its historical three-seed macro-F1 was $0.429\pm0.004$; explicit low/high regime transfer produced negative paired changes in both directions. See `final_report.pdf` for definitions, baselines, uncertainty, limitations, and the full interpretation.

The report-ready evidence is stored under `artifacts/final/`: the frozen manifest,
the committed `frozen/cnn_lstm.pt` and `frozen/baselines.npz` checkpoints,
historical and prospective JSON, cross-regime CSV, qualitative examples, artifact
index, reveal marker, and final figures.

To compile the four-page course report plus references:

```bash
tectonic final_report.tex
```

The submission artifact is `final_report.pdf`.

## Earlier progress checkpoint

The earlier progress-report pipeline:

- downloads hourly BTCUSDT and ETHUSDT OHLCV data from the public Binance API;
- builds 13 causal return, volume, RSI, MACD, Bollinger, and volatility features;
- creates 96-hour windows and down/flat/up next-hour labels;
- uses chronological 70/15/15 splits with training-only thresholds and normalization;
- evaluates momentum and logistic-regression baselines;
- trains an 18,067-parameter PyTorch CNN-LSTM with validation-based early stopping;
- saves metrics, confusion matrices, learning curves, and qualitative prediction examples.

That historical checkpoint used the same July 2023–June 2026 source range. On 7,702 held-out windows, logistic regression reached 48.0% accuracy and the single-seed checkpoint CNN-LSTM reached 44.4% accuracy (42.5% macro-F1). These are preserved progress-report results, not the final three-seed or prospective metrics above.

## Run in Google Colab

[Open the progress-report notebook in Google Colab](https://colab.research.google.com/github/Saberartoria77/APS-360-Project/blob/main/notebooks/progress_report.ipynb).

Restart the runtime, run every cell, and verify that `artifacts/metrics/results.json` reports `"dry_run": false` before relying on its outputs.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py --start 2023-07-01 --end 2026-07-01 --epochs 12
```

The collector caches downloaded CSV files under `data/raw/`, which is intentionally excluded from Git. Subsequent runs reuse that cache.

To compile the archived progress report with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
tectonic progress_report.tex
```

The archived progress artifact is `progress_report.pdf`; the current submission artifact is `final_report.pdf`.

## Repository map

- `notebooks/progress_report.ipynb` — Colab-ready end-to-end workflow.
- `run_experiment.py` — experiment orchestration and artifact generation.
- `run_final_experiment.py` — guarded historical freeze and one-time prospective evaluation.
- `src/data.py` — Binance collection, causal features, labels, splits, and windows.
- `src/baselines.py` — momentum and logistic-regression baselines.
- `src/models.py` — CNN-LSTM architecture.
- `src/training.py` — deterministic training, validation, and early stopping.
- `src/evaluation.py` — metrics, confusion matrices, and qualitative examples.
- `artifacts/metrics/` — genuine run configuration, metrics, and selected predictions.
- `artifacts/figures/` — report-ready generated figures.
- `progress_report.tex`, `progress_report.pdf` — APS360 progress-report source and compiled draft.
- `artifacts/final/` — genuine frozen manifest, final metrics, examples, and figures.
- `final_report.tex`, `final_report.pdf` — final-report source and submission PDF.
- `tests/` — leakage, alignment, model-shape, artifact, notebook, and report-integrity checks.

## Verification

```bash
python -m pytest -v
```

The report-integrity tests cross-check its headline accuracy and macro-F1 values against the saved genuine experiment output.

## Responsible use

This is a classification feasibility study, not trading advice. Accuracy is not profitability, and the results should not be used to make financial decisions. Before academic submission, the student must rerun or inspect the notebook, understand the implementation and results, revise claims into wording they can defend, and follow course rules for acknowledging external or AI assistance.

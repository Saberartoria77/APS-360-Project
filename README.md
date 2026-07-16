# Cross-Regime Generalization in Cryptocurrency Direction Prediction

An APS360 project studying the feasibility and robustness of next-hour BTC/ETH direction classification with a hybrid CNN-LSTM. The long-term question is how predictive performance changes under market regime shift; the current repository contains the completed progress-report checkpoint.

## Progress checkpoint

The reproducible pipeline currently:

- downloads hourly BTCUSDT and ETHUSDT OHLCV data from the public Binance API;
- builds 13 causal return, volume, RSI, MACD, Bollinger, and volatility features;
- creates 96-hour windows and down/flat/up next-hour labels;
- uses chronological 70/15/15 splits with training-only thresholds and normalization;
- evaluates momentum and logistic-regression baselines;
- trains an 18,067-parameter PyTorch CNN-LSTM with validation-based early stopping;
- saves metrics, confusion matrices, learning curves, and qualitative prediction examples.

The genuine July 2023–June 2026 run contains 52,608 raw hourly rows. On 7,702 held-out windows, logistic regression reached 48.0% accuracy and the CNN-LSTM reached 44.4% accuracy (42.5% macro-F1). The neural model did not yet beat the strongest baseline, but improved recall for directional classes and demonstrated that the complete neural pipeline is feasible. Full regime-shift testing is intentionally deferred to the final project.

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

To compile the course-format report with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
tectonic progress_report.tex
```

The compiled submission draft is `progress_report.pdf`. Its main text is three pages; references occupy page 4.

## Repository map

- `notebooks/progress_report.ipynb` — Colab-ready end-to-end workflow.
- `run_experiment.py` — experiment orchestration and artifact generation.
- `src/data.py` — Binance collection, causal features, labels, splits, and windows.
- `src/baselines.py` — momentum and logistic-regression baselines.
- `src/models.py` — CNN-LSTM architecture.
- `src/training.py` — deterministic training, validation, and early stopping.
- `src/evaluation.py` — metrics, confusion matrices, and qualitative examples.
- `artifacts/metrics/` — genuine run configuration, metrics, and selected predictions.
- `artifacts/figures/` — report-ready generated figures.
- `progress_report.tex`, `progress_report.pdf` — APS360 progress-report source and compiled draft.
- `tests/` — leakage, alignment, model-shape, artifact, notebook, and report-integrity checks.

## Verification

```bash
python -m pytest -v
```

The report-integrity tests cross-check its headline accuracy and macro-F1 values against the saved genuine experiment output.

## Responsible use

This is a classification feasibility study, not trading advice. Accuracy is not profitability, and the results should not be used to make financial decisions. Before academic submission, the student must rerun or inspect the notebook, understand the implementation and results, revise claims into wording they can defend, and follow course rules for acknowledging external or AI assistance.

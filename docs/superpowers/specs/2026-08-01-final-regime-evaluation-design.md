# Final Regime Evaluation Design

## Objective

Complete a focused, rigorous APS360 final project by August 6, 2026, and leave behind a portfolio-quality data science and machine learning case study. The final scientific question is:

> How much does next-hour BTC/ETH direction-classification performance change across volatility regimes and on a genuinely unseen future month?

The work must satisfy the final-report rubric while preserving the existing progress-report pipeline and its leakage controls.

## Success Criteria

The project is complete when:

1. The existing majority, momentum, logistic-regression, and CNN-LSTM models are evaluated with the same labels, windows, and metrics.
2. Volatility regimes are defined causally using thresholds learned only from historical training data.
3. Low-to-high and high-to-low regime transfer are measured on the historical held-out split.
4. The final CNN-LSTM and baselines are frozen before July 2026 data is downloaded or inspected.
5. The frozen models are evaluated once on July 1 through July 31, 2026 data, with no subsequent tuning.
6. CNN-LSTM variability is reported across three fixed random seeds.
7. Results, figures, and qualitative examples are generated reproducibly from saved artifacts.
8. All automated tests pass and every number in the four-page final report is traceable to a saved result.

## Scope

### Included

- One-hour-ahead, three-class prediction: down, flat, or up.
- BTCUSDT and ETHUSDT hourly data.
- Existing 96-hour windows and 13 causal input features.
- A training-derived majority-class baseline, momentum, logistic regression, and the existing CNN-LSTM.
- Historical low/medium/high volatility analysis.
- Explicit low-regime versus high-regime transfer experiments.
- Three CNN-LSTM seeds: 42, 43, and 44.
- A genuinely unseen July 2026 prospective evaluation.
- Final-report figures, tables, qualitative examples, limitations, and ethical discussion.

### Excluded

- Additional prediction horizons.
- Transformers or additional neural architectures.
- Trading simulations, transaction costs, or profitability claims.
- Hyperparameter searches after the design is frozen.
- A dashboard before the course submission.

Portfolio README improvements and an optional interactive dashboard are post-submission work.

## Data Protocol

### Historical development data

- Source: Binance public hourly klines for BTCUSDT and ETHUSDT.
- Range: July 1, 2023 through June 30, 2026.
- Existing chronological 70/15/15 train/validation/test partitions remain unchanged.
- Feature normalization and the down/flat/up return threshold are fitted on training data only.
- Hyperparameters and seed selection use validation results only.
- Historical test results are used for final analysis, not model selection.

### Genuinely unseen prospective data

- Scored range: July 1, 2026 00:00 UTC through July 31, 2026 23:00 UTC.
- Pre-July observations may be downloaded only as causal context for the first 96-hour windows and rolling indicators. No pre-July target is part of the prospective score.
- July data must not be downloaded, summarized, plotted, or inspected until the model artifacts and experiment configuration are frozen.
- After prospective results are revealed, no model, threshold, regime definition, feature, or selection rule may change.
- If data are missing, the pipeline reports the missing timestamps and scores only valid contiguous windows; it never fills future-aware values.

The repository currently has no local raw-data cache, so the prospective data has not been examined in this workspace as of August 1, 2026.

## Causal Regime Definition

For each symbol, compute trailing 168-hour realized volatility as the standard deviation of hourly log returns using observations available through the forecast origin. The regime value for a sample is attached at the end of its 96-hour input window; the next-hour outcome is never used.

For each symbol separately, calculate the 33rd and 67th percentiles of 168-hour volatility using historical training rows only:

- low: volatility at or below the 33rd percentile;
- medium: volatility between the thresholds;
- high: volatility above the 67th percentile.

The same frozen per-symbol thresholds classify validation, historical test, and July prospective samples. Per-symbol thresholds prevent asset identity from becoming a proxy for regime because ETH and BTC have different volatility scales.

## Experiment Matrix

### Experiment A: Global model robustness

Train each model on the complete historical training split, select neural checkpoints using the complete validation split, and evaluate the historical test predictions:

- overall;
- by low, medium, and high regime;
- separately for BTC and ETH.

This measures whether a generally trained model degrades in particular regimes.

### Experiment B: Explicit cross-regime transfer

Create low-regime and high-regime training/validation subsets using the causal forecast-origin regime labels. For the CNN-LSTM and logistic regression:

- train on low, test on low;
- train on low, test on high;
- train on high, test on high;
- train on high, test on low.

The primary transfer statistic is the macro-F1 change from the matching-regime evaluation to the opposite-regime evaluation. Medium-regime-specific training is excluded to keep the experiment focused.

### Experiment C: Prospective July evaluation

Before accessing July data, select one CNN-LSTM seed using minimum historical validation loss. Freeze its state dictionary and all preprocessing metadata. Evaluate the frozen majority, momentum, logistic-regression, and selected CNN-LSTM models once on July targets:

- overall;
- by volatility regime;
- separately for BTC and ETH when sample counts are sufficient.

The report will present the prospective result even if it is worse than expected. No July-dependent correction is allowed.

## Metrics and Selection Rules

### Quantitative metrics

- Primary metric: macro-F1, because the three classes are imbalanced.
- Secondary metrics: accuracy and down/flat/up recall.
- CNN-LSTM historical results: mean and standard deviation across seeds 42, 43, and 44.
- Deterministic baselines: one result per evaluation slice.
- Every result includes the number of evaluated samples and class counts.

The majority baseline predicts the most frequent historical training class. It is not hard-coded to flat.

### Qualitative examples

For the validation-selected CNN-LSTM, choose examples mechanically rather than editorially:

- highest-confidence correct prediction in each non-empty regime;
- highest-confidence error in each non-empty regime.

The report will show a compact subset that best demonstrates regime-dependent behavior and will state the selection rule. Saved CSV artifacts retain the full selected set.

## Components and Data Flow

### Regime module

A focused regime module will:

- compute trailing realized volatility;
- fit per-symbol training thresholds;
- assign regimes to samples by symbol and forecast origin;
- validate that no future timestamp contributes to a regime label.

### Dataset utilities

Dataset helpers will subset an existing `DatasetBundle` without changing window contents, timestamps, symbols, scaler parameters, or the label threshold. Empty or single-class subsets fail with a clear message before model fitting.

### Baselines

The baseline module will add a training-derived majority predictor and expose fitted logistic-regression models for reuse on prospective data. Momentum remains parameter-free apart from the frozen class threshold.

### Model persistence and inference

The final pipeline will save:

- CNN-LSTM state dictionary;
- selected seed and training configuration;
- feature order and window length;
- scaler mean and scale;
- direction-label threshold;
- per-symbol regime thresholds;
- development date bounds and prospective cutoff;
- validation selection evidence.

Loading rejects incompatible feature orders, tensor shapes, or missing metadata.

### Final experiment runner

A separate final runner will orchestrate historical training, regime evaluation, model freezing, and prospective scoring. Historical and prospective stages will be separate commands so July data cannot be fetched accidentally before the freeze artifact exists.

The prospective command refuses to run unless it finds a complete frozen manifest created by the historical command.

### Artifacts

The final run will create stable, report-ready outputs under `artifacts/final/`:

- `manifest.json`;
- `historical_results.json`;
- `cross_regime_results.csv`;
- `prospective_results.json`;
- `qualitative_examples.csv`;
- frozen model files;
- a compact model/regime diagram;
- a regime-performance comparison figure;
- a prospective confusion-matrix figure;
- a training-variability figure or table source.

## Error Handling and Integrity Guards

- Reject non-monotonic, duplicate, missing, or timezone-naive market timestamps.
- Reject regime thresholds fitted from anything except historical training samples.
- Reject prospective scoring before a frozen manifest exists.
- Reject prospective targets outside July 2026.
- Reject metadata whose feature order, window size, or model dimensions differ from the loaded model.
- Record library versions, configuration, seed, sample counts, and date bounds in saved artifacts.
- Never silently fall back to synthetic data during a genuine run.

## Testing Strategy

Tests will be written before implementation and will cover:

1. Regime labels are unchanged when future returns are modified.
2. Regime thresholds use training data only and are fitted separately by symbol.
3. Dataset subsets preserve sample alignment and metadata.
4. The majority baseline uses the training majority class.
5. The regime experiment matrix contains all four low/high train-test combinations.
6. Model persistence round-trips predictions within numerical tolerance.
7. Prospective scoring fails without a frozen manifest.
8. Prospective scoring includes only July target timestamps.
9. A dry run produces every required final artifact.
10. Final report metrics match the saved genuine results.

The existing 25 tests remain required.

## Final Report Plan

The main text is limited to four pages; references are unlimited.

- Page 1: introduction, at least five related works, and one clear model/regime diagram.
- Page 2: data processing, causal regime definition, architecture, training, and baselines.
- Page 3: historical, cross-regime, and prospective quantitative results.
- Page 4: qualitative examples, critical discussion, limitations, ethics, and future work.

Every rubric category must appear explicitly. The discussion will emphasize what changed across regimes, why the model did or did not beat simple baselines, and what was learned. Accuracy is not profitability, and no trading recommendation will be made.

## Deadline-Safe Schedule

- August 1: approve and commit this design; write the implementation plan.
- August 2: implement regime logic, majority baseline, persistence, and tests.
- August 3: implement the final runner, complete dry-run verification, run historical experiments, and freeze the selected model.
- August 4: download and score July data once; generate final tables and figures.
- August 5: write and compile the four-page report; cross-check all numbers.
- August 6: visual QA, complete rerun where feasible, submit early, and retain the remaining time only as a buffer.

The Quercus deadline timezone must be confirmed. Operationally, August 5 is treated as the target submission date.

## Post-Submission Portfolio Layer

After the course submission, convert the repository into an internship case study with a concise README, one prominent regime-shift visualization, clean reproduction commands, and short interview narratives. A dashboard is optional and must not compete with the final-report deadline.

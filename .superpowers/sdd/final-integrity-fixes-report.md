# Final Integration Integrity Fixes Report

Date: 2026-08-02 (Asia/Shanghai)

Base head: `073888b`

Final implementation head before this report: `0e54cf3`

## Scope and non-negotiable chronology

- The genuine prospective command was **not** invoked during this correction.
- Commit `c547192` was not amended or rewritten.
- The frozen global CNN-LSTM weights, fitted baselines, feature order, scaling,
  label threshold, regime thresholds, selected seed, and July interval were not
  tuned or changed.
- The only experiment rerun was the explicitly permitted historical-only
  transfer recomputation from cached data ending at `2026-07-01T00:00:00Z`
  exclusive.
- The final report remained exactly four main pages plus one references page.

## Commits

1. `9b4ec86` — `fix: bind frozen package to exact checkpoints`
2. `2fc2f6b` — `fix: enforce one-time paired evaluation integrity`
3. `18f17cd` — `feat: add historical-only transfer recomputation`
4. `c373c3b` — `exp: bind paired transfer evidence to checkpoints`
5. `e030cc0` — `docs: archive post-reveal prospective provenance`
6. `0e54cf3` — `docs: explain sealed final evaluation`

## Fix 1: portable frozen-state provenance

The exact small checkpoint binaries are now committed:

```text
670265c722336bfd4c1fb4d0ac035ed4fad9a657427f3d26733f96f433ffced6  artifacts/final/frozen/cnn_lstm.pt
8cb39d6b8ab30c7482ed83ebc0d4840f52c6bff5591f2089cd29ba2dcf59f84f  artifacts/final/frozen/baselines.npz
```

`FrozenManifest` schema v2 records both digests. Loading validates both files
before PyTorch or NumPy deserialization. Saving writes staged model and baseline
files, computes their digests, then publishes an fsync'd manifest atomically and
last. Tests cover round-trip equality, tampering before deserialization, missing
or invalid digests, and committed-package loading.

The genuine prospective result now contains a clearly labelled
`post-reveal archival correction`. It names the original historical freeze
commit and does not claim the hashes were present in `c547192`.

## Fix 2: durable one-time and output-mode protocol

- Genuine historical execution refuses an output directory containing a reveal
  marker or prospective artifacts before fetching or training.
- Genuine prospective execution validates the package, then exclusively creates
  and fsyncs `.prospective-reveal.json` before market-data access. A failed run
  retains this marker.
- Dry runs refuse the canonical genuine directory and any directory containing
  genuine state.
- Repeated prospective runs refuse before package loading, fetching, or training.
- Prospective JSON, qualitative CSV, and all three figures are first written to
  a temporary directory. They are validated and bound by SHA-256/size in
  `prospective_artifact_index.json`, then published.
- Synthetic and genuine states cannot share an output directory.

The committed genuine artifact index validates all five published files. The
controlled prospective JSON file hash changed only because archival provenance
was added:

```text
before file SHA-256: 396cba4e92043c820a520d6eed0d06cd8b5c9ab8e1c6c121c4d6de7e8550cc89
after  file SHA-256: 5b0cca2c744e1e6940934a5124064db71553ff7db06819bbaba8de8defd69585
```

The canonical sorted JSON hash of `{data, selected_model, slices}` is unchanged:

```text
cbc08e362e994eb7aa30a9b311c138d3ce085e2c1251f6e167ba081131ffccc0
```

The qualitative CSV and all figures are byte-identical to the initial checkpoint:

```text
f427dc02c45fe9c03b484410f585c728bb68ca76c0663ad744ab571e7bd687d7  qualitative_examples.csv
1b18a830a0f08eebcaabfcc70b40017ad098f67db48f53c352a0f7789cf0fe31  model_regime_diagram.png
c3ecd8ee3b5c5bce450b0c2c9373930123d4fee2b95bece82b1e7a7f5db1cfd5  prospective_confusion.png
d80f59546d82cdd4dd0a21c59220fc33432dfa2f1442014e346a8e992556555c  regime_performance.png
```

## Fix 3: strict same-checkpoint paired transfer

Cross-regime CNN evaluation now trains exactly once for each
`(training regime, seed)`. The exact returned in-memory checkpoint is hashed and
reused for matching and opposite test regimes. Both rows record the same digest,
and paired-delta calculation rejects a missing or mismatched digest.

The saved genuine transfer evidence lacked proof of checkpoint identity, so the
maintenance-only command below was executed once:

```text
../../.venv/bin/python run_final_experiment.py recompute-transfer --output-dir artifacts/final --epochs 12
```

The command requires existing pre-July caches and can replace only
`historical_results.json` and `cross_regime_results.csv`. It does not load the
prospective cache, call the prospective stage, or save a frozen package. Protected
frozen/prospective hashes were checked immediately before and after and were
identical.

The corrected paired numbers were exactly equal to the original numbers:

```text
low -> high before/after: -0.08119333753112472 +/- 0.005052308417758101
high -> low before/after: -0.05676323320095291 +/- 0.0045295256838280345
```

Therefore no transfer value, result interpretation, chart, table, or report claim
required numerical revision. The historical files changed only to add the shared
checkpoint digests and explicit recomputation provenance.

## Independent prospective reproduction audit

A read-only audit loaded the now-committed checkpoint package, read the existing
July cache directly, rebuilt causal features/windows/regimes, and performed frozen
inference without invoking `run_prospective_stage`. It proved exact equality for:

- the full saved `slices` dictionary for all four models;
- all 1,488 labels and low/medium/high regime totals;
- every saved confusion matrix and derived metric;
- all six qualitative rows and their probabilities; and
- both frozen binary digests and all five artifact-index bindings.

Audit output:

```text
FINAL_INTEGRITY_AUDIT_OK 1488 July samples 6 qualitative rows 5 digest-bound artifacts
```

The frozen CNN's saved overall July metrics remain:

```text
accuracy: 0.489247311827957
macro-F1: 0.40899071062724995
recall down/flat/up: 0.36807817589576547 / 0.6271981242672919 / 0.24390243902439024
```

## Verification

Focused red/green checkpoints:

- persistence: `24 passed`;
- persistence plus final experiment: `51 passed`, then `52 passed` after the
  historical-only recomputation test;
- report/provenance: `15 passed`.

Fresh full suite:

```text
../../.venv/bin/python -m pytest -q
94 passed in 11.85s
```

Fresh network-blocked dry reproduction disabled DNS and socket connection APIs,
ran a one-epoch synthetic historical stage followed by a synthetic prospective
stage in a new temporary directory, and validated every artifact binding:

```text
NETWORK_BLOCKED_DRY_OK 5 artifacts 1488 samples
```

Deterministic report build:

```text
env SOURCE_DATE_EPOCH=1785658321 tectonic final_report.tex
```

Tectonic exited zero with three non-blocking underfull-box warnings. `pdfinfo`
confirmed five letter-size pages, unencrypted. The output is byte-identical to
the previously approved report:

```text
761ec3cbcbe00cc1e5fea22a3220cce404dc1a651ca74f106fdff7440056e2c5  final_report.pdf
```

All five pages were rendered at 170 dpi and inspected. Page 4 was additionally
rendered at 320 dpi to confirm the complete `Qualitative Results` heading. There
is no clipping, overlap, missing figure, broken reference, or fifth main-text
page. Pypdf independently confirmed pages 1--4 contain Sections 1--9 and page 5
contains only references.

`git diff --check` passed and the worktree was clean before writing this report.

## Concerns and boundaries

- Raw Binance caches remain intentionally ignored. A clean clone can load the
  committed frozen package and inspect all saved evidence, but reproducing the
  independent July inference audit requires the separately retained raw cache.
- The prospective JSON file hash necessarily changed for the explicit archival
  metadata addition; its prediction/metric semantic payload did not change.
- The genuine prospective runner was never rerun, and no recovery override was
  introduced. Any future recovery after a failed genuine reveal must be an
  explicit documented maintenance decision.
- The report remains an educational robustness study and does not establish
  profitability or provide financial advice.

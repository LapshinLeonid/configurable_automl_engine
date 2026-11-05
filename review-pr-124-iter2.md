# Review: PR 124 — `quantile` scaling (`QuantileTransformer`) with fold-size-aware clipping

**Iteration 2 review** (head commit `507e75b`, "Extract shared tree-scaling INFO helper").
Previous iteration reviewed commit `c49b805` with resolution approve.

## Verdict: BLOCK (merge conflicts with `main`)

The code changes themselves remain sound; the blocker is integration state, not code quality.

## Blocking finding

### 1. Merge conflicts with current `main` (4 files)

`main` gained two merged PRs after this branch forked from `25060b4`:

- `125015d` — ExperimentResult unified data model (!121)
- `17c96b4` — `outliers` setting: presets, config, pipeline (Winsorizer → imputer → scaler) (!122)

`git merge-tree` / test merge confirms content conflicts in:

- `API_REFERENCE.md` (the "Composition of a preset" section was rewritten by !122;
  this PR adds the quantile block to the same section)
- `config.schema.json` (both PRs regenerated it; `PreprocessingOverride` gained
  an `outliers` field on main)
- `src/configurable_automl_engine/preprocessing/builder.py` (!122 added
  `Winsorizer` import / `_build_winsorizer`; this PR adds the quantile import
  and changes `_build_scaler(...)` signature to accept `random_state`; the
  `num_steps` list literal is also touched by both sides)
- `src/configurable_automl_engine/preprocessing_presets.py` (!122 added the
  `OutlierSpec` union right after `scaling_type()` — exactly where this PR
  inserts `warn_tree_class_scaling_ineffective`; `PreprocessingPreset` /
  `PreprocessingOverride` / `_format_scaling` / `__str__` also overlap)

**Action required:** rebase onto `main` (or merge `main` into the branch), resolve
the four conflicts, regenerate `config.schema.json` with `schema_generator.py`
(the conflict must NOT be resolved by hand-editing the JSON), and re-run the
quality gates.

**Rebase-specific attention points** (beyond mechanical conflict resolution):

- `PreprocessingPreset.__str__` / `_format_scaling` on main now also render the
  outlier spec; after the rebase the quantile formatting must coexist with it
  (e.g. `outliers=quantile(0.01, 0.99) scaling=quantile(dist=normal, n=1000)`).
- The numeric pipeline order on main is `[Winsorizer] -> SimpleImputer(add_indicator) -> [scaler]`;
  the quantile scaler stays in the scaler slot, so the fold-size-aware clipping
  logic is unaffected — but a combined test (`outliers: quantile` + `scaling: quantile`,
  small fold, indicator columns) would be a cheap regression guard for the merged
  pipeline.
- The `trees`-class INFO helper and the `outliers` WARNING prefix must not
  collide in the two call sites (`trainer_data`, `trial_pipeline`).

## Verification performed in this session (commit `507e75b`)

- Environment: Python 3.11.18, scikit-learn 1.9.1, project installed with `[all,test]`.
- `ruff check src/` — clean; `ruff format --check src/` — clean (82 files).
- `mypy src` — Success: no issues in 83 source files (after installing
  pandas-stubs/scipy-stubs/types-PyYAML; the stubs are env dependencies, not code issues).
- `import-linter` — 4 contracts kept, 0 broken.
- Full `pytest tests/` — **4504 passed, 2 skipped** (~10.5 min), matching the PR description.
- `schema_generator.py` on the PR head produces a byte-identical `config.schema.json` (in sync).
- Manual behavior checks:
  - sklearn emits `UserWarning: n_quantiles (1000) is greater than the total
    number of samples (30)` without the wrapper — the wrapper's clipping is
    genuinely needed.
  - `_FoldSafeQuantileTransformer(n_quantiles=1000)` on a 30-row fold is
    output-identical to `QuantileTransformer(n_quantiles=30)` and exposes
    `n_quantiles_ == 30`.
- The HPO INFO line fires once per phase (`plan_preprocessing` is called from
  `runner` before the trial loop), not per trial — confirmed in source.
- `warn_tree_class_scaling_ineffective` is called in both `trainer_data.build_trainer_preprocessor`
  and `hpo/trial_pipeline._build_trial_preprocessor`; tests assert exactly one line.

## Checklist

- Functionality: issue 170 implemented (spec + fold-safe wrapper + builder wiring +
  tree-class INFO). ✅
- Tests: positive/negative/edge cases for the spec; wrapper behavior (parity with
  sklearn, warning suppression, clipping log, pickle, determinism, feature names,
  NotFittedError/empty input); LOO/k-fold integration; INFO-line tests for both
  call sites plus negatives. ✅ (after the rebase, add the outliers+quantile
  combination test suggested above)
- Types: strict mypy clean, no unjustified `Any` in the new code (`X: Any` in the
  wrapper mirrors the surrounding transformer style). ✅
- Docstrings: Google-style, English, accurate — the `subsample` default divergence
  from sklearn (f2c8c55) is now correctly documented in the spec, the wrapper and
  API_REFERENCE. ✅
- Docs/schema: API_REFERENCE and config.schema.json updated and in sync on the
  branch head. ✅ (will need regeneration after the rebase)

## Non-blocking findings (carried over from iteration 1, still valid)

1. **`quantile.py:96`** — `int(np.asarray(X).shape[0])` can materialize a copy of
   the input (e.g. a DataFrame column subset). `X.shape[0]` (or
   `np.asarray(X, copy=False)`) suffices for counting rows. Minor: this runs once
   per fit, but the input can be large.
2. **`quantile.py:26,95`** — use of the private
   `sklearn.utils.validation._check_feature_names`. The project pins
   `scikit-learn` without an upper bound in some extras; a private-API rename
   upstream would break at import time. A comment justifying it (or a light
   fallback) would make the coupling explicit. (Low priority.)
3. **`preprocessing/__init__.py`** — `_FoldSafeQuantileTransformer` (a private
   name) is re-exported in the package `__all__`. Stylistic; either keep it
   module-private or rename it to a public name if it is meant to be part of the
   package surface.
4. **`preprocessing_presets.py:347`** — `warn_tree_class_scaling_ineffective`
   logs at INFO; the `warn_` prefix suggests a WARNING. Consider
   `note_tree_class_scaling_ineffective` / `log_tree_class_scaling_ineffective`
   (naming nit, introduced by commit 507e75b).

## Files

- Full report: `review-pr-124-iter2.md` (this file)
- Short summary: `review-pr-124-iter2-summary.txt`

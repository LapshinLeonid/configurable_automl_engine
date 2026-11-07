# Review of PR #132 (T8: MLflowReporter), round 5 — iteration 0d25bd9

**Verdict: block (one blocking finding).** Rounds 3–4 addressed the key-length
findings, but introduced a regression: the `…` (U+2026 ELLIPSIS) clamp marker
does not pass MLflow's own name-validation regex, so any clamped key fails the
report exactly like the pre-fix behavior did — via a different failure mode.
One targeted fix plus a name-validity regression test return the PR to
approve-ready state.

## Environment verification (python 3.11.18, feat/mlflow-reporter-t8 @ 0d25bd9, worktree)

- `pytest tests/test_mlflow_reporter.py` — 29 passed (mlflow NOT installed;
  the suite is independent of the package as designed);
- full `pytest tests/` — **4701 passed, 2 skipped** (~11.7 min; the deviation
  from the PR-declared 4792 is dependency-version drift, zero failures);
- `mypy src` — **0 errors** (94 files, strict; with
  pandas-stubs/scipy-stubs/types-PyYAML);
- `ruff check src/`, `ruff format --check src/` — clean; `ruff check` and
  `ruff format --check` on `tests/test_mlflow_reporter.py` — clean;
- `lint-imports` — 4 contracts KEPT;
- `tests/test_module_structure.py` — 98 passed
  (`mlflow_reporter.py` = 466 lines, within the 1000-line limit);
- lazy import re-verified: without mlflow installed the package imports and
  the N1 tests pass without skips; the AST regression
  `test_core_never_imports_mlflow` keeps the core clean.

## End-to-end smoke on the real client (mlflow 3.18.0, sqlite tracking)

- Basic paths: `nested=False` into an active run and `nested=True` both land
  all metrics/params/tags and the three artifacts; the nested run is named
  after the experiment `run_id` and carries the parent linkage;
- **Regression reproduced:** a 300-character phase name (ASCII and Cyrillic
  variants) →
  `MlflowException: Invalid value "automl.score.ridge.p…p" for parameter 'name'
  supplied: Names may only contain alphanumerics, underscores (_), dashes (-),
  periods (.), spaces ( ), colon(:) and slashes (/).` → per the failure policy
  `report` returns `None` (metrics logged, params and artifacts lost) — the
  very scenario the clamping fix was meant to close, now failing one step
  later;
- Windows-regex conformance of every key in real runs was checked
  (`^[/\w.\- ]*$`): the only violating keys are the ones containing `…`.

## Findings

1. **Blocking — `…` (U+2026) in clamped keys fails MLflow's
   `validate_param_and_metric_name` regex** (`^[/\w.\- :]*$`; verified against
   the installed mlflow 3.18.0 and its 2.x line). `_SEGMENT_ELLIPSIS` is used
   by `_clamp_segment` for the phase segment, the flattened parameter
   sub-keys and the additional-metric names; any key that goes through the
   clamp becomes invalid, and `log_metric`/`log_params` raise
   `INVALID_PARAMETER_VALUE` → per the failure policy the whole report
   degrades to `None`. The fake-module tests cannot catch this: `FakeMLflow`
   does not validate key names, and the B5/B5c assertions only check
   `len(key) <= 250` and the presence of `…`. The regression was introduced
   in `aaa521d`/`0d25bd9`; before it a long phase also degraded the report,
   but because of the raw >250 length. Fix options: an ASCII marker allowed
   by the regex (`...`, `-`, `~`), plus a regression test that runs every
   generated key through `mlflow.utils.validation.validate_param_and_metric_name`
   (or its regex copy) — that would catch both the current bug and future
   ones. Non-ASCII segments without clamping (valid `\w` names, as in
   `test_unicode_params_and_tags`) are fine — the regex accepts unicode word
   characters; the problem is only `…`.
   (Inline comment at `mlflow_reporter.py:141`.)
2. **Non-blocking — the whole-key tail-cut inherits `…`** (`_clamp_key`,
   `key[:_MAX_KEY_LENGTH]`): a composed key whose segment was clamped still
   carries the invalid marker after the final guard. The same ASCII-marker
   fix covers it; verify both places.
   (Inline comment at `mlflow_reporter.py:223`.)
3. **Non-blocking — the PR description is stale**: it says "24 tests" (29 in
   fact after B5–B5e) and still shows the `<algorithm>:<phase>` colon
   separator (the code and docs use `__`); the key-clamping policy itself is
   not mentioned in the description. (Comment on the PR.)

## Confirmed resolved from rounds 1–4

- Windows-safe `__` phase separator (round 2), with the rationale documented
  in the docstring and API_REFERENCE.md;
- partial-run ordering under the failure policy documented (round 3);
- `test_artifacts_disabled` regression guard for metrics/params (round 2);
- lint-clean test file, self-contained API_REFERENCE.md example, base.py
  wording, re-reporting policy (rounds 1–2);
- whole-key clamp as the final guard (round 4) — the guard itself works
  (the B5e test proves the composed length bound); only the marker charset
  is wrong.

## Checklist

- [x] Functionality: T8 requirements met (issue #49); prior notes addressed;
- [x] Tests: positive/negative/boundary coverage, independent of mlflow;
- [x] Types: strict mypy clean;
- [x] Docstrings: Google-style English; clamping policy documented;
- [ ] Blocking: replace the `…` clamp marker with a regex-valid one and add
      a name-validity regression test.

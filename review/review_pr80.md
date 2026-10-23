# Review — PR #80 (fix: гасить ошибки фолдов в pooled/pruning оценке HPO, issue #135)

Reviewer: code-reviewer (manual, in-session)
Date: 2026-10-23
PR head reviewed: `b52ced8` (merge of `6a5a1e9` = latest main into the PR branch)
Base: `main` (`6a5a1e9`)

## Verdict: approve (blocking remarks resolved; one non-blocking note)

## Verification performed

- **Merge state**: PR head `b52ced8` = merge commit of main `6a5a1e9` into
  `fix/issue-135-hpo-fold-errors`. `git merge-base HEAD origin/main` ==
  `origin/main` HEAD → main is fully contained; `git merge-tree --write-tree`
  exits 0 (no conflicts); an actual test merge into the worktree auto-merges
  cleanly (`tuner.py`, `tests/test_tuner.py`, `tests/test_tuner_pruning.py`
  all merge; the `test_manual_range_types.py` pytest import is present on both
  sides and deduplicates). Previous blocker №1 resolved.
- **Fatal exception contract**: all fold-skip catch blocks now re-raise
  `(MemoryError, RuntimeError, InvalidDataError)` and swallow only non-fatal
  types (`TypeError`, `KeyError`, `IndexError`, `ValueError`/`LinAlgError`).
  Verified empirically on PR head: `MemoryError` from fold fit propagates out
  of both `_evaluate_pooled_cv` and `_evaluate_with_intermediate_reports`;
  `MemoryError` from per-fold scorer on the additive and the non-poolable
  paths propagates. HPO-level test `test_optimize_memoryerror_fold_disqualifies_algorithm`
  confirms 5× `MemoryError` → `InvalidAlgorithmError` (circuit breaker works).
  Previous blocker №2 resolved.
- **Full suite on PR head**: `pytest tests/ -q` → **3165 passed**. Target
  files: `test_tuner.py` + `test_tuner_pruning.py` + `test_manual_range_types.py`
  → 225 passed.
- **ruff**: `ruff check src/` clean; `ruff format --check src/` clean.
- **mypy src**: only pre-existing `import-untyped` stub errors for
  pandas/scipy/yaml — identical set on `main` (comparison via `/tmp/tuner_main.py`).

## What the PR correctly does (checked against issue #135)

- A fold failing in fit/predict with a non-fatal exception is skipped:
  its rows do not enter the pooled OOF vector, a warning
  `Fold N failed to fit or predict: … excluded from the evaluation` is logged,
  and the trial continues. Issue #135's requirement is satisfied.
- All folds failed at fit/predict → `ValueError` "All N fold(s) failed …" —
  non-fatal contract preserved (trial PRUNED at the caller, algorithm not
  disqualified). Covered by unit tests in both evaluation paths.
- Scoring-path asymmetry (`fits OK, every score impossible` → `nan` →
  `HPO_WORST_SCORE`) is documented and covered by
  `test_evaluate_pooled_cv_scoring_all_folds_failed_returns_nan` with the
  once-only fallback WARNING.
- The 5 duplicated log.warning blocks were unified into `_warn_failed_fold`
  (previous non-blocking remark addressed).
- `algo_name`/`phase_name`/`warn_pooled_fallback_once` from main #136 are
  preserved and used on both paths; the merged code contains no conflict
  markers and no lost main functionality (complexity-guard `p_size`,
  leaf-regularization, log-scale, boundary warnings all intact).

## Non-blocking notes (optional follow-ups)

1. **Transient per-fold fallback in the pruning pooled branch still swallows
   `Exception`** (pre-existing, `tuner.py` ≈ lines 728–731, unchanged by this
   PR): when `pooled_scorer_value` raises `ValueError` on the accumulated
   buffer, the per-fold `scorer(...)` call inside the fallback is wrapped in
   `except Exception: pass`, so a `MemoryError`/`RuntimeError`/`InvalidDataError`
   raised by the per-fold scorer on that particular fold would still be
   swallowed (I reproduced this with a contrived scorer whose raw metric always
   raises ValueError and whose per-fold `__call__` raises MemoryError →
   function returns NaN instead of propagating). For the stock `make_scorer`
   metrics this needs a simultaneously pooled-impossible and per-fold-fatal
   scorer — very narrow — and the code predates this PR, but it slightly
   contradicts the PR's own "fatal types are NOT swallowed on the pruning path"
   claim. A one-line `except (MemoryError, RuntimeError, InvalidDataError): raise`
   before the generic catch would close it.
2. Class-level per-test counters (`_folds_fitted`/`_folds_predicted`) remain
   in the new regression tests — works without xdist, fragile if
   `pytest-xdist` is enabled later (previous reviewer's remark; acknowledged).
3. `if n_folds_total > 0 and n_failed_folds == n_folds_total` — the `> 0`
   guard is dead protection (`iter_splits` raises for `n_folds < 2`), harmless.
4. Commit `57e099c` (add `import pytest` to `test_manual_range_types.py`) is
   now redundant after the main merge — main already carries the same import
   (`917034a`); the merge deduped it cleanly, no harm.

## Conclusion

The regression from issue #135 is fixed with the exact requested semantics,
the two blocking review remarks (rebased merge with main, fatal-type circuit
breaker) are resolved and covered by tests, and the full test suite passes
(3165). Approving.

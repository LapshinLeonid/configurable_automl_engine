# Review archive

This directory archives the per-PR review transcripts (`review-pr-*.md`).
The transcripts are kept verbatim as review-round records and are not
updated in place; the resolution status of every finding is listed below,
and current quality gates live in CI.

**Status: all findings are resolved.** Per-file status:

- `review-pr-130-round2.md` (T7, reporter framework) — resolved
  (approve round).
- `review-pr-132-round2.md` … `review-pr-132-round6.md` (T8, MLflow
  reporter) — resolved; the round-6 verdict is *approve*. The
  reported-failure policy (`report` returning the "reported-failed"
  `None` artifact on a tracking-server failure) is documented as part
  of the base `Reporter.report` contract in
  `src/configurable_automl_engine/experiment/reporting/base.py` and
  pinned by contract tests in `tests/test_mlflow_reporter.py` and
  `tests/test_reporters.py`.
- `review-pr-136-round1.md` (T10, ensemble) — resolved. Per-finding
  status:
  - BLOCKER-1 (per-member positive-weight check) — fixed;
  - MAJOR-1 (`top_k_from_leaderboard` winner insertion) — fixed: the
    terminal winner absent from the top-K is inserted at its **score
    position** (not appended with the last rank), ranks are renumbered
    from the resulting order; covered by
    `tests/test_ensemble.py::test_p4_winner_inserted_at_score_position`;
  - MAJOR-2 (kwargs drift to `_fit_finalist`) — accepted as documented
    behavior: `build_ensemble_from_context` resolves the validation
    settings explicitly from `prepared`; adding a new required
    `_fit_finalist` parameter is caught by mypy, and the semantics
    of the resolution are single-sourced in `PreparedData`;
  - MINOR-1 (WARNING naming the facade consequence) — fixed;
  - MINOR-2 (explicit guard for a missing `model_path`) — fixed at
    the assembly level plus the `ExperimentResult` consistency
    invariant rejecting an ensemble section without a winner artifact
    path (`tests/test_experiment_result.py`);
  - MINOR-3 (dangling "## Ensemble" heading) — fixed: the empty-members
    check precedes the heading;
  - N-1 (API_REFERENCE.md ensemble section) — documented;
  - N-2 (mypy stubs) — environment follow-up.

Current quality gates live in CI.

---
name: metric-honesty-invariants
description: "Invariants of honest score estimation and data-leak prevention for any change that touches metrics, validation, prediction masks, scaling, target transforms, feature selection, HPO or bootstrap. Run before committing such a change: verify row-masking semantics, common-row configuration comparison, holdout/future isolation from pre-verification decisions, adaptive-HPO statistical claims, and metric docstring completeness."
---

# metric-honesty-invariants

## Purpose

Instill the invariants of **honest quality estimation** so that new features
(metrics, prediction masks, scaling, target transforms, HPO, bootstrap,
feature selection) do not violate them. Violations of these invariants are
semantic bugs: the score is formally computed, tests are green, lint and type
checks are clean — yet the estimate itself is wrong (inflated, non-comparable,
or leaked). Reviewers cannot reliably catch them by reading the diff, and each
one historically required a separate fix task and pull request.

When to run: on **any** change that affects one of

- metric computation or a custom metric (`common/metrics.py`, `common/bbc_cv.py`,
  `common/custom_metrics.py`, `trainer_scoring.py`);
- prediction coverage or masking of validation rows (OOF buffers, fold masks,
  holdout scoring in `training_engine/`);
- preprocessing decisions made from data (`auto` modes, scaling, quantile
  clipping, feature selection, outlier detection);
- the configuration search or its statistical evaluation (HPO evaluation,
  pruning, BBC-CV, leaderboard).

Apply the invariants below to the whole affected data path — from the raw
train/validation split to the score that reaches the leaderboard or report —
not only to the edited function.

## The invariants

### 1. Row masking: mask only rows that were never meant to be scored

Distinguish two fundamentally different situations for a row without a usable
prediction:

- **The row did not participate in validation by the design of the split
  strategy** (e.g. the train part of `train_test_split` has no out-of-fold
  prediction) — a normal, expected gap in coverage. Masking such rows is
  correct and allowed, even though the values themselves may be finite.
- **The model was required to predict the row but returned NaN or Inf** —
  this is a quality failure of the model (`InvalidPredictionError`), not an
  empty slot to be skipped. It must never be silently excluded from the
  metric, because excluding it inflates the score: a model that predicted
  10 of 100 required rows could otherwise score better than a model that
  predicted all 100.

Concretely:

- Masking conditions may select rows by **coverage of the split design**
  (was the row in the validation part at all?), never by the **quality of
  the prediction**. `np.isfinite(pred)` as a score mask is always wrong
  where the split design defines which rows must be predicted (the
  `expected` mask); the `expected=None` legacy contract of
  `common/metrics.py` (`oof_rmse`/`pooled_scorer_value`), which drops any
  non-finite pair, is a documented exception — do not "fix" its existing
  call sites in passing, and do not extend it to new code.
- NaN/Inf on a required row raises `InvalidPredictionError` (or folds/trials
  are penalized as failed, per the module's established error policy) — it is
  never replaced by zero, a mean, or any imputation, and never dropped.
- If a new metric or scorer takes a mask parameter, the mask must be derived
  from the split design (an explicit `expected` argument filled by the
  caller), not computed inside the metric from the prediction values.

### 2. Configurations are compared on a common set of rows

When several model configurations are compared, every configuration should be
scored on the **same set of rows** — the rows *every* configuration predicted.
A configuration must never be scored on its own private row subset, and a
configuration with non-finite predictions must be penalized by the shrinking
of the common usable row subset (down to an explicit error when fewer than two
usable rows remain), not by being quietly re-scored on what it did manage to
predict. If a row is dropped for one configuration it is dropped for all of
them.

**Current state:** BBC-CV (`common/bbc_cv.py`) is the comparison path that
implements this contract today.

**Compliant by construction:** the leaderboard ranks candidates by
`AlgorithmRun.score` — the user-semantic best-trial CV estimate of each
algorithm's phase (`experiment/leaderboard.py::build_leaderboard`, derived in
`training_engine/run_result.py` as `to_user_value(metric, best_trial.score)`);
the `cv_best` select step ranks the working set by the same raw trial score
(`training_engine/select_steps.py`). A fold whose predictions are non-finite
makes the whole trial estimate NaN — the worst outcome under the penalty
policy of `hpo/evaluation.py` — so a partially-predicting configuration is
penalized, not re-scored on a private subset. The precondition: this stays
common-row only while the validation strategy and the splits seed are shared
across algorithms (a single `general.validation_strategy`,
`splits_seed = random_state if random_state is not None else 42` in
`trainer_scoring.py`); a per-algorithm strategy/seed override would break it.

**Known gap:** the `rmse_oof`-ranking paths — `select_robust_winner` and the
finalists-pool / fallback ranking in `training_engine/robust_winner.py`
(`_best_by_rmse_oof`, fed by `trainer.oof_score_`). Per-algorithm OOF
coverage can differ there: a fold whose predictions are non-finite is not
collected into that algorithm's OOF buffer by `trainer_scoring.py`
(`_collect_fold_oof`), and `_finalize_oof` computes `oof_rmse` under the
`expected=None` legacy contract (non-finite pairs dropped), so finalists can
effectively be compared over different row subsets. These are the paths this
invariant targets — do not report them as compliant, and do not silently
"fix" them in an unrelated change either. Do not extend the gap claim to the
leaderboard: its ranking does not use `rmse_oof` at all, and auditing it for
OOF coverage (or "aligning" its ranking with the contract) is a mistake.

**The rule for new code:** every new comparison path (an ensemble, a second
leaderboard, a new bootstrap) must preserve the common-row contract where it
applies; if you touch an existing comparison path, either keep its current
semantics unchanged or align it with the contract explicitly.

### 3. No holdout or future information in decisions made before verification

All data-driven decisions that happen **before** the external estimate must
use only data the estimation strategy legitimately exposes at that point:
train data, or out-of-fold data. Forbidden:

- resolving an `auto` configuration placeholder (target transform, scaling
  mode, outlier preset) over the full target vector including the future
  holdout — resolve it after the outer split, on the training part only;
  when no outer split exists (no hold-out reserved), resolve it per
  cross-validation fold from the fold's own train part (the
  `AutoTargetTransformer` pattern) instead of globally on the full data —
  the validation-fold values must not influence the choice the same rows
  are scored with;
- fitting any statistic used by preprocessing (means, quantiles, skewness,
  category lists, feature-selection rankings) on validation, OOF or holdout
  rows;
- tuning hyperparameters or selecting the winner using holdout rows in any
  way (even "just to check").

**Carve-out — domain-safety guards for the final refit.** A data-dependent
decision is also legitimate when it protects the final fit on all data from
crashing after the search budget is spent. The documented example is
`_fallback_transform_for_full_y` (`training_engine/data_prep.py`): the `auto`
method choice itself uses only the train part of the outer split, and then
the resolved method is guarded against the FULL target domain (train +
holdout), downgrading a domain-restricted method (e.g. `log1p` →
`yeo-johnson`) with a WARNING when holdout rows violate the domain. This
guard reads holdout values by design; it is not a leak. Do not flag it or
remove it — removing it would make the final refit fail on data that violates
the method's domain.

When adding an `auto` mode or any data-dependent decision, verify the
decision point: the *method/configuration choice* must be possible to keep
unchanged when every future/holdout value changes (the test pattern in the
next section encodes this check), and any holdout-reading behavior on the
path must be an explicit, documented domain-safety guard of the kind above.

### 4. Statistical claims require reproducing the search process

Do not claim that an estimate compensates for the adaptivity of a search
procedure unless the procedure itself is reproduced. Concretely: after
adaptive HPO, the set of evaluated configurations depends on the original
sample (each trial's configuration and pruning decision were chosen using
scores on it). A plain bootstrap of the collected OOF score matrix resamples
rows but does not reproduce the search, so its output is biased. BBC-CV is
the accepted estimator for this case; its behavior is pinned by the
`tests/test_bbc_cv.py` suite, which runs in the CI `pytest` job — keep that
suite passing. If a new estimation or
aggregation feature (a new bootstrap, a resampled leaderboard, a confidence
interval) makes a fairness claim, either reproduce the selection process
inside it or state in the docstring that it does not compensate for
adaptivity.

### 5. Every new metric, mask or transform declares its honesty semantics

A new metric, mask, scorer, transform or estimation path must state in its
docstring, explicitly:

- **which rows participate** in the value (all validation rows? the common
  row set? rows with finite targets?) and who may legitimately mask rows;
- **how NaN/Inf predictions are handled** — error, penalty, or (only where
  the split design allows) masking, and that no substitution with zeros
  happens;
- **units and sign** (error vs score, direction, what "better" means).

This section is documentation of a contract, not a description of the code
path: a reviewer must be able to verify the honesty invariants from the
docstring alone.

## Test patterns

New behavior in this area is verified with tests of these shapes, following
the existing suites (`tests/test_non_finite_predictions.py`,
`tests/test_bbc_cv.py`, `tests/test_target_transform_auto.py`,
`tests/test_val_score_semantics.py`, `tests/test_honest_scores.py`):

1. **Partial coverage cannot score well.** A model that predicts a valid
   value for only a small subset of the required rows must fail
   (`InvalidPredictionError`), be penalized as a failed fold/trial, or
   produce a score no better than a fully-failing model — never a good score
   computed on its own subset.
2. **No-substitution.** Non-finite predictions are never replaced by zeros or
   other imputations: assert the score path raises or penalizes instead of
   asserting a numeric value that could also arise from substituted zeros.
3. **Leak poisoning.** Poison (shuffle, set to extreme values) every
   row/column that must not influence a decision (holdout targets, future
   values) and assert the decision is unchanged. Scope the assertion to the
   **choice** decisions specifically — the `auto` transform *method choice*
   (`resolve_target_transform`, which reads only the train part of the outer
   split), fitted statistics, selected configuration. The deliberate
   domain-safety guard `_fallback_transform_for_full_y` is exempt: it reads
   the full target by design and may legitimately change on poisoned holdout
   values (see the carve-out in invariant 3).
4. **Common-row comparison.** Give one configuration non-finite predictions
   on rows other configurations predict finitely; assert the comparison
   penalizes it (shrinking common row set, error below the others) and does
   not score it on a private subset.
5. **Each monitor pins its own invariant class.** The Sanity Gate benchmark
   (`benchmarks/run_sanity_gate_benchmark.py`, a mandatory CI step) monitors
   model **degeneracy** — flat/constant predictions, dead features, overfit,
   the audit circuits A–D of `training_engine/sanity_gate.py` — and guards
   winner selection; it does **not** pin metric masking or leak semantics, and
   a masking or holdout-isolation regression can pass its acceptance
   conditions unchanged. The honesty monitors for invariants 1–3 are the
   pytest suites (`tests/test_non_finite_predictions.py`,
   `tests/test_target_transform_auto.py`, `tests/test_honest_scores.py`,
   `tests/test_bbc_cv.py`), which run in the CI `pytest` job. The actionable
   rule: extend the suite or benchmark that actually pins the changed
   behavior instead of weakening its acceptance criteria, and state which
   monitor covers which invariant class — do not assign all of them to the
   benchmark.

## Self-check before committing

Answer for the changed code, item by item; any "no" is a blocker:

- [ ] Every mask in the changed code selects rows by split-design coverage,
      never by prediction finiteness (the documented `expected=None` legacy
      contract in `common/metrics.py` is the only exception; do not extend it
      to new code).
- [ ] Every row the model was required to predict with a non-finite
      prediction leads to `InvalidPredictionError` or an explicit penalty —
      no silent exclusion, no zero substitution.
- [ ] Every multi-configuration comparison **introduced** by the change scores
      all configurations on the same rows. Known non-compliant paths
      (`select_robust_winner` / finalists-pool ranking by own `rmse_oof`) are
      left as-is or aligned with the contract explicitly, never silently;
      compliant paths (the leaderboard ranking by `AlgorithmRun.score` over
      shared seeded splits) are not "audited" or realigned in passing.
- [ ] Every data-driven **choice** on the path to the external estimate uses
      only train/OOF data; an `auto` placeholder is resolved after the outer
      split on the training part — or, when no outer split exists, per CV
      fold from the fold's own train part (the `AutoTargetTransformer`
      pattern of `preprocessing/target_transform.py`). Any holdout-reading
      behavior is an explicit, documented domain-safety guard for the final
      refit (the `_fallback_transform_for_full_y` pattern).
- [ ] Any adaptivity-compensation claim is backed by reproducing the search
      (BBC-CV) or is explicitly disclaimed in the docstring.
- [ ] Every new metric/mask/transform docstring states rows, NaN/Inf
      handling, units and sign.
- [ ] Tests of the patterns above exist for the new behavior and fail when
      the invariant is broken (mutation-checked, per the pre-submit
      verification skill).

## Expected effect

- New metric/validation/preprocessing features satisfy the invariants from
  the start, instead of surfacing later as dedicated fix tasks for masking,
  leak or adaptivity-fairness bugs.
- Blocking review findings on score honesty are prevented before review
  rather than discovered in it.
- The CI monitors keep their distinct roles: the Sanity Gate benchmark step
  pins degeneracy-audit behavior (winner selection), the regression test
  suites (`tests/test_non_finite_predictions.py`, `tests/test_bbc_cv.py`,
  `tests/test_honest_scores.py` and others) pin score-honesty semantics —
  both stay stable because the behavior each of them pins is not silently
  changed.

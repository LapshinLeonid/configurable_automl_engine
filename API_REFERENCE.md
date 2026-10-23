# Configuration File Structure

The system uses a typed YAML or JSON config based on Pydantic.

Scheme: [config.schema.json](config.schema.json).

Some Pydantic validation rules cannot be described in the schema. See [config_parser.py](/src/configurable_automl_engine/training_engine/config_parser.py) for details.

The file must contain two sections: "general" and "algorithms". Additionally, it may include an optional "oversampling" section.

## General section

The `general` section may include the following attributes:

* `phases` — required section (array) of hyperparameter optimization phases.
* `comparison_metric` — (optional) accuracy metric for model comparison. Defaults to `"r2"` if not specified.
* `additional_metrics` — (optional) list of additional quality metrics computed for the final trained model and returned together with the training results (see [Additional metrics](#additional-metrics-additional_metrics)).
* `path_to_model` — (optional) path to save the best model.
* `serialization_format` — (optional) format for saving the model (`"pickle"` or `"joblib"`).
* `log_to_file` — (optional) path to the log file.
* `validation_strategy` — (optional) strategy for evaluating model accuracy (`"train_test_split"`, `"k_fold"`, `"loo"`, `"auto"`). With `"auto"` the engine automatically picks between LOO, k-fold and train-test split based on the number of observations and features.
* `n_folds` — (optional) number of folds for cross-validation; used only if `validation_strategy = "k_fold"`. Ignored when `validation_strategy = "auto"`.
* `max_workers` — (optional) maximum number of threads/processes. If not specified, the number of CPU cores is used.
* `parallel_mode` — (optional) parallelism mode (`"threads"` or `"processes"`). Defaults to `"threads"`.
* `parallel_strategy` — (optional) controls parallel execution strategy. Defaults to `"algorithms"`.
* `phase_timeout` — (optional) global timeout for the entire HPO phase in seconds (minimum 1.0). If `null`, defaults to 3600. Supported only in `"processes"` mode, where it is **hard** (hung workers are force-terminated). In `"threads"` mode setting it is a **config error**: an already running Python function cannot be interrupted, so a soft timeout does not save time and only discards the ready results of completed algorithms.
* `task_timeout` — (optional) per-task timeout in seconds (minimum 1.0). If `null`, uses `phase_timeout`. Supported only in `"processes"` mode (hard). In `"threads"` mode setting it is a **config error**: the deadline counted from submit caused tasks queued behind `max_workers` to "expire" before even starting.
* `pruning` — (optional) early-stopping (pruning) settings for Optuna trials. Disabled by default.
* `categorical_encoding` — (optional) categorical encoding strategy: `"one_hot"` (default), `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`. See [Categorical encoding](#categorical-encoding).
* `high_cardinality_threshold` — (optional, `>= 0`) cardinality threshold for the automatic high-cardinality mode. Must be set together with `high_cardinality_encoding`.
* `high_cardinality_encoding` — (optional) encoding strategy for high-cardinality columns (`"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`). Must be set together with `high_cardinality_threshold`.
* `hashing_n_components` — (optional, `>= 1`, default `16`) number of binary columns per categorical column when `"hashing"` encoding is used.
* `target_encoding_smoothing` — (optional, `>= 0`, default `20.0`) smoothing parameter `m` of target encoding.
* `target_encoding_fallback` — (optional, default `null`) fallback value of target encoding for categories unseen in training; `null` means the global target mean.
* `sanity_gate` — (optional) thresholds and settings of the Sanity Gate audit used by select phases with `strategy: "sanity_gate"`. The section is a pure source of thresholds — the global switch `mode` was removed. See [Sanity Gate configuration](#sanity-gate-configuration-sanity_gate).
* `complexity_guard` — (optional) settings of the adaptive complexity guard for tree/boosting search spaces on small data. See [Complexity Guard configuration](#complexity-guard-configuration-complexity_guard).

### Early stopping (`pruning`)

The optional `pruning` block enables Optuna-based early stopping of unpromising trials during the HPO phase. When enabled, the tuner publishes an intermediate quality score after each natural validation step (e.g., after each cross-validation fold — the cumulative running mean), and the configured pruner may prune a hopeless trial before its full evaluation finishes. Pruned trials are **not** errors: they do not count as fatal failures, do not disqualify the algorithm, and do not stop the optimization. The best model is selected only from fully completed trials using the same metric-maximization rule as without pruning. If all trials are pruned, the standard "no successful trials" scenario applies (no new exceptions).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable` | `bool` | `false` | Master switch for early stopping. When `false` (default), every trial runs to completion — identical behavior to configurations without this block. |
| `strategy` | `string` | `"median"` | Pruning strategy: `"median"` (Optuna `MedianPruner`) or `"hyperband"` (Optuna `HyperbandPruner`). |
| `min_steps` | `int` | `1` | Minimum number of intermediate steps (e.g., CV folds) before the first pruning decision. For `"median"` it maps to `n_warmup_steps`; for `"hyperband"` it maps to `min_resource`. Must be ≥ 1. |
| `n_startup_trials` | `int` | `5` | Number of completed trials before pruning is enabled (`"median"` only; Optuna `n_startup_trials`). Must be ≥ 1. |
| `reduction_factor` | `int` | `3` | Budget reduction factor between Hyperband rounds (`"hyperband"` only; controls aggressiveness). Must be ≥ 2. |

Example:

```yaml
general:
  validation_strategy: "k_fold"
  n_folds: 5
  pruning:
    enable: true
    strategy: "median"
    min_steps: 2
    n_startup_trials: 5
  phases:
    - name: "search"
      n_trials: 50
      action: "all_algorithms"
    - name: "final_select"
      n_trials: 1
      action: "select"
      strategy: "cv_best"
      output: "winner"
```

Validation rules (rejected at config load with a clear message):

* Unknown `strategy` (anything other than `"median"` / `"hyperband"`).
* Non-positive `min_steps`, `n_startup_trials`, or `reduction_factor < 2`.
* Conflict with validation settings: `pruning.enable: true` + `validation_strategy: "k_fold"` + `min_steps > n_folds` (the pruner would never get enough intermediate steps).

Behavior by validation strategy:

* `k_fold` / `loo` — pruning works on natural per-fold (per-sample) steps.
* `auto` — pruning is applied when `auto` resolves to `k_fold` or `loo`; if it resolves to `train_test_split`, pruning is not applied (see below).
* `train_test_split` (hold-out) — there are no intermediate steps, so the pruner is **not** applied; the configuration is accepted with a warning, and every trial runs to completion. This behavior is safe and documented.

### Structure of a pipeline phase (`phases`)

* `n_trials` — number of HPO iterations for phases `all_algorithms` / `refine` (ignored for `select`, but the field is required for schema uniformity).
* `name` — (optional) user-defined name of the phase.
* `action` — (optional) action of the phase, default is `"all_algorithms"`.
* `strategy` — (only for `action: "select"`) selection strategy: `"cv_best"` or `"sanity_gate"`.
* `output` — (only for `action: "select"`) what the select step emits: `"winner"`, `"top_k"` or `"pool"`.
* `k` — (only for `output: "top_k"`) number of candidates to keep, `>= 1`.

### Allowed actions for a pipeline phase

> **Note:** both explicit pipeline actions are executed:
> `refine` (see the contract below) and `select` (both
> strategies `cv_best` / `sanity_gate` and all output forms `winner` /
> `top_k` / `pool`).

* `"all_algorithms"` — for each algorithm, performs `n_trials` hyperparameter optimization attempts.
* `"refine"` — performs `n_trials` warm-started HPO attempts for **every** candidate of the incoming working set. Input: the current working set of the pipeline — the result of the last HPO phase or the output of the previous select step. Only `params` are used for warm-start: each candidate's current parameters are passed to the tuner as `initial_params` (`enqueue_trial` guarantees score monotonicity); incoming scores are ignored. Monotonicity holds because every HPO phase evaluates models on the SAME folds: the data-split/preprocessor seed is the fixed `RANDOM_STATE` for all phases (as in the final `ModelTrainer` fit), while only the TPE sampler seed varies per phase (`RANDOM_STATE + phase_ordinal`) so that startup points do not repeat. The service key `use_feature_selection` keeps being carried inside `params`. Output: the same set of names with updated `(score', params')` records; the output replaces the working set. There is no narrowing in refine — no winners, no top-K, no corridors. A per-algorithm failure excludes only that algorithm (circuit breaker `disqualified_algorithms`, the others continue); if refine wipes out the whole working set (all candidates failed), the phase ends with `RuntimeError` — a fallback to pre-refine parameters would hide the regression (a deliberate design decision). Refine runs like a regular HPO phase: in parallel across algorithms where applicable. As the first phase `refine` is accepted: the incoming working set is then the full set of enabled algorithms (no warm-start from a previous phase).
* `"select"` — an explicit selection step. Required fields: `strategy` (`"cv_best"` — ranking by CV score, or `"sanity_gate"` — finalists pool + `ModelSanityGate` audit + ranking by `RMSE_oof`) and `output` (`"winner"` — a single winner, terminal step; `"top_k"` — top-K candidates, requires integer `k >= 1`; `"pool"` — the whole corridor finalists pool). The step runs synchronously (never inside `run_parallel`) and its output **replaces** the pipeline working set — subsequent `refine` / `select` steps operate on it. The combination `strategy: "cv_best"` + `output: "pool"` is valid: it builds the corridor finalists pool (corridor δ ∩ `top_k_candidates`, thresholds from the `sanity_gate` section) ranked by CV score **without** the `ModelSanityGate` audit.

Execution semantics:

* `strategy: "cv_best"` — ranking by raw CV score (no audit, no gate section in the report): `output: "winner"` — the CV leader (documented tie-break: configuration order); `output: "top_k"` — the first `k` candidates of the ranking; `output: "pool"` — the corridor finalists pool.
* `strategy: "sanity_gate"` — the two-stage selection: finalists pool → every finalist trained on **100% of the data exactly once** → `ModelSanityGate` audit → `select_robust_winner` ranking of the passed candidates by `RMSE_oof` (formal fallback criterion when the whole pool is disqualified). Ranking metric of the strategy — `RMSE_oof`. `output: "winner"` — the robust winner (terminal step; its full-fit trainer is reused for the artifact — the winner is **not** retrained); `output: "top_k"` — the first `k` candidates of the `RMSE_oof` ranking; `output: "pool"` — every candidate that passed the audit (or the full fallback ranking when all are disqualified). Gate disqualifications are fed into the circuit breaker (`disqualified_algorithms`) on a par with HPO failures. A repeated sanity select (e.g. after `refine`) trains its finalists from scratch — stale full-fit trainers are never reused.
* An empty incoming working set (nothing to rank/audit) raises a `RuntimeError`.

Validation rules (rejected at config load with a clear message):

* Deprecated action `"refine_winner"` — rejected with a migration hint (no silent translation): replace it with explicit `select` + `refine` phases.
* `select` cannot be the first phase (there is nothing to rank/audit yet).
* No phases are allowed after a terminal select (`output: "winner"`).
* The **last phase must be a `select` with `output: "winner"`** — the final artifact is built from the result of the terminal select step.
* `output: "top_k"` without `k >= 1` — error; `k` set for `output: "winner" | "pool"` — error; `strategy` / `output` / `k` set for non-select phases — error.
* The parameter `k` is deliberately **not** cross-checked against `sanity_gate.top_k_candidates` at the config level: the finalists pool is capped by `top_k_candidates` first, and the exact `k`-vs-pool interplay (trim / cap) is defined at execution time.

Typical pipeline (see the migration table in README.md):

```yaml
general:
  phases:
    - name: "coarse"
      n_trials: 100
      action: "all_algorithms"
    - name: "gate_select"
      n_trials: 1
      action: "select"
      strategy: "sanity_gate"
      output: "top_k"
      k: 3
    - name: "refine"
      n_trials: 200
      action: "refine"
    - name: "gate_final"
      n_trials: 1
      action: "select"
      strategy: "sanity_gate"
      output: "winner"
```

### Allowed values for `comparison_metric`

* `"nrmse"`
* `"rmse"`
* `"neg_root_mean_squared_error"`
* `"mae"`
* `"mse"`
* `"r2"`

### Finalists pool Top-K (`select_finalists`)

The engine replaces the "blind" first-place CV pick with a deterministic
**finalists pool** used by the two-stage winner selection: candidates whose CV
score falls inside the δ corridor from the leader, intersected with the hard
`top_k_candidates` cap. The pool is built by `select_finalists` in
`training_engine/component.py` from the HPO phase results (raw optimizer
scores); the corridor is expressed in **user** metric semantics (see
`metrics.user_direction` / `metrics.to_user_value`).

Corridor form (`corridor_mode`):

* `"multiplicative"` — min-better metrics (errors): `Score_CV(m) <=
  Score_CV_best * (1 + δ)`; max-better metrics (R² etc.): `Score_CV(m) >=
  Score_CV_best * (1 - δ)`.
* `"additive"` — for metrics where the multiplicative form is incorrect
  (zero/negative values: R² < 0, MAE ≈ 0): `|Score_CV(m) - Score_CV_best| <=
  δ * (max_score - min_score)`, the range computed over all valid candidates.
* `"auto"` (default) — chosen by the metric type: error → multiplicative,
  score metric with a possible sign → additive. For error metrics with a
  practically zero best score (MAE ≈ 0) the multiplicative form would collapse
  the pool to the leader, so the additive form is used instead.
* If the multiplicative corridor excludes the leader itself (e.g. a max-better
  metric with a negative best score such as R² < 0), the additive form is used
  instead.

Combination of filters (explicit): pool = (candidates inside the corridor) ∩
(first `top_k_candidates` by CV score). If the corridor yields fewer candidates
than `top_k_candidates`, the pool contains exactly those — no topping up beyond
the corridor, except the optional family-diversity rule:

* `enforce_family_diversity` (default `false`) — when the pool represents
  exactly one algorithm family (linear / ensembles / kernel-GPR) **and that
  family is permitted by `allowed_families`**, the best representative of each
  missing family is added from the *extended* corridor
  (`family_diversity_multiplier * δ`, default 1.5), never exceeding
  `top_k_candidates`. The added families are also restricted to
  `allowed_families`; when the pool's single family is not permitted, no
  top-up happens at all.

Algorithms disqualified by the circuit breaker never enter the pool.
Determinism: the pool is ordered by the raw score descending (optimizer
semantics, same ordering as `select_winner`); ties keep the configuration
order, which is the documented tie-break. Scores exactly on the corridor
boundary are included (equality never fails).

Edge cases: an empty result set raises `RuntimeError`; fewer than `top_k` valid
algorithms produce a pool of what is available; `top_k_candidates = 1` yields
the single CV leader. The pool function is standalone: it is used by select
steps with `strategy: "sanity_gate"`; thresholds come from the
`general.sanity_gate` section.

### Two-stage winner selection with Sanity Gate (`select_robust_winner`)

Select phases with `strategy: "sanity_gate"` replace the
"blind" CV-first-place pick with a two-stage selection:

1. **Pool** — built by `select_finalists` from the HPO phase results.
2. **Finalist training** — every pool candidate is trained on **100% of the
   data exactly once** (`ModelTrainer.fit`, which also produces the OOF vector
   `oof_predictions_` and `RMSE_oof`); the same training result is reused for
   `RMSE_full`, the sanity audit and the final report (no duplicated training).
3. **Audit** — each finalist is checked by `ModelSanityGate`: circuits A/B
   (diversity / unique) are computed on the **OOF vector**, circuit D
   (generalization gap) uses the full-fit predictions, circuit C (dead
features) inspects the model. Circuit D violations are routed by
   `general.sanity_gate.generalization_gap_mode`: the default
   `"warn"` records them in `soft_warnings` (no disqualification — the
   candidate keeps its chance to win by `RMSE_oof`), `"reject"` turns them
   into disqualification reasons. In `"warn"` mode this covers an **infinite**
   gap as well (`RMSE_full ≈ 0` perfect in-sample fit, `RMSE_oof ≈ 0`);
   `RMSE_oof ≈ 0` additionally raises a `WARNING` log record
   (suspiciously perfect out-of-fold fit, possible target leakage). In both
   modes a missing `y_pred_full` and the absence of finite pairs stay
   `reasons`. The total audit phase can be capped by
   `general.sanity_gate.audit_time_budget_seconds` (0 = no limit):
   when the budget is exhausted, the remaining candidates are skipped with an
   explicit reason.
4. **Ranking** — among the candidates that passed every check the winner is
   the one with the **best (minimum) `RMSE_oof`**. CV score and `RMSE_full`
   do NOT participate in the ranking (the v2 composite score
   `α·Score_CV + (1−α)·Score_full` was removed).

Selection step modes (`select_robust_winner` mode):

The mode is a property of the select step handling, **not** a config field
(`general.sanity_gate.mode` was removed) and there is no way to
set it from the config: the mode is fixed inside the select-step
implementation — select steps with `strategy: "sanity_gate"` always run the
**`"active"`** mode. The underlying `select_robust_winner` machinery exposes
two modes:

* `"off"` — old `select_winner` behavior: no pool, no audit (legacy direct
  calls only; `strategy: "cv_best"` select steps do not use
  `select_robust_winner` at all).
* `"active"` — disqualifications are applied; the winner is the best `RMSE_oof`
  among the candidates that passed the audit.

The `"warn_only"` mode was removed together with the
report keys `sanity_gate_warn_only` and `winner_if_active`: diagnostics are
always available in the gate section of each select-step report, and the
decision to narrow the working set is made explicitly by the select step
itself.

Fallback (formalized): if every candidate is rejected, the "least problematic"
model wins — minimum number of violated circuits, then minimum severity of the
violations (circuit significance order fixed in code: D > A > B > C), then the
best `RMSE_oof`. The run never fails; a `WARNING`
with all reasons is logged.

Failure of a finalist's full-data training is handled: the candidate is
skipped, the reason is logged, and selection continues with the next candidate.
If no finalist can be trained at all, a `RuntimeError` is raised.

Result reporting: every select step with `strategy: "sanity_gate"` adds its
own section to `result["sanity_gate"]` keyed by the **phase name** — see
[Result keys](#result-keys) below. Select steps with
`strategy: "cv_best"` add no gate section.

#### Result keys

The `sanity_gate` key of the `train_best_model` return value is a dict keyed
by the names of select phases with `strategy: "sanity_gate"`; select steps
with `strategy: "cv_best"` add no sections, and with
no sanity-gate select steps the key is absent entirely:

| Key of `result["sanity_gate"][<phase name>]` | Type | Description |
|-----|------|-------------|
| `strategy` | `str` | Always `"sanity_gate"`. |
| `generalization_gap_mode` | `str` | Applied circuit D mode: `"warn"` (default) or `"reject"`. In `"warn"` a gap violation — finite or infinite (`RMSE_full ≈ 0` / `RMSE_oof ≈ 0`) — appears as a `soft_warnings` entry and does NOT disqualify; in `"reject"` it is a `reasons` entry. |
| `pool` | `list[str]` | Ordered finalist names (best first). |
| `audit` | `dict[str, dict]` | Per-candidate audit results: `is_valid`, `reasons`, `soft_warnings`, `diversity_ratio`, `unique_ratio`, `dead_features_count`, `generalization_gap` (disabled/not-computable statistics are `null`), `generalization_gap_infinite` (`true` when the circuit D gap is `inf` — division by zero), `generalization_gap_mode`, `severity`. |
| `disqualified` | `dict[str, list[str]]` | Applied disqualifications `{algo: [reasons]}`; the same entries are also fed into the top-level `disqualified_algorithms` (circuit breaker). Warn-mode circuit D violations never appear here. |
| `fallback_used` | `bool` | `true` when the whole pool was disqualified and the winner was chosen by the formal fallback criterion. |
| `winner_rmse_oof` | `float \| null` | `RMSE_oof` of the winner (`null` when not computable). |
| `overhead_seconds` | `float` | Wall-clock overhead of the select step: finalist training on 100% of the data + the audit phase. |

The legacy keys `disqualified_by_sanity_gate`, `sanity_gate_warn_only` and
`sanity_gate_overhead_seconds` (and the `mode` / `winner_if_active` fields)
were removed together with the `warn_only` mode.

### Sanity Gate configuration (`sanity_gate`)

The optional `general.sanity_gate` block is a **pure source of thresholds**
for select phases with `strategy: "sanity_gate"` — corridor shape,
pool cap, audit limits and circuit thresholds. The global switch `mode`
(`"off" | "warn_only" | "active"`) was removed together with the
`warn_only` semantics: selection happens only through explicit select phases,
and gate diagnostics are available in the gate section of each select-step
report. An absent block (or an empty `{}`) is valid — the section is optional,
not a behavioral flag; a config with `sanity_gate.mode` is rejected with a
migration hint. The only routing flag the section keeps is
`generalization_gap_mode` — it tunes one circuit (D), not the
whole gate: with the default `"warn"` a gap violation — finite or infinite
(`RMSE_full ≈ 0` / `RMSE_oof ≈ 0`) — is a soft warning, and an
explicit `generalization_gap_mode: reject` restores the legacy strict
disqualification.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k_candidates` | `int` | `3` | Hard cap of the finalists pool size (≥ 1). Larger than the number of algorithms — allowed (pool of what is available). |
| `corridor_delta` | `float` | `0.15` | Relative corridor width δ from the leader, strictly in `(0, 1)`. |
| `corridor_mode` | `string` | `"auto"` | Corridor form: `"multiplicative"`, `"additive"` or `"auto"` (chosen by metric type; unknown metric — conservative additive form + warning). |
| `enforce_family_diversity` | `bool` | `false` | Optional soft top-up of missing algorithm families into the pool. |
| `algorithm_families` | `dict[str, str] \| null` | `null` | «algorithm → family» map for the family-diversity rule. |
| `allowed_families` | `list[str] \| null` | `null` | Families participating in the top-up (`null` — all allowed). |
| `family_diversity_multiplier` | `float` | `1.5` | Extended-corridor multiplier for the family top-up (> 0). |
| `min_prediction_diversity` | `float` | `0.15` | Circuit A threshold: `Var(y_pred_oof) / Var(y) >=` threshold, computed on **OOF predictions**, strictly in `(0, 1]`. Too strict values (e.g. `1.0` on noisy data) are not rejected by validation but cause false positives — documented consequence. |
| `adaptive_diversity` | `bool` | `false` | Adaptive mode of circuit A (comparison against the constant model). |
| `min_unique_count` | `int` | `5` | Absolute lower bound of `nunique` for circuit B (≥ 1). The final threshold never exceeds the number of distinct target values. |
| `min_unique_ratio` | `float` | `0.05` | Lower bound of the unique-prediction share for circuit B, in `(0, 1]`. Applied as `min_unique_ratio * N` **capped by the number of distinct target values** (a perfect model always reaches the cap). |
| `max_dead_feature_ratio` | `float` | `0.4` | Dead-feature share threshold of circuit C, in `[0, 1]` — a **signal** for linear models; standalone disqualification only when `dead_features_require_low_diversity=true` and diversity is low. |
| `dead_features_require_low_diversity` | `bool` | `true` | Combination rule of circuit C: disqualify only when "many dead features AND low diversity". |
| `zero_coef_tolerance` | `float` | `1e-12` | Coefficients with `abs(coef) <=` threshold count as zero (circuit C, linear path; ≥ 0). |
| `max_generalization_gap` | `float` | `1.5` | Circuit D threshold (≥ 1): `RMSE_oof / RMSE_full >` threshold means overfitting. |
| `generalization_gap_mode` | `string` | `"warn"` | Circuit D routing of a gap violation: `"warn"` — the violation is recorded in `soft_warnings` (no disqualification, `is_valid` stays `true`, `severity` ≤ 1), including an **infinite** gap from division by zero (`RMSE_full ≈ 0`, `RMSE_oof ≈ 0`); `"reject"` — legacy strict behavior: a reason and disqualification. In both modes a missing `y_pred_full` and the absence of finite pairs remain `reasons`. The field is deliberately NOT named `mode` (collision with the removed `sanity_gate.mode`, which the validator rejects with a migration hint). |
| `check_permutation_sensitivity` | `bool` | `true` | Enable the permutation sensitivity audit of circuit C for nonlinear models (cost control of the permutation audit). `false` — the permutation path is skipped, the linear `coef_` path remains. |
| `permutation_max_rows` | `int` | `5000` | Row subsampling for the permutation path (≥ 1). |
| `permutation_max_features` | `int` | `100` | Cap of checked columns in the permutation path (≥ 1). |
| `permutation_repeats` | `int` | `3` | Permutation repeats per column (≥ 1). |
| `permutation_seed` | `int` | `42` | Fixed permutation seed (determinism). |
| `permutation_tolerance` | `float` | `1e-3` | Relative "deadness" threshold of a feature in the permutation path (≥ 0). |
| `audit_time_budget_seconds` | `float` | `0` | Time budget of the audit phase in seconds; `0` — no limit (default). When the budget is exhausted, the remaining finalists are not audited and receive an explicit "audit skipped: budget exceeded" reason (they cannot win in `"active"` mode). |
| `check_diversity` | `bool` | `true` | Enable circuit A. |
| `check_unique` | `bool` | `true` | Enable circuit B. |
| `check_dead_features` | `bool` | `true` | Enable circuit C. |
| `check_generalization_gap` | `bool` | `true` | Enable circuit D. |

Validation rules (rejected at config load with a clear message):

* Unknown keys — the block has `extra="forbid"` (a typo is a hard error).
* `sanity_gate.mode` (removed) — rejected with a migration hint:
  "use an explicit select step" (`{action: 'select', strategy: 'sanity_gate',
  output: 'winner'}` as the last phase).
* Unknown `corridor_mode`.
* Unknown `generalization_gap_mode` (only `"warn"` and `"reject"` are
  accepted).
* Out-of-range thresholds: `top_k_candidates < 1`; `corridor_delta` outside
  `(0, 1)`; `min_prediction_diversity` outside `(0, 1]`;
  `max_dead_feature_ratio` outside `[0, 1]`; `min_unique_ratio` outside
  `(0, 1]`; `min_unique_count < 1`; `max_generalization_gap < 1`;
  `permutation_max_rows < 1`; `permutation_max_features < 1`;
  `permutation_repeats < 1`; `audit_time_budget_seconds < 0`.

Edge cases:

* `top_k_candidates` greater than the number of algorithms — allowed (pool of
  what is available).
* `corridor_mode: "auto"` with an unknown metric — conservative additive form
  plus a warning.
* Absent `generalization_gap_mode` — the default `"warn"` applies; an explicit
  `generalization_gap_mode: reject` fully reproduces the legacy strict
  behavior of circuit D (migration path).
* A finite `gap` strictly **equal** to `max_generalization_gap` — not a
  violation in either mode (`_fails_above`, equality always passes).
* Absent `sanity_gate` block or an empty `{}` — valid: the section is an
  optional threshold source, not a behavioral flag.
* A `sanity_gate` section without any select phase with
  `strategy: "sanity_gate"` — valid (thresholds are simply unused).

Example:

```yaml
general:
  comparison_metric: "r2"
  validation_strategy: "k_fold"
  n_folds: 5
  phases:
    - name: "search"
      n_trials: 50
      action: "all_algorithms"
    - name: "gate_select"
      n_trials: 1
      action: "select"
      strategy: "sanity_gate"   # pool + audit + ranking by RMSE_oof
      output: "winner"          # terminal step
  sanity_gate:
    top_k_candidates: 3        # pool size cap
    corridor_delta: 0.15       # relative corridor from the CV leader
    min_prediction_diversity: 0.15
    min_unique_count: 5
    min_unique_ratio: 0.05
    max_dead_feature_ratio: 0.4
    max_generalization_gap: 1.5
algorithms:
  ridge:
    enable: true
    limit_hyperparameters: true
  random_forest:
    enable: true
    limit_hyperparameters: true
```

Boundary behavior:

* **OOF coverage with `train_test_split`** — the OOF vector covers only the
  hold-out part (`oof_coverage_` < 1.0); train rows are `NaN` and masked, so
  circuits A/B and the `RMSE_oof` ranking are computed on the hold-out subset
  only. Prefer `k_fold` / `loo` for the most complete audit.
* **Small N** — the unique-ratio circuit is governed by the absolute
  `min_unique_count` bound; a constant target (`Var(y) == 0`) skips circuit A
  with a soft warning and circuit B takes over. The unique threshold
  (`max(min_unique_count, min_unique_ratio * N)`) is capped by the number of
  distinct target values at the prediction rounding granularity: a model
  predicting `y` exactly always reaches the cap and passes, so a
  low-cardinality target with a high `min_unique_ratio` cannot disqualify a
  perfect model.
* **Correlated features** — the permutation path may report "false-dead"
  features on correlated columns (the model borrows the information from a
  neighbour column): a high dead-feature share produces a soft warning, never
  a standalone disqualification.
* **Permutation cost controls** — `permutation_max_rows` / `permutation_max_features`
  bound the audit cost; `check_permutation_sensitivity: false` disables the
  whole permutation path (the linear `coef_` path of circuit C remains).

Migration note: the v2 statement tightened `corridor_delta` to the
open interval `(0, 1)` — the previously allowed value `0` (degenerate corridor
= leader only) is now rejected. Use a small positive value (e.g. `1e-9`) for
the same behavior. `permutation_max_rows` / `permutation_max_features` are
plain `int` (≥ 1) since v2; the old `null` ("no cap") form is replaced by
`check_permutation_sensitivity: false` when the permutation audit should be
skipped entirely. These changes apply to the pre-release schema only —
no released configuration is affected.

### Complexity Guard configuration (`complexity_guard`)

The optional `general.complexity_guard` block is an **adaptive complexity
guard** for the search spaces of trees and boosting on small data. It extends
the existing data-dependent clipping layer (`clip_search_space` in
`common/hyperopt_defaults.py`): on small data the default ranges of
tree/boosting hyperparameters are too wide (e.g. `max_depth` 3–20 or 2–32 on
a 15-row training fold), trials are spent on values that do not change the
model, the TPE surface has flat regions and the search drifts toward overfit
models. The guard tightens the ranges depending on the number of rows (`n` —
the effective training-fold rows) and the feature count.

Two feature counts are used deliberately:

* The size-mode ratio `n / p` uses the **original column count** (a
  categorical column counts as 1, before the one-hot expansion): the
  one-hot expansion must not push large-but-wide data into a smaller size
  mode.
* The rules that depend on the feature count — `max_features` and
  `colsample_bytree` — use **`p` after preprocessing** (what the model
  sees, including one-hot columns and missing-value indicators).

Two independent levels of restrictions (each toggled separately):

* **Level A — lossless clipping** (`lossless: true`): removes values that
  cannot change the model on data of the given size. For example,
  `max_depth=25` on 15 rows is equivalent to `max_depth=6`. Safe. Note: for
  `min_samples_leaf` the level A cap (`≤ floor(n / 2)`) is a *physical*
  bound — larger leaves are unreachable on that sample size — but the
  remaining values below the cap still change the model, so the level is
  "lossless" only in the sense that no reachable leaf size is removed.
* **Level B — regularization** (`regularize: true`): deliberately shifts the
  space toward simpler models on small data. A heuristic. For boosting this
  includes the `n_estimators` budget (see below) — the tree count together
  with `learning_rate` is the main complexity regulator of a boosting
  ensemble.
* **Time-only caps** (`n_estimators` ≤ 300 in the small mode) apply to the
  **bagging** ensembles only (random forest, extra trees) and are marked
  explicitly as "for time", not "for regularization": bagging averages
  independent trees, so more trees only reduce the variance of the ensemble.

The size mode is determined by `n` and the ratio `n / p`, where `p` is the
**original column count** (a categorical column counts as 1, before the
one-hot expansion):

* `"small"`: `n < small_n_threshold` or `n / p < small_np_ratio_threshold`;
* `"medium"`: `n < medium_n_threshold` or `n / p < medium_np_ratio_threshold`
  (and not small);
* `"ordinary"`: everything else — the spaces are not changed.

For example, `n = 4000` rows and 50 original columns (250 one-hot expanded)
give `n / p = 80 >= 20` — the `"ordinary"` mode and unchanged spaces,
while the old `p`-after-preprocessing counting produced `16 < 20` and the
`"medium"` restrictions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `string` | `"apply"` | Operation mode: `"apply"` (default) — restrictions are computed and applied; `"warn"` — computed and logged but NOT applied (the spaces stay unchanged); `"off"` — the guard is disabled. |
| `lossless` | `bool` | `true` | Enable level A (lossless clipping). |
| `regularize` | `bool` | `true` | Enable level B (regularization heuristic). |
| `user_range_policy` | `string` | `"user_wins"` | Policy for an **empty** intersection of a user range (`algo.hyperparameters`) with the guard range: `"user_wins"` (default) — the explicit user range stays and a warning is logged (an explicit setting beats the heuristic); `"engine_wins"` — the guard restriction is applied. |
| `small_n_threshold` | `int` | `50` | Row threshold of the `"small"` mode (≥ 1). |
| `small_np_ratio_threshold` | `float` | `5.0` | Ratio threshold of the `"small"` mode (> 0). `p` is the original column count (a categorical column counts as 1, before one-hot). |
| `medium_n_threshold` | `int` | `300` | Row threshold of the `"medium"` mode (≥ 1). |
| `medium_np_ratio_threshold` | `float` | `20.0` | Ratio threshold of the `"medium"` mode (> 0). `p` is the original column count (a categorical column counts as 1, before one-hot). |
| `size_basis` | `string` | `"rows"` | Basis of `n`: `"rows"` (default) — independent rows; `"groups"` — reserved for group validation (repeats of one group carry less information than independent rows); only `"rows"` is used today. |

Rules of the `"small"` mode (with `n = 15`, `p = 5` as the example):

* **Decision tree / Random forest / Extra trees**
  * `min_samples_leaf`: level A — upper bound ≤ `floor(n / 2)`; level B —
    lower bound 2, upper bound ≤ `floor(n / 4)` (example: 2–3). On `n < 8`
    the range `[2, floor(n / 4)]` is empty and collapses to the lower bound
    2, capped by the physical limit `floor(n / 2)` — the leaf regularization
    does not vanish (it never collapses to `min_samples_leaf = 1`);
  * `max_depth`: level A — ≤ `floor(n / leaf_min) − 1`; level B —
    ≤ `ceil(log2(n / leaf_min))` (example: A gives ≤ 6, B gives 2–3);
  * `max_features` (a fraction): level A — lower bound ≥ `1 / p` (at least one
    feature);
  * `n_estimators` (RF / Extra trees): capped only for time (e.g. 100–300) —
    bagging averages independent trees, so more trees only reduce variance,
    not a regularization.
* **Gradient boosting / XGBoost**
  * `max_depth`: level A — as for trees; level B — 1–3;
  * `min_samples_leaf` (Gradient boosting): as for trees;
  * `min_child_weight` (XGBoost): level B — lower bound 2–3;
  * `subsample`: level B — lower bound ≥ 0.7;
  * `colsample_bytree`: level A — lower bound ≥ `1 / p`;
  * `reg_lambda` (XGBoost): level B — `[1, 100]` on the log scale;
  * `n_estimators`: level B — the upper bound is derived from the space's
    `learning_rate` upper bound: `cap = 60 / lr_high`. For a boosting ensemble
    the number of trees together with `learning_rate` is the main complexity
    regulator (every tree is fitted on the residual of the previous ones), so
    this is a *regularization* restriction: it is disabled together with
    `regularize`, unlike the RF/Extra trees time-only cap. The cap depends on
    the `learning_rate` upper bound so that the product `n_estimators ×
    learning_rate` never exceeds the budget 60 for any sampled pair — e.g. with
    the default lr range `[0.01, 0.3]` the cap is `60 / 0.3 = 200`, with
    `[0.01, 1.0]` it is `60`, with `[0.01, 0.1]` it is `600`. A strict
    per-trial budget (sampling one parameter conditionally on the other) is not
    supported by the guard's per-parameter rules; the upper-bound derivation
    above is the closest the space mechanism allows. When `learning_rate` is
    absent from the space, the default upper bound 0.3 is assumed.
    `learning_rate` itself is not restricted by the guard.

Rules of the `"medium"` mode — the same but softer:
`min_samples_leaf` lower bound 2; `max_depth` ≤ `ceil(log2(n / leaf_min)) + 2`;
XGBoost `max_depth` ≤ 5. In the `"ordinary"` mode the spaces are unchanged.

Interaction with user ranges (`limit_hyperparameters` + `hyperparameters`):

* The default spaces adapt automatically. The default XGBoost space
  (`DEFAULT_SPACES['xgboosting']`) includes `min_child_weight` (1–10, float)
  and `reg_lambda` (1e-3–100, log scale), so the level B rules of the guard
  fire on the defaults (both are tightened to `≥ 2` / `[1, 100]` on small
  data) and are reported in the log and the `complexity_guard` report — they
  are not restricted only when the user overrides them with a conflicting
  range and `user_range_policy: "user_wins"` keeps the override.
* A range the user set explicitly is intersected with the guard range; on an
  **empty** intersection the `user_range_policy` decides (by default the user
  range stays and a warning is logged). The strict "engine wins" mode is
  enabled via `user_range_policy: "engine_wins"`.
* The result is never empty: `low <= high`, the int/float type and the log
  scale are preserved; a range collapsed to one value fixes the parameter
  (logged). Integer-like parameters (leaf sizes, depths, tree counts) never
  go below their physical minimum of 1 — on extremely small `n` an empty
  range is fixed at 1, never at 0 (sklearn rejects 0 with
  `InvalidParameterError`; XGBoost silently overrides `max_depth=0`).

Transparency:

* Per (algorithm, phase, parameter) one INFO log record: "was → became" with
  the reason, e.g. `max_depth 3–20 → 2–3: n=15, p=5, mode=small, leaf_min=2`.
* The same records are aggregated into the `train_best_model` report under the
  `complexity_guard` key: `{"parameters": [ ... ]}` — **one record per
  restricted parameter** (the A/B rules of the same parameter are merged, and
  identical records of repeated phases are not duplicated) with `algo`,
  `param`, `before`, `after`, `levels` (the list of applied rule levels, e.g.
  `["A", "B"]` or `["B"]`), `size_mode`, `n`, `p`, `leaf_min`, `reason`,
  `applied`, `conflict`, `fixed`.

Note on `max_features`: the parameter accepts both a float (a fraction of the
features, clamped to `(0, 1]`) and an int (a plain feature count, physical
minimum 1). The guard treats them accordingly — an int user range is never
collapsed to a fixed `1`.

Note on `min_samples_leaf`: sklearn also accepts a float in `(0, 1)` as a
fraction of the sample size. The guard's rules are expressed in absolute rows,
so a fractional range is converted to rows (`value × n`) before a rule applies
and divided by `n` afterwards — a user fraction range stays in `(0, 1]` and is
never widened to a row count (a float `>= 1` is a row count and is treated as
integer-like).

Determinism and parallelism:

* The guard is computed from the same inputs (`n`, `p`, the config) in every
  HPO phase, so the TPE history transfer between phases
  (`_transfer_previous_trials`) stays valid — no `HPOHistoryError`.
* In the subprocesses the restrictions are identical: the inputs are computed
  deterministically and travel with the phase snapshot.

Validation rules (rejected at config load with a clear message):

* Unknown keys — the block has `extra="forbid"` (a typo is a hard error).
* `small_n_threshold` must be `< medium_n_threshold`.
* `small_np_ratio_threshold` must be `< medium_np_ratio_threshold`.
* Thresholds must be positive; `mode` must be one of `off`/`warn`/`apply`;
  `user_range_policy` — one of `user_wins`/`engine_wins`; `size_basis` — one
  of `rows`/`groups`.

Example:

```yaml
general:
  phases: [...]
  complexity_guard:
    mode: "apply"
    lossless: true
    regularize: true
    user_range_policy: "user_wins"
    small_n_threshold: 50
    small_np_ratio_threshold: 5.0
    medium_n_threshold: 300
    medium_np_ratio_threshold: 20.0
    size_basis: "rows"
```

### Additional metrics (`additional_metrics`)

The optional `general.additional_metrics` parameter is a list of extra quality metrics whose values are computed for the **final trained model** (after the winner is selected) and returned together with the training results.

Rules and behavior:

* **Optional & backward compatible.** If the parameter is absent or empty, the system behaves exactly as before: no `additional_metrics` key is added to the results.
* **Allowed values** — the same set as for `comparison_metric` (see above). An unknown metric name fails config validation before training starts (`ValidationError` with the list of supported values).
* **Informational only.** Additional metrics do **not** participate in hyperparameter optimization, model comparison, or winner selection. The winner is determined solely by the main `comparison_metric`.
* **Duplicates.** If the same metric is listed several times, it is computed and returned exactly once (first-occurrence order is kept). Duplicates are removed at config validation time.
* **Overlap with the main metric.** If an additional metric coincides with the main comparison metric (including aliases of the same metric, e.g. `"rmse"` and `"neg_root_mean_squared_error"`), it is excluded from the list — its value is already returned once as `score`. A warning is logged.
* **Post-validation list.** Both rules above are applied at config validation time, so the resulting `general.additional_metrics` (as exposed on the validated config object and forwarded to the final `ModelTrainer`) may differ from the list the user provided: duplicates are removed and the comparison metric (with its aliases) is dropped. Only the metrics present in the final list are computed and returned.
* **Computation.** Each additional metric is evaluated for the final model with the **same validation method as the comparison metric**: on a hold-out split (`train_test_split`) or as the CV estimate (`k_fold`/`loo`) — never on the full training set. The CV estimate is the mean over folds for additive metrics (MAE and similar per-sample means) and the pooled value over the out-of-fold vector for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D², explained_variance, max error, median absolute error and their aliases), matching the `val_score` semantics. Error-type metrics (RMSE, MAE, MSE, NRMSE) are returned as positive natural values, score-type metrics (R²) as-is. Non-finite values (e.g., `inf` from NRMSE on a constant target) are returned as-is and do not affect training completion or artifact saving.
* **Failure tolerance.** If a single additional metric cannot be computed (scorer error), it is skipped with a warning and does not abort training, model selection, or artifact saving.

Example:

```yaml
general:
  comparison_metric: "rmse"
  additional_metrics: ["r2", "mae", "mse"]
  phases:
    - name: "search"
      n_trials: 50
      action: "all_algorithms"
```

Result format (`train_best_model` return value):

```python
{
    "algorithm": "ridge",
    "score": 0.123,              # main comparison metric in natural (user-facing) semantics
    "metric": "rmse",            # user-facing name of the comparison metric
    "params": {...},
    "model_path": "model.pkl",
    "additional_metrics": {"r2": 0.98, "mae": 0.07, "mse": 0.009},  # only when non-empty
}
```

Metric semantics: `score` always reflects the natural value of the
metric as the user knows it — positive RMSE/MAE/MSE/NRMSE, plain R². The
internal sklearn negative scorers (`neg_root_mean_squared_error` and other
`neg_*`) are an optimizer detail: their inverted values never leak into
`result["score"]`. `metric` carries the user-facing name of the comparison
metric from the configuration (`general.comparison_metric`).

The `additional_metrics` key is a dict mapping each configured metric name to its computed value for the final model (same natural semantics as `score`).

## Algorithms section

The `algorithms` section is a dictionary where the key is the algorithm name, and the value is a set of configurations for that algorithm.

Supported algorithms:

* `"elasticnet"`
* `"sgdregressor"`
* `"decision_tree"`
* `"random_forest"`
* `"extra_trees"`
* `"gradient_boosting"`
* `"adaboost"`
* `"poissonregressor"`
* `"gammaregressor"`
* `"tweedieregressor"`
* `"gaussian_process_regression"`
* `"isotonic_regression"`
* `"nearest_neighbors_regression"`
* `"svr"`
* `"ardregression"`
* `"glm"`
* `"ridge"`
* `"lasso"`
* `"xgboosting"`

Short aliases are also supported: `"dt"`, `"rf"`, `"et"`, `"gb"`, `"ab"`, `"sgd"`, `"knn"`, `"gpr"`, `"ard"`, `"xgboost"`, and others.

Algorithm configuration consists of:

* `enable` — boolean flag, whether hyperparameter search is performed for the algorithm.
* `limit_hyperparameters` — (optional) boolean flag to set limits for hyperparameter search.
* `hyperparameters` — (optional) hyperparameter value constraints, unique to each algorithm. Keys must match hyperparameters accepted by the estimator constructor (including legacy names from `LEGACY_PARAM_MAPPINGS`, e.g. `n_iter` → `max_iter` for ARDRegression). The allowed set is derived from the estimator signature via [`get_allowed_hyperparameters`](src/configurable_automl_engine/models.py), not from the default search space — so parameters outside `DEFAULT_SPACES` (e.g. `min_samples_split`, `max_features`) are valid as long as the estimator accepts them. Unknown names (typos) are rejected at configuration validation.

A numeric range is written as `[low, high]` or with an explicit type: `[low, high, "float"]`, `[low, high, "float_log"]` or `[low, high, "int"]`. When the type is **not** written explicitly, it is taken from the **parameter's type** (the default search space or the estimator signature), not from how the boundary numbers look — so `[1, 50]` for a real-valued parameter (e.g. `C` of an SVR or `alpha` of a Tweedie) stays a real range instead of silently becoming an integer one. The **logarithmic scale** (`float_log`) is used by default when the parameter is searched on a log scale by default — a scale parameter (`alpha`, `C`, `epsilon`, `gamma`, `learning_rate`) or a parameter whose default search space entry is `float_log` (e.g. `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` of an ARD regression or `eta0` of an SGD regressor) — and both bounds are positive; an explicit type (`[1, 50, "int"]`) is always honored. An explicit `[low, high, "float_log"]` requires `low > 0` (the log scale needs strictly positive bounds).

When the best trial of an HPO phase lies at the boundary of a numeric search range, the tuner logs a warning — "the true optimum may lie beyond the range, widen it" — signalling that the search was cut off. The check covers dict-based numeric search spaces (YAML/`SearchSpaceEntry`, including the `space_overrides` dict form of a direct `optimize()` call); function-based spaces (custom `space_fn`) and categorical parameters are not checked.
* `preprocessing` — (optional) explicit override of the feature preprocessing preset for this algorithm. Takes priority over the automatic class-based selection. May override the whole preset or only some fields:

```yaml
algorithms:
  ridge:
    enable: true
    preprocessing:
      imputation_strategy: median   # optional: "mean" | "median"
      scaling: standard              # optional: "standard" | "robust" | "none"
```

Invalid values in the `preprocessing` block are rejected at configuration validation with a clear message.

## Adaptive Feature Preprocessing (preprocessing presets)

The engine automatically chooses the feature preprocessing strategy based on the selected regression algorithm. A **preprocessing preset** declaratively describes the numeric-feature handling rules: the missing-value imputation strategy and whether/how scaling is applied. Each supported algorithm is assigned to an **algorithm class** with common preprocessing requirements; membership is fixed in the default mapping table (see [`preprocessing_presets.py`](src/configurable_automl_engine/preprocessing_presets.py)) — adding a new algorithm only requires registering it there.

### Algorithm classes

| Class | Algorithms | Imputation | Scaling |
|-------|-----------|------------|---------|
| `scale_sensitive` — scale-sensitive models (linear, kernel, distance-based) | `elasticnet`, `sgdregressor`, `ridge`, `lasso`, `ardregression`, `svr`, `nearest_neighbors_regression`, `gaussian_process_regression` | `mean` (standard) | `standard` (always applied) |
| `trees` — trees and tree ensembles | `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `adaboost`, `xgboosting` | `median` (robust to outliers) | `none` (never applied) |
| `glm_skewed` — GLMs with skewed distributions | `poissonregressor`, `gammaregressor`, `tweedieregressor`, `glm` | `median` (skewness-aware, robust to outliers) | `robust` (RobustScaler) |
| `univariate` — specialized univariate algorithms | `isotonic_regression` | `median` | `none` |
| `default` — fallback for supported algorithms without an explicit registration | any supported algorithm missing from the mapping | `mean` | `standard` |

Unknown algorithm names (typos) are rejected with a `ValueError` listing the available algorithms, instead of silently falling back to the default preset.

### Composition of a preset

Each preset defines:

* `imputation_strategy` — missing-value imputation strategy for numeric features: `mean` (standard) or `median` (robust to outliers / skewed distributions).
* `scaling` — whether scaling is needed and its type: `standard` (`StandardScaler`), `robust` (`RobustScaler`, robust to outliers), or `none` (numeric features are passed to the model without scaling).

### Selection and override rules

* The preset is resolved automatically from the algorithm name (aliases are supported) before training starts; the decision is made once, so there is no measurable overhead. Algorithm names are validated against the supported registry — unknown names raise `ValueError`.
* The same preset is applied at every lifecycle stage — hyperparameter optimization (HPO), final training, and prediction — through the single resolution point used by both `tuner.optimize` and `ModelTrainer`.
* Explicit user override (the `preprocessing` block in the algorithm config, or the `preprocessing_override` argument of `ModelTrainer`/`tuner.optimize`) takes priority over the automatic selection. Fields left unset keep the automatically chosen values.
* The resolved preset is logged (INFO level) and stored on the trainer as `preprocessing_preset`, so the applied preprocessing can always be determined from logs.

## Oversampling section

The optional `oversampling` section may include:

* `enable` — (optional) enable oversampling.
* `multiplier` — (optional) factor to increase dataset size (minimum 1.0).
* `algorithm` — (optional) oversampling algorithm.

### Supported oversampling algorithms

* `"random"`
* `"smote"`
* `"adasyn"`

Gaussian noise injection is **not a separate algorithm** — it is enabled with the `add_noise` flag of `DataOversampler` (intensity is controlled by `noise_level`, default `0.01`) and applies to numeric features **after** resampling, for any of the three algorithms. Integer columns are cast to float when `add_noise=True`, and categorical columns are excluded from the noise. Noise is not supported on sparse input (e.g., the output of the `hashing` encoder): combining `add_noise=True` with a sparse matrix raises `TypeError`.


# API Reference

## `train_best_model(config, df, target, model_path_override)`

The main entry point for the AutoML pipeline. Validates data, runs multi-phase hyperparameter optimization, selects the best algorithm, and saves the final model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str`, `Path`, `Config`, or `dict` | Configuration — path to YAML/JSON file, `Config` object, or dictionary. |
| `df` | `pd.DataFrame` | Input dataset. |
| `target` | `str` or `None` | Name of the target column. Defaults to `"target"`. |
| `model_path_override` | `str`, `Path`, or `None` | Alternative path to save the model. |

**Returns:** `dict[str, Any]` with keys `"algorithm"`, `"score"`, `"metric"`, `"params"`, `"model_path"`. `"score"` is the comparison metric in natural user-facing semantics (positive RMSE/MAE/MSE/NRMSE, plain R² — internal `neg_*` sklearn scorers are inverted back), `"metric"` is its user-facing name from `general.comparison_metric`. When `general.additional_metrics` is configured (non-empty after deduplication and exclusion of the comparison metric), an additional key `"additional_metrics"` maps each extra metric name to its value computed for the final model (same natural semantics). Every select step with `strategy: "sanity_gate"` adds its own section under `sanity_gate` keyed by the phase name — see [Result keys](#result-keys); when any algorithm was disqualified (HPO failures or gate audits), the top-level `disqualified_algorithms` dict names the reasons. When the adaptive complexity guard adapted any tree/boosting range, an additional key `"complexity_guard"` is present: `{"parameters": [ ... ]}` — one "before → after" record per restricted parameter (see [Complexity Guard configuration](#complexity-guard-configuration-complexity_guard)).

**Raises:** `TypeError` for unsupported config types; `RuntimeError` if no algorithm succeeds.

---

## `optimize(algo_name, X, y, ...)`

Runs hyperparameter optimization for a single algorithm using Optuna.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algo_name` | `str` | — | Algorithm name (e.g., `"random_forest"`). |
| `X` | `np.ndarray` or `pd.DataFrame` | — | Feature matrix. |
| `y` | `np.ndarray`, `pd.Series`, or `pd.DataFrame` | — | Target vector. |
| `data_oversampling` | `bool` | `False` | Enable oversampling. |
| `data_oversampling_multiplier` | `float` | `1.0` | Oversampling multiplier. |
| `data_oversampling_algorithm` | `str` | `"random"` | Oversampling algorithm (`"random"`, `"smote"`, `"adasyn"`). |
| `metric` | `str` | `"r2"` | Metric to maximize. |
| `val_method` | `ValidationStrategy` or `str` | `"k_fold"` | Validation method (legacy parameter). |
| `validation_strategy` | `ValidationStrategy` or `str` | `"k_fold"` | Validation method (`"train_test_split"`, `"k_fold"`, `"loo"`, `"auto"`). |
| `train_test_split_test_size` | `float` | `0.2` | Test set size when using `train_test_split` validation. Ignored when using `"auto"`. |
| `n_folds` | `int` | `5` | Number of CV folds. |
| `n_trials` | `int` | `50` | Number of Optuna trials. |
| `random_state` | `int` or `None` | `42` | Random seed of the data splits and the preprocessor (CV folds, `train_test_split`, pruning folds, hashing / target-encoding cross-fit). |
| `sampler_seed` | `int` or `None` | `None` | Seed of the TPE sampler, separated from `random_state`: each HPO phase may use a different `sampler_seed` (e.g. `RANDOM_STATE + phase_ordinal`) while the data splits stay fixed. `None` — the sampler falls back to `random_state` (backward-compatible). |
| `space_overrides` | `dict[str, Callable]` or `None` | `None` | Custom search space overrides. |
| `initial_params` | `dict[str, Any]` or `None` | `None` | Hyperparameters from the previous HPO phase, enqueued as the first trial (`enqueue_trial`) — the "monotonicity anchor" between phases. |
| `previous_trials` | `list[optuna.trial.FrozenTrial]` or `None` | `None` | Trials of the previous HPO phase to be transferred into the new study via `study.add_trials` (the TPE history warm-start). Transferred BEFORE `enqueue_trial`/`optimize`; they feed the TPE sampler but never count as the current phase's own trials. Empty list — no-op. Trials whose distributions are incompatible with the current search space raise `HPOHistoryError` (fail fast). |
| `preprocessing_override` | `dict` or `PreprocessingOverride` or `None` | `None` | Explicit override of the preprocessing preset. Partial (`{"scaling": "none"}`) or full overrides are supported; takes priority over automatic class-based selection. |
| `pruning` | `dict[str, Any]` or `None` | `None` | Early-stopping settings: `{"enable": bool, "strategy": "median"\|"hyperband", "min_steps": int, "n_startup_trials": int, "reduction_factor": int}`. When `None` or `enable: false`, trials run to completion (identical to previous behavior). |
| `feature_selection_cfg` | `FeatureSelectionCfg`, `dict`, or `None` | `None` | Feature selection config (method, percentile, `min_features`, etc.). `None` means `FeatureSelectionCfg()` (`mode='disabled'`). In `'always'` mode every trial pipeline (and the final model) contains the `feature_selector` step; in `'disabled'` it never does; in `'auto'` Optuna decides per trial via the `use_feature_selection` categorical (recorded in `best_params`). For `isotonic_regression` the flag is forcibly turned off. |
| `complexity_guard_cfg` | `ComplexityGuardConfig`, `dict`, or `None` | `None` | Settings of the adaptive complexity guard for tree/boosting search spaces (from `general.complexity_guard`). `None` — the guard is disabled (the exact legacy behavior for direct callers). When enabled, the dict-based search spaces of trees/boostings are tightened depending on `n` (effective training-fold rows) and the feature counts: the size mode uses the original column count (`p_size`, a categorical column counts as 1), the `max_features`/`colsample_bytree` rules use `p` (features after preprocessing). |
| `user_hyperparameter_keys` | `list[str]` or `None` | `None` | Names of parameters set explicitly by the user (`algo.hyperparameters`); the guard intersects its ranges with these user ranges and applies `user_range_policy` on an empty intersection. |
| `fit_best_model` | `bool` | `True` | Whether to train the final model on all provided data after the search. Set to `False` when only the best parameters and score are needed (e.g., the training engine's HPO phases — the final model is trained by its own trainer): the redundant full-data fit is skipped and the `model` field of the result is `None`. |

**Returns:** `OptimizeResult` (a dataclass with `model`, `params`, `score`, `trials` and `complexity_guard`) when the phase produced a valid result; `None` when **no trial of the current phase succeeded** (e.g., `n_trials=0`, or every trial is pruned via a non-fatal error). `model` is `None` when `fit_best_model=False` (the full-data fit is skipped); otherwise it is the trained model with the best parameters. `trials` always contains ONLY the current phase's own Optuna trials (transferred `previous_trials` are excluded), so the caller can warm-start the next phase without duplicating computations. `complexity_guard` is the list of the adaptive complexity guard records for the phase (`None` when the guard is disabled or produced no records). `None` is the explicit failure signal — the orchestrator (`train_best_model`) then excludes the algorithm from candidates instead of treating the `HPO_WORST_SCORE` sentinel (`-3.4028235e38`) as a valid result; transferred history never masks a total failure of the current phase. The sentinel itself is returned as `score` only when trials complete but all produce a non-finite metric.

In `'auto'` mode `params` additionally contains the service key `use_feature_selection` (`True`/`False`) — the winner's decision about feature selection, which is preserved between multi-phase HPO runs (passed to the next phase via `initial_params`/`enqueue_trial`) and consumed by `train_best_model` for the final fit. The key is absent in `'always'`/`'disabled'` modes and for `isotonic_regression` (feature selection is forcibly turned off there). The key is never passed to the base model constructor.

**Raises:** `ValueError` for invalid `n_trials`; `HyperoptError` for missing search spaces; `HPOHistoryError` when `previous_trials` are incompatible with the current search space (fail fast); `InvalidAlgorithmError` after 5 consecutive fatal failures.

---

## `create_model(algorithm, **hyperparams)`

Creates a scikit-learn compatible regressor instance by algorithm name.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `algorithm` | `str` | Algorithm name or alias (e.g., `"rf"`, `"xgboost"`, `"knn"`). |
| `**hyperparams` | `Any` | Hyperparameters passed to the estimator constructor. |

**Returns:** `RegressorMixin` instance.

**Raises:** `TypeError` if `algorithm` is not a string; `ValueError` for unknown algorithms; `ImportError` for missing optional dependencies.

**Notes:** Automatically cleans and remaps hyperparameters via `clean_hyperparameters()`. Sets `max_iter=10000` for SVR. Sets `kernel=RBF(1.0)` for GaussianProcessRegressor by default.

---

## `ModelTrainer`

Orchestrator class for training, validation, and serialization of regression models.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | `str` | `"elasticnet"` | Algorithm name. |
| `hyperparams` | `dict` or `None` | `None` | Model hyperparameters. |
| `metric` | `str` | `"r2"` | Evaluation metric. |
| `random_state` | `int` or `None` | `42` | Random seed. |
| `data_oversampling` | `bool` | `False` | Enable oversampling. |
| `data_oversampling_multiplier` | `float` | `1.0` | Oversampling multiplier. |
| `data_oversampling_algorithm` | `str` | `"random"` | Oversampling algorithm (`"random"`, `"smote"`, `"adasyn"`). |
| `serialization_format` | `SerializationFormat` | `pickle` | Save format. |
| `categorical_features` | `list[str]` or `None` | `None` | Categorical column names. |
| `numerical_features` | `list[str]` or `None` | `None` | Numerical column names. |
| `id_column` | `str` or `None` | `None` | ID column to exclude. |
| `encoding_strategy` | `str` | `"one_hot"` | Categorical encoding strategy: `"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`. |
| `additional_metrics` | `Iterable[str]` or `None` | `None` | Additional (informational) metrics computed for the trained model with the same validation method as the main metric (hold-out split or the CV estimate — the fold mean for additive metrics, the pooled OOF value for non-additive ones — never a train score). Any iterable of strings is accepted (list, tuple, set, generator); names are normalized to lowercase. Stored in `additional_scores` after `fit()`. |
| `validation_strategy` | `str` or `ValidationStrategy` | `"train_test_split"` | Validation strategy for the final model quality estimate: `"train_test_split"` (one hold-out split), `"k_fold"` (CV fold mean), `"loo"` or `"auto"`. The engine path always passes the already-resolved strategy. |
| `n_folds` | `int` | `5` | Number of folds for `"k_fold"` (`>= 2`). |
| `test_size` | `float` or `int` | `0.2` | Hold-out size: a fraction in `(0, 1)` or an integer number of rows (like sklearn `train_test_split`). |
| `preprocessing_override` | `dict` or `PreprocessingOverride` or `None` | `None` | Explicit override of the preprocessing preset. Partial or full overrides are supported; takes priority over the automatic class-based selection. |
| `high_cardinality_threshold` | `int` or `None` | `None` | Cardinality threshold for the automatic high-cardinality mode (`>= 0`). Columns with more unique values than the threshold are encoded with `high_cardinality_encoding`; the rest use `encoding_strategy`. Must be set together with `high_cardinality_encoding`. `None` disables the mode. |
| `high_cardinality_encoding` | `str` or `None` | `None` | Encoding strategy for high-cardinality columns (`"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`). Must be set together with `high_cardinality_threshold`. |
| `hashing_n_components` | `int` | `16` | Number of binary columns per categorical column when `"hashing"` encoding is used (`>= 1`). Controls output dimensionality regardless of column cardinality. |
| `target_encoding_smoothing` | `float` | `20.0` | Smoothing parameter `m` of target encoding (`>= 0`). With `m=0` the raw per-category mean is used; larger values pull rare categories toward the global mean. |
| `target_encoding_fallback` | `float` or `None` | `None` | Fallback value for categories unseen in training under target encoding. `None` means the global target mean. |

**Key attributes:**

* `preprocessing_preset` — the resolved `PreprocessingPreset` (algorithm class, imputation strategy, scaling type) applied during `fit()`. Useful for observability and serialization.

**Key methods:**

* `fit(X, y)` — Train the model with full preprocessing pipeline.
* `predict(X)` — Make predictions on new data.
* `save(path)` — Serialize to disk.
* `load(path)` — Load from disk (class method).

**Attributes after `fit()`:**

* `val_score` — value of the main metric computed on data the model has **not** been trained on: a hold-out split score (`train_test_split`) or the CV estimate (`k_fold`/`loo`) — the same validation method used by HPO to compare models. The CV estimate is the mean over folds for additive metrics and the pooled value over the out-of-fold vector for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D², explained_variance, max error, median absolute error and their aliases). It is **not** a train score. Error-type metrics are returned as positive natural values (sign-corrected). When the pooled value is impossible (a scorer without the sklearn raw metric attributes, or no finite OOF pairs), the engine falls back to the per-fold mean and logs a WARNING once per algorithm and phase (e.g. an HPO phase name or the final fit).
* `additional_scores` — `dict[str, float]` with values of the configured additional metrics computed with the same validation method as `val_score` (empty if none were set).

**Categorical encoding**

Categorical columns are encoded by a single shared preprocessor used both during HPO and final training. Five strategies are available via the `general.categorical_encoding` config key (or the `encoding` argument of `tuner.optimize`):

* `one_hot` (default) — each category becomes a binary column via `OneHotEncoder`. Unknown categories on prediction are ignored.
* `ordinal` — each categorical column becomes a single numeric column via `OrdinalEncoder` (categories ordered alphabetically). Unknown categories map to `-1`, so prediction never fails. Does not expand the number of columns but imposes an artificial ordering on categories.
* `target` — target encoding (`TargetEncodingTransformer`): each category is replaced by the smoothed mean of the target for that category, `(n_cat * mean_cat + m * global_mean) / (n_cat + m)`, where `m = general.target_encoding_smoothing`. All statistics are computed on the training part only, so no information from validation/test data leaks into the encoding — including cross-validation, where the preprocessor is refitted on every fold. To also avoid within-train leakage (a row's own target inflating its encoded feature, critical for high-cardinality columns with few rows per category), `fit_transform` uses cross-fitting: each training row is encoded with statistics computed on a fold that excludes it, while the full-sample statistics are fixed on the encoder for subsequent `transform` on new data. Categories absent from the training data map to `general.target_encoding_fallback` (default: the global target mean). One output column per input column.
* `frequency` — frequency encoding (`FrequencyEncodingTransformer`): each category is replaced by its relative frequency in the training data. Deterministic; unknown categories map to `0.0`. One output column per input column.
* `hashing` — feature hashing (`HashingEncodingTransformer`): each category is deterministically hashed (blake2b, salt derived from the random seed) into `general.hashing_n_components` binary columns per input column. The output dimensionality is fixed and independent of column cardinality — unlike one-hot, high-cardinality columns do not cause feature explosion. Unknown categories are hashed the same way, so prediction never fails. Hash collisions (different categories landing in the same column) are a documented limitation.

**Sparse output of `hashing`**

The hashing transformer returns a `scipy.sparse.csr_matrix` (COO→CSR, `nnz = n_samples * n_columns`, values `1.0`) instead of a dense `float64` matrix — exactly like sklearn's standard `FeatureHasher`. Building a dense `(n_rows, n_cols * n_components)` matrix would consume gigabytes of memory on real high-cardinality tables (OOM risk). `SplitCategoricalEncoder` is sparse-aware: parts are joined with `sp.hstack` when any encoder produces a sparse matrix.

Because `ColumnTransformer` uses the default `sparse_threshold=0.3`, its output becomes a `csr_matrix` whenever a sparse part is present. The models `gaussian_process_regression`, `isotonic_regression` and `ardregression` reject sparse input, so for them the preprocessor is automatically built with `force_dense_output=True` (`sparse_threshold=0.0`), returning a dense matrix (flag derived per algorithm by `models.requires_dense_input` in both `ModelTrainer` and `tuner.optimize`). `DataOversampler` handles sparse input as well: `random` oversampling stays sparse, while `smote`/`adasyn` convert the input to dense (they require dense distance matrices by nature).

**High-cardinality mode**

When `general.high_cardinality_threshold` and `general.high_cardinality_encoding` are set together, columns whose number of unique values strictly exceeds the threshold are encoded with `high_cardinality_encoding`, while the remaining categorical columns keep `categorical_encoding`. The split is computed at fit time from the training data only (via `SplitCategoricalEncoder`), so the decision never depends on validation/test observations. A threshold of `0` marks every non-empty column as high-cardinality; a threshold larger than every column's cardinality effectively disables the mode.

**Edge cases**

* Columns with a single unique category or fully missing columns are handled without failures for all strategies (missing values are imputed with the most frequent category before encoding).
* NaNs in categorical columns are converted to strings and treated as ordinary categories.
* Configurations without the new parameters and models saved before the introduction of these strategies keep their previous behavior (backward compatibility).
* `target`/`frequency`/`hashing` encode columns before oversampling (the preprocessor step precedes the sampler step in the pipeline), so SMOTE/ADASYN always receive already-encoded numeric features.

---

## `run_parallel(func, args_seq, ...)`

Executes a function in parallel across multiple workers with shared memory and disk persistence support.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | — | Function to execute. |
| `args_seq` | `Iterable[Sequence]` | `[()]` | Sequence of argument tuples. |
| `kwargs_seq` | `Iterable[Mapping]` | — | Sequence of keyword argument dicts. |
| `max_workers` | `int` or `None` | `None` | Max worker count (defaults to CPU count). |
| `mode` | `str` | `"threads"` | `"threads"` or `"processes"`. |
| `timeout` | `float` or `None` | `None` | Global timeout in seconds. Used only in `"processes"` mode (hard: hung workers are force-terminated). `None` — no limit. In `"threads"` mode it is ignored with a warning — a running Python function cannot be interrupted, so a soft timeout did not save time and only discarded ready results. |
| `task_timeout` | `float` or `None` | `None` | Per-task timeout in seconds. Used only in `"processes"` mode. In `"threads"` mode it is ignored with a warning: the deadline counted from submit made queued tasks "expire" before starting. |
| `shared_args_indices` | `list[int]` or `None` | `None` | Indices of DataFrame args to share via shared memory. |
| `disk_args_indices` | `list[int]` or `None` | `None` | Indices of DataFrame args to persist to disk. |

**Returns:** `list[Any]` — results in the same order as inputs; failed tasks return `None`.

---

## `train_model(cfg_or_algo, metric_or_testsize, params_or_metric, ...)`

Legacy facade function for training a single model. Accepts either a config dictionary or positional arguments.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg_or_algo` | `dict` or `str` | Config dict or algorithm name. |
| `metric_or_testsize` | `str` or `float` | Metric name or test size. |
| `params_or_metric` | `dict` or `str` | Hyperparameters or metric name. |
| `X` | `Any` | Feature matrix. |
| `y` | `Any` | Target vector. |
| `enable_logging` | `bool` | Enable result logging. |
| `random_state` | `int` or `None` | Random seed. |
| `log_path` | `str`, `Path`, or `None` | Log file path. Used only when `enable_logging=True`: file logging is initialized via `setup_logging` (one file handler per process — the first path wins), and the training-result message and library logs are written to this file. In the config-dict branch the `log_path` key takes precedence over this argument, except when present with a `None` value (falls back to the argument). |
| `feature_selection_cfg` (keyword-only) | `FeatureSelectionCfg`, `dict`, or `None` | Feature selection config (method, percentile, `min_features`, etc.). In the config-dict branch the `feature_selection_cfg` key takes precedence; a key present with a `None` value is treated as unset and falls back to this argument; `None` means `FeatureSelectionCfg()` (`mode='disabled'`). |
| `feature_selection_active` (keyword-only) | `bool` or `None` | Explicit feature selection activity flag (takes precedence over the config mode). In the config-dict branch the `feature_selection_active` key takes precedence; a key present with a `None` value is treated as unset and falls back to this argument; non-bool values are rejected by `ModelTrainer`; `None` defers the decision to the config. |

**Returns:** `float` — validation metric value computed on hold-out data (or the CV estimate: the fold mean for additive metrics, the pooled OOF value for non-additive ones) with the `ModelTrainer` validation strategy — an honest out-of-sample estimate, not a train score.

---

## `DataOversampler`

Oversampling wrapper compatible with imbalanced-learn pipelines.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multiplier` | `float` | `1.0` | Dataset size multiplier. |
| `algorithm` | `str` | `"random"` | Algorithm (`"random"`, `"smote"`, `"adasyn"`). |
| `add_noise` | `bool` | `False` | Add Gaussian noise to numeric features. |
| `balance` | `bool` | `False` | Balance all classes to majority class size. |
| `random_state` | `int` or `None` | `42` | Random seed. |
| `noise_level` | `float` | `0.01` | Noise intensity relative to std. |

**Key methods:**

* `fit_resample(X, y)` — Resample with bypass of `_check_X_y` for categorical data support.
* `oversample(data, target)` — DataFrame-level oversampling interface.

---
"""The HPO run of a single algorithm (``_OptimizeRunner``).

Holds the state of one ``optimize()`` call and the run loop: the Optuna
study with the TPE history transfer, the objective with the circuit breaker
for fatal failures, the best-trial selection and the final fit of the best
model. The setup is a chain of pure steps that return plans
(``hpo.plans``): ``plan_validation`` (here: input validation and the
validation strategy) → ``plan_preprocessing`` (``hpo.trial_pipeline``) →
``plan_search_space`` (``hpo.search_space``) → ``plan_pruning``
(``hpo.pruning``); the trial evaluation is ``score_trial``
(``hpo.evaluation``). Only the runner's own methods mutate the runner. It
must not import ``configurable_automl_engine.tuner`` (the façade imports
this module).
"""

from __future__ import annotations

import logging as _logging
from typing import Any, NoReturn, cast

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FrozenTrial, Trial
from scipy import sparse as sp

from configurable_automl_engine.common.metrics import (
    is_non_additive_metric,
)
from configurable_automl_engine.common.validation_utils import (
    check_target_numeric,
    get_effective_train_size,
    validate_features,
)
from configurable_automl_engine.feature_selection import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from configurable_automl_engine.models import (
    create_model,
    resolve_algorithm_name,
)
from configurable_automl_engine.validation import make_cv, norm_val_method

from .evaluation import (
    _build_scorer,
    _phase_oof_matrix,
    score_trial,
)
from .plans import (
    BestTrial,
    PreprocessingPlan,
    PruningPlan,
    SearchSpacePlan,
    ValidationPlan,
)
from .pruning import plan_pruning
from .request import HpoRequest
from .search_space import (
    _canonical_distributions,
    _enqueueable_initial_params,
    _warn_if_optimum_at_boundary,
    log_complexity_guard,
    plan_search_space,
)
from .trial_pipeline import (
    _resolve_apply_fs,
    _resolve_isotonic_feature_name,
    assemble_estimator,
    plan_preprocessing,
)
from .types import (
    WORST_SCORE_THRESHOLD,
    HPOHistoryError,
    HyperoptError,
    InvalidAlgorithmError,
    InvalidDataError,
    OptimizeResult,
)

log = _logging.getLogger(__name__)


def _validate_data(
    X: Any,
    y: Any,
    *,
    inf_policy: str = "as_missing",
    datetime_policy: str = "expand",
    drop_id_like_columns: bool = False,
) -> Any:
    """Validate the types and sizes of the input data X and y.

    The target ``y`` is additionally required to be a NUMERIC vector
    without NaN/±inf: a non-finite target makes per-fold evaluations depend
    on which rows happen to land in which fold (the folds containing the
    values crash, the rest look "clean"), so the HPO budget is spent on
    randomly clean folds and the final full-data fit fails with an unclear
    error. The numeric/finite/shape checks are shared with the pipeline
    entry point (``check_target_numeric``) so both callers reject a bad
    target identically.

    Feature diagnostics mirror the pipeline entry point and honor the
    configurable policies: duplicate column names are always rejected;
    ±inf in numeric features follows ``inf_policy`` (``"as_missing"`` by
    default, as in ``general.inf_policy``; ``"error"`` rejects the data);
    datetime/timedelta columns follow ``datetime_policy`` (``"expand"`` by
    default — the derived components keep them from being silently dropped
    by the preprocessor); ID-like columns follow ``drop_id_like_columns``.
    The ID-like WARNING is suppressed here — it is emitted by the pipeline
    entry point (``prepare_dataset``) for engine runs and by the standalone
    ``optimize()`` for direct callers.

    Args:
        X (Any): Feature matrix (expected np.ndarray or pd.DataFrame).
        y (Any): Target variable vector
            (expected np.ndarray, pd.Series or pd.DataFrame).
        inf_policy: Policy for ±inf in numeric features (default
            ``"as_missing"`` — replace with NaN and warn; ``"error"`` —
            reject with InvalidDataError).
        datetime_policy: Policy for datetime/timedelta columns (default
            ``"expand"`` — replace with derived components).
        drop_id_like_columns: Whether ID-like columns are dropped (default
            ``False``).
    Returns:
        The (possibly repaired) feature matrix ``X``: for a ``pd.DataFrame``
        with datetime columns the expanded frame is returned; otherwise the
        original ``X`` is returned unchanged.
    Raises:
        InvalidDataError: If the data types are not supported
            or the sizes of X and y do not match, or the target y is
            non-numeric or contains NaN/±inf, or the features contain
            duplicate column names / infinite values (when the policy
            rejects them).
    """
    ok_types = (np.ndarray, pd.DataFrame)
    if not isinstance(X, ok_types):
        raise InvalidDataError("X must be a numpy.ndarray or pandas.DataFrame")
    if not isinstance(y, ok_types + (pd.Series,)):
        raise InvalidDataError(
            "y must be a numpy.ndarray, pandas.Series, or pandas.DataFrame"
        )
    if len(X) != len(y):
        raise InvalidDataError(f"Size mismatch: X={len(X)} and y={len(y)}")
    try:
        check_target_numeric(y)
    except ValueError as err:
        raise InvalidDataError(str(err)) from err
    # Feature diagnostics: duplicate columns are always rejected (a duplicated
    # name makes X[col] return a DataFrame — an unclear pandas error
    # downstream); ±inf / datetime / ID-like columns follow the configured
    # policies.
    if isinstance(X, pd.DataFrame):
        try:
            repaired, _ = validate_features(
                X,
                inf_policy=inf_policy,
                datetime_policy=datetime_policy,
                drop_id_like_columns=drop_id_like_columns,
                warn_id_like=False,
            )
        except ValueError as err:
            raise InvalidDataError(str(err)) from err
        return repaired
    return X


def _get_estimator(algo: str) -> Any:
    """Check the availability and validity of the given algorithm.

    Args:
        algo (str): Algorithm name.
    Returns:
        Any: Returns True if the model is created successfully by the base factory.
    Raises:
        InvalidAlgorithmError: If the algorithm is not supported or a required
            dependency (library) is missing.
    """
    try:
        # Try to create a model with minimal parameters to check existence
        create_model(algo)
        return True
    except (ValueError, ImportError) as err:
        # If the algorithm is not found in models.py or the package is not
        #  installed (e.g. XGBoost)
        raise InvalidAlgorithmError(f"Algorithm '{algo}' is not supported: {err}")


class _OptimizeRunner:
    """A single hyperparameter optimisation run, split into steps.

    ``optimize()`` remains the public entry point with its original signature
    and delegates all the work to this class through ``run_optimization``:

        run()
         ├─ plan_validation()        → ValidationPlan   (repaired X)
         ├─ plan_preprocessing()     → PreprocessingPlan (cleaned X; before
         │                             the search space: the complexity
         │                             guard needs p)
         ├─ plan_search_space()      → SearchSpacePlan
         ├─ plan_pruning()           → PruningPlan
         ├─ _transfer_previous_trials()  (TPE history warm-start)
         ├─ _objective() → score_trial() / handle_failure()  (Optuna loop)
         ├─ select_best_trial()      → BestTrial
         └─ _fit_best_model()  (skipped when ``fit_best_model=False``)

    The setup steps are pure functions; the runner stores their plans and
    passes them explicitly to the trial evaluation. When a step changes the
    feature matrix (the validation repairs it, the preprocessing plan drops
    uninformative columns) the plan returns the new matrix and ``run()``
    replaces ``self.X`` in one visible line.

    TPE history transfer: when ``previous_trials`` is passed, the trials are
    added to the study with ``study.add_trials`` AFTER ``create_study`` and
    BEFORE ``enqueue_trial``/``optimize``. The boundary of the current phase's
    own trials (``phase_start_number``) is fixed right after the transfer, so
    ``select_best_trial`` counts and ranks ONLY the trials of the current
    phase — the transferred history feeds the TPE sampler but cannot mask a
    total failure of the current phase (the no-fallback refine contract).
    """

    MAX_FATAL_FAILURES = 5
    """Limit of consecutive fatal failures before the algorithm is disqualified."""

    validation: ValidationPlan
    preprocessing: PreprocessingPlan
    search: SearchSpacePlan
    pruning: PruningPlan
    scorer: Any
    study: optuna.Study
    best: BestTrial
    best_model: Any

    def __init__(self, X: Any, y: Any, request: HpoRequest) -> None:
        """Store the data and the settings of a single optimisation run.

        Args:
            X (Any): The feature matrix.
            y (Any): The target vector.
            request (HpoRequest): The run settings (see ``optimize()`` for
                the semantics of every field).
        """
        self.request = request
        self.X = X
        self.y = y
        self.algo = request.algo_name.lower()
        # Whether the metric is non-additive (RMSE, MSE, R², nRMSE) and must
        # therefore be evaluated over the pooled out-of-fold vector instead of
        # the average of per-fold values (required for LOO, applied everywhere
        # with multiple folds). MAE and similar per-sample means stay on the
        # fold-average path (their fold mean equals the pooled value).
        self.pool_metric = is_non_additive_metric(request.metric)
        # The trial-number boundary between the transferred TPE history and the
        # current phase's own trials: fixed in run() right after
        # study.add_trials(previous_trials) and before enqueue_trial/optimize.
        self.phase_start_number = 0
        # Trials of the current phase only (number >= phase_start_number);
        # filled in run() after the Optuna loop and returned in OptimizeResult.
        self.phase_trials: list[FrozenTrial] = []
        self.consecutive_fatal_failures = 0
        # The out-of-fold predictions of every completed trial with a valid
        # score (the BBC-CV input): trial number -> OOF vector of len(X).
        self.trial_oof: dict[int, np.ndarray] = {}

    def run(self) -> OptimizeResult | None:
        """Run the full optimisation cycle, split into steps.

        Returns:
            OptimizeResult: The result of the phase — the final model, the
                best params, the best score, and the trials of the current
                phase ONLY (transferred ``previous_trials`` are excluded).
            None: The "no valid result" signal — ``n_trials=0`` or no
                COMPLETED trial of the current phase. Transferred history
                does NOT count toward the current phase validity.
        """
        if not self._validate_n_trials():
            return None

        request = self.request
        self.validation = plan_validation(self.X, self.y, request)
        self.X = self.validation.X
        # The preprocessor must be planned BEFORE the search space: the
        # adaptive complexity guard needs ``p`` — the number of features
        # after preprocessing (one-hot, missing-value indicators).
        self.preprocessing = plan_preprocessing(self.X, request)
        self.X = self.preprocessing.X
        self.search = plan_search_space(
            self.algo,
            self.y,
            space_override=(request.space_overrides or {}).get(self.algo),
            limit_hyperparameters=request.limit_hyperparameters,
            user_hyperparameter_keys=request.user_hyperparameter_keys,
            complexity_guard=request.complexity_guard,
            n_samples_eff=self.validation.n_samples_eff,
            preprocessing=self.preprocessing,
        )
        log_complexity_guard(self.search.guard_records, request.complexity_guard)
        self.scorer = _build_scorer(request.metric)
        self.pruning = plan_pruning(request.pruning, self.validation.method)

        # If early stopping is disabled, explicitly disable any pruner
        # (NopPruner) so that trials are guaranteed to run to completion.
        # The TPE sampler uses ``sampler_seed`` (per-phase, so startup points
        # do not repeat between HPO phases); the data splits and the
        # preprocessor keep ``random_state`` (RANDOM_STATE from the engine),
        # so every phase evaluates models on the same folds. ``None`` falls
        # back to ``random_state`` (the behavior of direct callers).
        sampler_seed = (
            request.sampler_seed
            if request.sampler_seed is not None
            else request.random_state
        )
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=sampler_seed),
            pruner=(
                self.pruning.pruner
                if self.pruning.active
                else optuna.pruners.NopPruner()
            ),
        )
        # TPE history transfer BEFORE enqueue_trial/optimize: the transferred
        # trials feed the TPE sampler (incl. the n_startup_trials window) but
        # are NOT part of the current phase's own trial set (the boundary
        # phase_start_number is fixed right after add_trials).
        self._transfer_previous_trials()
        self.phase_start_number = len(self.study.get_trials(deepcopy=False))
        # If there are parameters from the previous phase, enqueue them as the first trial
        if request.initial_params is not None:
            initial_params = _enqueueable_initial_params(
                request.initial_params, self.search.space_dict, self.algo
            )
            self.study.enqueue_trial(initial_params)
            log.info("Enqueued initial params from previous phase: %s", initial_params)
        try:
            self.study.optimize(self._objective, n_trials=request.n_trials)
        except InvalidAlgorithmError:
            log.warning(
                "Algorithm %s disqualified after %d consecutive fatal failures",
                self.algo,
                self.MAX_FATAL_FAILURES,
            )
            raise

        # The current phase's own trials (transferred ones have smaller
        # numbers and are excluded). Returned with the result so the caller
        # can warm-start the next phase with them.
        self.phase_trials = [
            t
            for t in self.study.get_trials(deepcopy=False)
            if t.number >= self.phase_start_number
        ]
        if not self.select_best_trial():
            return None
        self._log_best_result()
        oof_predictions = _phase_oof_matrix(self.phase_trials, self.trial_oof)
        # HPO-only mode (``fit_best_model=False``, e.g. the training engine's
        # phases): the caller needs only the best parameters and score — the
        # final model is trained later by its own trainer. The redundant
        # full-data fit is skipped, saving one full training per algorithm
        # per phase; the model slot of the OptimizeResult stays None.
        model = self._fit_best_model() if request.fit_best_model else None
        return OptimizeResult(
            model=model,
            params=self.best.params,
            score=self.best.score,
            trials=self.phase_trials,
            complexity_guard=self.search.guard_records or None,
            n_trials_evaluated=self.best.n_trials_evaluated,
            oof_predictions=oof_predictions,
        )

    # ─────────────────── run steps ─────────────────────────────────────────

    def _transfer_previous_trials(self) -> None:
        """Transfer the TPE history of the previous phase into the study.

        Fails fast (``HPOHistoryError``) instead of silently skipping the
        history:
        - when some transferred trial has a parameter whose distribution does
          not match the current search space (a search space change between
          phases would break the TPE state in an obscure way during the first
          trial of the new phase);
        - when the current search space cannot be reproduced by the
          distribution pre-check (an exotic custom ``space_fn``) — the
          transfer cannot be validated, and silently dropping the history
          would degrade refine to a cold run;
        - an empty list is a no-op (``previous_trials=[]`` is allowed).
        """
        previous_trials = self.request.previous_trials
        if not previous_trials:
            return
        canonical = _canonical_distributions(
            self.search.space_fn,
            fs_mode=self.validation.fs_mode,
            is_isotonic=self.validation.is_isotonic,
            algo=self.algo,
        )
        if canonical is None:
            # The search space cannot be reproduced by the pre-check (an
            # exotic custom ``space_fn``, e.g. one using legacy ``suggest_*``
            # methods the recorder does not implement). Transferring the
            # history blind would risk a silent ``study.add_trials`` desync,
            # and skipping the transfer would silently degrade refine to a
            # cold run — both violate the fail-fast contract: raise.
            raise HPOHistoryError(
                f"Cannot transfer TPE history for algorithm '{self.algo}': "
                "the search space cannot be reproduced for the distribution "
                "compatibility check (the space function uses constructs the "
                "check does not support). TPE history transfer requires a "
                "search space that can be replayed for validation."
            )
        for trial in previous_trials:
            for name, dist in trial.distributions.items():
                expected = canonical.get(name)
                if expected is None or expected != dist:
                    raise HPOHistoryError(
                        f"Cannot transfer trial #{trial.number} of algorithm "
                        f"'{self.algo}': parameter '{name}' has distribution "
                        f"{dist}, incompatible with the current search space "
                        f"({expected!r}). TPE history transfer requires the "
                        f"same search space between HPO phases."
                    )
        self.study.add_trials(previous_trials)
        log.info(
            "Transferred %d trial(s) from the previous phase", len(previous_trials)
        )

    def assemble_estimator(
        self,
        model_impl: Any,
        apply_fs: bool = False,
        feature_index: int | None = None,
    ) -> Any:
        """Assemble the trial pipeline; see ``trial_pipeline.assemble_estimator``."""
        return assemble_estimator(
            model_impl,
            self.preprocessing,
            is_isotonic=self.validation.is_isotonic,
            fs_cfg=self.validation.fs_cfg,
            oversampling=self.request.oversampling,
            random_state=self.request.random_state,
            apply_fs=apply_fs,
            feature_index=feature_index,
        )

    def evaluate_trial(self, trial: Trial, current_estimator: Any) -> float:
        """Evaluate the assembled trial pipeline and return the target metric.

        Delegates the evaluation to ``score_trial``. With ``collect_oof`` the
        out-of-fold predictions of the trial are kept in ``trial_oof`` when
        the trial produced a valid score (the worst-score sentinel — a failed
        fold or a non-finite metric — is not a selectable candidate and is
        not kept).

        Args:
            trial (Trial): The current Optuna trial object.
            current_estimator (Any): The pipeline/model of the current trial.
        Returns:
            float: The target metric value for the current parameter set.
        Raises:
            ValueError: Non-fatal evaluation error (the trial is pruned).
            MemoryError, RuntimeError, InvalidDataError: Fatal failures —
                counted in the ``handle_failure`` counter.
            optuna.TrialPruned: Pruning by the early stopping pruner.
        """
        request = self.request
        result = score_trial(
            current_estimator,
            self.X,
            self.y,
            trial=trial,
            validation=self.validation,
            scorer=self.scorer,
            pruning_active=self.pruning.active,
            pool_metric=self.pool_metric,
            random_state=request.random_state,
            algo_name=request.algo_name,
            phase_name=request.phase_name or "",
            collect_oof=request.collect_oof,
        )
        if result.oof is not None and result.value > WORST_SCORE_THRESHOLD:
            self.trial_oof[trial.number] = result.oof
        return result.value

    def _objective(self, trial: Trial) -> float:
        """Optuna objective function: evaluate_trial + handle_failure.

        Splits trial evaluation (``evaluate_trial``) and its failure handling
        (``handle_failure``) into separate steps. Hyperparameter generation and
        pipeline assembly run OUTSIDE try/except — as in the original
        implementation: errors from a user space function or the model
        constructor must propagate (trial FAIL and an exception from
        ``study.optimize``) rather than silently become pruning /
        disqualification. The consecutive fatal failure counter is reset on
        any non-fatal outcome: success, pruner pruning (optuna.TrialPruned
        from ``_evaluate_with_intermediate_reports``), or a non-fatal
        ValueError.

        Args:
            trial (Trial): The current Optuna trial object.
        Returns:
            float: The target metric value for the current parameter set.
        Raises:
            optuna.TrialPruned: If the trial is pruned (by an error or the pruner).
            InvalidAlgorithmError: When the fatal failure limit is exceeded.
        """
        # 1. hyperparameters, model and pipeline assembly — outside try/except
        #    (contract of the original code: their errors are not fatal
        #    evaluation failures and are not counted by the circuit breaker).
        params = self.search.space_fn(trial)
        model_kwargs = dict(params)
        # The model seed follows the run's ``random_state`` (the engine seed),
        # unless the user set ``random_state`` explicitly in the algorithm
        # hyperparameters — an explicit value always wins (including an
        # explicit None → a truly random fit, a WARNING is logged by
        # ``create_model``).
        if "random_state" not in model_kwargs:
            model_kwargs["random_state"] = self.request.random_state
        model = create_model(self.algo, **model_kwargs)
        current_estimator = self.assemble_estimator(
            model,
            apply_fs=_resolve_apply_fs(
                trial, self.validation.fs_mode, self.validation.is_isotonic
            ),
            feature_index=params.get("feature_index"),
        )

        fatal_failure = False
        try:
            return self.evaluate_trial(trial, current_estimator)
        except (ValueError, MemoryError, RuntimeError, InvalidDataError) as err:
            fatal_failure = not isinstance(err, ValueError)
            self.handle_failure(trial, err)
            # Unreachable: handle_failure always raises (NoReturn). The stub
            # is only needed by static analysis so the except branch has a
            # return value.
            return 0.0  # pragma: no cover
        finally:
            # Any non-fatal trial outcome breaks the sequence of fatal
            # failures: a successful _score return, pruner pruning
            # (optuna.TrialPruned), or a non-fatal ValueError. Without this
            # reset, the counter would accumulate errors separated by normal
            # trials and falsely disqualify the algorithm.
            if not fatal_failure:
                self.consecutive_fatal_failures = 0

    def handle_failure(self, trial: Trial, err: Exception) -> NoReturn:
        """Handle a trial evaluation failure according to the outcome contract.

        A non-fatal ``ValueError`` simply prunes the trial
        (``optuna.TrialPruned``) without counting it. ``MemoryError`` /
        ``RuntimeError`` / ``InvalidDataError`` are considered fatal: they
        increment the consecutive failure counter and disqualify the algorithm
        once the limit is reached (``InvalidAlgorithmError``).

        Args:
            trial (Trial): The current Optuna trial (for the user_attr "fail_reason").
            err (Exception): The caught evaluation exception.
        Raises:
            optuna.TrialPruned: Always (non-fatal or fatal failure).
            InvalidAlgorithmError: If the fatal failure limit is reached.
        """
        trial.set_user_attr("fail_reason", str(err))
        if isinstance(err, ValueError):
            raise optuna.TrialPruned() from err
        self.consecutive_fatal_failures += 1
        if self.consecutive_fatal_failures >= self.MAX_FATAL_FAILURES:
            raise InvalidAlgorithmError(
                f"Algorithm '{self.algo}' disqualified after "
                f"{self.consecutive_fatal_failures} consecutive fatal failures"
            ) from err
        raise optuna.TrialPruned() from err

    def select_best_trial(self) -> bool:
        """Select the best completed trial of the CURRENT phase.

        Explicit trial state counting: the presence of a valid result is
        determined by the number of completed (COMPLETED) trials of the
        current phase (``number >= phase_start_number``) rather than by the
        internal behavior of ``study.best_params``. Transferred trials
        (``previous_trials``) feed the TPE sampler but are NEVER counted or
        ranked here: they must not mask a total failure of the current phase
        (otherwise a refine phase that wiped out all its own trials would
        silently fall back to the previous phase's parameters — the forbidden
        no-fallback contract). Counters are logged for observability ("no
        valid result" must be explainable from the logs).

        The best trial is selected EXPLICITLY by its value among the current
        phase's COMPLETED trials; ``study.best_params`` is deliberately not
        used — it would rank the transferred trials too. The result is stored
        in ``self.best``.

        Returns:
            bool: True if there is a completed trial of the current phase;
                False — the "no valid result" contract (all OWN trials
                PRUNED/FAILED or an empty list).
        """
        own_trials = self.phase_trials or [
            t
            for t in self.study.get_trials(deepcopy=False)
            if t.number >= self.phase_start_number
        ]
        n_completed = sum(
            1 for t in own_trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        n_pruned = sum(
            1 for t in own_trials if t.state == optuna.trial.TrialState.PRUNED
        )
        n_failed = sum(1 for t in own_trials if t.state == optuna.trial.TrialState.FAIL)
        if self.pruning.active:
            log.info(
                "Early stopping: pruned %d of %d trials (algo=%s)",
                n_pruned,
                len(own_trials),
                self.algo,
            )
        log.info(
            "Trial states for algorithm '%s': completed=%d, pruned=%d, failed=%d",
            self.algo,
            n_completed,
            n_pruned,
            n_failed,
        )
        if n_completed == 0:
            # The "no valid result" contract: all OWN trials are pruned
            # (PRUNED), failed (FAILED), or the trial list is empty. Transferred
            # history is irrelevant here. Return False so the caller (_run_hpo)
            # can tell a failure from a valid result and exclude the algorithm
            # from the candidates.
            log.warning(
                "Algorithm '%s' produced no completed trials "
                "(completed=%d, pruned=%d, failed=%d); returning no result",
                self.algo,
                n_completed,
                n_pruned,
                n_failed,
            )
            return False

        best_trial = max(
            (t for t in own_trials if t.state == optuna.trial.TrialState.COMPLETE),
            key=lambda t: cast(float, t.value),
        )
        best_params = dict(best_trial.params)

        _warn_if_optimum_at_boundary(
            best_params,
            self.search,
            algo=self.algo,
            n_samples_eff=self.validation.n_samples_eff,
            preprocessing=self.preprocessing,
        )

        # The feature selection decision comes from the best trial's params
        # (mode='auto': the "use_feature_selection" key is already fixed by
        # Optuna) or from the configuration mode ('always' -> True, 'disabled'
        # -> False). For isotonic_regression selection is forcibly ignored.
        apply_fs = bool(
            best_params.get(
                "use_feature_selection",
                self.validation.fs_mode == FeatureSelectionMode.always,
            )
        )
        if self.validation.is_isotonic:
            apply_fs = False
            best_params = self._isotonic_best_params(best_params)
        # The base model constructor parameters are cleaned of the tuner's
        # auxiliary keys ("use_feature_selection" and the informational
        # "isotonic_feature" are not model hyperparameters; ``create_model``
        # rejects unknown names). They are built from the already filtered
        # best params (for isotonic_regression the key was excluded above)
        # so the source is single; the best trial's params are not mutated
        # and the key is guaranteed not to reach the model constructor.
        model_params = {
            k: v
            for k, v in best_params.items()
            if k not in ("use_feature_selection", "isotonic_feature")
        }
        self.best = BestTrial(
            params=best_params,
            score=float(cast(float, best_trial.value)),
            model_params=model_params,
            apply_fs=apply_fs,
            # The honest evaluation count reported next to the best-trial
            # score: the best score is the MAXIMUM over exactly these trials
            # (the more trials were evaluated on the same folds, the more
            # optimistic the score is expected to be). Transferred history
            # never counts.
            n_trials_evaluated=n_completed,
        )
        return True

    def _isotonic_best_params(self, best_params: dict[str, Any]) -> dict[str, Any]:
        """The reported best params of ``isotonic_regression``.

        The auxiliary ``use_feature_selection`` key is excluded (a guard
        against initial_params carried through enqueue_trial from the
        previous phase with the key), and the chosen NUMERICAL feature name
        is reported as ``isotonic_feature`` when it is resolvable from the
        preprocessor's ``get_feature_names_out`` (the numerical block
        occupies the last ``n_numeric`` columns).

        Args:
            best_params: The parameters of the best trial.

        Returns:
            dict[str, Any]: The reported params.
        """
        params = {k: v for k, v in best_params.items() if k != "use_feature_selection"}
        preprocessor = self.preprocessing.preprocessor
        if preprocessor is not None:
            feature_name = _resolve_isotonic_feature_name(
                preprocessor,
                self.preprocessing.n_numeric_eff,
                self.X,
                self.y,
                params.get("feature_index", 0),
            )
            if feature_name is not None:
                params["isotonic_feature"] = feature_name
        return params

    def _log_best_result(self) -> None:
        """Log the outcome of the best trial."""
        log.info(
            "Hyperopt: algo=%s | val=%s | score=%.5f | params=%s",
            self.algo,
            self.validation.method,
            self.best.score,
            self.best.params,
        )

    def _fit_best_model(self) -> Any:
        """Train the final model on all provided data.

        Assembles the pipeline with the best parameters (taking the feature
        selection decision from ``select_best_trial`` into account) and trains
        it on the full dataset. If oversampling was enabled, the final model
        goes through it as well. Skipped entirely when ``fit_best_model`` is
        False (HPO-only callers). The trained model is stored in
        ``self.best_model``.

        Returns:
            Any: The trained best model (``self.best_model``).
        """
        params = dict(self.best.model_params)
        if "random_state" not in params:
            params["random_state"] = self.request.random_state
        best_model = self.assemble_estimator(
            create_model(self.algo, **params),
            apply_fs=self.best.apply_fs,
            feature_index=self.best.model_params.get("feature_index"),
        )
        best_model.fit(self.X, self.y)

        self.best_model = best_model
        # Observability: the dimensionality and the storage format of the
        # preprocessed matrix the final model received (dense
        # one-hot on high-cardinality columns inflated the matrix with no log
        # signal). Best-effort — the log must never break the optimization.
        preprocessor = self.preprocessing.preprocessor
        if preprocessor is not None:
            try:
                matrix = preprocessor.transform(self.X)
                log.info(
                    "preprocessed matrix: %d × %d (%s)",
                    matrix.shape[0],
                    matrix.shape[1],
                    "sparse" if sp.issparse(matrix) else "dense",
                )
            except Exception:  # noqa: BLE001 — observability only
                log.debug(
                    "Could not log the preprocessed matrix dimensions for '%s'.",
                    self.algo,
                )
        return best_model

    # ─────────────────── private helpers ───────────────────────────────────

    def _validate_n_trials(self) -> bool:
        """Validate ``n_trials`` and the "no valid result" contract.

        Returns:
            bool: True if Optuna should run; False — return
                ``None`` (the "no valid result" signal) without running.
        Raises:
            ValueError: If ``n_trials`` is negative or not an integer.
        """
        n_trials = self.request.n_trials
        if not isinstance(n_trials, int) or n_trials < 0:
            raise ValueError(f"n_trials must be a non-negative integer, got {n_trials}")
        if n_trials == 0:
            # Empty trial list — the "no valid result" contract: return a
            # plain None without running Optuna and without exceptions
            # (including for algorithms without a search space).
            log.warning(
                "Algorithm '%s' requested with n_trials=0; no trials to run — "
                "returning no result",
                self.request.algo_name,
            )
            return False
        return True


def _normalize_feature_selection_cfg(value: Any) -> FeatureSelectionCfg:
    """Normalize the feature selection configuration of a run.

    ``None`` -> default (mode='disabled'), dict -> ``FeatureSelectionCfg``
    (validated at run time), ``FeatureSelectionCfg`` object -> as is.

    Args:
        value: The ``feature_selection_cfg`` of the request.

    Returns:
        FeatureSelectionCfg: The validated configuration.

    Raises:
        HyperoptError: If a dict with invalid values is passed.
        TypeError: If an object of an unsupported type is passed.
    """
    if value is None:
        return FeatureSelectionCfg()
    if isinstance(value, FeatureSelectionCfg):
        return value
    if isinstance(value, dict):
        try:
            return FeatureSelectionCfg.model_validate(value)
        except Exception as err:
            raise HyperoptError(f"Invalid feature_selection_cfg: {err}") from err
    raise TypeError(
        "feature_selection_cfg must be a FeatureSelectionCfg, dict or None, "
        f"got {type(value).__name__}"
    )


def plan_validation(X: Any, y: Any, request: HpoRequest) -> ValidationPlan:
    """Validate the input and resolve the validation strategy.

    Performs the data and algorithm availability checks, normalizes the
    feature selection configuration, and picks the CV strategy via
    ``make_cv``. The input data is not mutated: a repaired feature matrix
    (e.g. expanded datetime columns) is returned in the plan.

    Args:
        X (Any): The feature matrix.
        y (Any): The target vector.
        request (HpoRequest): The run settings.

    Returns:
        ValidationPlan: The repaired matrix and the resolved validation.

    Raises:
        InvalidDataError: If the data is invalid.
        InvalidAlgorithmError: If the algorithm is not available.
        HyperoptError: If the feature selection configuration is invalid.
    """
    algo = request.algo_name.lower()
    policies = request.policies
    # _validate_data may return a repaired feature matrix (e.g. datetime
    # columns expanded); the None return is kept for backward compatibility
    # with callers/mocks that treat the helper as a pure validator.
    validated = _validate_data(
        X,
        y,
        inf_policy=policies.inf_policy,
        datetime_policy=policies.datetime_policy,
        drop_id_like_columns=policies.drop_id_like_columns,
    )
    if isinstance(validated, (pd.DataFrame, np.ndarray)):
        X = validated

    _get_estimator(algo)

    # Isotonic regression is strictly univariate and uses its own
    # IsotonicDataTransformer, so selection is forcibly disabled for it
    # regardless of the mode.
    fs_cfg = _normalize_feature_selection_cfg(request.feature_selection_cfg)
    is_isotonic = resolve_algorithm_name(algo) == "isotonic_regression"
    log.info(
        "Feature selection mode for algorithm '%s': %s",
        algo,
        fs_cfg.mode.value,
    )

    # Determine how many rows the model actually "sees" during training
    # inside CV/Split, and resolve the validation strategy (incl. 'auto').
    n_samples = len(y)
    n_features = X.shape[1]
    n_samples_eff = get_effective_train_size(
        n_total=n_samples,
        strategy=request.val_method,
        n_folds=request.n_folds,
        test_size=request.train_test_split_test_size,
        n_features=n_features,
    )
    method, cv_obj, auto_decision = make_cv(
        n_samples,
        val_method=request.val_method,
        n_folds=request.n_folds,
        random_state=request.random_state,
        test_size=request.train_test_split_test_size,
        n_features=n_features,
    )
    # 'auto' is resolved exactly once (here, in make_cv). The resulting
    # decision is passed to iter_splits so that the test size is not
    # recomputed and cannot diverge between n_samples_eff and the actual
    # split.
    resolved_via_auto = norm_val_method(request.val_method) == "auto"
    if method == "train_test_split" and resolved_via_auto and auto_decision is not None:
        split_test_size: float | int = int(auto_decision["test_size"])
    else:
        split_test_size = request.train_test_split_test_size

    # A single fold count for the early stopping branch (k_fold). The
    # 'auto' strategy could resolve to kfold with a number of folds k that
    # differs from the original n_folds: in that case the pruning branch
    # must use the computed k, otherwise iter_splits inside
    # _evaluate_with_intermediate_reports would fall back to the default
    # n_folds and the evaluation would diverge from
    # cross_val_score(cv=cv_obj).
    if method == "k_fold":
        # Take the fold count from the ready cv_obj (for k_fold it is
        # guaranteed to be non-None — make_cv always creates a KFold in
        # this branch). One formula covers auto→kfold (k from the
        # decision), the 'auto' fallback without P (clamped to max(2, k)),
        # and an explicit k_fold.
        assert cv_obj is not None
        n_folds = int(cv_obj.get_n_splits())
    else:
        # Explicit loo / train_test_split: the fold count is not used.
        n_folds = request.n_folds
    return ValidationPlan(
        X=X,
        algo=algo,
        fs_cfg=fs_cfg,
        is_isotonic=is_isotonic,
        n_samples_eff=n_samples_eff,
        method=method,
        cv=cv_obj,
        auto_decision=auto_decision,
        resolved_via_auto=resolved_via_auto,
        n_folds=n_folds,
        test_size=request.train_test_split_test_size,
        split_test_size=split_test_size,
    )


def run_optimization(X: Any, y: Any, request: HpoRequest) -> OptimizeResult | None:
    """Run the hyperparameter search of one algorithm.

    The single entry point behind ``tuner.optimize`` and the training
    engine's HPO phases.

    Args:
        X (Any): The feature matrix.
        y (Any): The target vector.
        request (HpoRequest): The run settings.

    Returns:
        OptimizeResult | None: The phase result, or ``None`` — the "no valid
            result" signal (see ``optimize()``).
    """
    return _OptimizeRunner(X, y, request).run()

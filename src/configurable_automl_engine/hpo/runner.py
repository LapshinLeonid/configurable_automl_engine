"""The HPO run of a single algorithm (``_OptimizeRunner``).

Holds the state of one ``optimize()`` call and the run loop: input
validation and the validation strategy (``resolve_validation``), the Optuna
study with the TPE history transfer, the objective with the circuit breaker
for fatal failures, the best-trial selection and the final fit of the best
model. The search space, the trial evaluation,
the pruning configuration and the pipeline assembly are delegated to the
other ``hpo`` modules. It must not import ``configurable_automl_engine.tuner``
(the façade imports this module).
"""

from __future__ import annotations

import logging as _logging
from collections.abc import Callable
from typing import Any, NoReturn, cast

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FrozenTrial, Trial
from scipy import sparse as sp

from configurable_automl_engine.common.validation_utils import (
    check_target_numeric,
    get_effective_train_size,
    validate_features,
)
from configurable_automl_engine.models import (
    create_model,
    resolve_algorithm_name,
)
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from configurable_automl_engine.training_engine.metrics import (
    is_non_additive_metric,
)
from configurable_automl_engine.validation import make_cv, norm_val_method

from .evaluation import (
    _phase_oof_matrix,
    _score_trial,
)
from .pruning import (
    handle_pruning,
)
from .request import HpoRequest
from .search_space import (
    _canonical_distributions,
    _enqueueable_initial_params,
    _normalize_space_dict,
    _warn_if_optimum_at_boundary,
    resolve_search_space,
)
from .trial_pipeline import (
    _resolve_apply_fs,
    _resolve_isotonic_feature_name,
    assemble_estimator,
    build_trial_pipeline,
)
from .types import (
    WORST_SCORE_THRESHOLD,
    HPOHistoryError,
    HyperoptError,
    InvalidAlgorithmError,
    InvalidDataError,
    OptimizeResult,
)

# The records keep the logger name of the pre-split ``tuner`` module: the
# logger name is part of the observable behavior (handlers, filters and
# ``caplog`` assertions select records by it).
log = _logging.getLogger("configurable_automl_engine.tuner")


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
    and delegates all the work to this class. The monolithic logic of the
    original function is split into isolated steps:

        run()
         ├─ resolve_validation()
         ├─ build_trial_pipeline()               (before the search space:
         │                                        the complexity guard needs p)
         ├─ resolve_search_space()
         ├─ handle_pruning()
         ├─ transfer_previous_trials()        (TPE history warm-start)
         ├─ evaluate_trial() / handle_failure()  (in the Optuna loop)
         ├─ select_best_trial()
         └─ _fit_best_model()  (skipped when ``fit_best_model=False``)

    TPE history transfer: when ``previous_trials`` is passed, the trials are
    added to the study with ``study.add_trials`` AFTER ``create_study`` and
    BEFORE ``enqueue_trial``/``optimize``. The boundary of the current phase's
    own trials (``phase_start_number``) is fixed right after the transfer, so
    ``select_best_trial`` counts and ranks ONLY the trials of the current
    phase — the transferred history feeds the TPE sampler but cannot mask a
    total failure of the current phase (the no-fallback refine contract).

    The run state is stored in instance attributes: each step fills in its own
    fields, and subsequent steps (as well as the Optuna objective) read them
    without long argument chains.
    """

    MAX_FATAL_FAILURES = 5
    """Limit of consecutive fatal failures before the algorithm is disqualified."""

    def __init__(self, X: Any, y: Any, request: HpoRequest) -> None:
        """Store the data and the settings of a single optimisation run.

        Args:
            X (Any): The feature matrix.
            y (Any): The target vector.
            request (HpoRequest): The run settings (see ``optimize()`` for
                the semantics of every field).
        """
        self.request = request
        algo_name = request.algo_name
        metric = request.metric
        self.algo_name = algo_name
        self.X = X
        self.y = y
        self.metric = metric
        # Whether the metric is non-additive (RMSE, MSE, R², nRMSE) and must
        # therefore be evaluated over the pooled out-of-fold vector instead of
        # the average of per-fold values (required for LOO, applied everywhere
        # with multiple folds). MAE and similar per-sample means stay on the
        # fold-average path (their fold mean equals the pooled value).
        self.pool_metric = is_non_additive_metric(metric)
        self.val_method = request.val_method
        self.validation_strategy = None
        self.n_folds = request.n_folds
        self.n_trials = request.n_trials
        self.random_state = request.random_state
        # The seed of the TPE sampler. Deliberately separate from
        # ``random_state`` (which drives the data splits and the preprocessor):
        # each HPO phase may use a different ``sampler_seed`` so that TPE
        # startup points do not repeat, while the folds/splits stay the same
        # (RANDOM_STATE) — otherwise every phase would re-evaluate models on
        # other folds, breaking score monotonicity and mixing incomparable
        # estimates in the transferred TPE history. ``None`` (the default for
        # direct callers) falls back to ``random_state`` to preserve the old
        # behavior.
        self.sampler_seed = request.sampler_seed
        self.train_test_split_test_size = request.train_test_split_test_size
        self.space_overrides = request.space_overrides
        self.initial_params = request.initial_params
        self.previous_trials = request.previous_trials
        self.preprocessor = request.preprocessor
        self.categorical_features = request.categorical_features
        self.numerical_features = request.numerical_features
        self.encoding = request.encoding.encoding
        self.preprocessing_override = request.preprocessing_override
        self.pruning = request.pruning
        self.high_cardinality_threshold = request.encoding.high_cardinality_threshold
        self.high_cardinality_encoding = request.encoding.high_cardinality_encoding
        self.hashing_n_components = request.encoding.hashing_n_components
        self.target_encoding_smoothing = request.encoding.target_encoding_smoothing
        self.target_encoding_fallback = request.encoding.target_encoding_fallback
        self.feature_selection_cfg = request.feature_selection_cfg
        # The adaptive complexity guard config (trees/boosting on small data).
        # ``None`` — the defaults of ``general.complexity_guard`` (mode
        # 'apply'), so direct ``optimize`` callers get the same search spaces
        # as the engine; the guard is disabled with ``{"mode": "off"}``. The
        # training engine passes the config from ``general.complexity_guard``.
        self.complexity_guard_cfg = request.complexity_guard
        # Names of parameters set explicitly by the user
        # (``algo.hyperparameters``); the complexity guard treats them as
        # "user ranges" (intersected, with ``user_range_policy`` on conflict).
        # ``None`` (a direct call that did not pass the list) — resolved in
        # ``resolve_search_space`` to the keys of the dict override; the
        # training engine always passes the list (an empty one included).
        self.user_hyperparameter_keys: list[str] | None = (
            None
            if request.user_hyperparameter_keys is None
            else list(request.user_hyperparameter_keys)
        )
        # Restrict the search to the user-listed parameters only: the space
        # dict passed via ``space_overrides`` is used as-is (NO adaptive SVR
        # epsilon injection, NO DEFAULT_SPACES fallback). Mirrors
        # ``AlgoCfg.limit_hyperparameters``.
        self.limit_hyperparameters = request.limit_hyperparameters
        # Number of features after preprocessing (what the model sees).
        # Filled in build_trial_pipeline (falls back to the raw column count).
        self.n_features_eff: int = 1
        # Number of NUMERICAL features after preprocessing (one column per
        # numerical feature for isotonic — no one-hot expansion, no missing
        # indicators). Used to size the isotonic ``feature_index`` search
        # range: only numeric columns are meaningful inputs for the strictly
        # univariate model. ``None`` — not determinable (an external
        # preprocessor without explicit column lists).
        self.n_numeric_eff: int | None = None
        # Number of ORIGINAL columns (a categorical column counts as 1,
        # before the one-hot expansion). Used only by the complexity guard
        # for the size-mode ratio ``n / p``; filled in build_trial_pipeline.
        self.n_features_size: int = 1
        # Whether to train the best model on all data after the Optuna loop
        # (``_fit_best_model``). Set to False when the caller only needs the
        # best parameters and score (e.g. the training engine's HPO phase):
        # the full-data fit is then skipped and the ``model`` field of the
        # returned ``OptimizeResult`` is None.
        self.fit_best_model = request.fit_best_model
        # Whether to keep the out-of-fold predictions of every completed
        # trial (the BBC-CV input): trial number -> OOF vector of len(X).
        self.collect_oof = request.collect_oof
        self.trial_oof: dict[int, np.ndarray] = {}
        # The pipeline phase name (e.g. 'coarse'/'refine') for the fallback
        # WARNING deduplication: ``None`` for direct ``optimize()`` callers.
        self.phase_name = request.phase_name

        # Input-data diagnostics policies for the fail-fast feature checks:
        # defaults mirror the pipeline behavior for direct ``optimize()``
        # callers (S2 — the policies are configurable instead of hard-coded).
        self.inf_policy = request.policies.inf_policy
        self.datetime_policy = request.policies.datetime_policy
        self.drop_id_like_columns = request.policies.drop_id_like_columns

        # Names of columns dropped by cleaning uninformative features.
        # Filled in build_trial_pipeline for diagnostics; empty by default.
        self.dropped_features_: list[str] = []

        # Oversampling config for pipeline assembly (single point of truth).
        self.oversampling_config: dict[str, Any] = {
            "active": request.oversampling.enable,
            "params": request.oversampling.sampler_params(),
        }

        # ── Fields populated by the run() steps ───────────────────────────────
        self.algo: str = ""
        self.fs_cfg = FeatureSelectionCfg()
        self.fs_mode = self.fs_cfg.mode
        self.is_isotonic = False
        self.n_samples_eff: int = 0
        self.val_method_eff: str = ""
        self.cv_obj: Any = None
        self.auto_decision: dict[str, Any] | None = None
        self.resolved_via_auto = False
        self.effective_n_folds = self.n_folds
        # A placeholder until the search space is resolved (step 2); replaced
        # in resolve_search_space(). run() always calls resolve_search_space
        # before the Optuna loop, so the real placeholder call is impossible.
        self.space_fn: Callable[[Trial], dict[str, Any]] = lambda trial: {}
        # The resolved dict search space (parameter name -> SearchSpaceEntry or
        # constant) for dict-based spaces; empty for function/callable spaces.
        # Used to warn when the best trial lies at a range boundary.
        self.search_space_dict: dict[str, Any] = {}
        self.scorer: Any = None
        self.pruner: optuna.pruners.BasePruner | None = None
        self.pruning_active = False
        self.study: optuna.Study
        # The trial-number boundary between the transferred TPE history and the
        # current phase's own trials: fixed in run() right after
        # study.add_trials(previous_trials) and before enqueue_trial/optimize.
        self.phase_start_number = 0
        # Trials of the current phase only (number >= phase_start_number);
        # filled in run() after the Optuna loop and returned in OptimizeResult.
        self.phase_trials: list[FrozenTrial] = []
        self.consecutive_fatal_failures = 0
        self.best_params: dict[str, Any]
        self.best_score: float
        self.clean_model_params: dict[str, Any]
        self.best_apply_fs: bool
        # The number of trials actually evaluated by the phase (own COMPLETED
        # trials; transferred history never counts), reported in
        # ``OptimizeResult.n_trials_evaluated`` as the honest denominator next
        # to the best-trial score.
        self.n_trials_evaluated: int = 0
        # Log records of the adaptive complexity guard, filled in
        # resolve_search_space; travels with OptimizeResult for the report.
        self.complexity_guard_logs: list[dict[str, Any]] = []

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

        self.resolve_validation()
        # The preprocessor must be built BEFORE the search space is resolved:
        # the adaptive complexity guard needs ``p`` — the number of features
        # after preprocessing (one-hot, missing-value indicators).
        self.build_trial_pipeline()
        self.resolve_search_space()
        self.handle_pruning()

        # If early stopping is disabled, explicitly disable any pruner
        # (NopPruner) so that trials are guaranteed to run to completion.
        # The TPE sampler uses ``sampler_seed`` (per-phase, so startup points
        # do not repeat between HPO phases); the data splits and the
        # preprocessor keep ``random_state`` (RANDOM_STATE from the engine),
        # so every phase evaluates models on the same folds.
        sampler_seed = (
            self.sampler_seed if self.sampler_seed is not None else self.random_state
        )
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=sampler_seed),
            pruner=self.pruner if self.pruning_active else optuna.pruners.NopPruner(),
        )
        # TPE history transfer BEFORE enqueue_trial/optimize: the transferred
        # trials feed the TPE sampler (incl. the n_startup_trials window) but
        # are NOT part of the current phase's own trial set (the boundary
        # phase_start_number is fixed right after add_trials).
        self._transfer_previous_trials()
        self.phase_start_number = len(self.study.get_trials(deepcopy=False))
        # If there are parameters from the previous phase, enqueue them as the first trial
        if self.initial_params is not None:
            initial_params = _enqueueable_initial_params(self, self.initial_params)
            self.study.enqueue_trial(initial_params)
            log.info("Enqueued initial params from previous phase: %s", initial_params)
        try:
            self.study.optimize(self._objective, n_trials=self.n_trials)
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
        oof_predictions = _phase_oof_matrix(self)
        if not self.fit_best_model:
            # HPO-only mode (e.g. the training engine's phases): the caller
            # needs only the best parameters and score — the final model is
            # trained later by its own trainer. The redundant full-data fit
            # is skipped, saving one full training per algorithm per phase;
            # the model slot of the OptimizeResult stays None.
            return OptimizeResult(
                model=None,
                params=self.best_params,
                score=self.best_score,
                trials=self.phase_trials,
                complexity_guard=self.complexity_guard_logs or None,
                n_trials_evaluated=self.n_trials_evaluated,
                oof_predictions=oof_predictions,
            )
        return OptimizeResult(
            model=self._fit_best_model(),
            params=self.best_params,
            score=self.best_score,
            trials=self.phase_trials,
            complexity_guard=self.complexity_guard_logs or None,
            n_trials_evaluated=self.n_trials_evaluated,
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
        if not self.previous_trials:
            return
        canonical = _canonical_distributions(self)
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
        for trial in self.previous_trials:
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
        self.study.add_trials(self.previous_trials)
        log.info(
            "Transferred %d trial(s) from the previous phase", len(self.previous_trials)
        )

    def resolve_validation(self) -> None:
        """Resolve the validation strategy (step 1); see ``evaluation``."""
        resolve_validation(self)

    def _normalize_space_dict(self, space_dict: dict[str, Any]) -> dict[str, Any]:
        """Re-resolve inferred range types; see ``search_space._normalize_space_dict``."""
        return _normalize_space_dict(self, space_dict)

    def resolve_search_space(self) -> None:
        """Resolve the search space and the scorer (step 2); see ``search_space``."""
        resolve_search_space(self)

    def build_trial_pipeline(self) -> None:
        """Build the preprocessor for the HPO phase (step 3); see ``trial_pipeline``."""
        build_trial_pipeline(self)

    def assemble_estimator(
        self,
        model_impl: Any,
        apply_fs: bool = False,
        feature_index: int | None = None,
    ) -> Any:
        """Assemble the trial pipeline; see ``trial_pipeline.assemble_estimator``."""
        return assemble_estimator(self, model_impl, apply_fs, feature_index)

    def evaluate_trial(self, trial: Trial, current_estimator: Any) -> float:
        """Evaluate the assembled trial pipeline and return the target metric (step 4).

        Delegates the evaluation to ``_score_trial``. With ``collect_oof`` the
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
        if not self.collect_oof:
            return _score_trial(self, trial, current_estimator, None)
        oof = np.full(len(self.X), np.nan, dtype=float)
        score = _score_trial(self, trial, current_estimator, oof)
        if score > WORST_SCORE_THRESHOLD:
            self.trial_oof[trial.number] = oof
        return score

    def handle_pruning(self) -> None:
        """Configure early stopping (step 5); see ``pruning.handle_pruning``."""
        handle_pruning(self)

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
        params = self.space_fn(trial)
        model_kwargs = dict(params)
        # The model seed follows the run's ``random_state`` (the engine seed),
        # unless the user set ``random_state`` explicitly in the algorithm
        # hyperparameters — an explicit value always wins (including an
        # explicit None → a truly random fit, a WARNING is logged by
        # ``create_model``).
        if "random_state" not in model_kwargs:
            model_kwargs["random_state"] = self.random_state
        model = create_model(self.algo, **model_kwargs)
        current_estimator = self.assemble_estimator(
            model,
            apply_fs=_resolve_apply_fs(self, trial),
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
        """Handle a trial evaluation failure according to the outcome contract (step 6).

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

    def _warn_if_optimum_at_boundary(self, best_params: dict[str, Any]) -> None:
        """Warn when the best value lies at a range boundary; see ``search_space``."""
        _warn_if_optimum_at_boundary(self, best_params)

    def select_best_trial(self) -> bool:
        """Select the best completed trial of the CURRENT phase (step 7).

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
        used — it would rank the transferred trials too.

        Returns:
            bool: True if there is a completed trial of the current phase;
                False — the "no valid result" contract (all OWN trials
                PRUNED/FAILED or an empty list).
        """
        assert self.study is not None
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
        if self.pruning_active:
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

        # The honest evaluation count reported next to the best-trial score:
        # the best score is the MAXIMUM over exactly these trials (the more
        # trials were evaluated on the same folds, the more optimistic the
        # score is expected to be). Transferred history never counts.
        self.n_trials_evaluated = n_completed

        best_trial = max(
            (t for t in own_trials if t.state == optuna.trial.TrialState.COMPLETE),
            key=lambda t: cast(float, t.value),
        )
        self.best_score = float(cast(float, best_trial.value))
        best_params = dict(best_trial.params)

        self._warn_if_optimum_at_boundary(best_params)

        # The feature selection decision comes from the best trial's params
        # (mode='auto': the "use_feature_selection" key is already fixed by
        # Optuna) or from the configuration mode ('always' -> True, 'disabled'
        # -> False). For isotonic_regression selection is forcibly ignored.
        best_apply_fs = bool(
            best_params.get(
                "use_feature_selection", self.fs_mode == FeatureSelectionMode.always
            )
        )
        if self.is_isotonic:
            best_apply_fs = False
            # For isotonic_regression, the auxiliary key is also excluded from
            # the returned best_params (a guard against initial_params carried
            # through enqueue_trial from the previous phase with the
            # use_feature_selection key).
            best_params = {
                k: v for k, v in best_params.items() if k != "use_feature_selection"
            }
            # Report the chosen NUMERICAL feature name in the params
            # (``isotonic_feature``), resolved from the preprocessor's
            # get_feature_names_out (the numerical block occupies the last
            # ``n_numeric`` columns). Fitted only when the preprocessor is
            # available and the name is resolvable.
            if self.preprocessor is not None:
                feature_name = _resolve_isotonic_feature_name(
                    self, best_params.get("feature_index", 0)
                )
                if feature_name is not None:
                    best_params["isotonic_feature"] = feature_name
        self.best_params = best_params
        # The base model constructor parameters are cleaned of the tuner's
        # auxiliary keys ("use_feature_selection" and the informational
        # "isotonic_feature" are not model hyperparameters; ``create_model``
        # rejects unknown names). They are built from the already filtered
        # self.best_params (for isotonic_regression the key was excluded
        # above) so the source is single; the best trial's params are
        # not mutated and the key is guaranteed not to reach the model
        # constructor.
        self.clean_model_params = dict(self.best_params)
        self.clean_model_params.pop("use_feature_selection", None)
        self.clean_model_params.pop("isotonic_feature", None)
        self.best_apply_fs = best_apply_fs
        return True

    def _log_best_result(self) -> None:
        """Log the outcome of the best trial (shared by both return paths)."""
        log.info(
            "Hyperopt: algo=%s | val=%s | score=%.5f | params=%s",
            self.algo,
            self.val_method_eff,
            self.best_score,
            self.best_params,
        )

    def _fit_best_model(self) -> Any:
        """Train the final model on all provided data (step 8).

        Assembles the pipeline with the best parameters (taking the feature
        selection decision from ``select_best_trial`` into account) and trains
        it on the full dataset. If oversampling was enabled, the final model
        goes through it as well. Skipped entirely when ``fit_best_model`` is
        False (HPO-only callers). The trained model is stored in
        ``self.best_model``.

        Returns:
            Any: The trained best model (``self.best_model``).
        """
        clean_params = dict(self.clean_model_params)
        if "random_state" not in clean_params:
            clean_params["random_state"] = self.random_state
        best_model = self.assemble_estimator(
            create_model(self.algo, **clean_params),
            apply_fs=self.best_apply_fs,
            feature_index=self.clean_model_params.get("feature_index"),
        )
        best_model.fit(self.X, self.y)

        self.best_model = best_model
        # Observability: the dimensionality and the storage format of the
        # preprocessed matrix the final model received (dense
        # one-hot on high-cardinality columns inflated the matrix with no log
        # signal). Best-effort — the log must never break the optimization.
        if self.preprocessor is not None:
            try:
                matrix = self.preprocessor.transform(self.X)
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
        if not isinstance(self.n_trials, int) or self.n_trials < 0:
            raise ValueError(
                f"n_trials must be a non-negative integer, got {self.n_trials}"
            )
        if self.n_trials == 0:
            # Empty trial list — the "no valid result" contract: return a
            # plain None without running Optuna and without exceptions
            # (including for algorithms without a search space).
            log.warning(
                "Algorithm '%s' requested with n_trials=0; no trials to run — "
                "returning no result",
                self.algo_name,
            )
            return False
        return True


def resolve_validation(runner: _OptimizeRunner) -> None:
    """Resolve the validation strategy and related quantities (step 1).

    Performs input normalization (the ``validation_strategy`` alias), data
    and algorithm availability checks, normalization of the feature
    selection configuration, and picks the CV strategy via ``make_cv``.
    The results are stored in instance attributes for later steps.
    """
    # the alias takes precedence
    if runner.validation_strategy is not None:
        runner.val_method = runner.validation_strategy

    runner.algo = runner.algo_name.lower()
    # _validate_data may return a repaired feature matrix (e.g. datetime
    # columns expanded); the None return is kept for backward compatibility
    # with callers/mocks that treat the helper as a pure validator.
    _validated = _validate_data(
        runner.X,
        runner.y,
        inf_policy=runner.inf_policy,
        datetime_policy=runner.datetime_policy,
        drop_id_like_columns=runner.drop_id_like_columns,
    )
    if isinstance(_validated, (pd.DataFrame, np.ndarray)):
        runner.X = _validated

    _get_estimator(runner.algo)

    # Normalize the feature selection configuration: None -> default
    # (mode='disabled'), dict -> FeatureSelectionCfg (validated at run
    # time), FeatureSelectionCfg object -> as is. Isotonic regression is
    # strictly univariate and uses its own IsotonicDataTransformer, so
    # selection is forcibly disabled for it regardless of the mode.
    fs_cfg = runner.feature_selection_cfg
    if fs_cfg is None:
        runner.fs_cfg = FeatureSelectionCfg()
    elif isinstance(fs_cfg, FeatureSelectionCfg):
        runner.fs_cfg = fs_cfg
    elif isinstance(fs_cfg, dict):
        try:
            runner.fs_cfg = FeatureSelectionCfg.model_validate(fs_cfg)
        except Exception as err:
            raise HyperoptError(f"Invalid feature_selection_cfg: {err}") from err
    else:
        raise TypeError(
            "feature_selection_cfg must be a FeatureSelectionCfg, dict or None, "
            f"got {type(fs_cfg).__name__}"
        )
    runner.fs_mode = runner.fs_cfg.mode
    runner.is_isotonic = resolve_algorithm_name(runner.algo) == "isotonic_regression"
    log.info(
        "Feature selection mode for algorithm '%s': %s",
        runner.algo,
        runner.fs_mode.value,
    )

    # Determine how many rows the model actually "sees" during training
    # inside CV/Split, and resolve the validation strategy (incl. 'auto').
    n_samples = len(runner.y)
    n_features = runner.X.shape[1]
    runner.n_samples_eff = get_effective_train_size(
        n_total=n_samples,
        strategy=runner.validation_strategy or runner.val_method,
        n_folds=runner.n_folds,
        test_size=runner.train_test_split_test_size,
        n_features=n_features,
    )
    runner.val_method_eff, runner.cv_obj, runner.auto_decision = make_cv(
        n_samples,
        val_method=runner.val_method,
        n_folds=runner.n_folds,
        random_state=runner.random_state,
        test_size=runner.train_test_split_test_size,
        n_features=n_features,
    )
    # 'auto' is resolved exactly once (here, in make_cv). The resulting
    # decision is passed to iter_splits so that the test size is not
    # recomputed and cannot diverge between n_samples_eff and the actual
    # split.
    runner.resolved_via_auto = norm_val_method(runner.val_method) == "auto"

    # A single fold count for the early stopping branch (k_fold). The
    # 'auto' strategy could resolve to kfold with a number of folds k that
    # differs from the original n_folds: in that case the pruning branch
    # must use the computed k, otherwise iter_splits inside
    # _evaluate_with_intermediate_reports would fall back to the default
    # n_folds and the evaluation would diverge from
    # cross_val_score(cv=cv_obj).
    if runner.val_method_eff == "k_fold":
        # Take the fold count from the ready cv_obj (for k_fold it is
        # guaranteed to be non-None — make_cv always creates a KFold in
        # this branch). One formula covers auto→kfold (k from the
        # decision), the 'auto' fallback without P (clamped to max(2, k)),
        # and an explicit k_fold.
        assert runner.cv_obj is not None
        runner.effective_n_folds = int(runner.cv_obj.get_n_splits())
    else:
        # Explicit loo / train_test_split: the fold count is not used.
        runner.effective_n_folds = runner.n_folds


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

"""HPO phase execution of the training pipeline (Training Engine).

Builds the HPO request of one algorithm (``_build_hpo_request``) and runs
the hyperparameter search directly (``_run_hpo``), executes an HPO phase
for every candidate (``_hpo_phase_worker`` — a module-level function so that
it can be pickled in the "processes" mode) and sorts the worker results into
a ``WorkerOutcome`` that ``component`` applies to the phase context. It also holds the target-domain guard, the BBC-CV memory
budget check and the complexity guard record merging. It may import ``state``
only and must not import ``data_prep``, ``finalists``, ``select_steps``,
``artifact`` or ``component``.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from typing import Any, cast

import numpy as np
import pandas as pd
from optuna.trial import FrozenTrial

from configurable_automl_engine.common.metrics import (
    BBC_MAX_OOF_CELLS,
    direction_label,
    to_user_value,
)
from configurable_automl_engine.common.settings import (
    EncodingSettings,
    InputPolicies,
    OversamplingSettings,
)
from configurable_automl_engine.hpo.request import (
    HpoRequest,
    as_complexity_guard_config,
)
from configurable_automl_engine.hpo.runner import run_optimization
from configurable_automl_engine.hpo.search_space import build_search_space
from configurable_automl_engine.hpo.target_domain import check_target_domain
from configurable_automl_engine.hpo.types import (
    HPOHistoryError,
    InvalidAlgorithmError,
    OptimizeResult,
)
from configurable_automl_engine.preprocessing import EncodingStrategy
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
)

from .state import (
    PhaseResult,
    PreparedData,
    TargetDomainOutcome,
    WorkerOutcome,
    _HpoPhaseSpec,
    _HPOWorkerResult,
)

_LOG = logging.getLogger(__name__)


def _algorithms_as_dict(algorithms_cfg: Any) -> dict[str, AlgoCfg]:
    """Convert an AlgorithmsConfig into a plain dictionary {name: AlgoCfg}."""
    # model_fields via an instance also works, but it is deprecated
    # (PydanticDeprecatedSince211, removal in V3.0) — use the class
    return {
        name: algo_cfg
        for name in type(algorithms_cfg).model_fields
        if (algo_cfg := getattr(algorithms_cfg, name)) is not None
    }


# --------------------------------------------------------------------------- #
#  HPO call                                                                   #
# --------------------------------------------------------------------------- #
def _build_hpo_request(
    algo_name: str,
    algo_cfg: AlgoCfg,
    spec: _HpoPhaseSpec,
    *,
    initial_params: dict[str, Any] | None = None,
    previous_trials: list[FrozenTrial] | None = None,
) -> HpoRequest:
    """Build the HPO request of one algorithm in one phase.

    The engine passes the user ranges of the algorithm
    (``AlgoCfg.hyperparameters``) as the search space override; the run
    merges them over ``DEFAULT_SPACES`` with ``build_search_space`` (or keeps
    only them with ``limit_hyperparameters: true``) — the same merge a direct
    ``optimize()`` call uses. The HPO phase needs only the best parameters
    and score, so the full-data fit of the best model is skipped
    (``fit_best_model=False``): the engine trains the final model itself with
    ``ModelTrainer``.

    Args:
        algo_name (str): The algorithm name.
        algo_cfg (AlgoCfg): The algorithm configuration.
        spec (_HpoPhaseSpec): The phase snapshot.
        initial_params (dict[str, Any] | None): The previous-phase parameters
            enqueued as the first trial (refine).
        previous_trials (list[FrozenTrial] | None): The previous-phase TPE
            history (refine).

    Returns:
        HpoRequest: The run settings.
    """
    hyperparameters = algo_cfg.hyperparameters
    return HpoRequest(
        algo_name=algo_name,
        metric=spec.metric_sklearn,
        n_trials=spec.phase.n_trials,
        val_method=spec.resolved_validation,
        n_folds=spec.resolved_n_folds,
        train_test_split_test_size=spec.resolved_test_size,
        # The data-split seed is the engine seed in every phase (the same
        # folds for every phase and the final fit); the TPE sampler seed
        # changes per phase.
        random_state=spec.random_state,
        sampler_seed=spec.sampler_seed,
        space_overrides=(
            {algo_name: dict(hyperparameters)} if hyperparameters else None
        ),
        # The names of parameters the user set explicitly: the complexity
        # guard treats them as "user ranges" (intersected, with
        # user_range_policy on an empty intersection).
        user_hyperparameter_keys=list(hyperparameters or {}),
        limit_hyperparameters=bool(algo_cfg.limit_hyperparameters),
        complexity_guard=as_complexity_guard_config(spec.complexity_guard_cfg),
        initial_params=initial_params,
        previous_trials=previous_trials or None,
        categorical_features=spec.categorical_features,
        numerical_features=spec.numerical_features,
        encoding=EncodingSettings(
            encoding=cast(EncodingStrategy, spec.encoding),
            high_cardinality_threshold=spec.high_cardinality_threshold,
            high_cardinality_encoding=cast(
                EncodingStrategy | None, spec.high_cardinality_encoding
            ),
            hashing_n_components=spec.hashing_n_components,
            target_encoding_smoothing=spec.target_encoding_smoothing,
            target_encoding_fallback=spec.target_encoding_fallback,
        ),
        preprocessing_override=algo_cfg.preprocessing,
        feature_selection_cfg=spec.feature_selection_cfg,
        oversampling=OversamplingSettings(
            enable=spec.oversampling_enabled,
            multiplier=spec.oversampling_multiplier,
            algorithm=spec.oversampling_algorithm,
            target_bins=spec.oversampling_target_bins,
            add_noise=spec.oversampling_add_noise,
            noise_level=spec.oversampling_noise_level,
        ),
        pruning=spec.pruning_cfg,
        policies=InputPolicies(
            inf_policy=spec.inf_policy,
            datetime_policy=spec.datetime_policy,
            drop_id_like_columns=spec.drop_id_like_columns,
        ),
        fit_best_model=False,
        phase_name=spec.phase.name,
        collect_oof=spec.collect_oof,
    )


def _run_hpo(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    request: HpoRequest,
) -> PhaseResult | None:
    """Run the hyperparameter search of one algorithm and map the outcome.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        request (HpoRequest): The run settings.

    Returns:
        PhaseResult | None: The phase result — score, params and the phase's
            own Optuna trials; or ``None`` if HPO produced no valid result
            or failed (the algorithm is excluded from candidates; NO
            fallback to the previous phase's parameters).

    Raises:
        InvalidAlgorithmError: If the algorithm is incompatible with the
            data (the circuit breaker disqualifies it).
        HPOHistoryError: If the transferred TPE history is incompatible
            (fail fast — the pipeline fails with a clear message).
    """
    algo_name = request.algo_name
    try:
        result = run_optimization(X, y, request)
    except (InvalidAlgorithmError, HPOHistoryError):
        # The disqualification is applied by the worker; TPE history
        # transfer problems are configuration errors: fail fast with a clear
        # message instead of silently excluding the algorithm.
        raise
    except Exception as err:  # noqa: BLE001 — any other failure excludes the algorithm
        _LOG.error("Algorithm %s failed during HPO: %s", algo_name, err, exc_info=True)
        # Return None so the main loop knows the algorithm didn't pass.
        return None
    return _to_phase_result(result, algo_name)


def _to_phase_result(
    result: OptimizeResult | None, algo_name: str
) -> PhaseResult | None:
    """Map the result of an HPO run onto a ``PhaseResult``.

    Args:
        result (OptimizeResult | None): The run result.
        algo_name (str): The algorithm name.

    Returns:
        PhaseResult | None: The phase result, or ``None`` when the run
            produced no valid result.
    """
    # No valid result: either all trials failed (optuna.TrialPruned) or the
    # run failed otherwise. Return None so the main loop knows the algorithm
    # didn't pass: the dummy candidate with score -3.4e38 and params=None
    # should no longer end up in phase_results.
    if result is None:
        _LOG.error(
            "Algorithm %s failed during HPO: no valid result (tuner returned None)",
            algo_name,
        )
        return None
    if result.score is None or result.params is None:
        _LOG.error(
            "Algorithm %s failed during HPO: no valid result "
            "(best_score=%r, best_params=%r)",
            algo_name,
            result.score,
            result.params,
        )
        return None
    return PhaseResult(
        score=result.score,
        params=result.params,
        # The phase's own trials (transferred history excluded).
        trials=list(result.trials),
        complexity_guard=list(result.complexity_guard or []),
        # The honest number of trials the best score was selected from (own
        # COMPLETED trials of the phase).
        n_trials_evaluated=int(result.n_trials_evaluated or 0),
        oof_predictions=result.oof_predictions,
    )


def _execute_hpo_phase(
    X: pd.DataFrame,
    y: pd.Series,
    request: HpoRequest,
) -> PhaseResult | None:
    """Run a specific HPO phase for an algorithm (module-level function).

    Provides the phase logging around ``_run_hpo``. The function is
    module-level (not a closure) so that the ``_hpo_phase_worker`` worker is
    serializable in the "processes" mode. The feature matrix ``X`` is passed
    as a separate argument: in the processes mode ``run_parallel`` transfers
    it through shared memory / disk, so it is not pickled for every
    algorithm.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        request (HpoRequest): The run settings of the algorithm in the phase.

    Returns:
        PhaseResult | None: The phase result (score, params, trials), or
            ``None`` when HPO did not return a valid result (the algorithm
            is excluded from candidates).

    Raises:
        Exception: Exceptions from ``_run_hpo`` are logged and re-raised
            (including ``InvalidAlgorithmError`` and ``HPOHistoryError``).
    """
    algo = request.algo_name
    phase_name = request.phase_name
    _LOG.info(f"=== {phase_name} phase: {algo} ({request.n_trials} tries) ===")

    try:
        result = _run_hpo(X=X, y=y, request=request)

        if result is None:
            return None

        # Log the value in user semantics: for neg_*-metrics (e.g.
        # neg_root_mean_squared_error) the raw scorer value is inverted, so
        # the natural (positive) value is shown to the user. The direction
        # accompanies the value explicitly ("min better" for errors): the
        # optimizer's greater_is_better flag is not used.
        disp = to_user_value(request.metric, result.score)
        _LOG.info(
            f"{phase_name} {algo:15} | score {disp:.5f} "
            f"({direction_label(request.metric)}) | params {result.params}"
        )
        return result
    except Exception as err:
        _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
        raise


def _hpo_phase_worker(
    algo_name: str,
    algo_cfg: AlgoCfg,
    spec: _HpoPhaseSpec,
    X: pd.DataFrame,
) -> _HPOWorkerResult:
    """Worker for parallel or sequential execution of an HPO task.

    A module-level function (not a closure): serializable in the "processes"
    mode. Does not mutate the shared context: the disqualification is
    returned in the result's ``disqualify_reason`` and applied in the main
    process (otherwise a write from another process would be lost).

    Args:
        algo_name (str): The algorithm name.
        algo_cfg (AlgoCfg): The algorithm configuration.
        spec (_HpoPhaseSpec): Snapshot of the HPO phase parameters.
        X (pd.DataFrame): The feature matrix (transferred through shared
            memory / disk by ``run_parallel`` in the "processes" mode instead
            of being pickled inside ``spec`` for every algorithm).

    Returns:
        _HPOWorkerResult: The task result: score/params/trials on success,
            ``disqualify_reason`` on a circuit breaker disqualification,
            otherwise a result without a score and reason.
    """
    # Determine the warm-start for refine: each candidate warm-starts from
    # its OWN previous-phase record — initial_params (the enqueue_trial
    # anchor, which guarantees score monotonicity) AND previous_trials (the
    # TPE history transferred via study.add_trials). The prev_results
    # snapshot lives in the immutable spec (frozen at phase definition time),
    # so the worker uses the state at phase start. Workers only READ this
    # dict; writes to phase_results happen after all phase workers finish,
    # and at the end of the phase phase_results is re-bound to the filtered
    # valid_results (objects are never mutated) — therefore the snapshot is
    # stable in both parallel and sequential modes.
    init_params = None
    previous_trials: list[FrozenTrial] | None = None
    if spec.phase.action == "refine" and algo_name in spec.prev_results:
        prev = spec.prev_results[algo_name]
        init_params = prev.params
        previous_trials = prev.trials or None

    request = _build_hpo_request(
        algo_name,
        algo_cfg,
        spec,
        initial_params=init_params,
        previous_trials=previous_trials,
    )
    try:
        result = _execute_hpo_phase(X, spec.y, request)

        if result is None:
            return _HPOWorkerResult(name=algo_name)

        return _HPOWorkerResult(
            name=algo_name,
            score=result.score,
            params=result.params,
            trials=list(result.trials),
            complexity_guard=list(result.complexity_guard or []),
            n_trials_evaluated=int(result.n_trials_evaluated or 0),
            oof_predictions=result.oof_predictions,
        )
    except InvalidAlgorithmError as err:
        # Circuit breaker: the algorithm is disqualified after
        # MAX_FATAL_FAILURES consecutive fatal errors. This is NOT an abort
        # of the whole run — the algorithm is excluded from the candidates,
        # the rest continue training (same as a None result).
        _LOG.warning(
            "Algorithm %s disqualified in phase %s: %s",
            algo_name,
            spec.phase.name,
            err,
        )
        return _HPOWorkerResult(name=algo_name, disqualify_reason=str(err))
    except HPOHistoryError as err:
        # TPE history transfer problems are configuration errors: fail fast
        # with a clear message instead of silently excluding the algorithm
        # (the pipeline must not continue with a degraded refine). The error
        # is carried in the result (a string, serializable in the "processes"
        # mode) and re-raised in the main process by
        # ``_consume_worker_results``: raising here would be swallowed by
        # ``run_parallel``, which converts worker exceptions to ``None``.
        _LOG.error(
            "Algorithm %s cannot transfer TPE history in phase %s: %s",
            algo_name,
            spec.phase.name,
            err,
        )
        return _HPOWorkerResult(name=algo_name, history_error=str(err))
    except Exception as e:  # noqa: BLE001
        _LOG.warning(f"Algorithm {algo_name} failed in phase {spec.phase.name}: {e}")
        return _HPOWorkerResult(name=algo_name)


def _consume_worker_results(
    results: Iterable[_HPOWorkerResult | None],
) -> WorkerOutcome:
    """Sort worker results into phase results and disqualifications.

    In the "processes" mode the worker cannot write to the ``PhaseContext``
    (changes in the child process would be lost), so disqualifications are
    returned in the results; the main process (``component``) merges the
    returned outcome. ``None`` results (a task timeout/failure in the pool)
    are ignored. A ``history_error`` result re-raises ``HPOHistoryError`` in
    the main process: TPE history transfer problems are configuration errors
    and must fail the run even in the parallel mode (where ``run_parallel``
    swallows worker exceptions).

    Args:
        results (Iterable[_HPOWorkerResult | None]): Worker results.

    Returns:
        WorkerOutcome: The successful results, the circuit breaker additions
            and the OOF matrices of the successful results (worker order).

    Raises:
        HPOHistoryError: If a worker reported a TPE history transfer error
            (fail fast — the pipeline must not continue with a degraded
            refine).
    """
    phase_results: dict[str, PhaseResult] = {}
    disqualified: dict[str, str] = {}
    oof_pool: list[np.ndarray] = []
    for res in results:
        if res is None:
            continue
        if res.history_error is not None:
            raise HPOHistoryError(res.history_error)
        if res.disqualify_reason is not None:
            disqualified[res.name] = res.disqualify_reason
            continue
        if res.score is None or res.params is None:
            continue
        # Every evaluated configuration was a candidate of the selection, so
        # its OOF predictions enter the BBC-CV pool even if the algorithm is
        # later dropped by a select step.
        if res.oof_predictions is not None:
            oof_pool.append(res.oof_predictions)
        phase_results[res.name] = PhaseResult(
            score=res.score,
            params=res.params,
            trials=list(res.trials or []),
            complexity_guard=list(res.complexity_guard or []),
            n_trials_evaluated=int(res.n_trials_evaluated or 0),
            oof_predictions=res.oof_predictions,
        )
    return WorkerOutcome(
        results=phase_results, disqualified=disqualified, oof_pool=oof_pool
    )


def _apply_target_domain_guard(
    all_algorithms: dict[str, AlgoCfg],
    y: pd.Series,
    disqualified: Collection[str] = (),
) -> TargetDomainOutcome:
    """Exclude or restrict GLMs whose loss domain does not fit the target.

    Runs once, before the first HPO phase. ``gammaregressor`` (``y > 0``) and
    ``poissonregressor`` (``y >= 0``) on a target outside their domain — and
    ``tweedieregressor`` / ``glm`` whose whole ``power`` search range is
    invalid for the target — would fail on every fold of every trial, so they
    are returned for the circuit breaker with a reason and a single WARNING,
    and no trial is ever run for them. A Tweedie ``power`` range that is only
    partly invalid is clipped to its valid part instead (the algorithm stays
    a candidate).

    Args:
        all_algorithms (dict[str, AlgoCfg]): The configured algorithms.
        y (pd.Series): The target vector.
        disqualified (Collection[str]): The already disqualified algorithms
            (not checked).

    Returns:
        TargetDomainOutcome: The algorithms with clipped search spaces applied
            (unchanged configs are returned as is; the input is not mutated)
            and the new disqualifications "algorithm → reason".
    """
    guarded: dict[str, AlgoCfg] = {}
    new_disqualified: dict[str, str] = {}
    for name, a_cfg in all_algorithms.items():
        if not a_cfg.enable or name in disqualified:
            guarded[name] = a_cfg
            continue
        space = build_search_space(
            name,
            a_cfg.hyperparameters,
            limit=bool(a_cfg.limit_hyperparameters),
        )
        check = check_target_domain(name, space, y)
        if check.disqualify_reason is not None:
            _LOG.warning(
                "Algorithm %s disqualified before HPO: %s",
                name,
                check.disqualify_reason,
            )
            new_disqualified[name] = check.disqualify_reason
            guarded[name] = a_cfg
            continue
        if check.clipped_entries:
            for message in check.messages:
                _LOG.warning("Target domain guard: %s", message)
            hyperparameters = {
                **(a_cfg.hyperparameters or {}),
                **check.clipped_entries,
            }
            a_cfg = a_cfg.model_copy(update={"hyperparameters": hyperparameters})
        guarded[name] = a_cfg
    return TargetDomainOutcome(algorithms=guarded, disqualified=new_disqualified)


def _bbc_enabled(
    cfg: Config, prepared: PreparedData, algorithms: dict[str, AlgoCfg]
) -> bool:
    """Decide whether the OOF predictions of every trial are kept for BBC-CV.

    BBC-CV is the honest estimate when no outer hold-out is reserved. The
    memory cost is known before HPO: ``n_rows`` floats per evaluated trial,
    at most the sum of ``n_trials`` over the HPO phases times the number of
    enabled algorithms. Above ``BBC_MAX_OOF_CELLS`` the estimate is skipped
    with a WARNING.

    Args:
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data (hold-out already split).
        algorithms (dict[str, AlgoCfg]): The configured algorithms.

    Returns:
        bool: True when the OOF predictions must be collected.
    """
    if prepared.holdout_X is not None:
        return False
    n_algorithms = sum(1 for a in algorithms.values() if a.enable)
    n_trials = sum(p.n_trials for p in cfg.general.phases if p.action != "select")
    cells = n_algorithms * n_trials * len(prepared.y)
    if cells > BBC_MAX_OOF_CELLS:
        _LOG.warning(
            "BBC-CV estimate skipped: keeping the OOF predictions of up to %d "
            "trials x %d rows exceeds the budget of %d values. Reserve an "
            "outer hold-out (general.holdout_fraction) for an honest estimate.",
            n_algorithms * n_trials,
            len(prepared.y),
            BBC_MAX_OOF_CELLS,
        )
        return False
    return True


def _merge_complexity_guard_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge per-(algo, param, level) guard records into one record per parameter.

    The guard applies several rules to the same parameter (level A then
    level B), producing one record per rule, and the same restrictions repeat
    in every HPO phase (the guard is deterministic between phases). The report
    documents "one record per restricted parameter", so the raw per-rule/per
    -phase list is merged:

    * the ``after`` range of a later rule (the final state of the parameter)
      becomes the record's ``after``;
    * the ``before`` of the first record and the final ``after`` span the
      whole adaptation;
    * the ``levels`` field lists all applied rule levels (e.g. ``["A", "B"]``);
    * a record produced in an earlier phase is not duplicated by a later one
      (identical (algo, param, before, after) collapses).

    Args:
        records: Raw guard records (per rule and per phase).

    Returns:
        list[dict[str, Any]]: One record per (algo, param) with ``levels``
            aggregating the applied rule levels.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        key = (rec.get("algo", "?"), rec.get("param", "?"))
        cur = merged.get(key)
        if cur is None:
            merged[key] = dict(rec)
            cur = merged[key]
            cur["levels"] = [rec["level"]] if rec.get("level") else []
            continue
        # Same parameter in a later phase or a later rule of the same phase:
        # the final range of the parameter is the latest ``after``.
        cur["after"] = rec["after"]
        if rec.get("level") and rec["level"] not in cur["levels"]:
            cur["levels"].append(rec["level"])
        # The aggregate "before" is the earliest one (the starting range
        # before any rule of the guard was applied).
        if cur.get("before") == cur.get("after") and rec.get("before") != rec.get(
            "after"
        ):
            cur["before"] = rec["before"]
        # The guard restriction is "applied" if any rule actually applied it.
        cur["applied"] = bool(cur.get("applied") or rec.get("applied"))
        cur["fixed"] = bool(cur.get("fixed") or rec.get("fixed"))
        if rec.get("conflict"):
            cur["conflict"] = True
    return list(merged.values())

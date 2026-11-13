"""Cross-field validation rules of the ``general`` config section.

Each rule is a small function with explicit inputs (the ``GeneralCfg``
fields it checks); it raises ``ValueError`` with the user-facing message, or
returns the normalized value. ``GeneralCfg`` calls them from thin
``@model_validator`` / ``@field_validator`` methods, so the model keeps only
its fields. The module builds on :mod:`.phases`; it must not import
:mod:`.general` or :mod:`.root`.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Mapping
from typing import Any, TypeVar

from configurable_automl_engine.common.definitions import (
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.metrics import (
    AVAILABLE_METRICS,
    check_custom_metric_name,
    to_sklearn_name,
)
from configurable_automl_engine.training_engine.config.phases import (
    HPOPhaseCfg,
    PruningCfg,
)

logger = logging.getLogger(__name__)

#: The computational budget guard of the multi-split validation: a total of
#: ``sum(phase.n_trials) * n_folds * n_repeats`` fit evaluations above the
#: WARN threshold is allowed with a WARNING, above the ERROR threshold the
#: config is rejected (a prohibitive run is caught at validation, not after
#: hours of computation).
MULTI_SPLIT_BUDGET_WARN_THRESHOLD = 2000
MULTI_SPLIT_BUDGET_ERROR_THRESHOLD = 10000

#: The sanity limit of ``n_folds * n_repeats`` (splits per trial): beyond it
#: the per-repeat folds cannot hold a meaningful number of validation rows.
MULTI_SPLIT_SPLITS_PER_TRIAL_LIMIT = 50

_MetricT = TypeVar("_MetricT", bound=str)


def check_custom_metric_names(v: Any) -> Any:
    """Check the keys of ``general.custom_metrics`` before the specs load.

    Run before the entries are validated, so a reserved or malformed name is
    reported even when its callable cannot be imported.

    Returns:
        The value unchanged.
    """
    if isinstance(v, Mapping):
        for name in v:
            check_custom_metric_name(name)
    return v


def check_metric_references(
    comparison_metric: str,
    additional_metrics: list[str],
    custom_metrics: Mapping[str, object],
) -> None:
    """Check that every metric name is a built-in or a declared custom metric.

    Names are matched exactly (custom metric names are lowercase). A custom
    metric that no metric field refers to is reported with a WARNING.

    Raises:
        ValueError: If a name is neither built-in nor in ``custom_metrics``.
    """
    allowed = [*AVAILABLE_METRICS, *custom_metrics]
    for field, name in [
        ("comparison_metric", comparison_metric),
        *(("additional_metrics", m) for m in additional_metrics),
    ]:
        if name not in allowed:
            raise ValueError(
                f"{field}: unknown metric {name!r}; expected one of "
                f"{', '.join(repr(m) for m in allowed)} (declare a user metric "
                "in general.custom_metrics)"
            )
    used = {comparison_metric, *additional_metrics}
    for name in custom_metrics:
        if name not in used:
            logger.warning(
                "general.custom_metrics.%s is declared but not used by "
                "comparison_metric or additional_metrics.",
                name,
            )


def deduplicate_additional_metrics(v: list[_MetricT]) -> list[_MetricT]:
    """Remove duplicates from the list of additional metrics.

    Documented behavior: the same metric specified several times is
    computed and returned exactly once (the order of the first
    occurrence is preserved). Deduplication is done using the
    lowercased string representation of the metric: ``ComparisonMetric``
    items are strings, so calling ``.lower()`` is type-safe and does not
    depend on an arbitrary element type.

    Args:
        v: The ``additional_metrics`` value.

    Returns:
        The metrics without repetitions, in the order of first occurrence.
    """
    seen: set[str] = set()
    result: list[_MetricT] = []
    for name in v:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result


def exclude_comparison_from_additional(
    comparison_metric: str, additional_metrics: list[_MetricT]
) -> list[_MetricT]:
    """Exclude the main comparison metric from the additional metrics list.

    The value of the main comparison metric is already returned in the
    training results as ``score``, so duplicating it in
    ``additional_metrics`` is unnecessary. The comparison is done by the
    sklearn name of the metric, so aliases of the same metric
    (for example, 'rmse' and 'neg_root_mean_squared_error') count as a
    match.

    Args:
        comparison_metric: ``general.comparison_metric``.
        additional_metrics: ``general.additional_metrics``.

    Returns:
        The additional metrics without the comparison metric (a WARNING is
        logged when it was there).
    """
    comparison_sklearn = to_sklearn_name(comparison_metric)
    filtered = [
        m for m in additional_metrics if to_sklearn_name(m) != comparison_sklearn
    ]
    if len(filtered) != len(additional_metrics):
        logger.warning(
            "additional_metrics contains the comparison metric '%s'; "
            "its value is already returned as 'score' and will not be "
            "duplicated in 'additional_metrics'.",
            comparison_metric,
        )
    return filtered


def check_n_folds(
    serialization_format: SerializationFormat,
    n_folds: int,
    validation_strategy: ValidationStrategy,
) -> None:
    """Check the logical consistency of validation and serialization settings.

    Validation logic:
    1. Checks that the `joblib` package is available when the corresponding format is selected.
    2. Ensures `n_folds` is a positive number for any strategy.
    3. Strict check for `k_fold`: the number of blocks must be at least 2.

    Args:
        serialization_format: ``general.serialization_format``.
        n_folds: ``general.n_folds``.
        validation_strategy: ``general.validation_strategy``.

    Raises:
        ValueError: If the 'joblib' package is not installed or `n_folds` is
            invalid for the selected strategy.
    """
    if serialization_format == SerializationFormat.joblib and not is_installed(
        "joblib"
    ):
        raise ValueError(
            "serialization_format='joblib' requires"
            " the 'joblib' package to be installed"
        )
    if n_folds < 1:
        raise ValueError("`n_folds` must be at least 1")
    # Strict validation only for k-fold
    if validation_strategy == ValidationStrategy.k_fold and n_folds < 2:
        raise ValueError("n_folds must be ≥ 2 for k-fold validation")


def check_multi_split(
    n_repeats: int,
    split_aggregation: str,
    validation_strategy: ValidationStrategy,
    n_folds: int,
    phases: list[HPOPhaseCfg],
) -> None:
    """Check the multi-split (repeated) validation settings.

    Validation logic:
    1. ``n_repeats`` must be a positive integer (>= 1). The ``ge=1`` field
       constraint rejects the broken values at parsing time; this rule
       re-checks the invariants for direct model construction and reports a
       config-validation (not runtime) error.
    2. ``split_aggregation`` must be 'mean' or 'median' (the type constraint
       rejects other values at parsing time; re-checked here for the same
       reason).
    3. ``n_repeats > 1`` with 'loo' is meaningless (deterministic folds) —
       a WARNING, the repeats are ignored downstream.
    4. The computational budget guard: a trial on 'k_fold'/'loo' runs
       ``n_folds * n_repeats`` fit evaluations, a trial on
       'train_test_split' — ``n_repeats`` (a repeat of the hold-out is ONE
       split, the fold count does not multiply it). The HPO phases run
       ``n_trials`` trials each. A total budget above the guard threshold
       is a config error; a budget in the WARN zone is allowed with a
       WARNING. With 'auto' the fold-counting variant of the strategy is
       assumed (the pessimistic estimate: 'auto' may resolve to k-fold).
    5. The sanity bound of ``n_folds * n_repeats`` (splits per trial for
       the fold-based strategies) — beyond it the per-repeat fold cannot
       hold a meaningful number of validation rows.
    6. The ``n_samples`` check of issue-211 (``n_folds * n_repeats`` must
       not exceed the number of samples) is a data-level check — the config
       does not know ``n_samples``; it is enforced at runtime by
       ``validation.iter_splits`` (a WARNING when a tiny sample cannot give
       independent repeats).

    Args:
        n_repeats: ``general.n_repeats``.
        split_aggregation: ``general.split_aggregation``.
        validation_strategy: ``general.validation_strategy``.
        n_folds: ``general.n_folds``.
        phases: ``general.phases`` (the HPO trial counts for the budget
            guard).

    Raises:
        ValueError: If ``n_repeats`` / ``split_aggregation`` are invalid, or
            the estimated computational budget exceeds the guard threshold.
    """
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise ValueError(
            f"n_repeats must be a positive integer (>= 1), got {n_repeats!r}"
        )
    if split_aggregation not in ("mean", "median"):
        raise ValueError(
            f"split_aggregation must be 'mean' or 'median', got {split_aggregation!r}"
        )
    if n_repeats == 1:
        return
    if validation_strategy == ValidationStrategy.loo:
        logger.warning(
            "general.n_repeats=%d with validation_strategy='loo': LOO has "
            "deterministic folds, the repeats are meaningless — they will "
            "be IGNORED (the effective n_repeats is 1).",
            n_repeats,
        )
        return
    # The computational budget guard: total fit evaluations of the HPO
    # phases (trials x splits-per-trial). Only the HPO phases carry trials.
    trials = sum(
        phase.n_trials
        for phase in phases
        if phase.action in ("all_algorithms", "refine")
    )
    # A hold-out repeat is ONE split per trial: the fold count does not
    # multiply it. For 'auto' the pessimistic (fold-based) estimate is used:
    # 'auto' may resolve to k-fold on the actual data.
    folds_per_trial = (
        1 if validation_strategy == ValidationStrategy.train_test_split else n_folds
    )
    splits_per_trial = folds_per_trial * n_repeats
    total_budget = trials * splits_per_trial
    if total_budget > MULTI_SPLIT_BUDGET_ERROR_THRESHOLD:
        raise ValueError(
            f"the estimated multi-split validation budget is prohibitive: "
            f"{trials} trials x "
            + (
                f"{n_folds} folds x {n_repeats} repeats = "
                if folds_per_trial > 1
                else f"{n_repeats} hold-out repeats = "
            )
            + f"{total_budget} fit evaluations (the guard threshold is "
            f"{MULTI_SPLIT_BUDGET_ERROR_THRESHOLD}). Reduce general.n_repeats, "
            "the phase n_trials, or general.n_folds."
        )
    if total_budget > MULTI_SPLIT_BUDGET_WARN_THRESHOLD:
        logger.warning(
            "the multi-split validation budget is high: %d trials x %s = "
            "%d fit evaluations (WARN threshold %d). Every repeat "
            "multiplies the validation cost of every trial.",
            trials,
            (
                f"{n_folds} folds x {n_repeats} repeats"
                if folds_per_trial > 1
                else f"{n_repeats} hold-out repeats"
            ),
            total_budget,
            MULTI_SPLIT_BUDGET_WARN_THRESHOLD,
        )
    if folds_per_trial > 1 and splits_per_trial > MULTI_SPLIT_SPLITS_PER_TRIAL_LIMIT:
        raise ValueError(
            f"n_folds ({n_folds}) x n_repeats ({n_repeats}) = "
            f"{splits_per_trial} splits per trial exceeds the sanity limit "
            f"{MULTI_SPLIT_SPLITS_PER_TRIAL_LIMIT}: the per-repeat fold "
            "would hold fewer rows than a meaningful validation fold. "
            "Decrease general.n_folds or general.n_repeats."
        )


def check_pruning_consistency(
    pruning: PruningCfg, validation_strategy: ValidationStrategy, n_folds: int
) -> None:
    """Check the consistency of early stopping settings with validation.

    Validation logic:
    1. For 'k_fold' the pruner must be able to fire before the evaluation
       finishes: min_steps cannot exceed n_folds.
    2. For 'train_test_split' there are no natural steps — early stopping
       is not applied; we emit a warning rather than an error (the
       behavior is documented and safe).

    Args:
        pruning: ``general.pruning``.
        validation_strategy: ``general.validation_strategy``.
        n_folds: ``general.n_folds``.

    Raises:
        ValueError: If the pruning settings conflict with validation.
    """
    if not pruning.enable:
        return

    if validation_strategy == ValidationStrategy.k_fold and pruning.min_steps > n_folds:
        raise ValueError(
            f"pruning.min_steps ({pruning.min_steps}) cannot be greater "
            f"than general.n_folds ({n_folds}) for k-fold validation: "
            "the pruner would never get enough intermediate steps. "
            "Decrease pruning.min_steps or increase general.n_folds."
        )

    if validation_strategy == ValidationStrategy.train_test_split:
        logger.warning(
            "pruning.enable=true with validation_strategy='train_test_split': "
            "early stopping will NOT be applied because hold-out evaluation "
            "has no intermediate steps."
        )


def validate_phase_pipeline(phases: list[HPOPhaseCfg]) -> None:
    """Cross-validate the sequence of pipeline phases.

    Validation logic (an empty phase list is not checked — the guard
    against missing phases stays in the ``train_best_model`` runtime):
    1. ``select`` cannot be the first phase: before any search there is
       nothing to audit/rank.
    2. Phases after a terminal select (``output: winner``) are forbidden:
       such a step yields a single winner, and any subsequent
       fine-tuning would hide the selection result.
    3. The last phase must be a ``select`` with ``output: winner``:
       the final artifact is built only from the result of the terminal
       selection step.

    Args:
        phases: ``general.phases``.

    Raises:
        ValueError: If the phase sequence violates the pipeline rules.
    """
    if not phases:
        return

    if phases[0].action == "select":
        raise ValueError(
            f"phase '{phases[0].name}' (index 0): action 'select' cannot "
            "be the first phase — there is nothing to rank/audit yet. "
            "Start the pipeline with an 'all_algorithms' (or 'refine') "
            "phase."
        )

    for index, phase in enumerate(phases[:-1]):
        if phase.action == "select" and phase.output == "winner":
            raise ValueError(
                f"phase '{phase.name}' (index {index}): select with "
                "output 'winner' is terminal — no phases are allowed "
                "after it. Move it to the last position or use "
                "output 'top_k'/'pool' to keep the pipeline going."
            )

    last = phases[-1]
    if not (last.action == "select" and last.output == "winner"):
        raise ValueError(
            f"the last phase '{last.name}' (action '{last.action}') must "
            "be a 'select' phase with output 'winner': the final artifact "
            "is built from the result of the terminal select step. "
            "Migration: append "
            "{action: 'select', strategy: 'cv_best', output: 'winner'} "
            "as the last phase. See 'Migrating older configs' in API_REFERENCE.md."
        )


def check_high_cardinality_encoding(
    threshold: int | None, hc_encoding: str | None
) -> None:
    """Check the consistency of the high-cardinality encoding settings.

    Validation logic:
    1. ``high_cardinality_threshold`` and ``high_cardinality_encoding``
       must be set together (otherwise the behavior is ambiguous — the
       configuration is rejected before training starts).
    2. The values of ``categorical_encoding`` / ``high_cardinality_encoding``
       are validated by the ``EncodingStrategy`` type (an unsupported
       strategy is rejected during config parsing).

    Args:
        threshold: ``general.high_cardinality_threshold``.
        hc_encoding: ``general.high_cardinality_encoding``.

    Raises:
        ValueError: If only one of the high-cardinality encoding
            parameters is set.
    """
    if (threshold is None) != (hc_encoding is None):
        raise ValueError(
            "general.high_cardinality_threshold and "
            "general.high_cardinality_encoding must be set together: "
            f"got threshold={threshold!r}, encoding={hc_encoding!r}. "
            "Provide both parameters to enable automatic high-cardinality "
            "encoding, or omit both to disable it."
        )


def check_start_method(start_method: str | None, parallel_mode: str) -> None:
    """Check that ``start_method`` applies and is available here.

    Args:
        start_method: ``general.start_method``.
        parallel_mode: ``general.parallel_mode``.

    Raises:
        ValueError: If ``start_method`` is set with
            ``parallel_mode='threads'``, or the platform does not support
            it (``fork`` on Windows).
    """
    if start_method is None:
        return
    if parallel_mode == "threads":
        raise ValueError(
            f"general.start_method={start_method!r} applies only to "
            "parallel_mode='processes'; remove it or switch "
            "general.parallel_mode to 'processes'."
        )
    available = multiprocessing.get_all_start_methods()
    if start_method not in available:
        raise ValueError(
            f"general.start_method={start_method!r} is not available "
            f"on this platform (available: {', '.join(available)})."
        )


def check_threads_timeouts(
    parallel_mode: str, phase_timeout: float | None, task_timeout: float | None
) -> None:
    """Forbid timeouts in threads mode.

    In ``parallel_mode='threads'`` timeouts are meaningless and harmful:
    an already running Python function cannot be interrupted, so a soft
    timeout does not save time — it only forces the phase to wait for
    ``shutdown(wait=True)`` completion of all algorithms and then
    discards their ready results. Therefore setting any of
    ``phase_timeout`` / ``task_timeout`` together with
    ``parallel_mode='threads'`` is a configuration error (a breaking
    change; backward compatibility is not preserved).

    Args:
        parallel_mode: ``general.parallel_mode``.
        phase_timeout: ``general.phase_timeout``.
        task_timeout: ``general.task_timeout``.

    Raises:
        ValueError: If ``parallel_mode='threads'`` and at least one of the
            timeouts ``phase_timeout`` / ``task_timeout`` is set.
    """
    if parallel_mode == "threads" and (
        phase_timeout is not None or task_timeout is not None
    ):
        offenders = []
        if phase_timeout is not None:
            offenders.append(f"general.phase_timeout={phase_timeout!r}")
        if task_timeout is not None:
            offenders.append(f"general.task_timeout={task_timeout!r}")
        raise ValueError(
            "timeouts are not supported in parallel_mode='threads': an "
            "already-running Python function cannot be "
            "interrupted, so a soft timeout does not save time and only "
            "discards the ready results of completed algorithms. "
            f"Offending settings: {', '.join(offenders)}. "
            "Remove the timeouts or switch general.parallel_mode to "
            "'processes' (where timeouts are hard and enforced)."
        )

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
from typing import TypeVar

from configurable_automl_engine.common.definitions import (
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.metrics import to_sklearn_name
from configurable_automl_engine.training_engine.config.phases import (
    HPOPhaseCfg,
    PruningCfg,
)

logger = logging.getLogger(__name__)

_MetricT = TypeVar("_MetricT", bound=str)


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

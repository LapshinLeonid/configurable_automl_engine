"""Runtime validation of a structurally valid configuration.

The contract ``Config`` checks the shape of the data. The checks in this
module depend on the installed packages, the estimator signatures, the
process-local custom-metric registry and the platform, so they stay in the
engine and run at every public entry point exactly once.
"""

from __future__ import annotations

import logging
import multiprocessing

from automl_contracts.config import AlgoCfg, Config
from automl_contracts.config.general_rules import check_n_folds as _check_n_folds
from automl_contracts.definitions import SerializationFormat, ValidationStrategy

from configurable_automl_engine.common.definitions import ALGO_PACKAGE_MAPPING
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.metrics import validate_custom_metrics
from configurable_automl_engine.hpo.default_spaces import (
    normalize_hyperparameter_entry,
)
from configurable_automl_engine.models import get_allowed_hyperparameters

logger = logging.getLogger(__name__)


def check_validation_settings(
    serialization_format: SerializationFormat,
    n_folds: int,
    validation_strategy: ValidationStrategy,
) -> None:
    """Check validation and serialization settings.

    Args:
        serialization_format: ``general.serialization_format``.
        n_folds: ``general.n_folds``.
        validation_strategy: ``general.validation_strategy``.

    Raises:
        ValueError: If ``joblib`` is missing or ``n_folds`` is invalid for
            the selected strategy.
    """
    if serialization_format == SerializationFormat.joblib and not is_installed(
        "joblib"
    ):
        raise ValueError(
            "serialization_format='joblib' requires"
            " the 'joblib' package to be installed"
        )
    _check_n_folds(n_folds, validation_strategy)


def check_start_method(start_method: str | None, parallel_mode: str) -> None:
    """Check that ``start_method`` applies and is available here.

    Args:
        start_method: ``general.start_method``.
        parallel_mode: ``general.parallel_mode``.

    Raises:
        ValueError: If the method is set for threads mode or is unavailable
            on the current platform.
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

    Args:
        parallel_mode: ``general.parallel_mode``.
        phase_timeout: ``general.phase_timeout``.
        task_timeout: ``general.task_timeout``.

    Raises:
        ValueError: If threads mode is combined with any timeout.
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


def get_required_package(algo_name: str) -> str | None:
    """Return the optional package required by ``algo_name``, if any."""
    return ALGO_PACKAGE_MAPPING.get(algo_name)


def get_unknown_hyperparameters(algo_cfg: AlgoCfg, algo_name: str) -> list[str]:
    """Return hyperparameters incompatible with ``algo_name``.

    An unknown algorithm or a constructor accepting ``**kwargs`` makes the
    check pass (the allowed set cannot be determined).
    """
    if algo_cfg.hyperparameters is None:
        return []
    allowed = get_allowed_hyperparameters(algo_name)
    if allowed is None:
        return []
    return [key for key in algo_cfg.hyperparameters if key not in allowed]


def _check_algorithm_dependencies(cfg: Config) -> None:
    for name in type(cfg.algorithms).model_fields:
        algo_cfg = getattr(cfg.algorithms, name)
        if algo_cfg is None or not algo_cfg.enable:
            continue
        required_pkg = get_required_package(name)
        if required_pkg and not is_installed(required_pkg):
            raise ValueError(
                f"Algorithm '{name}' is enabled, but the package "
                f"'{required_pkg}' is not installed. "
                f"Please run: pip install {required_pkg}"
            )


def _check_hyperparameter_compatibility(cfg: Config) -> None:
    errors: list[str] = []
    for name in type(cfg.algorithms).model_fields:
        algo_cfg = getattr(cfg.algorithms, name)
        if algo_cfg is None or not algo_cfg.enable:
            continue
        unknown = get_unknown_hyperparameters(algo_cfg, name)
        if unknown:
            allowed = sorted(get_allowed_hyperparameters(name) or set())
            errors.append(
                f"Algorithm '{name}': unknown hyperparameters {unknown}. "
                f"Allowed parameters: {allowed}"
            )
    if errors:
        raise ValueError("\n".join(errors))


def _normalize_hyperparameter_types(cfg: Config) -> None:
    for name in type(cfg.algorithms).model_fields:
        algo_cfg = getattr(cfg.algorithms, name)
        if algo_cfg is None or not algo_cfg.enable:
            continue
        hyperparams = algo_cfg.hyperparameters
        if not hyperparams:
            continue
        for param, entry in list(hyperparams.items()):
            normalized = normalize_hyperparameter_entry(name, param, entry)
            if normalized is not entry:
                hyperparams[param] = normalized


def validate_runtime_config(cfg: Config) -> Config:
    """Run every environment-dependent check on a structural ``Config``.

    The function mutates ``cfg`` in place while normalizing hyperparameter
    ranges and returns the same object, so callers that own a configuration
    (``read_config``/``load_config``) see the normalized values.

    Args:
        cfg: A structurally valid contract configuration.

    Returns:
        The same validated configuration object.

    Raises:
        ValueError: If an enabled algorithm depends on a missing package, its
            hyperparameters are incompatible with the estimator, or a
            timeout/start-method rule is violated.
        PydanticCustomError: If a custom metric declaration is invalid.
    """
    _check_algorithm_dependencies(cfg)
    _check_hyperparameter_compatibility(cfg)
    _normalize_hyperparameter_types(cfg)
    validate_custom_metrics(cfg.general.custom_metrics)
    check_start_method(cfg.general.start_method, cfg.general.parallel_mode)
    check_threads_timeouts(
        cfg.general.parallel_mode,
        cfg.general.phase_timeout,
        cfg.general.task_timeout,
    )
    check_validation_settings(
        cfg.general.serialization_format,
        cfg.general.n_folds,
        cfg.general.validation_strategy,
    )
    return cfg


__all__ = [
    "check_start_method",
    "check_threads_timeouts",
    "check_validation_settings",
    "get_required_package",
    "get_unknown_hyperparameters",
    "validate_runtime_config",
]

"""Early stopping (pruning) configuration of the HPO runner.

Normalizes the ``general.pruning`` settings and builds the Optuna pruner
(median / hyperband) with the pure ``plan_pruning``. It may import
``hpo.types`` and ``hpo.plans`` only and must not import ``hpo.runner`` or
``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

import logging as _logging
from typing import Any

import optuna

from .plans import PruningPlan
from .types import (
    HyperoptError,
)

log = _logging.getLogger(__name__)


# ═══════════════ early stopping (pruning) helpers ═══════════════════════════
_DEFAULT_PRUNING_CONFIG: dict[str, Any] = {
    "enable": False,
    "strategy": "median",
    "min_steps": 1,
    "n_startup_trials": 5,
    "reduction_factor": 3,
}


def _normalize_pruning_config(
    pruning: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize the early stopping configuration into a dict with valid fields.

    Accepts a settings dict (e.g. forwarded from ``training_engine``) or
    None. Returns ``None`` if early stopping is disabled or the config is not
    provided — in that case the tuner behaves identically to the current one.

    Args:
        pruning (dict[str, Any] | None): A dict with enable, strategy,
            min_steps, n_startup_trials, reduction_factor keys (or a subset).
    Returns:
        dict[str, Any] | None: The normalized pruner configuration, or None.
    """
    if not pruning:
        return None
    enabled = bool(pruning.get("enable", False))
    if not enabled:
        return None
    return {
        "strategy": str(pruning.get("strategy", _DEFAULT_PRUNING_CONFIG["strategy"])),
        "min_steps": int(
            pruning.get("min_steps", _DEFAULT_PRUNING_CONFIG["min_steps"])
        ),
        "n_startup_trials": int(
            pruning.get("n_startup_trials", _DEFAULT_PRUNING_CONFIG["n_startup_trials"])
        ),
        "reduction_factor": int(
            pruning.get("reduction_factor", _DEFAULT_PRUNING_CONFIG["reduction_factor"])
        ),
    }


def _build_pruner(cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    """Create an Optuna pruner object from a normalized configuration.

    Args:
        cfg (dict[str, Any]): Normalized early stopping configuration
            (the result of ``_normalize_pruning_config``).
    Returns:
        optuna.pruners.BasePruner: A configured Optuna pruner.
    Raises:
        HyperoptError: If an unknown pruning strategy is given or the
            parameter values are invalid.
    """
    strategy = cfg["strategy"]
    min_steps = cfg["min_steps"]
    if min_steps < 1:
        raise HyperoptError(
            f"pruning.min_steps must be a positive integer, got {min_steps}"
        )
    if strategy == "median":
        n_startup_trials = cfg["n_startup_trials"]
        if n_startup_trials < 1:
            raise HyperoptError(
                "pruning.n_startup_trials must be a positive integer, "
                f"got {n_startup_trials}"
            )
        # n_warmup_steps=min_steps: steps are numbered from 1, so the first
        # pruning decision is possible exactly after min_steps reports.
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=min_steps,
        )
    if strategy == "hyperband":
        reduction_factor = cfg["reduction_factor"]
        if reduction_factor < 2:
            raise HyperoptError(
                "pruning.reduction_factor must be at least 2 (Hyperband), "
                f"got {reduction_factor}"
            )
        return optuna.pruners.HyperbandPruner(
            min_resource=min_steps,
            reduction_factor=reduction_factor,
        )
    raise HyperoptError(
        f"Unknown pruning strategy: '{strategy}'. Must be 'median' or 'hyperband'."
    )


def plan_pruning(pruning: dict[str, Any] | None, method: str) -> PruningPlan:
    """Configure early stopping (pruning) for the selected validation.

    Normalizes the early stopping configuration and builds an Optuna
    pruner. Pruning applies only when there are natural evaluation steps
    (k_fold / loo); for hold-out (train_test_split) there are no
    intermediate steps, so early stopping is not applied (documented
    behavior).

    Args:
        pruning (dict[str, Any] | None): The early stopping settings.
        method (str): The effective validation method.

    Returns:
        PruningPlan: The pruner and whether it is applied.

    Raises:
        HyperoptError: If the settings are invalid.
    """
    pruning_cfg = _normalize_pruning_config(pruning)
    pruner: optuna.pruners.BasePruner | None = None
    if pruning_cfg is not None:
        pruner = _build_pruner(pruning_cfg)
        log.info(
            "Early stopping enabled: strategy=%s, min_steps=%d, "
            "n_startup_trials=%d, reduction_factor=%d",
            pruning_cfg["strategy"],
            pruning_cfg["min_steps"],
            pruning_cfg["n_startup_trials"],
            pruning_cfg["reduction_factor"],
        )
    active = pruner is not None and method != "train_test_split"
    if pruner is not None and not active:
        log.warning(
            "Early stopping is enabled, but validation strategy '%s' has no "
            "intermediate steps — pruning will not be applied.",
            method,
        )
    return PruningPlan(pruner=pruner, active=active)

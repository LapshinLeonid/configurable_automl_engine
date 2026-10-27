"""Early stopping (pruning) configuration of the HPO runner.

Normalizes the ``general.pruning`` settings and builds the Optuna pruner
(median / hyperband). It may import ``hpo.types`` only and must not import
``hpo.runner`` or ``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

# ─────────────────────────────── stdlib
import logging as _logging
from typing import Any

# ──────────────────────────── third-party
import optuna

# ──────────────────────────── project
from .types import (
    HyperoptError,
)

# The records keep the logger name of the pre-split ``tuner`` module: the
# logger name is part of the observable behavior (handlers, filters and
# ``caplog`` assertions select records by it).
log = _logging.getLogger("configurable_automl_engine.tuner")


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

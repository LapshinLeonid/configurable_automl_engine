"""Search spaces of the HPO runner.

The data-dependent default spaces (the KNN space bounded by the effective
sample size, the SVR space with the adaptive epsilon), the conversion of a
dict search space into Optuna suggestions (``_apply_dynamic_space``) and the
distribution recorder used to validate the TPE history transfer. It may
import ``hpo.types`` only and must not import ``hpo.runner`` or
``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

# ─────────────────────────────── stdlib
import logging as _logging
from collections.abc import Callable
from typing import Any, cast

# ──────────────────────────── third-party
import optuna
from optuna.trial import Trial

# ──────────────────────────── project
from configurable_automl_engine.common.hyperopt_defaults import (
    DEFAULT_SPACES,
    FloatSpace,
    SearchSpaceEntry,
    compute_svr_epsilon_bounds,
)

# The records keep the logger name of the pre-split ``tuner`` module: the
# logger name is part of the observable behavior (handlers, filters and
# ``caplog`` assertions select records by it).
log = _logging.getLogger("configurable_automl_engine.tuner")


# ═══════════════════════════════════ search spaces ═══════════════════════════


# ══════════ KNN-space depends on the sample size ══════════
def _make_knn_space(n_samples: int) -> Callable[[Trial], dict[str, Any]]:
    """Create a search space generator for the KNN algorithm."""

    def _space(t: Trial) -> dict[str, Any]:
        # Cap n_neighbors by the physical limit of the training set (N_eff - 1).
        # Use n_samples_eff instead of the total number of rows in the dataset.
        physical_limit = max(1, n_samples - 1)
        max_k = int(min(30, physical_limit))

        return {
            "n_neighbors": t.suggest_int("n_neighbors", 1, max_k),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
            "p": t.suggest_int("p", 1, 2),
        }

    return _space


# ══════════ SVR space: epsilon depends on the spread of y ══════════
def _svr_adaptive_epsilon_entry(y: Any) -> SearchSpaceEntry:
    """Build a SearchSpaceEntry for SVR epsilon with bounds scaled by y.

    The bounds are computed from the spread of ``y`` (σ(y) by default) ONCE —
    before the Optuna loop (called from ``resolve_search_space``, not from
    ``space_fn``): otherwise every trial would have its own bounds and the
    TPE sampler would lose determinism.

    Args:
        y (Any): Target variable vector.
    Returns:
        SearchSpaceEntry: float_log distribution of epsilon within adaptive bounds.
    """
    low, high = compute_svr_epsilon_bounds(y)
    log.info("SVR adaptive epsilon bounds: [%.6g, %.6g] (from y spread)", low, high)
    return SearchSpaceEntry.model_validate([low, high, "float_log"])


def _make_svr_space_dict(y: Any) -> dict[str, Any]:
    """Build the default SVR search space dict with adaptive epsilon.

    The default space is taken from ``DEFAULT_SPACES["svr"]``, and epsilon is
    replaced by a float_log distribution within bounds derived from the spread
    of ``y_train``.

    Args:
        y (Any): Target variable vector.
    Returns:
        dict[str, Any]: A dict of ``SearchSpaceEntry`` objects.
    """
    space = DEFAULT_SPACES["svr"].copy()
    space["epsilon"] = _svr_adaptive_epsilon_entry(y)
    return space


# ═════════════════════════════ helper utilities ═════════════════════════════


def _apply_dynamic_space(trial: Trial, space_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a configuration dict into model parameters via Optuna methods.

    Args:
        trial (Trial): The current Optuna trial object.
        space_dict (dict[str, Any]): A dict of SearchSpaceEntry objects
            (with bounds and distribution types) or constant values.
    Returns:
        dict[str, Any]: The concrete hyperparameter values for this iteration.
    """
    params: dict[str, Any] = {}
    for key, value in space_dict.items():
        # If this is a SearchSpaceEntry
        # (use the low, high, dist_type, step properties)
        if hasattr(value, "dist_type"):
            low, high = value.low, value.high
            dist_type = value.dist_type
            step = value.step
            if dist_type == "int":
                low_val, high_val = int(cast(float, low)), int(cast(float, high))
                params[key] = trial.suggest_int(
                    key, low_val, high_val, step=int(step) if step is not None else 1
                )
            elif dist_type == "float":
                params[key] = trial.suggest_float(
                    key,
                    float(low),
                    float(high),
                    step=float(step) if step is not None else None,
                )
            elif dist_type == "float_log":
                FloatSpace.validate_log_low(float(low))
                params[key] = trial.suggest_float(
                    key, float(low), float(high), log=True
                )
            elif dist_type == "categorical":
                # Extract the options list from the options attribute or nested config
                options = getattr(value, "options", None)
                if options is None and hasattr(value, "config"):
                    options = getattr(value.config, "options", None)

                # If still None, fall back to low (for compatibility)
                if options is None:
                    options = low

                params[key] = trial.suggest_categorical(key, options)
        else:
            # If it is a plain value (constant), use it as is
            params[key] = value
    return params


class _DistributionRecorder:
    """Record the canonical distributions of the phase search space.

    A lightweight stand-in for an Optuna ``Trial`` that only collects the
    distributions of every ``suggest_*`` call (``suggest_int`` /
    ``suggest_float`` / ``suggest_categorical``). Running ``space_fn`` on the
    recorder yields the distributions the current phase would produce — they
    are compared against the distributions of ``previous_trials`` BEFORE
    ``study.add_trials`` so that an incompatible history fails fast with a
    clear message instead of a silent skip (or a confusing error during the
    first trial of the new phase).
    """

    def __init__(self) -> None:
        """Initialize an empty distribution recorder."""
        self.distributions: dict[str, Any] = {}
        # Trial-compatible attributes that user space functions may read
        # (e.g. ``trial.number`` to alternate between good/bad points).
        self.number = -1

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
        """Record an ``IntDistribution`` and return ``low`` (a valid value)."""
        self.distributions[name] = optuna.distributions.IntDistribution(
            low=low, high=high, step=step
        )
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Record a ``FloatDistribution`` and return ``low`` (a valid value)."""
        self.distributions[name] = optuna.distributions.FloatDistribution(
            low=low, high=high, step=step, log=log
        )
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Record a ``CategoricalDistribution`` and return the first choice."""
        self.distributions[name] = optuna.distributions.CategoricalDistribution(
            choices=list(choices)
        )
        return choices[0]

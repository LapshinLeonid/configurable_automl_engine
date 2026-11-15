"""Static metric names and metric metadata of the contract.

The engine extends this set with runtime scorer objects, dynamic global NRMSE
and user-defined metrics. The contract itself only needs the stable built-in
names, the alias map and the fixed directions used when validating result
payloads.
"""

from __future__ import annotations

from .definitions import MetricDirection

#: The built-in metric names accepted by ``comparison_metric`` and
#: ``additional_metrics``. The list type is preserved for backward
#: compatibility; consumers must not mutate it.
AVAILABLE_METRICS: list[str] = [
    "nrmse",
    "rmse",
    "neg_root_mean_squared_error",
    "mae",
    "mse",
    "r2",
]

ALIAS_TO_SKLEARN: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
}


def to_sklearn_name(name: str) -> str:
    """Map a user metric alias to the corresponding sklearn name.

    Args:
        name: User-facing metric name.

    Returns:
        The lowercased alias target, or the lowercased name when it has no
        alias.
    """
    return ALIAS_TO_SKLEARN.get(name.lower(), name.lower())


#: Fixed direction of the built-in metrics in user semantics. The engine-only
#: ``global_nrmse`` scorer is included because result payloads may refer to it.
BUILTIN_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "nrmse": MetricDirection.minimize,
    "rmse": MetricDirection.minimize,
    "neg_root_mean_squared_error": MetricDirection.minimize,
    "mae": MetricDirection.minimize,
    "mse": MetricDirection.minimize,
    "r2": MetricDirection.maximize,
    "global_nrmse": MetricDirection.minimize,
}

__all__ = [
    "ALIAS_TO_SKLEARN",
    "AVAILABLE_METRICS",
    "BUILTIN_METRIC_DIRECTIONS",
    "to_sklearn_name",
]

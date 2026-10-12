"""
Factory for regression models.
Added short aliases (dt, rf, etc.) and new algorithms.
"""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    ARDRegression,
    ElasticNet,
    GammaRegressor,
    Lasso,
    PoissonRegressor,
    Ridge,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine.common.dependency_utils import is_installed

Algorithm = str

logger = logging.getLogger("configurable_automl_engine.models")

# ----------------------------------------------------------------------------- #
#                           Algorithm map (long keys)                           #
# ----------------------------------------------------------------------------- #


def _get_factory() -> dict[str, Any]:
    factory = {
        "elasticnet": ElasticNet,
        "sgdregressor": SGDRegressor,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "extra_trees": ExtraTreesRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "adaboost": AdaBoostRegressor,
        "poissonregressor": PoissonRegressor,
        "gammaregressor": GammaRegressor,
        "tweedieregressor": TweedieRegressor,
        "gaussian_process_regression": GaussianProcessRegressor,
        "isotonic_regression": IsotonicRegression,
        "nearest_neighbors_regression": KNeighborsRegressor,
        "svr": SVR,
        "ardregression": ARDRegression,
        "glm": TweedieRegressor,
        "ridge": Ridge,
        "lasso": Lasso,
        "xgboosting": None,
    }

    if is_installed("xgboost"):
        from xgboost import XGBRegressor

        factory["xgboosting"] = XGBRegressor
    return factory


AVAILABLE_ALGORITHMS = [
    "elasticnet",
    "sgdregressor",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "adaboost",
    "poissonregressor",
    "gammaregressor",
    "tweedieregressor",
    "gaussian_process_regression",
    "isotonic_regression",
    "nearest_neighbors_regression",
    "svr",
    "ardregression",
    "glm",
    "ridge",
    "lasso",
    "xgboosting",
]

_FACTORY = _get_factory()

# ----------------------------------------------------------------------------- #
#                      Sparse compatibility of algorithms                       #
# ----------------------------------------------------------------------------- #
# Algorithms that reject sparse input matrices (sklearn calls check_array
# with accept_sparse=False): GaussianProcessRegressor, IsotonicRegression,
# and ARDRegression require dense data. With hashing encoding (sparse
# output), the preprocessor must return a dense matrix (force_dense_output)
# for these algorithms.
SPARSE_UNSUPPORTED_ALGORITHMS: frozenset[str] = frozenset(
    {
        "gaussian_process_regression",
        "isotonic_regression",
        "ardregression",
    }
)


def requires_dense_input(algorithm: str) -> bool:
    """Return ``True`` if the algorithm does not support sparse input.

    Used when building the preprocessor (``build_preprocessor``): for
    algorithms that reject ``scipy.sparse``, ``force_dense_output`` is
    enabled so that hashing encoding returns a dense matrix
    (GPR/Isotonic/ARD physically cannot work with sparse data).

    Args:
        algorithm: Algorithm name or alias (e.g. ``'gpr'``,
            ``'isotonic'``, ``'ard'``).

    Returns:
        ``True`` — the model requires dense input; ``False`` — sparse
        matrices (csr) are supported.
    """
    return resolve_algorithm_name(algorithm) in SPARSE_UNSUPPORTED_ALGORITHMS


# ----------------------------------------------------------------------------- #
#               Legacy parameter mappings (API drift)                          #
# ----------------------------------------------------------------------------- #
LEGACY_PARAM_MAPPINGS: dict[str, dict[str, str]] = {
    # ARDRegression: sklearn renamed `n_iter` -> `max_iter` in 1.2+
    "ardregression": {"n_iter": "max_iter"},
}

# ----------------------------------------------------------------------------- #
#                                 Short aliases                                 #
# ----------------------------------------------------------------------------- #
_ALIASES: dict[str, str] = {
    "dt": "decision_tree",
    "rf": "random_forest",
    "et": "extra_trees",
    "gb": "gradient_boosting",
    "ab": "adaboost",
    "elasticnet": "elasticnet",
    "sgd": "sgdregressor",
    "knn": "nearest_neighbors_regression",
    "gpr": "gaussian_process_regression",
    "gaussianprocessregressor": "gaussian_process_regression",
    "svr": "svr",
    "isotonic": "isotonic_regression",
    "ard": "ardregression",
    "glm": "glm",
    "xgboost": "xgboosting",
    "ridge_regression": "ridge",
    "lasso": "lasso",
}


# ----------------------------------------------------------------------------- #
#                  Signature introspection (cached)                             #
# ----------------------------------------------------------------------------- #


@lru_cache(maxsize=32)
def _get_constructor_param_info(cls: type) -> tuple[frozenset[str], bool]:
    """Return accepted constructor param names and whether cls accepts **kwargs.

    The result is cached per *class* (not per instance) so that repeated
    introspection during HyperOpt sweeps is O(1) after the first call.

    Returns
    -------
    tuple[frozenset[str], bool]
        ``(accepted_param_names, accepts_var_kwargs)``
    """
    sig = inspect.signature(cls)
    accepted: set[str] = set()
    accepts_var_kwargs = False
    for name, param in sig.parameters.items():
        if name == "self":  # pragma: no cover
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
        else:
            accepted.add(name)
    return frozenset(accepted), accepts_var_kwargs


# ----------------------------------------------------------------------------- #
#                  Parameter cleaning                                           #
# ----------------------------------------------------------------------------- #


def clean_hyperparameters(
    algo_key: str,
    estimator_cls: type,
    hyperparams: dict[str, Any],
) -> dict[str, Any]:
    """Filter and remap hyperparameters to match the estimator constructor.

    1. **Remap** legacy keys (e.g. ``n_iter`` → ``max_iter`` for ARDRegression)
       using :data:`LEGACY_PARAM_MAPPINGS`.
    2. **Drop** keys that are not accepted by the constructor, unless the
       constructor accepts ``**kwargs``.
    3. **Log** every remap and drop at DEBUG level.

    Parameters
    ----------
    algo_key : str
        Normalised algorithm key (e.g. ``"ardregression"``).
    estimator_cls : type
        The sklearn estimator class.
    hyperparams : dict[str, Any]
        Raw hyperparameter dict from a sweep / config.

    Returns
    -------
    dict[str, Any]
        A clean dict containing only valid constructor kwargs.
    """
    accepted_params, accepts_var_kwargs = _get_constructor_param_info(estimator_cls)
    mappings = LEGACY_PARAM_MAPPINGS.get(algo_key, {})
    cleaned: dict[str, Any] = {}

    for key, value in hyperparams.items():
        # 1. Remap legacy keys
        if key in mappings:
            new_key = mappings[key]
            if new_key not in hyperparams and (
                new_key in accepted_params or accepts_var_kwargs
            ):
                cleaned[new_key] = value
                logger.debug("Remapped '%s' -> '%s' for %s", key, new_key, algo_key)
            continue

        # 2. Keep valid keys; drop unknown ones
        if key in accepted_params or accepts_var_kwargs:
            cleaned[key] = value
        else:
            logger.debug("Dropped unknown param '%s' for %s", key, algo_key)

    return cleaned


# ----------------------------------------------------------------------------- #
#                  Allowed hyperparameter lookup (validation)                   #
# ----------------------------------------------------------------------------- #


@lru_cache(maxsize=64)
def get_allowed_hyperparameters(algo_key: str) -> frozenset[str] | None:
    """Return the set of hyperparameter names valid for the estimator.

    Single source of truth for configuration validation: the allowed set is
    derived from the estimator constructor signature (same criterion as
    :func:`clean_hyperparameters`) plus the legacy parameter names from
    :data:`LEGACY_PARAM_MAPPINGS` that would be remapped to a still-existing
    constructor parameter.

    Returns ``None`` when validation must be skipped:
    - the algorithm is unknown or its optional dependency is missing
      (estimator class is ``None``);
    - the constructor accepts ``**kwargs``, in which case
      :func:`clean_hyperparameters` keeps every key, so no name can be
      considered unknown.

    The result is cached per algorithm key.

    Args:
        algo_key: Normalised or aliased algorithm key (e.g. ``'rf'``,
            ``'xgboost'``).

    Returns:
        Frozen set of accepted hyperparameter names, or ``None`` if the
        allowed set cannot be determined or is unrestricted.
    """
    resolved = resolve_algorithm_name(algo_key)
    estimator_cls = _FACTORY.get(resolved)
    if estimator_cls is None:
        return None

    accepted_params, accepts_var_kwargs = _get_constructor_param_info(estimator_cls)
    if accepts_var_kwargs:
        return None

    allowed = set(accepted_params)
    for legacy, target in LEGACY_PARAM_MAPPINGS.get(resolved, {}).items():
        if target in accepted_params:
            allowed.add(legacy)
    return frozenset(allowed)


# ----------------------------------------------------------------------------- #
#                  Public factory                                                 #
# ----------------------------------------------------------------------------- #


def resolve_algorithm_name(algorithm: str) -> str:
    """Normalize an algorithm name: lowercase + alias expansion.

    Args:
        algorithm: Algorithm name or alias (e.g. ``'RF'``, ``'xgboost'``).

    Returns:
        Canonical algorithm name for lookup in the registry of supported
        algorithms (:data:`AVAILABLE_ALGORITHMS`).
    """
    algo_key = algorithm.lower()
    return _ALIASES.get(algo_key, algo_key)


def create_model(
    algorithm: Algorithm = "elasticnet", **hyperparams: Any
) -> RegressorMixin:
    """
    Returns an instance of the selected regressor.
    The algorithm is given as a string and is case-insensitive. Aliases are
    supported. If the algorithm is not found or requires a missing package,
    raises ValueError/ImportError.
    """
    if not isinstance(algorithm, str):
        raise TypeError(f"Algorithm must be a string, got: {type(algorithm).__name__}")

    algo_key = resolve_algorithm_name(algorithm)

    if algo_key not in _FACTORY:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    estimator_cls = _FACTORY[algo_key]
    if estimator_cls is None:
        # For example, XGBoost can be None if the library is not installed
        raise ImportError(
            f"Algorithm '{algo_key}' requires an additional package. "
            "Install it or exclude it from the list."
        )

    # For GaussianProcessRegressor (gpr), set the RBF(1.0) kernel by default
    if algo_key == "gaussian_process_regression" and "kernel" not in hyperparams:
        hyperparams["kernel"] = RBF(1.0)

    # If max_iter is not set, add a cap (e.g. 10,000 iterations) so that the
    # LibSVM C-code exits the loop by itself if it hangs
    if algo_key == "svr" and "max_iter" not in hyperparams:
        hyperparams["max_iter"] = 10000

    # Clean / remap hyperparameters before instantiation
    hyperparams = clean_hyperparameters(algo_key, estimator_cls, hyperparams)

    return estimator_cls(**hyperparams)

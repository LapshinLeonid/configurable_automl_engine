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
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
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

#: Default random seed injected into stochastic estimators by
#: :func:`create_model` (trees, forests, boosting, elasticnet, ridge, ...).
#: Mirrors ``validation.RANDOM_STATE`` so that the models evaluated during HPO
#: and the final ``ModelTrainer`` fit are reproducible: without a fixed seed
#: every ``fit()`` would sample fresh trees/bootstraps/feature subsets and
#: identical ``train_best_model`` runs would yield different scores and
#: different best parameters.
MODEL_RANDOM_STATE = 42

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


#: Supported Gaussian process kernel families for the ``kernel_type``
#: hyperparameter (the small HPO search space of GPR).
GPR_KERNEL_TYPES: tuple[str, ...] = ("rbf", "matern", "rational_quadratic")

_DEFAULT_BOUNDS = (1e-3, 1e3)


def _make_gpr_kernel(kernel_type: str) -> Any:
    """Build the GPR kernel for a given kernel family name.

    Every kernel is composed as ``amplitude * base_kernel + white_noise``:
    an explicit output-scale amplitude, a base kernel with an adjusted
    length scale, and an explicit noise component. The default kernel is
    ``ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-5)``.

    Args:
        kernel_type: Kernel family name — one of ``'rbf'``, ``'matern'``
            or ``'rational_quadratic'`` (any other name falls back to
            RBF with a warning).

    Returns:
        A compound sklearn kernel (``Sum`` of a ``Product`` and a
        ``WhiteKernel``).
    """
    base: Any
    if kernel_type == "matern":
        base = Matern(
            length_scale=1.0,
            length_scale_bounds=_DEFAULT_BOUNDS,
            nu=1.5,
        )
    elif kernel_type == "rational_quadratic":
        base = RationalQuadratic(
            length_scale=1.0,
            length_scale_bounds=_DEFAULT_BOUNDS,
            alpha=1.0,
            alpha_bounds=_DEFAULT_BOUNDS,
        )
    else:
        if kernel_type != "rbf":
            logger.warning(
                "Unknown GPR kernel type %r, falling back to 'rbf'", kernel_type
            )
        base = RBF(length_scale=1.0, length_scale_bounds=_DEFAULT_BOUNDS)
    return ConstantKernel(
        1.0, constant_value_bounds=_DEFAULT_BOUNDS
    ) * base + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e1))


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


def remap_legacy_param_names(
    algo_key: str, hyperparams: dict[str, Any]
) -> dict[str, Any]:
    """Remap legacy hyperparameter names in a dict, once (before the Optuna loop).

    Renames keys listed in :data:`LEGACY_PARAM_MAPPINGS` (e.g. ``n_iter`` →
    ``max_iter`` for ARDRegression) in a search-space / parameter dict.
    Unlike :func:`clean_hyperparameters`, which runs on every model
    instantiation and logs every remap, this function normalizes the names a
    single time while the search space is being prepared — otherwise an Optuna
    sweep of ~100 trials would emit ~100 identical "Remapped ..." DEBUG lines.

    If both the legacy and the modern name are present, the legacy key is
    dropped (the modern name wins), matching the priority used by
    :func:`clean_hyperparameters`.

    Args:
        algo_key: Algorithm name or alias (e.g. ``'ard'``).
        hyperparams: Parameter dict whose keys are hyperparameter names; the
            values are passed through unchanged (``SearchSpaceEntry`` objects,
            raw values, etc.).

    Returns:
        A new dict with legacy keys renamed to their modern equivalents.
        The input dict is not mutated.
    """
    resolved = resolve_algorithm_name(algo_key)
    mappings = LEGACY_PARAM_MAPPINGS.get(resolved, {})
    if not mappings:
        return dict(hyperparams)

    remapped: dict[str, Any] = {}
    for key, value in hyperparams.items():
        if key in mappings:
            new_key = mappings[key]
            # The modern name wins: if it is already present, keep it and
            # drop the legacy key (same priority as clean_hyperparameters).
            if new_key in hyperparams:
                continue
            remapped[new_key] = value
        else:
            remapped[key] = value
    return remapped


@lru_cache(maxsize=128)
def get_param_numeric_type(algo_key: str, param: str) -> str | None:
    """Return ``'int'`` or ``'float'`` for a numeric constructor parameter.

    The numeric type is inferred from the type of the parameter's default
    value in the estimator constructor signature (sklearn models carry no
    parameter annotations). ``None`` is returned when the type cannot be
    determined:
    - the algorithm is unknown or its optional dependency is missing
      (estimator class is ``None``);
    - the parameter is absent from the signature;
    - the default is non-numeric (e.g. a string or ``None``);
    - the default is a boolean (not a numeric range).

    The result is cached per (algorithm, parameter) pair.

    Args:
        algo_key: Normalised or aliased algorithm key (e.g. ``'svr'``,
            ``'xgboost'``).
        param: Constructor parameter name.

    Returns:
        ``'int'``, ``'float'``, or ``None`` when the type is unknown.
    """
    resolved = resolve_algorithm_name(algo_key)
    estimator_cls = _FACTORY.get(resolved)
    if estimator_cls is None:
        return None
    signature = inspect.signature(estimator_cls)
    p = signature.parameters.get(param)
    if p is None or p.default is inspect.Parameter.empty:
        return None
    default = p.default
    if isinstance(default, bool):
        return None
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    return None


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
    constructor parameter. Tuner-level data-adapter hints are added as
    documented exceptions: ``kernel_type`` for
    ``gaussian_process_regression``, ``feature_index`` for
    ``isotonic_regression``.

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
    # The kernel family selector of GaussianProcessRegressor: a
    # tuner-level hint translated to a concrete kernel by
    # ``create_model`` (not a constructor parameter).
    if resolved == "gaussian_process_regression":
        allowed.add("kernel_type")
    # ``feature_index`` of IsotonicRegression: not a constructor parameter
    # but a data-adapter hint consumed by ``IsotonicDataTransformer`` (the
    # algorithmic feature chosen for the strictly univariate isotonic model).
    if resolved == "isotonic_regression":
        allowed.add("feature_index")
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

    # Deterministic seeding of stochastic estimators. Trees, forests, boosting
    # and the randomized linear models draw randomness at fit time; with a
    # fresh default RNG every fit() is different, so identical
    # ``train_best_model`` runs would produce different scores/parameters
    # (and the refine monotonicity promise would break). A fixed seed is
    # injected for every estimator that accepts ``random_state`` (including
    # XGBoost, whose constructor accepts **kwargs), unless the caller set the
    # parameter explicitly — an explicit value always wins. An EXPLICIT
    # ``random_state=None`` (a truly random fit, sklearn semantics) is
    # respected: the seed is not injected, and a WARNING is logged so that the
    # non-reproducible fit is visible.
    if "random_state" in hyperparams:
        if hyperparams["random_state"] is None:
            logger.warning(
                "Algorithm '%s': explicit random_state=None — the fit is "
                "NOT reproducible (a fresh RNG is drawn at fit time).",
                algo_key,
            )
    else:
        accepted_params, accepts_var_kwargs = _get_constructor_param_info(estimator_cls)
        if "random_state" in accepted_params or accepts_var_kwargs:
            hyperparams["random_state"] = MODEL_RANDOM_STATE

    # For GaussianProcessRegressor (gpr), set a proper default kernel
    # (amplitude × RBF + white noise), enable target normalization and
    # several optimizer restarts with a fixed seed. A bare RBF(1.0) without
    # an amplitude/noise component and without y-normalization produces a
    # "mean predictor" (score ≈ std(y)), while GPR is one of the most
    # suitable models on small samples and provides prediction intervals.
    if algo_key == "gaussian_process_regression":
        # The HPO search space exposes a categorical ``kernel_type``
        # (rbf/matern/rational_quadratic) instead of a raw kernel object —
        # Optuna stores only JSON-serializable categorical values, so the
        # abstract name is translated into a concrete kernel here.
        kernel_type = hyperparams.pop("kernel_type", "rbf")
        if "kernel" not in hyperparams:
            hyperparams["kernel"] = _make_gpr_kernel(kernel_type)
    if algo_key == "gaussian_process_regression" and "normalize_y" not in hyperparams:
        hyperparams["normalize_y"] = True
    if (
        algo_key == "gaussian_process_regression"
        and "n_restarts_optimizer" not in hyperparams
    ):
        hyperparams["n_restarts_optimizer"] = 2

    # If max_iter is not set, add a cap (e.g. 10,000 iterations) so that the
    # LibSVM C-code exits the loop by itself if it hangs
    if algo_key == "svr" and "max_iter" not in hyperparams:
        hyperparams["max_iter"] = 10000

    # IsotonicRegression defaults to out_of_bounds='nan': validation points
    # outside the training range get NaN predictions, so the default hold-out
    # (train_test_split) fails with "Validation scoring failed for all folds:
    # Input contains NaN". With out_of_bounds='clip' the same data scores a
    # finite R² (≈0.968). Unless the caller set the parameter explicitly,
    # "clip" is used so that training works out of the box. An explicit value
    # is validated here (sklearn accepts only 'nan'/'clip'/'raise') so a typo
    # fails fast at model creation, before any fold evaluation.
    if algo_key == "isotonic_regression":
        raw_oob = hyperparams.get("out_of_bounds", "clip")
        if raw_oob not in ("nan", "clip", "raise"):
            raise ValueError(
                f"out_of_bounds must be one of ('nan', 'clip', 'raise') for "
                f"IsotonicRegression, got {raw_oob!r}"
            )
        hyperparams["out_of_bounds"] = raw_oob

    # Clean / remap hyperparameters before instantiation
    hyperparams = clean_hyperparameters(algo_key, estimator_cls, hyperparams)

    return estimator_cls(**hyperparams)

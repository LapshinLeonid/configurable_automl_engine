"""
Фабрика регрессионных моделей.
Добавлены короткие алиасы (dt, rf и т.д.) и новые алгоритмы.
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
#                       Карта алгоритмов (длинные ключи)                       #
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
        "xgboosting": None
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
#               Legacy parameter mappings (API drift)                          #
# ----------------------------------------------------------------------------- #
LEGACY_PARAM_MAPPINGS: dict[str, dict[str, str]] = {
    # ARDRegression: sklearn renamed `n_iter` -> `max_iter` in 1.2+
    "ardregression": {"n_iter": "max_iter"},
}

# ----------------------------------------------------------------------------- #
#                       Короткие псевдонимы (алиасы)                            #
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
    "lasso": "lasso"
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
    sig = inspect.signature(cls.__init__)
    accepted: set[str] = set()
    accepts_var_kwargs = False
    for name, param in sig.parameters.items():
        if name == "self":
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
            if new_key not in hyperparams and (new_key in accepted_params or accepts_var_kwargs):
                cleaned[new_key] = value
                logger.debug(
                    "Remapped '%s' -> '%s' for %s", key, new_key, algo_key
                )
            continue

        # 2. Keep valid keys; drop unknown ones
        if key in accepted_params or accepts_var_kwargs:
            cleaned[key] = value
        else:
            logger.debug("Dropped unknown param '%s' for %s", key, algo_key)

    return cleaned


# ----------------------------------------------------------------------------- #
#                  Public factory                                                 #
# ----------------------------------------------------------------------------- #

def create_model(algorithm: Algorithm = "elasticnet", 
                 **hyperparams: Any
                 ) -> RegressorMixin:
    """
    Возвращает экземпляр выбранного регрессора.
    Алгоритм задаётся строкой, не зависит от регистра. Поддерживаются алиасы.
    Если алгоритм не найден или требует отсутствующий пакет, 
    бросает ValueError/ImportError.
    """
    if not isinstance(algorithm, str):
        raise TypeError(f"Алгоритм должен быть строкой, получено: {type(algorithm).__name__}")  

    algo_key = algorithm.lower()
    algo_key = _ALIASES.get(algo_key, algo_key)

    if algo_key not in _FACTORY:
        raise ValueError(f"Неизвестный алгоритм: {algorithm!r}")

    estimator_cls = _FACTORY[algo_key]
    if estimator_cls is None:
        # Например, XGBoost может быть None, если библиотека не установлена
        raise ImportError(
            f"Алгоритм '{algo_key}' требует дополнительного пакета. "
            "Установите его или исключите из списка."
        )

    # Для GaussianProcessRegressor (gpr) по умолчанию ставим ядро RBF(1.0)
    if algo_key == "gaussian_process_regression" and "kernel" not in hyperparams:
        hyperparams["kernel"] = RBF(1.0)

    # Если max_iter не задан, ставим ограничение (например, 10 000 итераций),
    # чтобы C-код LibSVM сам выходил из цикла при заклинивании
    if algo_key == "svr" and "max_iter" not in hyperparams:
        hyperparams["max_iter"] = 10000

    # Clean / remap hyperparameters before instantiation
    hyperparams = clean_hyperparameters(algo_key, estimator_cls, hyperparams)

    return estimator_cls(**hyperparams)

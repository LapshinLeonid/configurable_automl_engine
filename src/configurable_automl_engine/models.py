"""
Фабрика регрессионных моделей.
Добавлены короткие алиасы (dt, rf и т.д.) и новые алгоритмы.
"""

from __future__ import annotations
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.linear_model import (
    ElasticNet,
    SGDRegressor,
    ARDRegression,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
    Ridge,
    Lasso
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor)

from configurable_automl_engine.common.dependency_utils import is_installed

Algorithm = str

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
        "lasso": Lasso
    }
        
    if is_installed("xgboost"):
        from xgboost import XGBRegressor
        factory["xgboosting"] = XGBRegressor
    return factory

_FACTORY = _get_factory()

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
        raise ValueError(f"Некорректный алгоритм: {algorithm!r}")

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

    return estimator_cls(**hyperparams)

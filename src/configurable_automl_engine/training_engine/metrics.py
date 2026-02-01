"""
Metric registry + helper utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Поддержаны кастомные метрики ``rmse`` и ``nrmse``.
  - ``rmse`` → стандартный корень из MSE, но для sklearn-совместимого скоринга
    регистрируем *отрицательный* вариант (чтобы чем выше, тем лучше).
  - ``nrmse`` → RMSE, нормированный на диапазон (max-min) таргета.
                Если диапазон < 1e-6, возвращает 0.0, избегая ±Inf.

После импорта этого файла любой вызов::

    from sklearn.metrics import get_scorer
    get_scorer("nrmse")

будет работать без исключений, а Optuna / GridSearchCV смогут честно
оптимизировать метрику.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, get_scorer as sklearn_get_scorer

# --------------------------------------------------------------------------- #
#  Сами метрики
# --------------------------------------------------------------------------- #
def _rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (меньше → лучше)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

class NRMSEZeroRangeError(ValueError):
    """Raised when y_true has zero range inside a CV-split."""

def _nrmse(y_true, y_pred):
    """
    Normalised RMSE with *split-local* min–max-scale.

    Raises
    ------
    NRMSEZeroRangeError
        If max(y_true) == min(y_true);  раньше возвращали 0.0.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    denom = np.max(y_true) - np.min(y_true)

    if denom < 1e-6:          # раньше просто отдавали 0.0
        raise NRMSEZeroRangeError(
            "Cannot compute NRMSE: target is constant within the split "
            f"(len={len(y_true)}, value={y_true[0]!r})."
        )

    return rmse / denom

# --------------------------------------------------------------------------- #
#  Частный реестр «сырой» (без переворота знака и прочего)
# --------------------------------------------------------------------------- #
_METRICS: Dict[str, Callable] = {
    # «меньше → лучше»
    "rmse": _rmse,
    "nrmse": _nrmse,

    # «больше → лучше»
    "r2": r2_score,

    # alias: «больше → лучше» (отрицательный RMSE)
    "neg_root_mean_squared_error": lambda y_t, y_p: -_rmse(y_t, y_p),
}

# --------------------------------------------------------------------------- #
#  Реестр готовых объектов-скореров для использования в sklearn API
# --------------------------------------------------------------------------- #
_SCORER_OBJECTS: Dict[str, Callable] = {
    "nrmse": make_scorer(_nrmse, greater_is_better=False),
    "neg_root_mean_squared_error": make_scorer(
        lambda y_t, y_p: -_rmse(y_t, y_p),
        greater_is_better=True,
    ),
    "rmse": make_scorer(
        lambda y_t, y_p: -_rmse(y_t, y_p),
        greater_is_better=True,
    ),
}

# Какие метрики интерпретируются как «больше — тем лучше»
_GREATER_IS_BETTER = {"r2", "neg_root_mean_squared_error"}


# --------------------------------------------------------------------------- #
#  Public helpers — могут пригодиться снаружи
# --------------------------------------------------------------------------- #
def get_metric(name: str) -> Callable:
    """
    Вернуть «сырую» функцию-метрику (без обёртки make_scorer).

    Raises
    ------
    KeyError : если запрошенной метрики нет.
    """
    lname = name.lower()
    if lname not in _METRICS:
        raise KeyError(f"Metric '{name}' not implemented.")
    return _METRICS[lname]


def is_greater_better(name: str) -> bool:
    """
    True, если метрику следует **максимизировать**.

    Нужна, например, когда вручную считаем score и хотим понять,
    ищем min или max.
    """
    return name.lower() in _GREATER_IS_BETTER

def get_scorer_object(name: str) -> Callable | str:
    """
    Возвращает объект-скорер для использования в GridSearchCV или cross_validate.
    Для кастомных метрик возвращает Scorer-объект, для стандартных — строку.
    """
    lname = name.lower()
    # Если это наша кастомная метрика (nrmse, rmse и т.д.)
    if lname in _SCORER_OBJECTS:
        return _SCORER_OBJECTS[lname]
    
    # В остальных случаях возвращаем имя как есть (sklearn сам найдет встроенную метрику)
    return sklearn_get_scorer(lname)

# --------------------------------------------------------------------------- #
#  Приведение пользовательских alias-ов к тому, что понимает sklearn
# --------------------------------------------------------------------------- #
_ALIAS_TO_SKLEARN = {
    "rmse": "neg_root_mean_squared_error",  # т.к. sklearn оптимизирует «чем выше — тем лучше»
    # nrmse регистрируется напрямую
}


def to_sklearn_name(name: str) -> str:
    """
    Вернуть имя, совместимое с ``sklearn.metrics.get_scorer``.
    Если alias не найден — возвращаем в lower-case как есть.
    """
    return _ALIAS_TO_SKLEARN.get(name.lower(), name.lower())

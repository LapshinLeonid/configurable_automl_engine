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
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# --------------------------------------------------------------------------- #
#  Достаём внутренний реестр scorers (имя модуля менялось в sklearn 1.3)
# --------------------------------------------------------------------------- #
try:  # scikit-learn ≤ 1.2
    from sklearn.metrics._scorer import SCORERS as _SK_SCORERS
except ImportError:  # scikit-learn ≥ 1.3
    from sklearn.metrics._scorer import _SCORERS as _SK_SCORERS  # type: ignore


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

# Какие метрики интерпретируются как «больше — тем лучше»
_GREATER_IS_BETTER = {"r2", "neg_root_mean_squared_error"}

# --------------------------------------------------------------------------- #
#  Регистрируем в SCORERS всё, чего там ещё нет
# --------------------------------------------------------------------------- #
if "nrmse" not in _SK_SCORERS:
    _SK_SCORERS["nrmse"] = make_scorer(_nrmse, greater_is_better=False)

# Честный отрицательный RMSE (у sklearn уже есть встроенный,
# но добавим на всякий случай, если его переименуют)
if "neg_root_mean_squared_error" not in _SK_SCORERS:
    _SK_SCORERS["neg_root_mean_squared_error"] = make_scorer(
        lambda y_t, y_p: -_rmse(y_t, y_p),
        greater_is_better=True,
    )

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

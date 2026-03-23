"""
    Regression Metric Ecosystem: Профессиональный реестр и адаптер метрик.
    Модуль расширяет стандартный набор `scikit-learn` 
    кастомными реализациями RMSE и NRMSE, 
    обеспечивая их бесшовную интеграцию в процессы автоматического 
    подбора гиперпараметров (GridSearchCV, Optuna) и кросс-валидацию.
    
    Ключевые возможности:
        1. Dual-Normalization NRMSE: 
            Поддержка двух стратегий нормализации — локальной 
            (динамический размах внутри фолда) 
            и глобальной (фиксированный размах всего датасета).
        2. Scorer API Compatibility: Автоматическая инверсия знака метрик-ошибок 
           (меньше -> лучше) в формат скореров (больше -> лучше) 
            для корректной оптимизации.
        3. Numeric Stability: Встроенная защита от деления на ноль 
           при встрече с константным таргетом — возврат `inf` 
           вместо ошибки для сохранения стабильности пайплайна.
        4. Smart Discovery: Единая точка доступа `get_scorer_object` 
           с механизмом алиасов,  упрощающая вызов кастомных метрик 
           по коротким именам (например, "rmse").
"""


from __future__ import annotations

from typing import Callable, Dict, Any, cast, Union

import numpy as np
import logging
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error,
                             r2_score, 
                             make_scorer, 
                             get_scorer as sklearn_get_scorer
                            )

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Сами метрики
# --------------------------------------------------------------------------- #
def _rmse(
        y_true: np.ndarray, 
        y_pred: np.ndarray
        ) -> float:
    """Рассчитать корень из среднеквадратичной ошибки (RMSE).
    Логика расчета:
    1. Вычисляется MSE с использованием стандартной функции `mean_squared_error`.
    2. Из результата извлекается квадратный корень через `np.sqrt`.
    3. Результат принудительно приводится к типу float для обеспечения консистентности.
    Args:
        y_true (np.ndarray): Истинные значения целевой переменной.
        y_pred (np.ndarray): Предсказанные значения модели.
    Returns:
        float: Значение RMSE (чем меньше, тем лучше).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

class NRMSEZeroRangeError(ValueError):
    """Raised when y_true has zero range inside a CV-split."""


# NRMSE для одного фолда
def _nrmse(
        y_true: Any,
        y_pred: Any
        ) -> float:
    """Рассчитать локальный нормализованный RMSE (NRMSE).
    Логика расчета:
    1. Вычисляется стандартное значение RMSE для текущей выборки.
    2. Определяется диапазон (max - min) на основе 
    переданных истинных значений `y_true`.
    3. Обработка константного таргета: если диапазон < 1e-6, 
    возвращается `inf` (бесконечная ошибка) 
    с логированием предупреждения, чтобы избежать деления на ноль.
    4. RMSE делится на вычисленный локальный диапазон.
    Args:
        y_true (Any): Истинные значения целевой переменной в рамках сплита.
        y_pred (Any): Предсказанные значения модели.
    Returns:
        float: Значение NRMSE, нормализованное на локальный диапазон.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.max(y_true) - np.min(y_true)

    if denom < 1e-6:
        logger.warning(
            f"NRMSE: target is constant (range < 1e-6) for split of size "
            f"{len(y_true)}. Returning +inf (will be inverted to -inf by scorer)."
        )
        return float('inf')
    return float(rmse / denom)

def _global_nrmse(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_range: float
) -> float:
    """Рассчитать глобальный нормализованный RMSE 
    с использованием фиксированного диапазона.
    Логика расчета:
    1. Валидация входного диапазона: если `target_range` меньше порога 1e-6, 
    возвращается `inf` для предотвращения численной нестабильности.
    2. Вычисляется стандартное значение RMSE.
    3. Ошибка нормализуется на заранее вычисленный глобальный диапазон 
    (не зависящий от текущего сплита).
    Args:
        y_true (np.ndarray): Истинные значения целевой переменной.
        y_pred (np.ndarray): Предсказанные значения модели.
        target_range (float): Глобальный размах (max - min) всего набора данных.
    Returns:
        float: Значение NRMSE, сопоставимое между разными фолдами кросс-валидации.
    """
    if target_range < 1e-6:
        logger.warning("Global target_range is too small. Returning +inf.")
        return float('inf')
        
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return rmse / target_range

def get_global_nrmse_scorer(global_y: np.ndarray) -> Callable[..., Any]:
    """Создать объект-скорер для глобального NRMSE, совместимый с Scikit-Learn.
    Логика создания:
    1. Вычисляются минимальное и максимальное значения 
    из переданного полного вектора `global_y`.
    2. Рассчитывается глобальный диапазон `target_range`.
    3. Формируется объект-скорер через `make_scorer`, 
    куда упаковывается функция `_global_nrmse`.
    4. Устанавливается флаг `greater_is_better=False`, 
    чтобы sklearn инвертировал метрику для максимизации.
    Args:
        global_y (np.ndarray): Полный вектор целевой переменной 
        для расчета глобального диапазона.
    Returns:
        Any: Объект Scorer для использования в GridSearchCV или cross_validate.
    """
    y_max = np.max(global_y)
    y_min = np.min(global_y)
    target_range = float(y_max - y_min)
    
    scorer = make_scorer(
        _global_nrmse, 
        greater_is_better=False, 
        target_range=target_range
    )

    return cast(Callable[..., Any], scorer)

# --------------------------------------------------------------------------- #
#  Частный реестр «сырой» (без переворота знака и прочего)
# --------------------------------------------------------------------------- #
_METRICS: Dict[str, Callable[..., Any]] = {
    # «меньше → лучше»
    "rmse": _rmse,
    "nrmse": _nrmse,
    "global_nrmse": _global_nrmse, 

    # «больше → лучше»
    "r2": r2_score,

    # alias: «больше → лучше» (отрицательный RMSE)
    "neg_root_mean_squared_error": lambda y_t, y_p: -_rmse(y_t, y_p),
}


# --------------------------------------------------------------------------- #
#  Реестр готовых объектов-скореров для использования в sklearn API
# --------------------------------------------------------------------------- #
_SCORER_OBJECTS: Dict[str, Callable[..., Any]] = {
    # Для ошибок устанавливаем greater_is_better=False, 
    # sklearn сам будет возвращать отрицательные значения для максимизации
    "nrmse": make_scorer(_nrmse, greater_is_better=False),
    "rmse": make_scorer(_rmse, greater_is_better=False),
    "neg_root_mean_squared_error": make_scorer(_rmse, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "mse": make_scorer(mean_squared_error, greater_is_better=False),

}

# Какие метрики ИЗНАЧАЛЬНО интерпретируются как «чем выше — тем лучше»
# (те, которые не требуют инверсии знака для понимания пользователем)
_GREATER_IS_BETTER = {
    "r2", 
    "explained_variance", 
    "adjusted_r2"
}


# --------------------------------------------------------------------------- #
#  Public helpers — могут пригодиться снаружи
# --------------------------------------------------------------------------- #
def get_metric(name: str) -> Callable[..., Any]:
    """Получить «сырую» функцию расчета метрики из внутреннего реестра.
    Логика получения:
    1. Имя метрики приводится к нижнему регистру для исключения ошибок поиска.
    2. Проверяется наличие имени в реестре `_METRICS`.
    3. Если метрика не найдена, инициируется исключение `KeyError`.
    Args:
        name (str): Название метрики (например, 'rmse' или 'nrmse').
    Returns:
        Callable[..., Any]: Функция, принимающая (y_true, y_pred) и возвращающая float.
    """
    lname = name.lower()
    if lname not in _METRICS:
        raise KeyError(f"Metric '{name}' not implemented.")
    return _METRICS[lname]


def is_greater_better(name: str) -> bool:
    """Определить, является ли метрика максимизируемой (Score) 
    или минимизируемой (Error).
    
    Args:
        name (str): Название метрики.
    Returns:
        bool: True для метрик типа R2, False для метрик типа RMSE/MAE.
    """
    lname = name.lower()
    # Если метрика явно в списке "больше-лучше"
    if lname in _GREATER_IS_BETTER:
        return True
    # Если метрика содержит префиксы ошибки или наши стандартные регрессионные ошибки
    if any(prefix in lname for prefix in ["rmse", "nrmse", "error", "mae", "mse"]):
        return False
    # По умолчанию для sklearn scorers, если не уверены, 
    # проверяем наличие 'neg_' в названии (стандарт sklearn для ошибок)
    if lname.startswith("neg_"):
        return False
        
    return True # Default fallback для R2-подобных метрик

def get_scorer_object(name: str,
                      global_y: np.ndarray | None = None
                      ) -> Callable[..., Any] | str:
    """Получить объект-скорер, готовый для использования в инструментах sklearn.
    Логика получения:
    1. Обработка 'global_nrmse': требует обязательного наличия `global_y` 
    для создания динамического скорера.
    2. Поиск в реестре `_SCORER_OBJECTS`: возвращает преднастроенные кастомные скореры.
    3. Fallback: если имя не в реестре, используется 
    стандартный `sklearn.metrics.get_scorer`.
    Args:
        name (str): Название требуемого скорера.
        global_y (np.ndarray | None): Опциональный массив таргета для глобальных метрик.
    Returns:
        Union[Callable[..., Any], str]: Объект Scorer или системная строка sklearn.
    """
    lname = name.lower()

    # Спец-обработка для глобального NRMSE
    if lname == "global_nrmse":
        if global_y is None:
            raise ValueError(
                "For 'global_nrmse', 'global_y' must be passed to get_scorer_object")
        return get_global_nrmse_scorer(global_y)

    # Если это наша кастомная метрика (nrmse, rmse и т.д.)
    if lname in _SCORER_OBJECTS:
        return _SCORER_OBJECTS[lname]
    
    # В остальных случаях возвращаем имя как есть 
    # (sklearn сам найдет встроенную метрику)
    scorer = sklearn_get_scorer(lname)
    return cast("Union[Callable[..., Any], str]", scorer)

# --------------------------------------------------------------------------- #
#  Приведение пользовательских alias-ов к тому, что понимает sklearn
# --------------------------------------------------------------------------- #
_ALIAS_TO_SKLEARN = {
    # т.к. sklearn оптимизирует «чем выше — тем лучше»
    # nrmse регистрируется напрямую
    "rmse": "neg_root_mean_squared_error",  
}


def to_sklearn_name(name: str) -> str:
    """Привести пользовательский алиас метрики к системному названию sklearn.
    Логика преобразования:
    1. Имя приводится к нижнему регистру.
    2. Выполняется поиск по словарю `_ALIAS_TO_SKLEARN` 
    (например, 'rmse' -> 'neg_root_mean_squared_error').
    3. Если алиас не найден, возвращается исходное имя в нижнем регистре.
    Args:
        name (str): Пользовательское название метрики.
    Returns:
        str: Имя метрики, распознаваемое внутренними механизмами sklearn.
    """
    return _ALIAS_TO_SKLEARN.get(name.lower(), name.lower())

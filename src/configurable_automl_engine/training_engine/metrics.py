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

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Сами метрики
# --------------------------------------------------------------------------- #
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def oof_rmse(y_true: Any, y_pred: Any) -> float:
    """Рассчитать RMSE по выровненному OOF-вектору целиком (issue #62).

    В отличие от усреднения по фолдам, метрика считается по всему вектору
    out-of-fold предсказаний сразу: каждая строка входит в расчёт ровно один
    раз. Пары, где ``y_true`` или ``y_pred`` не являются конечными числами
    (NaN/None/inf — непокрытые строки ``train_test_split`` либо сбойные
    предсказания отдельных фолдов), отбрасываются, чтобы конкатенация фолдов
    разной длины «не разъезжалась» по индексам.

    Args:
        y_true (Any): Истинные значения, выровненные с ``y_pred`` по позиции.
        y_pred (Any): OOF-предсказания, выровненные с ``y_true`` по позиции.

    Returns:
        float: RMSE по всем валидным парам (меньше — лучше).

    Raises:
        ValueError: Если после отбрасывания нефинитных пар не осталось ни
            одной валидной пары (OOF-оценка невозможна).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        raise ValueError("No valid (y_true, y_pred) pairs for OOF RMSE.")
    return _rmse(yt[mask], yp[mask])


class NRMSEZeroRangeError(ValueError):
    """Raised when y_true has zero range inside a CV-split."""


# NRMSE для одного фолда
def _nrmse(y_true: Any, y_pred: Any) -> float:
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
        return float("inf")
    return float(rmse / denom)


def _global_nrmse(y_true: np.ndarray, y_pred: np.ndarray, target_range: float) -> float:
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
        return float("inf")

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
        _global_nrmse, greater_is_better=False, target_range=target_range
    )

    return cast(Callable[..., Any], scorer)


# --------------------------------------------------------------------------- #
#  Частный реестр «сырой» (без переворота знака и прочего)
# --------------------------------------------------------------------------- #
_METRICS: dict[str, Callable[..., Any]] = {
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
# Для каждой метрики рядом со скорером явно хранится направление
# оптимизации (is_greater_better): True — «больше — лучше» (r2, а также
# инвертированные neg_-метрики), False — «меньше — лучше» (rmse, mae и т.д.).
# Направление больше НЕ выводится эвристикой из подстрок имени — только из
# этого реестра либо из объекта-скорера sklearn.
_SCORER_OBJECTS: dict[str, tuple[Callable[..., Any], bool]] = {
    # Для ошибок устанавливаем greater_is_better=False,
    # sklearn сам будет возвращать отрицательные значения для максимизации
    "nrmse": (make_scorer(_nrmse, greater_is_better=False), False),
    "rmse": (make_scorer(_rmse, greater_is_better=False), False),
    "neg_root_mean_squared_error": (make_scorer(_rmse, greater_is_better=False), True),
    "mae": (make_scorer(mean_absolute_error, greater_is_better=False), False),
    "mse": (make_scorer(mean_squared_error, greater_is_better=False), False),
    "r2": (make_scorer(r2_score, greater_is_better=True), True),
}

AVAILABLE_METRICS = list(_SCORER_OBJECTS.keys())


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


def _resolve_scorer(name: str) -> Callable[..., Any]:
    """Разрешить имя метрики в фактический объект-скорер.

    Логика разрешения:
    1. Для метрик собственного реестра ``_SCORER_OBJECTS`` возвращается
       преднастроенный объект ``make_scorer`` (первый элемент кортежа).
    2. Иначе имя делегируется ``sklearn.metrics.get_scorer``.

    Args:
        name (str): Название метрики.

    Returns:
        Callable[..., Any]: Объект-скорер, совместимый с API sklearn.

    Raises:
        ValueError: Если имя метрики не найдено ни в реестре, ни в sklearn
            (с понятным сообщением в стиле ``get_metric``).
    """
    lname = name.lower()
    if lname in _SCORER_OBJECTS:
        return _SCORER_OBJECTS[lname][0]
    try:
        return cast(Callable[..., Any], sklearn_get_scorer(lname))
    except ValueError as err:
        raise ValueError(
            f"Metric '{name}' not implemented. Use one of the registered "
            "metrics or a valid sklearn scorer name "
            "(see sklearn.metrics.get_scorer_names())."
        ) from err


@lru_cache(maxsize=256)
def _sklearn_direction(name: str) -> bool:
    """Определить направление оптимизации по объекту-скореру sklearn.

    Кэшируется только «стабильная» часть: реестр скореров sklearn не
    изменяется в рантайме, поэтому результат для конкретного имени
    детерминирован и безопасен для кэширования. В отличие от неё,
    собственный реестр ``_SCORER_OBJECTS`` может быть расширен/изменён
    пользователем, поэтому он читается без кэша (см. ``is_greater_better``).

    У объекта-скорера доступен флаг направления (``_sign``): +1 — «больше —
    лучше», -1 — инвертированная ошибка (neg_-префикс), которую оптимизатор
    по-прежнему максимизирует.

    Args:
        name (str): Название метрики в нижнем регистре.

    Returns:
        bool: True, если значение метрики максимизируется.

    Raises:
        ValueError: Если метрика неизвестна sklearn (см. ``_resolve_scorer``).
    """
    scorer = _resolve_scorer(name)
    sign = getattr(scorer, "_sign", None)
    if sign is not None:
        return bool(sign > 0) or name.startswith("neg_")
    return True


def is_greater_better(name: str) -> bool:
    """Определить направление оптимизации метрики: «больше — лучше»?

    Направление определяется по фактическому объекту-скореру, а не по
    подстрокам имени:
    1. Для метрик собственного реестра ``_SCORER_OBJECTS`` направление
       хранится явно рядом со скорером (rmse/mae/mse/nrmse — False,
       r2 и neg_root_mean_squared_error — True). Реестр читается каждый раз
       без кэша: он изменяем, и результат обязан отражать актуальное
       состояние (ревью PR #18).
    2. Для остальных имён используется объект ``sklearn.metrics.get_scorer``:
       все скореры sklearn устроены так, что большее значение лучше —
       метрики-ошибки уже инвертированы в neg_-скореры (``_sign == -1``),
       score-метрики возвращаются как есть (``_sign == +1``). Этот путь
       стабилен и кэшируется (``_sklearn_direction``).

    Args:
        name (str): Название метрики.

    Returns:
        bool: True, если значение метрики максимизируется (r2, neg_*-метрики),
            False для метрик-ошибок (RMSE, MAE, MSE, NRMSE, global_nrmse).

    Raises:
        ValueError: Если метрика неизвестна ни реестру, ни sklearn.
    """
    lname = name.lower()
    # 1. Явное направление из собственного реестра — без кэша.
    if lname in _SCORER_OBJECTS:
        return _SCORER_OBJECTS[lname][1]
    # 2. Динамический глобальный NRMSE — ошибка («меньше — лучше»), как и
    #    обычный NRMSE. Скорер требует global_y и не резолвится sklearn,
    #    поэтому обрабатывается явно (синхронно с to_user_value).
    if lname == "global_nrmse":
        return False
    # 3. Стабильные sklearn-скореры — с кэшированием результата.
    return _sklearn_direction(lname)


def is_error_metric(name: str) -> bool:
    """Является ли метрика ошибкой в пользовательском представлении.

    Определяется по фактическому объекту-скореру (атрибут ``_sign``):
    инвертированные скореры ошибок (-RMSE, -MAE, все neg_-метрики)
    возвращают пользователю положительное значение ошибки → True;
    score-метрики (R² и т.п.) возвращаются как есть → False.

    Args:
        name (str): Название метрики.

    Returns:
        bool: True для метрик-ошибок (RMSE, MAE, MSE, NRMSE, neg_*-метрики),
            False для score-метрик (R², accuracy и т.п.).

    Raises:
        ValueError: Если метрика неизвестна ни реестру, ни sklearn.
    """
    lname = name.lower()
    if lname == "global_nrmse":
        return True
    scorer = _resolve_scorer(lname)
    sign = getattr(scorer, "_sign", None)
    return sign is not None and sign < 0


def direction_label(name: str) -> str:
    """Вернуть человекочитаемое направление метрики для логов (issue #54, T6).

    Пользовательские значения метрик (``to_user_value``) в логах обязаны
    сопровождаться направлением: для ошибок — ``"min better"``, для score-
    метрик — ``"max better"``. Направление берётся из ``user_direction``
    (пользовательская семантика), а не из флага оптимизатора
    ``greater_is_better`` — для neg_-метрик эти семантики расходятся
    (оптимизатор максимизирует -RMSE, а пользователю меньшее значение лучше).

    Args:
        name (str): Название метрики.

    Returns:
        str: ``"min better"`` для ошибок, ``"max better"`` для score-метрик.

    Raises:
        ValueError: Если метрика неизвестна ни реестру, ни sklearn.
    """
    return "min better" if user_direction(name) == "minimize" else "max better"


def user_direction(name: str) -> Literal["minimize", "maximize"]:
    """Определить направление метрики в пользовательской семантике (issue #54).

    В отличие от ``is_greater_better`` (семантика оптимизатора — «больше —
    лучше» для всех скореров), возвращает направление для значений в
    пользовательском представлении (``to_user_value``):

    - метрики-ошибки (RMSE, MAE, MSE, NRMSE, global_nrmse и все neg_-метрики)
      → ``"minimize"``: в пользовательской семантике меньшее значение лучше;
    - score-метрики (R², accuracy и т.п.) → ``"maximize"``: большее лучше.

    Направление требуется формуле коридора пула финалистов (T3, issue #65):
    мультипликативная форма коридора зависит от того, какое значение лучше —
    меньшее или большее (для R² с отрицательным лидером мультипликативная
    форма некорректна, см. ``select_finalists``).

    Args:
        name (str): Название метрики.

    Returns:
        Literal["minimize", "maximize"]: Направление в пользовательской
            семантике: "minimize" для ошибок, "maximize" для score-метрик.

    Raises:
        ValueError: Если метрика неизвестна ни реестру, ни sklearn.
    """
    lname = name.lower()
    # Ошибки инвертированы скорером, но пользователю возвращаются
    # естественными («меньше — лучше»): приоритет над greater_is_better,
    # который для neg_-метрик истинен (семантика оптимизатора). Для всех
    # остальных метрик пользовательская семантика — «больше лучше».
    if is_error_metric(lname):
        return "minimize"
    return "maximize"


def to_user_value(name: str, raw_value: float) -> float:
    """Привести «сырое» значение скорера к пользовательской семантике.

    Скореры ошибок (в т.ч. все neg_-метрики) возвращают инвертированное
    значение (-RMSE, -MAE, ...), чтобы оптимизатор мог максимизировать.
    Пользователю возвращается естественное значение метрики:
    RMSE → положительный RMSE, MAE → положительный MAE, R² → обычный R².

    Примечание (issue #32): нефинитные значения (например, -inf от NRMSE на
    константном таргете) инвертируются как обычно — это существующий контракт
    ``ModelTrainer.val_score``/``additional_scores``. Защита от протечки
    worst-score сентинела в отчёт обеспечивается финальным инвариантом
    winner-скора на уровне оркестратора (``component.is_valid_winner_score``),
    а не этой функцией.

    Args:
        name (str): Название метрики.
        raw_value (float): «Сырое» значение, возвращённое объектом-скорером.

    Returns:
        float: Значение метрики в пользовательском представлении.
    """
    value = float(raw_value)
    if name.lower() == "global_nrmse":
        # Динамический скорер NRMSE: ошибка, всегда возвращает -NRMSE.
        return -value
    scorer = _resolve_scorer(name)
    sign = getattr(scorer, "_sign", None)
    if sign is not None and sign < 0:
        return -value
    return value


def get_scorer_object(
    name: str, global_y: np.ndarray | None = None
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
                "For 'global_nrmse', 'global_y' must be passed to get_scorer_object"
            )
        return get_global_nrmse_scorer(global_y)

    # Если это наша кастомная метрика (nrmse, rmse и т.д.)
    if lname in _SCORER_OBJECTS:
        return _SCORER_OBJECTS[lname][0]

    # В остальных случаях возвращаем имя как есть
    # (sklearn сам найдет встроенную метрику)
    scorer = sklearn_get_scorer(lname)
    return cast("Callable[..., Any] | str", scorer)


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

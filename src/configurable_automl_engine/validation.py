"""
Validation Engine: Единая фабрика разбиений для обучения и оптимизации (Optuna).
Данный модуль инкапсулирует логику разделения данных на обучающие 
и проверочные выборки. 
Ключевой особенностью является интеллектуальное управление стратегиями: система 
автоматически оценивает достаточность данных и может понижать сложность валидации 
(например, откат с K-Fold на Train-Test Split), чтобы предотвратить статистическую 
недостоверность на малых выборках.
Поддерживаемые стратегии (ValidationStrategy):
    • train_test_split : Классическое разделение (по умолчанию 80/20).
    • k_fold           : Перекрестная проверка (K-Fold CV) с фиксированным сидом.
    • loo              : Leave-One-Out (валидация на каждом объекте), для малых выборок.
Особенности реализации:
    1. Robustness: Автоматический fallback-механизм в `make_cv` предотвращает падение 
       процесса обучения при малом количестве наблюдений (N < 2*Folds).
    2. Consistency: Все генераторы используют единый RANDOM_STATE для обеспечения 
       воспроизводимости экспериментов и честного сравнения моделей.
    3. Type Safety: Полная поддержка Enum ValidationStrategy и строгих Type Hints.
    4. Integration: Полная совместимость с интерфейсами sklearn.model_selection.
"""

from __future__ import annotations

from typing import Generator, Tuple, Union

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

from configurable_automl_engine.common.definitions import ValidationStrategy

from configurable_automl_engine.common.validation_utils import (
    validate_df_not_empty
)

import logging as logging
log = logging.getLogger(__name__)

RANDOM_STATE = 42  # фиксируем сид в одном месте

# ═════════════════════════════════════ exceptions ════════════════════════════

class ValidationError(Exception):
    """Базовая ошибка модуля."""

class InvalidDataError(ValidationError):
    """Некорректный X / y."""


# ═══════════════════════════════ helper CV factory ═══════════════════════════
def norm_val_method(val_method: ValidationStrategy | str) -> str:
    """
    Приводит входной метод к единому строковому идентификатору.
    Это центральная точка для маппинга Enum -> String.
    """
    # Если передан объект Enum, сразу возвращаем его строковое значение
    if isinstance(val_method, ValidationStrategy):
        return val_method.value  
    # Если передана строка, пытаемся привести её к нижнему регистру
    return str(val_method).lower()

def make_cv(  # noqa: C901  (читаемо и так)
    n_samples: int,
    *,
    val_method: ValidationStrategy | str,
    n_folds: int,
    random_state: int | None,
    test_size: float,
) -> tuple[str, model_selection.BaseCrossValidator | None]:
    """
    Фабрика объектов валидации scikit-learn с механизмом адаптации под объем данных.
    Args:
        n_samples: Общее количество объектов в выборке.
        val_method: Желаемая стратегия 
        (Enum или строка: 'k_fold', 'loo', 'train_test_split').
        n_folds: Количество фолдов (используется для 'k_fold').
        random_state: Инициализатор генератора случайных чисел для воспроизводимости.
    Returns:
        tuple: Кортеж, содержащий:
            - final_method (str): Реально выбранный метод 
            (может отличаться от запрошенного при fallback).
            - cv_object (BaseCrossValidator | None): Объект валидатора sklearn 
            или None, если выбран 'train_test_split'.
    Note:
        Если n_samples < max(4, 2 * n_folds), стратегия 'k_fold' будет автоматически 
        заменена на 'train_test_split' для сохранения статистической значимости.
    """
    method = norm_val_method(val_method)

    if method == "train_test_split":
        return "train_test_split", None

    if method == "k_fold":
        # Логика защиты: K-Fold требует минимум 2*k образцов для репрезентативности.
        # Число 4 — абсолютный минимум для корректного расчета дисперсии.
        min_required = max(4, 2 * n_folds)
        if n_samples < min_required:
            log.warning(
                "Insufficient samples (%d) for %d-fold CV (min required: %d). "
                "Falling back to 'train_test_split' with test_size=%.2f.",
                n_samples, n_folds, min_required, test_size
            )
            return "train_test_split", None
        return "k_fold", KFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    if method == "loo":
        if n_samples < 2:
            raise InvalidDataError(
                "Leave-One-Out validation requires at least 2 samples.")
        return "loo", LeaveOneOut()

    raise ValueError(
        "Unknown validation method. Must be 'train_test_split', 'k_fold' or 'loo'"
    )


def iter_splits(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series] = None,
    *,
    method: ValidationStrategy | str  = "k_fold",
    n_folds: int = 5,
    test_size: float = 0.2, 
    random_state: int | None = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Генерирует итеративные разбиения данных на обучающую и валидационную выборки.
    Функция инкапсулирует логику кросс-валидации и простого разделения, возвращая 
    готовые подмножества данных для каждой итерации.
    Args:
        X: Признаки (массив numpy или DataFrame pandas).
        y: Целевая переменная. Если None, возвращает None для y_train и y_val.
        method: Стратегия разделения. 
        Поддерживаются 'k_fold', 'loo', 'train_test_split'.
        n_folds: Количество блоков для кросс-валидации.
        test_size: Доля валидационной выборки 
        (используется только при методе 'train_test_split').
        random_state: Зерно для фиксации случайности при перемешивании.
    Yields:
        Tuple: Кортеж из четырех элементов:
            - X_train: Данные для обучения.
            - X_val: Данные для проверки.
            - y_train: Метки для обучения (или None).
            - y_val: Метки для проверки (или None).
    Raises:
        InvalidDataError: Если входные данные пусты или размеры X и y не совпадают.
        ValidationError: При критической ошибке инициализации объекта валидации.
    """
    method_str = norm_val_method(method)
    
    # 1. Валидация входных данных
    if isinstance(X, pd.DataFrame):
        validate_df_not_empty(X)
    elif len(X) == 0:
        raise InvalidDataError("Input array X is empty.")
    if y is not None and len(X) != len(y):
        raise InvalidDataError(f"X and y length mismatch: {len(X)} != {len(y)}")
    # 2. Получение финальной стратегии (с учетом возможного отката/fallback)
    final_method, cv = make_cv(
        len(X), 
        val_method=method_str, 
        n_folds=n_folds, 
        random_state=random_state,
        test_size=test_size
    )
    # 3. Генерация разбиений
    if final_method == "train_test_split":
        # y может быть None, sklearn.train_test_split это корректно обрабатывает
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=random_state,
        )
        yield X_tr, X_te, y_tr, y_te
    elif final_method in ("k_fold", "loo"):
        if cv is None:
            raise ValidationError(
                f"Critical error: CV object is not"
                f" initialized for strategy '{final_method}'"
            )
        # Объединяем логику для всех BaseCrossValidator объектов (KFold, LeaveOneOut)
        for train_idx, test_idx in cv.split(X):
            # Безопасная индексация X (Pandas vs Numpy)
            if hasattr(X, 'iloc'):
                x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
            else:
                x_tr, x_te = X[train_idx], X[test_idx]
            
            # Безопасная индексация y (с учетом того, что он может быть None)
            y_tr, y_te = None, None
            if y is not None:
                if hasattr(y, 'iloc'):
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    y_tr, y_te = y[train_idx], y[test_idx]
            
            yield x_tr, x_te, y_tr, y_te

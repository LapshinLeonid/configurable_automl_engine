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
    • auto             : Автоматический выбор между LOO/k_fold/train_test_split
                         на основе размера выборки (N) и числа признаков (P).
Особенности реализации:
    1. Robustness: Автоматический fallback-механизм в `make_cv` предотвращает падение
       процесса обучения при малом количестве наблюдений (N < 2*Folds).
    2. Consistency: Все генераторы используют единый RANDOM_STATE для обеспечения
       воспроизводимости экспериментов и честного сравнения моделей.
    3. Type Safety: Полная поддержка Enum ValidationStrategy и строгих Type Hints.
    4. Integration: Полная совместимость с интерфейсами sklearn.model_selection.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.common.validation_utils import (
    choose_validation_method,
    resolve_auto_no_features_fallback,
    validate_df_not_empty,
)

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


def make_cv(
    n_samples: int,
    *,
    val_method: ValidationStrategy | str,
    n_folds: int,
    random_state: int | None,
    test_size: float,
    n_features: int | None = None,
) -> tuple[str, model_selection.BaseCrossValidator | None, dict[str, Any] | None]:
    """
    Фабрика объектов валидации scikit-learn с механизмом адаптации под объем данных.
    Args:
        n_samples: Общее количество объектов в выборке.
        val_method: Желаемая стратегия
        (Enum или строка: 'k_fold', 'loo', 'train_test_split', 'auto').
        n_folds: Количество фолдов (используется для 'k_fold').
        random_state: Инициализатор генератора случайных чисел для воспроизводимости.
        n_features: Количество признаков P. Требуется для стратегии 'auto'.
    Returns:
        tuple: Кортеж, содержащий:
            - final_method (str): Реально выбранный метод
            (может отличаться от запрошенного при fallback).
            - cv_object (BaseCrossValidator | None): Объект валидатора sklearn
            или None, если выбран 'train_test_split'.
            - decision (dict[str, Any] | None): Разрешённое решение стратегии
            'auto' (от choose_validation_method) для синхронизации test_size
            между точками принятия решения. None для не-auto стратегий.
    Note:
        Если n_samples < max(4, 2 * n_folds), стратегия 'k_fold' будет автоматически
        заменена на 'train_test_split' для сохранения статистической значимости.
    """
    method = norm_val_method(val_method)

    if method == "auto":
        return _resolve_auto_cv(
            n_samples=n_samples,
            n_features=n_features,
            n_folds=n_folds,
            random_state=random_state,
            test_size=test_size,
        )

    if method == "train_test_split":
        return "train_test_split", None, None

    if method == "k_fold":
        # Логика защиты: K-Fold требует минимум 2*k образцов для репрезентативности.
        # Число 4 — абсолютный минимум для корректного расчета дисперсии.
        min_required = max(4, 2 * n_folds)
        if n_samples < min_required:
            log.warning(
                "Insufficient samples (%d) for %d-fold CV (min required: %d). "
                "Falling back to 'train_test_split' with test_size=%.2f.",
                n_samples,
                n_folds,
                min_required,
                test_size,
            )
            return "train_test_split", None, None
        return (
            "k_fold",
            KFold(n_splits=n_folds, shuffle=True, random_state=random_state),
            None,
        )

    if method == "loo":
        if n_samples < 2:
            raise InvalidDataError(
                "Leave-One-Out validation requires at least 2 samples."
            )
        return "loo", LeaveOneOut(), None

    raise ValueError(
        "Unknown validation method. Must be 'train_test_split', 'k_fold', 'loo' or 'auto'"
    )


def _resolve_auto_cv(
    *,
    n_samples: int,
    n_features: int | None,
    n_folds: int,
    random_state: int | None,
    test_size: float,
) -> tuple[str, model_selection.BaseCrossValidator | None, dict[str, Any] | None]:
    """Resolve the 'auto' validation strategy into a concrete sklearn CV object.

    Uses choose_validation_method(n_samples, n_features) to pick between LOO,
    k-fold and train-test split. When P is unavailable/invalid, safely falls
    back to the default k-fold behaviour with a warning. The resolved decision
    (when available) is returned so callers can reuse the computed test size.
    """
    if n_features is None or n_features <= 0:
        log.warning(
            "Auto validation requires n_features (P); got %r. "
            "Falling back to k_fold with n_folds=%d.",
            n_features,
            n_folds,
        )
        method, k = resolve_auto_no_features_fallback(n_samples, n_folds)
        if method == "train_test_split":
            return "train_test_split", None, None
        return (
            "k_fold",
            KFold(n_splits=k, shuffle=True, random_state=random_state),
            None,
        )

    decision = choose_validation_method(n_samples, n_features)
    method = decision["method"]

    # Предупреждения (low confidence) единообразно логируются для всех веток.
    if "warning" in decision:
        log.warning("Auto validation: %s", decision["warning"])

    if method == "LOO":
        return "loo", LeaveOneOut(), decision

    if method == "kfold":
        k = max(2, int(decision["k"]))
        return (
            "k_fold",
            KFold(n_splits=k, shuffle=True, random_state=random_state),
            decision,
        )

    if method == "train_test_split":
        return "train_test_split", None, decision

    # method == "invalid" (N < 2)
    raise InvalidDataError(decision.get("reason", "Insufficient data for validation."))


def iter_splits(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series | None = None,
    *,
    method: ValidationStrategy | str = "k_fold",
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Генерирует итеративные разбиения данных на обучающую и валидационную выборки.
    Функция инкапсулирует логику кросс-валидации и простого разделения, возвращая
    готовые подмножества данных для каждой итерации.
    Args:
        X: Признаки (массив numpy или DataFrame pandas).
        y: Целевая переменная. Если None, возвращает None для y_train и y_val.
        method: Стратегия разделения.
        Поддерживаются 'k_fold', 'loo', 'train_test_split', 'auto'.
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
    n_features = X.shape[1] if method_str == "auto" else None
    final_method, cv, decision = make_cv(
        len(X),
        val_method=method_str,
        n_folds=n_folds,
        random_state=random_state,
        test_size=test_size,
        n_features=n_features,
    )
    # 3. Генерация разбиений
    if final_method == "train_test_split":
        # Для 'auto' используем динамический размер теста, вычисленный
        # choose_validation_method (целое число строк), а не фиксированный.
        effective_test_size: float | int = test_size
        if method_str == "auto" and decision is not None:
            effective_test_size = int(decision["test_size"])
        # y может быть None, sklearn.train_test_split это корректно обрабатывает
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=effective_test_size,
            shuffle=True,
            random_state=random_state,
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
            if hasattr(X, "iloc"):
                x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
            else:
                x_tr, x_te = X[train_idx], X[test_idx]

            # Безопасная индексация y (с учетом того, что он может быть None)
            y_tr, y_te = None, None
            if y is not None:
                if hasattr(y, "iloc"):
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    y_tr, y_te = y[train_idx], y[test_idx]

            yield x_tr, x_te, y_tr, y_te

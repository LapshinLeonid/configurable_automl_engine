"""
Единая фабрика разбиений для Optuna-objective.

Поддерживает:
    • train_test_split  – 80 / 20, shuffle, фикс. сид.
    • k_fold            – KFold, shuffle, фикс. сид.
    • loo               – Leave-One-Out
"""

from __future__ import annotations

from pathlib import Path

from typing import Generator, Tuple

import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split


from configurable_automl_engine.common.definitions import ValidationStrategy

import importlib
import logging as _logging
import sys
from logging.handlers import RotatingFileHandler

RANDOM_STATE = 42  # фиксируем сид в одном месте

# ═════════════════════════════════════ exceptions ════════════════════════════

class ValidationError(Exception):
    """Базовая ошибка модуля."""

class InvalidDataError(ValidationError):
    """Некорректный X / y."""


# ═════════════════════════════ pseudo-safe logging init ══════════════════════
if not hasattr(_logging, "handlers"):  # edge-case в Jupyter
    sys.modules.pop("logging", None)
    _logging = importlib.import_module("logging")
logging = _logging  # alias

# ═══════════════════════════════════ logging setup ═══════════════════════════
_ROOT = Path(__file__).resolve().parents[2]
_LOG_FILE = _ROOT / "logs" / "validation.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_handler = RotatingFileHandler(
    _LOG_FILE, maxBytes=1_000_000, backupCount=10, encoding="utf-8", delay=True
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

log = logging.getLogger("validation")
log.setLevel(logging.INFO)
log.propagate = False
if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
    log.addHandler(_handler)

# ═══════════════════════════════ helper CV factory ═══════════════════════════
def norm_val_method(val_method: ValidationStrategy | str) -> str:
    """
    Нормализует метод валидации в строковый формат.
    Приоритет отдается объектам ValidationStrategy для обеспечения типобезопасности.
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
) -> tuple[str, model_selection.BaseCrossValidator | None]:
    """
    Возвращает (фактический_метод, cv-объект|None).

    • k-fold → train_test_split, если наблюдений < 2 × k или < 4.
    • train_test_split → cv=None.
    • loo → LeaveOneOut().
    """
    method = norm_val_method(val_method)

    if method == "train_test_split":
        return "train_test_split", None

    if method == "k_fold":
        min_required = max(4, 2 * n_folds)
        if n_samples < min_required:
            log.warning(
                "Мало наблюдений (%d) для %d-fold CV — переключаюсь "
                "на train_test_split",
                n_samples,
                n_folds,
            )
            return "train_test_split", None
        return "k_fold", KFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    if method == "loo":
        if n_samples < 2:
            raise InvalidDataError("Для leave-one-out нужно ≥ 2 наблюдений")
        return "loo", LeaveOneOut()

    raise ValueError(
        "val_method должен быть 'train_test_split', 'k_fold' или 'loo'"
    )


def iter_splits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "k_fold",
    n_folds: int = 5,
    random_state: int | None = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Генерирует пары (X_train, X_test, y_train, y_test) 
    согласно выбранному способу валидации.

    Parameters
    ----------
    X, y : массивы одинаковой длины
    method : {"train_test_split", "k_fold", "loo"}
    n_folds : int, >1 — используется **только** при method="k_fold"
    """
    method = method.lower()
    if method not in {"train_test_split", "k_fold", "loo"}:
        raise ValueError(f"Unknown validation method: {method}")

    if method == "train_test_split":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True,
            random_state=random_state,
            # stratify=y  # не требовалось ТЗ
        )
        yield X_tr, X_te, y_tr, y_te

    elif method == "k_fold":
        if n_folds < 2:
            raise ValueError("n_folds must be ≥2 for k_fold validation.")
        kf = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )
        for train_idx, test_idx in kf.split(X):
            yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    else:  # "loo"
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X):
            yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]

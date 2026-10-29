"""
Validation Engine: A unified split factory for training and optimization (Optuna).
This module encapsulates the logic of splitting data into training
and validation sets.
Its key feature is intelligent strategy management: the system
automatically assesses data sufficiency and can lower the validation complexity
(e.g. fall back from K-Fold to Train-Test Split) to prevent statistical
unreliability on small samples.
Supported strategies (ValidationStrategy):
    • train_test_split : Classic split (80/20 by default).
    • k_fold           : Cross-validation (K-Fold CV) with a fixed seed.
    • loo              : Leave-One-Out (validation on each sample), for small samples.
    • auto             : Automatic selection between LOO/k_fold/train_test_split
                         based on the sample size (N) and the number of features (P).
Implementation features:
    1. Robustness: The automatic fallback mechanism in `make_cv` prevents the training
       process from crashing with a small number of observations (N < 2*Folds).
    2. Consistency: All generators use a single RANDOM_STATE to ensure
       experiment reproducibility and fair model comparison.
    3. Type Safety: Full support for the ValidationStrategy Enum and strict Type Hints.
    4. Integration: Full compatibility with the sklearn.model_selection interfaces.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any, Literal, overload

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

RANDOM_STATE = 42  # fix the seed in a single place

#: Validation methods supported by ``make_cv``/``iter_splits`` in the current
#: version. A single point of extension for the validation strategy interface
#: (edge case "time series"): for a new strategy (e.g. an
#: order-aware ``TimeSeriesSplit``) it is enough to add the method to this
#: tuple and a branch to ``make_cv`` — ``ModelTrainer`` and the OOF channel will
#: pick it up automatically, without rewriting.
SUPPORTED_VALIDATION_METHODS: tuple[str, ...] = (
    "train_test_split",
    "k_fold",
    "loo",
    "auto",
)

# ═════════════════════════════════════ exceptions ════════════════════════════


class ValidationError(Exception):
    """A base error of the module."""


class InvalidDataError(ValidationError):
    """Invalid X / y."""


# ═══════════════════════════════ helper CV factory ═══════════════════════════
def norm_val_method(val_method: ValidationStrategy | str) -> str:
    """
    Brings the input method to a single string identifier.
    This is the central point for the Enum -> String mapping.
    """
    # If an Enum object is passed, immediately return its string value
    if isinstance(val_method, ValidationStrategy):
        return val_method.value
    # If a string is passed, try to lower-case it
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
    A factory of scikit-learn validation objects with an adaptation mechanism for data size.
    Args:
        n_samples: The total number of samples.
        val_method: The desired strategy
        (Enum or string: 'k_fold', 'loo', 'train_test_split', 'auto').
        n_folds: The number of folds (used for 'k_fold').
        random_state: The random number generator initializer for reproducibility.
        n_features: The number of features P. Required for the 'auto' strategy.
    Returns:
        tuple: A tuple containing:
            - final_method (str): The actually selected method
            (may differ from the requested one on fallback).
            - cv_object (BaseCrossValidator | None): The sklearn validator
            object, or None if 'train_test_split' is selected.
            - decision (dict[str, Any] | None): The resolved 'auto' strategy
            decision (from choose_validation_method) for synchronizing test_size
            between decision points. None for non-auto strategies.
    Note:
        If n_samples < max(4, 2 * n_folds), the 'k_fold' strategy is automatically
        replaced by 'train_test_split' to preserve statistical significance.
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
        # Protection logic: K-Fold requires a minimum of 2*k samples for representativeness.
        # The number 4 is the absolute minimum for a correct variance computation.
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
        "Unknown validation method. Must be one of "
        f"{', '.join(SUPPORTED_VALIDATION_METHODS)}"
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

    # Warnings (low confidence) are logged uniformly for all branches.
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


# The ``iter_splits`` result: a tuple of four subsets (X_train, X_val,
# y_train, y_val) or of five when ``include_indices=True`` — the fifth element
# is an ``np.ndarray`` of positional indices of the validation rows in the
# original data (assembly of OOF predictions in ``ModelTrainer``).
# Precise typing via ``@overload`` with ``Literal``: calls with
# ``include_indices=False`` (default) get a 4-element tuple,
# with ``include_indices=True`` — a 5-element one.
Split = tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
SplitWithIndices = tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray
]


@overload
def iter_splits(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series | None = None,
    *,
    method: ValidationStrategy | str = "k_fold",
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int | None = 42,
    include_indices: Literal[False] = False,
) -> Generator[Split, None, None]: ...


@overload
def iter_splits(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series | None = None,
    *,
    method: ValidationStrategy | str = "k_fold",
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int | None = 42,
    include_indices: Literal[True],
) -> Generator[SplitWithIndices, None, None]: ...


def iter_splits(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series | None = None,
    *,
    method: ValidationStrategy | str = "k_fold",
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int | None = 42,
    include_indices: bool = False,
) -> Generator[Split | SplitWithIndices, None, None]:
    """
    Generates iterative splits of the data into training and validation sets.
    The function encapsulates the cross-validation and simple split logic, returning
    ready subsets of data for each iteration.
    Args:
        X: Features (a numpy array or a pandas DataFrame).
        y: The target variable. If None, returns None for y_train and y_val.
        method: The split strategy.
        Supported: 'k_fold', 'loo', 'train_test_split', 'auto'.
        n_folds: The number of folds for cross-validation.
        test_size: The fraction of the validation set
        (used only with the 'train_test_split' method).
        random_state: The seed for fixing randomness during shuffling.
        include_indices: If True, each tuple is extended with a fifth element —
            an ``np.ndarray`` of positional indices of the validation rows in the
            original ``X`` (the order corresponds to the order of ``X_val``). Used for
            assembling OOF predictions aligned with the original row order.
            Defaults to False — the behavior is fully compatible
            with the old 4-element contract.
    Yields:
        Tuple: A tuple of four (or five when ``include_indices=True``)
            elements:
            - X_train: Data for training.
            - X_val: Data for validation.
            - y_train: Labels for training (or None).
            - y_val: Labels for validation (or None).
            - test_idx (only with ``include_indices=True``): The positional
              indices of the ``X_val`` rows in the original ``X``.
    Raises:
        InvalidDataError: If the input data is empty or the sizes of X and y do not match.
        ValidationError: On a critical error during validation object initialization.
    """
    method_str = norm_val_method(method)

    # 1. Input data validation
    if isinstance(X, pd.DataFrame):
        validate_df_not_empty(X)
    elif len(X) == 0:
        raise InvalidDataError("Input array X is empty.")
    if y is not None and len(X) != len(y):
        raise InvalidDataError(f"X and y length mismatch: {len(X)} != {len(y)}")
    # 2. Obtain the final strategy (taking into account a possible fallback)
    n_features = X.shape[1] if method_str == "auto" else None
    final_method, cv, decision = make_cv(
        len(X),
        val_method=method_str,
        n_folds=n_folds,
        random_state=random_state,
        test_size=test_size,
        n_features=n_features,
    )
    # 3. Generate the splits: one generator of (train_idx, test_idx) pairs per
    # strategy, one loop that yields the subsets.
    if final_method == "train_test_split":
        pairs = _holdout_pairs(
            len(X),
            test_size=test_size,
            method_str=method_str,
            decision=decision,
            random_state=random_state,
        )
    elif final_method in ("k_fold", "loo"):
        if cv is None:
            raise ValidationError(
                f"Critical error: CV object is not"
                f" initialized for strategy '{final_method}'"
            )
        # Combine the logic for all BaseCrossValidator objects (KFold, LeaveOneOut)
        pairs = cv.split(X)
    else:
        return
    for train_idx, test_idx in pairs:
        x_tr, x_te = _take(X, train_idx), _take(X, test_idx)
        # y may be None: then y_train and y_val are None as well
        y_tr = None if y is None else _take(y, train_idx)
        y_te = None if y is None else _take(y, test_idx)
        if include_indices:
            yield x_tr, x_te, y_tr, y_te, np.asarray(test_idx)
        else:
            yield x_tr, x_te, y_tr, y_te


def _take(obj: Any, idx: np.ndarray) -> Any:
    """Select rows by position (pandas via ``iloc``, numpy by indexing).

    Args:
        obj: A pandas object or a numpy array.
        idx: Positional row indices.

    Returns:
        Any: The selected rows, of the same type as ``obj``.
    """
    if hasattr(obj, "iloc"):
        return obj.iloc[idx]
    return obj[idx]


def _holdout_pairs(
    n_samples: int,
    *,
    test_size: float,
    method_str: str,
    decision: dict[str, Any] | None,
    random_state: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the single (train_idx, test_idx) pair of the hold-out split.

    The POSITIONAL indices are split once, so that with ``include_indices=True``
    ``iter_splits`` returns the real positions of the validation rows in the
    original X, and the subsets themselves are obtained by indexing with the
    same positions. This guarantees the consistency of X_val/y_val/test_idx and
    correct behavior with ``random_state=None`` (two independent
    train_test_split calls would yield different permutations).

    Args:
        n_samples: The number of rows.
        test_size: The requested validation fraction.
        method_str: The normalized requested method.
        decision: The ``choose_validation_method`` decision ('auto' only).
        random_state: The shuffling seed.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: One (train_idx, test_idx) pair.
    """
    # For 'auto' use the dynamic test size computed by
    # choose_validation_method (an integer number of rows), not a fixed one.
    effective_test_size: float | int = test_size
    if method_str == "auto" and decision is not None:
        effective_test_size = int(decision["test_size"])
    train_idx, test_idx = train_test_split(
        np.arange(n_samples),
        test_size=effective_test_size,
        shuffle=True,
        random_state=random_state,
    )
    return [(train_idx, test_idx)]

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
    5. Multi-split (repeated) validation: ``make_cv``/``iter_splits`` accept
       ``n_repeats`` (the ``general.n_repeats`` config field). With
       ``n_repeats = R > 1`` the base strategy is re-drawn R times with the
       deterministic seeds ``random_state + r`` (r = 0..R-1), so two runs with
       the same ``random_state`` produce identical sets of repeated splits.
       Every yielded split of a repeated run is a ``RepeatedSplit`` — the same
       4/5-element tuple as before, tagged with the ``repeat_index`` attribute.
       With ``n_repeats = 1`` the behavior is byte-in-byte the classic
       single-split one (plain tuples, no tags). The 'loo' strategy has
       deterministic folds, so repeating it only multiplies the cost: repeats
       are ignored with a WARNING. The per-split estimates are combined
       downstream by ``aggregate_split_scores`` (the ``general.split_aggregation``
       config field).
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

#: Aggregation policies of the per-repeat (per-split) scores
#: (``general.split_aggregation``): 'median' (default) — robust to a
#: "lucky" split on a micro-sample; 'mean' — the classic fold-average.
SUPPORTED_SPLIT_AGGREGATIONS: tuple[str, ...] = ("mean", "median")

#: The repeat index attribute name of a repeated split tuple
#: (see ``with_repeat_index``).
REPEAT_INDEX_ATTR = "repeat_index"

#: The sklearn seed range: random_state must be in [0, 2**32 - 1]. The
#: repeated seeds ``random_state + r`` are wrapped into this range so a
#: valid base seed with ``r > 0`` cannot leave it and crash the splits.
SKLEARN_SEED_MODULUS = 2**32


class RepeatedSplit(tuple):
    """A split tuple tagged with its repeat index (attribute-only tag).

    A lightweight ``tuple`` subclass: the elements and the unpacking are
    identical to the classic split tuples, the only addition is the
    ``repeat_index`` attribute (set by ``iter_splits`` when ``n_repeats > 1``,
    read via ``repeat_index_of``). The attribute lives in the instance
    ``__dict__`` (empty ``__slots__``: a ``tuple`` subtype cannot declare
    nonempty slots).
    """

    repeat_index: int = 0


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


def resolved_n_repeats(n_repeats: int | None, method: ValidationStrategy | str) -> int:
    """Resolve the effective number of repeats for a validation strategy.

    The 'loo' strategy has deterministic folds (Leave-One-Out draws nothing):
    repeating it only multiplies the computation by an identical factor, so
    ``n_repeats > 1`` is ignored for it with a WARNING via
    ``logging.getLogger(__name__)`` — the repeats are NOT applied.

    Args:
        n_repeats: The requested number of repeats (``None`` → 1). Must be a
            positive integer; non-integer values (e.g. ``2.7``) are rejected
            instead of being silently coerced by ``int()``.
        method: The validation strategy (Enum or string).

    Returns:
        The effective number of repeats, always >= 1.

    Raises:
        ValueError: If ``n_repeats`` is not a positive integer.
    """
    if n_repeats is None:
        return 1
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int):
        # A wrong type is reported as a config-validation ValueError (the
        # same policy as ``trainer_params._validate_validation_strategy``),
        # not a TypeError: the value reaches here from user configuration.
        raise ValueError(f"n_repeats must be a positive integer, got {n_repeats!r}")  # noqa: TRY004
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats!r}")
    if n_repeats > 1 and norm_val_method(method) == "loo":
        log.warning(
            "validation strategy 'loo' has deterministic folds: "
            "n_repeats=%d is meaningless (repeating it would only multiply "
            "the computation by identical splits) — the repeats are IGNORED "
            "and the effective n_repeats is 1.",
            n_repeats,
        )
        return 1
    return n_repeats


def repeated_seed(random_state: int | None, repeat_index: int) -> int | None:
    """The seed of repeat ``repeat_index``: ``random_state + repeat_index``.

    Deterministic repeat seeds are the core invariant of the multi-split
    validation: two runs with the same ``random_state`` produce identical
    sets of repeated splits. ``random_state=None`` stays None (a
    non-deterministic run is not reproducible by design). The result is
    wrapped into the sklearn seed range ``[0, 2**32 - 1]``: the base seed
    ``general.random_state`` is valid on its own (sklearn accepts
    ``2**32 - 1``), but ``random_state + r`` with ``r > 0`` can leave the
    range and crash the split construction — the wrap keeps every repeat
    seed valid and deterministic (a wrap of ``2**32`` is a no-op for the
    in-range seeds of ordinary configs).

    Args:
        random_state: The base seed (``general.random_state``).
        repeat_index: The zero-based repeat index r (0..R-1).

    Returns:
        ``(random_state + repeat_index) mod 2**32``, or ``None``.
    """
    if random_state is None:
        return None
    return (int(random_state) + int(repeat_index)) % SKLEARN_SEED_MODULUS


def with_repeat_index(
    split: Split | SplitWithIndices, repeat_index: int
) -> Split | SplitWithIndices:
    """Tag a split tuple with its repeat index (returns the same length).

    The tag is an attribute on the yielded tuple, so the classic
    ``for X_tr, X_te, y_tr, y_te, ... in iter_splits(...)`` unpacking and the
    OOF channel (``include_indices=True``) keep working unchanged; a caller
    that needs the repeat reads it via ``repeat_index_of``. Built-in tuples
    reject attributes, so a lightweight ``tuple`` subclass of the same
    length and the same elements is returned.

    Args:
        split: The split tuple yielded by ``iter_splits``.
        repeat_index: The zero-based repeat index r.

    Returns:
        The same-length tuple carrying the ``repeat_index`` attribute.
    """
    tagged = RepeatedSplit(split)
    tagged.repeat_index = int(repeat_index)
    return tagged


def repeat_index_of(split: Split | SplitWithIndices) -> int:
    """The repeat index of a split yielded by ``iter_splits`` (0 for untagged).

    Args:
        split: A split tuple (tagged or plain).

    Returns:
        The zero-based repeat index carried by the ``repeat_index``
        attribute, 0 when the split is untagged (``n_repeats = 1``).
    """
    return int(getattr(split, REPEAT_INDEX_ATTR, 0))


def aggregate_split_scores(
    scores: list[float] | tuple[float, ...],
    aggregation: str = "median",
) -> float:
    """Combine the per-repeat (per-split) scores into a single estimate.

    Pure function of the scores and the policy: 'median' (default) is
    robust to a "lucky" split on a micro-sample, 'mean' is the classic
    average. Non-finite scores (NaN from failed folds, ±inf from degenerate
    metric values) are excluded before the aggregation with a WARNING (the
    count and the policy are logged once per call); a non-finite repeat
    cannot make the combined estimate fail — only reduce the evidence.

    Args:
        scores: The raw per-repeat scores.
        aggregation: 'mean' or 'median' (an unknown value is a ValueError).

    Returns:
        The aggregated score, or NaN when no finite score is available.
    """
    if aggregation not in SUPPORTED_SPLIT_AGGREGATIONS:
        raise ValueError(
            "Unknown split aggregation "
            f"{aggregation!r}; expected one of {SUPPORTED_SPLIT_AGGREGATIONS}"
        )
    values = [float(s) for s in scores]
    finite = [s for s in values if np.isfinite(s)]
    if len(finite) != len(values):
        log.warning(
            "split_aggregation=%s: %d of %d per-repeat scores are "
            "non-finite (NaN/±inf) and are EXCLUDED from the aggregation.",
            aggregation,
            len(values) - len(finite),
            len(values),
        )
    if not finite:
        return float("nan")
    if aggregation == "mean":
        return float(np.mean(finite))
    return float(np.median(finite))


def make_cv(
    n_samples: int,
    *,
    val_method: ValidationStrategy | str,
    n_folds: int,
    random_state: int | None,
    test_size: float,
    n_features: int | None = None,
    n_repeats: int | None = None,
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
        n_repeats: The number of repeats of the base strategy
            (R = 1 — the classic single-split behavior). The repeats are
            materialized by ``iter_splits`` with the wrapped seeds
            ``repeated_seed(random_state, r)``; ``make_cv`` only validates
            the count (a non-positive or non-integer value is a
            ValueError). The 'loo' + repeats guard is applied by
            ``iter_splits`` against the RESOLVED method ('auto' can resolve
            to LOO).
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
    # Validate the repeat count at the single extension point (a non-positive
    # or non-integer value is rejected for every caller). The 'loo' +
    # repeats check is NOT applied here: with 'auto' the actual method is
    # only known after the resolution below — the caller (``iter_splits``)
    # re-resolves the repeats against the RESOLVED method, so auto → LOO
    # also gets the deterministic-folds guard.
    if n_repeats is not None:
        resolved_n_repeats(n_repeats, "k_fold")

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
    n_repeats: int | None = None,
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
    n_repeats: int | None = None,
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
    n_repeats: int | None = None,
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
            Defaults to False — only the 4-element tuples are yielded.
        n_repeats: The number of repeats of the base strategy with the
            deterministic seeds ``random_state + r`` (r = 0..R-1). 1 (default)
            — the classic single-split behavior (plain tuples, byte-in-byte
            the pre-multi-split output). R > 1 — the base strategy is
            re-drawn R times and every yielded tuple carries the
            ``repeat_index`` attribute (read via ``repeat_index_of``), so
            the existing unpacking-based consumers keep working unchanged.
            Repeats are resolved against the RESOLVED strategy: the 'loo'
            strategy (explicit or resolved from 'auto') has deterministic
            folds — any ``n_repeats > 1`` is ignored with a WARNING.
    Yields:
        Tuple: A tuple of four (or five when ``include_indices=True``)
            elements:
            - X_train: Data for training.
            - X_val: Data for validation.
            - y_train: Labels for training (or None).
            - y_val: Labels for validation (or None).
            - test_idx (only with ``include_indices=True``): The positional
              indices of the ``X_val`` rows in the original ``X``.
            With ``n_repeats > 1`` every tuple is tagged with the
            ``repeat_index`` attribute (the zero-based repeat r).
    Raises:
        InvalidDataError: If the input data is empty or the sizes of X and y do not match.
        ValidationError: On a critical error during validation object initialization.
        ValueError: If ``n_repeats`` is not a positive integer.
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
        n_repeats=n_repeats,
    )
    # 2a. Resolve the repeats against the RESOLVED method: 'auto' can land on
    # LOO (a micro-sample), and LOO folds are deterministic — repeating them
    # would only multiply the computation by identical splits. The guard is
    # applied here (the single point where the resolved method is known) so
    # it covers every strategy resolution path, explicit or 'auto'.
    repeats = resolved_n_repeats(n_repeats, final_method)
    # 2b. The n_samples sanity check (issue-211): a repeat cycle of the
    # k-fold strategy evaluates every row R * n_splits times — on a tiny
    # sample this is both a wasted budget and a statistics-free estimate.
    # The config rule cannot know n_samples, so the check lives at the
    # single split point (a WARNING: the splits are still produced).
    if repeats > 1 and final_method == "k_fold":
        n_splits_expected = int(cv.get_n_splits(X)) if cv is not None else n_folds
        if len(X) < repeats * n_splits_expected:
            log.warning(
                "n_samples=%d is smaller than n_folds * n_repeats (%d x %d = "
                "%d) for the repeated k-fold validation: every row would be "
                "validated more times than there are rows; the repeats do "
                "not add independent evidence on this sample size.",
                len(X),
                n_splits_expected,
                repeats,
                repeats * n_splits_expected,
            )
    # 3. Generate the splits: one generator of (train_idx, test_idx) pair
    # sets per repeat, one loop that yields the subsets. The repeat wrapper
    # is built into this single point of extension: with repeats = 1 the
    # seed is exactly ``random_state`` and the output is byte-in-byte the
    # classic single-split one (plain, untagged tuples).
    if final_method == "train_test_split":
        # Each repeat is its own pair set (a single hold-out split per
        # repeat, seed ``random_state + r``).
        pair_sets = _holdout_pairs(
            len(X),
            test_size=test_size,
            method_str=method_str,
            decision=decision,
            random_state=random_state,
            n_repeats=repeats,
        )
    elif final_method in ("k_fold", "loo"):
        if cv is None:
            raise ValidationError(
                f"Critical error: CV object is not"
                f" initialized for strategy '{final_method}'"
            )
        # Combine the logic for all BaseCrossValidator objects (KFold,
        # LeaveOneOut). Each repeat re-draws the folds with its deterministic
        # seed ``repeated_seed(random_state, r)`` (wrapped into the sklearn
        # seed range); the fold count (incl. the 'auto' fallback k) is taken
        # from the base cv object so all repeats share the resolved
        # structure. LOO has deterministic folds — one cv object (the
        # repeats were already resolved away in ``resolved_n_repeats``).
        n_splits = int(cv.get_n_splits(X))
        pair_sets = []
        for r in range(repeats):
            seed = repeated_seed(random_state, r)
            cv_r = (
                KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if final_method == "k_fold"
                else cv
            )
            pair_sets.append(cv_r.split(X))
    else:
        return
    for repeat_idx, repeat_pairs in enumerate(pair_sets):
        for train_idx, test_idx in repeat_pairs:
            x_tr, x_te = _take(X, train_idx), _take(X, test_idx)
            # y may be None: then y_train and y_val are None as well
            y_tr = None if y is None else _take(y, train_idx)
            y_te = None if y is None else _take(y, test_idx)
            if include_indices:
                split: Split | SplitWithIndices = (
                    x_tr,
                    x_te,
                    y_tr,
                    y_te,
                    np.asarray(test_idx),
                )
            else:
                split = (x_tr, x_te, y_tr, y_te)
            # The repeat tag: attribute-only (the tuple shape is unchanged),
            # so the unpacking consumers and the OOF channel keep their API.
            if repeats > 1:
                split = with_repeat_index(split, repeat_idx)
            yield split


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
    n_repeats: int = 1,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Return the pair sets of the hold-out split(s).

    Each element of the returned list is the pair set of one repeat: a list
    with the single ``(train_idx, test_idx)`` pair of that repeat.

    The POSITIONAL indices are split once per repeat, so that with
    ``include_indices=True`` ``iter_splits`` returns the real positions of
    the validation rows in the original X, and the subsets themselves are
    obtained by indexing with the same positions. This guarantees the
    consistency of X_val/y_val/test_idx and correct behavior with
    ``random_state=None`` (two independent train_test_split calls would
    yield different permutations).

    With ``n_repeats > 1`` every repeat r draws its own permutation with the
    deterministic wrapped seed ``repeated_seed(random_state, r)``; a set of
    identical repeats
    (possible when sklearn cannot vary the permutation on a tiny sample,
    e.g. a test part of 1 row) defeats the purpose of the multi-split
    validation and is reported with a WARNING — the splits are still
    returned (the caller decides whether the duplicated evidence matters).

    Args:
        n_samples: The number of rows.
        test_size: The requested validation fraction.
        method_str: The normalized requested method.
        decision: The ``choose_validation_method`` decision ('auto' only).
        random_state: The base shuffling seed (repeat r uses
            ``repeated_seed(random_state, r)``).
        n_repeats: The number of hold-out repeats (>= 1).
    """
    # For 'auto' use the dynamic test size computed by
    # choose_validation_method (an integer number of rows), not a fixed one.
    effective_test_size: float | int = test_size
    if method_str == "auto" and decision is not None:
        effective_test_size = int(decision["test_size"])
    if n_repeats <= 1:
        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=effective_test_size,
            shuffle=True,
            random_state=random_state,
        )
        # Uniform pair-set structure: one repeat = one set with one pair.
        return [[(train_idx, test_idx)]]
    pair_sets: list[list[tuple[np.ndarray, np.ndarray]]] = []
    seen: list[tuple[bytes, bytes]] = []
    for r in range(n_repeats):
        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=effective_test_size,
            shuffle=True,
            random_state=repeated_seed(random_state, r),
        )
        pair_sets.append([(train_idx, test_idx)])
        seen.append((train_idx.tobytes(), test_idx.tobytes()))
    if len(set(seen)) < n_repeats:
        log.warning(
            "Repeated hold-out validation produced identical splits for at "
            "least two of the %d repeats (n_samples=%d): the repeats do not "
            "add independent evidence on this sample size.",
            n_repeats,
            n_samples,
        )
    return pair_sets

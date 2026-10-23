"""
Regression Metric Ecosystem: a professional metric registry and adapter.
The module extends the standard `scikit-learn` set with custom
RMSE and NRMSE implementations, providing their seamless integration
into automatic hyperparameter tuning processes (GridSearchCV, Optuna)
and cross-validation.

Key features:
    1. Dual-Normalization NRMSE:
        Support for two normalization strategies — local
        (dynamic range within a fold)
        and global (fixed range of the whole dataset).
    2. Scorer API Compatibility: Automatic sign inversion of error metrics
       (lower -> better) into the scorer format (higher -> better)
        for correct optimization.
    3. Numeric Stability: Built-in division-by-zero protection
       when a constant target is encountered — returns `inf`
       instead of an error to keep the pipeline stable.
    4. Smart Discovery: A single entry point `get_scorer_object`
       with an alias mechanism simplifying the invocation of custom metrics
       by short names (e.g. "rmse").
"""

from __future__ import annotations

import logging
import threading
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
#  The metrics themselves
# --------------------------------------------------------------------------- #
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the root mean squared error (RMSE).
    Calculation logic:
    1. MSE is computed with the standard `mean_squared_error` function.
    2. The square root is taken from the result via `np.sqrt`.
    3. The result is forcibly cast to float for consistency.
    Args:
        y_true (np.ndarray): True values of the target variable.
        y_pred (np.ndarray): Model predictions.
    Returns:
        float: The RMSE value (the lower, the better).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def oof_rmse(y_true: Any, y_pred: Any) -> float:
    """Compute RMSE over the whole aligned OOF vector.

    Unlike averaging by folds, the metric is computed over the entire
    out-of-fold prediction vector at once: each row enters the calculation
    exactly once. Pairs where ``y_true`` or ``y_pred`` are not finite numbers
    (NaN/None/inf — uncovered rows of ``train_test_split`` or failing
    predictions of individual folds) are dropped so that the concatenation
    of folds of different lengths does not "drift" across indices.

    Args:
        y_true (Any): True values aligned with ``y_pred`` by position.
        y_pred (Any): OOF predictions aligned with ``y_true`` by position.

    Returns:
        float: RMSE over all valid pairs (lower is better).

    Raises:
        ValueError: If no valid pair remains after dropping non-finite pairs
            (an OOF estimate is impossible).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        raise ValueError("No valid (y_true, y_pred) pairs for OOF RMSE.")
    return _rmse(yt[mask], yp[mask])


# Metrics whose estimate cannot be obtained by averaging per-fold values:
# the pooled value over the concatenated OOF vector is required. RMSE, MSE,
# R² and nRMSE (including their sklearn aliases), RMSLE (sqrt of a mean —
# non-additive like RMSE), D² and explained_variance (fraction-of-variance
# scores, like R²), max error (max over the sample) and median absolute
# error (a sample median, not a per-sample mean) are non-additive; MAE (and
# similar per-sample means) remain additive — the fold mean equals the
# pooled value, so they are intentionally excluded.
_NON_ADDITIVE_METRICS = frozenset(
    {
        "rmse",
        "neg_root_mean_squared_error",
        "mse",
        "neg_mean_squared_error",
        "r2",
        "nrmse",
        "global_nrmse",
        # RMSLE — the square root of the mean squared LOG error:
        # non-additive in the same way as RMSE.
        "neg_root_mean_squared_log_error",
        # D² — a fraction-of-variance-explained score, like R².
        "d2_absolute_error_score",
        # explained_variance — a global fraction of variance explained
        # (1 - Var(res)/Var(y)), non-additive like R². Under LOO every fold
        # value is identically 1.0 (sklearn on a single pair), so the fold
        # mean would silently not depend on the model at all.
        "explained_variance",
        # max absolute error — the max over the sample, not a per-sample mean.
        "neg_max_error",
        # median absolute error — the median over the sample, not a
        # per-sample mean. Under LOO the fold mean of single-row medians
        # degenerates into MAE, silently substituting the metric.
        "neg_median_absolute_error",
    }
)


def is_non_additive_metric(name: str) -> bool:
    """Return whether a metric must be computed over the pooled OOF vector.

    For non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D²,
    explained_variance, max error, median absolute error) the value over the
    concatenated out-of-fold prediction vector differs from the average of
    the per-fold values (e.g. pooled RMSE² = mean of per-fold MSE, not the
    mean of per-fold RMSE; pooled R² is not the mean of fold R²; the nRMSE
    normalization range also changes; under LOO the fold R²/EV are undefined
    or constant, and the median degenerates into the mean). Such metrics are
    therefore computed over the whole vector at once. Additive metrics (MAE
    and similar per-sample means) stay on the fold-average path — their fold
    mean equals the pooled value.

    Args:
        name (str): Metric name (user or sklearn alias, case-insensitive;
            surrounding whitespace is ignored).

    Returns:
        bool: True for non-additive metrics, False for additive ones.
    """
    return name.lower().strip() in _NON_ADDITIVE_METRICS


def pooled_scorer_value(scorer: Any, y_true: Any, y_pred: Any) -> float:
    """Compute a scorer value over the whole aligned vector at once.

    Applies the raw metric function (with the scorer's sign and kwargs) over
    the concatenated vector, dropping non-finite pairs — the same masking
    contract as ``oof_rmse``. Used to estimate non-additive metrics (RMSE,
    MSE, R², nRMSE) over the pooled OOF vector instead of averaging per-fold
    values.

    Args:
        scorer (Any): A scikit-learn ``make_scorer`` object exposing
            ``_score_func``, ``_sign`` and ``_kwargs``.
        y_true (Any): True values aligned with ``y_pred`` by position.
        y_pred (Any): Predictions aligned with ``y_true`` by position.

    Returns:
        float: The raw (sign-applied) metric value over all valid pairs.

    Raises:
        ValueError: If no valid pair remains after dropping non-finite pairs,
            or if the scorer does not expose a raw ``_score_func`` (it cannot
            be pooled).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        raise ValueError("No valid (y_true, y_pred) pairs for the pooled metric.")
    score_func = getattr(scorer, "_score_func", None)
    if score_func is None:
        raise ValueError(
            "The scorer cannot be pooled: no raw metric function is exposed."
        )
    sign = getattr(scorer, "_sign", 1)
    kwargs = getattr(scorer, "_kwargs", None) or {}
    return float(sign * score_func(yt[mask], yp[mask], **kwargs))


# Fallback-warning deduplication: when a scorer cannot be pooled (no raw
# ``_score_func`` exposed, or the pooled value is impossible), the engine
# falls back to the per-fold mean. The WARNING is emitted exactly once per
# (algorithm, phase) pair — added to the set on first emission — so the
# metric degradation leaves a trace in the log without spamming one message
# per trial/fold. The set is guarded by a lock: the fallback
# paths run concurrently in the "processes"/threads parallel mode.
_pooled_fallback_lock = threading.Lock()
_pooled_fallback_warned: set[tuple[str, str]] = set()


def warn_pooled_fallback_once(*, algo_name: str, phase_name: str, reason: str) -> None:
    """Log a WARNING once per (algorithm, phase) about the pooled fallback.

    When the pooled evaluation of a non-additive metric is impossible (the
    scorer does not expose the private ``_score_func``/``_sign``/``_kwargs``
    attributes, or no valid OOF pairs remain), the engine silently returns
    the per-fold mean instead of the pooled value — the reported metric
    changes without a trace in the log. This helper emits a single WARNING
    for the first occurrence per (``algo_name``, ``phase_name``) pair and
    stays silent for the subsequent occurrences of the same pair.

    Args:
        algo_name (str): The algorithm being evaluated (e.g. 'ridge').
        phase_name (str): The pipeline phase (e.g. an HPO phase name, or
            'final_fit' for the final ``ModelTrainer`` fit).
        reason (str): The human-readable cause of the fallback.

    Returns:
        None
    """
    key = (algo_name, phase_name)
    with _pooled_fallback_lock:
        if key in _pooled_fallback_warned:
            return
        _pooled_fallback_warned.add(key)
    logger.warning(
        "Pooled evaluation of the metric is not available for algorithm "
        "'%s' in phase '%s': %s. Falling back to the per-fold mean — the "
        "reported value may differ from the pooled estimate.",
        algo_name,
        phase_name,
        reason,
    )


def reset_pooled_fallback_warnings() -> None:
    """Clear the warn-once deduplication state (test helper).

    ``warn_pooled_fallback_once`` logs at most once per (algorithm, phase)
    pair for the whole process lifetime. Tests exercise the fallback with
    different pairs, so they call this helper (e.g. in an autouse fixture)
    to reset the recorded pairs between tests.
    """
    with _pooled_fallback_lock:
        _pooled_fallback_warned.clear()


class NRMSEZeroRangeError(ValueError):
    """Raised when y_true has zero range inside a CV-split."""


# Per-fold NRMSE
def _nrmse(y_true: Any, y_pred: Any) -> float:
    """Compute the local normalized RMSE (NRMSE).
    Calculation logic:
    1. The standard RMSE for the current sample is computed.
    2. The range (max - min) is determined from the
    passed true values `y_true`.
    3. Constant-target handling: if the range < 1e-6,
    `inf` (infinite error) is returned
    with a warning logged to avoid division by zero.
    4. RMSE is divided by the computed local range.
    Args:
        y_true (Any): True values of the target variable within the split.
        y_pred (Any): Model predictions.
    Returns:
        float: The NRMSE value normalized by the local range.
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
    """Compute the global normalized RMSE
    using a fixed range.
    Calculation logic:
    1. Input range validation: if `target_range` is below the threshold 1e-6,
    `inf` is returned to prevent numerical instability.
    2. The standard RMSE is computed.
    3. The error is normalized by a precomputed global range
    (independent of the current split).
    Args:
        y_true (np.ndarray): True values of the target variable.
        y_pred (np.ndarray): Model predictions.
        target_range (float): Global range (max - min) of the entire dataset.
    Returns:
        float: An NRMSE value comparable across different cross-validation folds.
    """
    if target_range < 1e-6:
        logger.warning("Global target_range is too small. Returning +inf.")
        return float("inf")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return rmse / target_range


def get_global_nrmse_scorer(global_y: np.ndarray) -> Callable[..., Any]:
    """Create a Scikit-Learn compatible scorer object for the global NRMSE.
    Creation logic:
    1. The minimum and maximum values are computed
    from the passed full vector `global_y`.
    2. The global range `target_range` is calculated.
    3. A scorer object is built via `make_scorer`,
    wrapping the `_global_nrmse` function.
    4. The `greater_is_better=False` flag is set
    so that sklearn inverts the metric for maximization.
    Args:
        global_y (np.ndarray): Full target variable vector
        for computing the global range.
    Returns:
        Any: A Scorer object for use in GridSearchCV or cross_validate.
    """
    y_max = np.max(global_y)
    y_min = np.min(global_y)
    target_range = float(y_max - y_min)

    scorer = make_scorer(
        _global_nrmse, greater_is_better=False, target_range=target_range
    )

    return cast(Callable[..., Any], scorer)


# --------------------------------------------------------------------------- #
#  Private registry of "raw" metrics (without sign inversion etc.)
# --------------------------------------------------------------------------- #
_METRICS: dict[str, Callable[..., Any]] = {
    # "lower → better"
    "rmse": _rmse,
    "nrmse": _nrmse,
    "global_nrmse": _global_nrmse,
    # "higher → better"
    "r2": r2_score,
    # alias: "higher → better" (negative RMSE)
    "neg_root_mean_squared_error": lambda y_t, y_p: -_rmse(y_t, y_p),
}


# --------------------------------------------------------------------------- #
#  Registry of ready-made scorer objects for use in the sklearn API
# --------------------------------------------------------------------------- #
# For each metric the optimization direction (is_greater_better) is stored
# explicitly next to the scorer: True — "higher is better" (r2, as well as
# inverted neg_ metrics), False — "lower is better" (rmse, mae, etc.).
# The direction is NO LONGER inferred heuristically from name substrings — only
# from this registry or from the sklearn scorer object itself.
_SCORER_OBJECTS: dict[str, tuple[Callable[..., Any], bool]] = {
    # For errors set greater_is_better=False;
    # sklearn itself will return negative values for maximization
    "nrmse": (make_scorer(_nrmse, greater_is_better=False), False),
    "rmse": (make_scorer(_rmse, greater_is_better=False), False),
    "neg_root_mean_squared_error": (make_scorer(_rmse, greater_is_better=False), True),
    "mae": (make_scorer(mean_absolute_error, greater_is_better=False), False),
    "mse": (make_scorer(mean_squared_error, greater_is_better=False), False),
    "r2": (make_scorer(r2_score, greater_is_better=True), True),
}

AVAILABLE_METRICS = list(_SCORER_OBJECTS.keys())


# --------------------------------------------------------------------------- #
#  Public helpers — may be useful externally
# --------------------------------------------------------------------------- #
def get_metric(name: str) -> Callable[..., Any]:
    """Get the "raw" metric computation function from the internal registry.
    Lookup logic:
    1. The metric name is lowercased to avoid lookup errors.
    2. The name is checked against the `_METRICS` registry.
    3. If the metric is not found, a `KeyError` is raised.
    Args:
        name (str): Metric name (e.g. 'rmse' or 'nrmse').
    Returns:
        Callable[..., Any]: A function accepting (y_true, y_pred) and returning a float.
    """
    lname = name.lower()
    if lname not in _METRICS:
        raise KeyError(f"Metric '{name}' not implemented.")
    return _METRICS[lname]


def _resolve_scorer(name: str) -> Callable[..., Any]:
    """Resolve a metric name into an actual scorer object.

    Resolution logic:
    1. For metrics in the ``_SCORER_OBJECTS`` registry, the preconfigured
       ``make_scorer`` object is returned (the first tuple element).
    2. Otherwise the name is delegated to ``sklearn.metrics.get_scorer``.

    Args:
        name (str): Metric name.

    Returns:
        Callable[..., Any]: A scorer object compatible with the sklearn API.

    Raises:
        ValueError: If the metric name is found neither in the registry nor in
            sklearn (with a clear message in the style of ``get_metric``).
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
    """Determine the optimization direction from a sklearn scorer object.

    Only the "stable" part is cached: the sklearn scorers registry does not
    change at runtime, so the result for a given name is deterministic and
    safe to cache. In contrast, the ``_SCORER_OBJECTS`` registry can be
    extended/changed by the user, so it is read without caching
    (see ``is_greater_better``).

    The scorer object exposes a direction flag (``_sign``): +1 — "higher is
    better", -1 — inverted error (neg_ prefix), which the optimizer still
    maximizes.

    Args:
        name (str): Lowercased metric name.

    Returns:
        bool: True when the metric value is maximized.

    Raises:
        ValueError: If sklearn does not know the metric (see ``_resolve_scorer``).
    """
    scorer = _resolve_scorer(name)
    sign = getattr(scorer, "_sign", None)
    if sign is not None:
        return bool(sign > 0) or name.startswith("neg_")
    return True


def is_greater_better(name: str) -> bool:
    """Determine the optimization direction of a metric: "higher is better"?

    The direction is determined from the actual scorer object, not from
    name substrings:
    1. For metrics in the ``_SCORER_OBJECTS`` registry, the direction is
       stored explicitly next to the scorer (rmse/mae/mse/nrmse — False,
       r2 and neg_root_mean_squared_error — True). The registry is read each
       time without caching: it is mutable, and the result must reflect the
       current state.
    2. For the remaining names the ``sklearn.metrics.get_scorer`` object is
       used: all sklearn scorers are designed such that a higher value is
       better — error metrics are already inverted into neg_ scorers
       (``_sign == -1``), score metrics are returned as is (``_sign == +1``).
       This path is stable and cached (``_sklearn_direction``).

    Args:
        name (str): Metric name.

    Returns:
        bool: True when the metric value is maximized (r2, neg_* metrics),
            False for error metrics (RMSE, MAE, MSE, NRMSE, global_nrmse).

    Raises:
        ValueError: If the metric is unknown to both the registry and sklearn.
    """
    lname = name.lower()
    # 1. Explicit direction from the private registry — no cache.
    if lname in _SCORER_OBJECTS:
        return _SCORER_OBJECTS[lname][1]
    # 2. Dynamic global NRMSE — an error ("lower is better"), just like
    #    regular NRMSE. The scorer requires global_y and is not resolved by
    #    sklearn, so it is handled explicitly (in sync with to_user_value).
    if lname == "global_nrmse":
        return False
    # 3. Stable sklearn scorers — with result caching.
    return _sklearn_direction(lname)


def is_error_metric(name: str) -> bool:
    """Return whether a metric is an error in the user-facing representation.

    Determined from the actual scorer object (the ``_sign`` attribute):
    inverted error scorers (-RMSE, -MAE, all neg_ metrics) return a positive
    error value to the user → True; score metrics (R² and similar) are
    returned as is → False.

    Args:
        name (str): Metric name.

    Returns:
        bool: True for error metrics (RMSE, MAE, MSE, NRMSE, neg_* metrics),
            False for score metrics (R², accuracy, etc.).

    Raises:
        ValueError: If the metric is unknown to both the registry and sklearn.
    """
    lname = name.lower()
    if lname == "global_nrmse":
        return True
    scorer = _resolve_scorer(lname)
    sign = getattr(scorer, "_sign", None)
    return sign is not None and sign < 0


def direction_label(name: str) -> str:
    """Return a human-readable metric direction for logs.

    User-facing metric values (``to_user_value``) in logs must come with their
    direction: ``"min better"`` for errors, ``"max better"`` for score
    metrics. The direction is taken from ``user_direction`` (user semantics),
    not from the optimizer flag ``greater_is_better`` — for neg_ metrics these
    semantics diverge (the optimizer maximizes -RMSE, while a smaller value is
    better for the user).

    Args:
        name (str): Metric name.

    Returns:
        str: ``"min better"`` for errors, ``"max better"`` for score metrics.

    Raises:
        ValueError: If the metric is unknown to both the registry and sklearn.
    """
    return "min better" if user_direction(name) == "minimize" else "max better"


def user_direction(name: str) -> Literal["minimize", "maximize"]:
    """Determine the metric direction in user semantics.

    Unlike ``is_greater_better`` (optimizer semantics — "higher is better"
    for all scorers), returns the direction for values in the user-facing
    representation (``to_user_value``):

    - error metrics (RMSE, MAE, MSE, NRMSE, global_nrmse and all neg_ metrics)
      → ``"minimize"``: in user semantics a smaller value is better;
    - score metrics (R², accuracy, etc.) → ``"maximize"``: a larger value
      is better.

    The direction is required by the finalist-pool corridor formula:
    the multiplicative corridor form depends on which value is better —
    smaller or larger (for R² with a negative leader the multiplicative form
    is incorrect, see ``select_finalists``).

    Args:
        name (str): Metric name.

    Returns:
        Literal["minimize", "maximize"]: Direction in user semantics:
            "minimize" for errors, "maximize" for score metrics.

    Raises:
        ValueError: If the metric is unknown to both the registry and sklearn.
    """
    lname = name.lower()
    # Errors are inverted by the scorer but returned to the user as natural
    # values ("lower is better"): this has priority over greater_is_better,
    # which is true for neg_ metrics (optimizer semantics). For all other
    # metrics the user semantics is "higher is better".
    if is_error_metric(lname):
        return "minimize"
    return "maximize"


def to_user_value(name: str, raw_value: float) -> float:
    """Convert a "raw" scorer value to user semantics.

    Error scorers (including all neg_ metrics) return an inverted value
    (-RMSE, -MAE, ...) so that the optimizer can maximize. The natural metric
    value is returned to the user: RMSE → positive RMSE, MAE → positive MAE,
    R² → ordinary R².

    Note: non-finite values (e.g. -inf from NRMSE on a constant target) are
    inverted as usual — this is the existing contract of
    ``ModelTrainer.val_score``/``additional_scores``. Protection against the
    worst-score sentinel leaking into the report is provided by the final
    winner-score invariant at the orchestrator level
    (``component.is_valid_winner_score``), not by this function.

    Args:
        name (str): Metric name.
        raw_value (float): "Raw" value returned by the scorer object.

    Returns:
        float: The metric value in the user-facing representation.
    """
    value = float(raw_value)
    if name.lower() == "global_nrmse":
        # Dynamic NRMSE scorer: an error, always returns -NRMSE.
        return -value
    scorer = _resolve_scorer(name)
    sign = getattr(scorer, "_sign", None)
    if sign is not None and sign < 0:
        return -value
    return value


def get_scorer_object(
    name: str, global_y: np.ndarray | None = None
) -> Callable[..., Any] | str:
    """Get a scorer object ready for use in sklearn tools.
    Lookup logic:
    1. Handle 'global_nrmse': requires `global_y` to be present
    for creating the dynamic scorer.
    2. Search the `_SCORER_OBJECTS` registry: returns the preconfigured custom scorers.
    3. Fallback: if the name is not in the registry, the
    standard `sklearn.metrics.get_scorer` is used.
    Args:
        name (str): Name of the requested scorer.
        global_y (np.ndarray | None): Optional target array for global metrics.
    Returns:
        Union[Callable[..., Any], str]: A Scorer object or a system sklearn string.
    """
    lname = name.lower()

    # Special handling for the global NRMSE
    if lname == "global_nrmse":
        if global_y is None:
            raise ValueError(
                "For 'global_nrmse', 'global_y' must be passed to get_scorer_object"
            )
        return get_global_nrmse_scorer(global_y)

    # If this is our custom metric (nrmse, rmse, etc.)
    if lname in _SCORER_OBJECTS:
        return _SCORER_OBJECTS[lname][0]

    # Otherwise return the name as is
    # (sklearn will find the built-in metric itself)
    scorer = sklearn_get_scorer(lname)
    return cast("Callable[..., Any] | str", scorer)


# --------------------------------------------------------------------------- #
#  Mapping user aliases to what sklearn understands
# --------------------------------------------------------------------------- #
_ALIAS_TO_SKLEARN = {
    # since sklearn optimizes "the higher — the better",
    # nrmse is registered directly
    "rmse": "neg_root_mean_squared_error",
}


def to_sklearn_name(name: str) -> str:
    """Map a user metric alias to the system sklearn name.
    Conversion logic:
    1. The name is lowercased.
    2. A lookup is performed in the `_ALIAS_TO_SKLEARN` dict
    (e.g. 'rmse' -> 'neg_root_mean_squared_error').
    3. If the alias is not found, the original lowercased name is returned.
    Args:
        name (str): User-facing metric name.
    Returns:
        str: A metric name recognized by sklearn internals.
    """
    return _ALIAS_TO_SKLEARN.get(name.lower(), name.lower())

"""Hyperparameter Optimisation Module:
An engine for automated hyperparameter search.

The module provides a high-level wrapper around Optuna for automatic model
configuration tuning with support for dynamic search spaces, cross-validation
and integrated oversampling.

Key features:
    1. Hybrid Model Zoo: Full support for the AutoML engine's base model pool
       with extensions for the SGD, GaussianProcess, Isotonic and GLM families.
    2. Neural Filter: Automatic detection and exclusion of neural network
       architectures (alias "nn") to save resources in a classical ML pipeline.
    3. Adaptive Validation: Intelligent switching between k-fold,
       Leave-One-Out and Train-Test Split (80/20) depending on the sample size
       (fallback when n_samples < 2k).
    4. Integrated Oversampling: Seamless class-balancing integration through an
       Imbalance-Pipeline right inside the optimisation process.
    5. Dynamic Search Spaces: Support for both hard-coded spaces (e.g. the
       adaptive KNN space) and external configurations via YAML/SearchSpaceEntry.
    6. Metric Agnostic: Optimisation against any standard or custom Sklearn
       metric (R² by default).
"""

from __future__ import annotations

# ─────────────────────────────── stdlib
import logging as _logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NoReturn, cast

# ──────────────────────────── third-party
import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.trial import FrozenTrial, Trial
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
)

from configurable_automl_engine.common.complexity_guard import (
    ComplexityGuardConfig,
    apply_complexity_guard,
)

# ──────────────────────────── project
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.common.hyperopt_defaults import (
    DEFAULT_SPACES,
    FloatSpace,
    SearchSpaceEntry,
    clip_search_space,
    compute_svr_epsilon_bounds,
    normalize_hyperparameter_entry,
)
from configurable_automl_engine.common.validation_utils import get_effective_train_size
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    create_model,
    remap_legacy_param_names,
    requires_dense_input,
    resolve_algorithm_name,
)
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
    _clean_uninformative_features,
    build_preprocessor,
    detect_feature_types,
)
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    requires_missing_indicator,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.trainer import IsotonicDataTransformer
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from configurable_automl_engine.training_engine.metrics import (
    get_scorer_object,
    is_non_additive_metric,
    pooled_scorer_value,
    warn_pooled_fallback_once,
)
from configurable_automl_engine.validation import iter_splits, make_cv, norm_val_method

logging = _logging  # alias


# ═════════════════════════════════════ exceptions ════════════════════════════
class HyperoptError(Exception):
    """Base error of the hyperparameter optimisation module."""


class InvalidAlgorithmError(HyperoptError):
    """Error raised when an algorithm is not found or is marked as unused."""


class InvalidDataError(HyperoptError):
    """Error raised when invalid X or y data structures are passed."""


class HPOHistoryError(HyperoptError):
    """Error raised when TPE history cannot be transferred between HPO phases.

    Raised in three situations:
    - the tuner does not accept the ``previous_trials`` argument, but the
      pipeline needs to transfer history (e.g. a custom tuner in a refine
      phase) — fail fast instead of silently degrading refine to a cold run;
    - the transferred trials have distributions incompatible with the current
      search space (``study.add_trials`` desync) — fail fast instead of
      silently skipping the history;
    - the current search space cannot be reproduced by the distribution
      pre-check (an exotic custom ``space_fn``) — the transfer cannot be
      validated, and silently dropping the history would degrade refine to a
      cold run (the exact bug this feature fixes).
    """


@dataclass
class OptimizeResult:
    """Result of a single HPO phase run (self-documenting return type).

    Replaces the positional ``(model, params, score)`` tuple of ``optimize``:
    the fields have explicit names, and the current phase's trials travel with
    the result so the caller can feed them into the next phase
    (``previous_trials``).

    Attributes:
        model: The final model trained on all data with the best parameters.
            ``None`` when ``fit_best_model=False`` (the full-data fit is
            skipped for HPO-only callers); otherwise never ``None`` on a
            non-``None`` ``OptimizeResult``.
        params: The best found hyperparameters (optimizer semantics, as for
            the score).
        score: The best raw "optimizer" score of the phase (higher is
            better).
        trials: The frozen trials of the current phase ONLY — transferred
            ``previous_trials`` are excluded. Always present on a non-``None``
            result; the list may contain COMPLETED/PRUNED/FAILED trials of
            the phase.
        complexity_guard: Log records of the adaptive complexity guard for
            the phase (one per restricted parameter, "before → after" with
            the reason). ``None`` when the guard is disabled or produced no
            records.
    """

    model: Any | None
    params: dict[str, Any] | None
    score: float | None
    trials: list[FrozenTrial]
    complexity_guard: list[dict[str, Any]] | None = None


# ═══════════════════════════════════ logging setup ═══════════════════════════
log = logging.getLogger(__name__)


# ═══════════════════════════════════ constants ═══════════════════════════════
# The worst possible trial score — an analog of the float32 minimum or
# float('-inf'). Returned as the objective value when the real metric is not
# finite (NaN/+inf, e.g. an nrmse error). The value is practically unreachable
# for real metrics, hence used as a sentinel.
#
# Used in two places:
# 1. Inside `_objective`: a non-finite trial avg_score is replaced with this
#    constant (the trial completes, but with a deliberately worst score).
# 2. On the orchestrator side: the `valid_results` filter in
#    `training_engine/component.py` drops results whose score belongs to the
#    worst-score sentinel class — any score at or below WORST_SCORE_THRESHOLD —
#    treating the algorithm as failed.
HPO_WORST_SCORE = -3.4028235e38

# The lower bound of the "worst-score sentinel" class: the exact float32
# minimum. Any score <= this bound is treated as a sentinel — real metrics of
# such magnitude are practically impossible (documented threshold decision).
# Comparing against this bound (instead of the exact
# HPO_WORST_SCORE via math.isclose) also catches the raw
# float(np.finfo(np.float32).min) value that numpy-cast metrics or custom
# tuners may return: its relative difference from HPO_WORST_SCORE is
# ≈ 9.88e-9, which the old isclose(rel_tol=1e-9) check silently missed.
WORST_SCORE_THRESHOLD = float(np.finfo(np.float32).min)

# ═══════════════════════════════════ search spaces ═══════════════════════════


# ══════════ KNN-space depends on the sample size ══════════
def _make_knn_space(n_samples: int) -> Callable[[Trial], dict[str, Any]]:
    """Create a search space generator for the KNN algorithm."""

    def _space(t: Trial) -> dict[str, Any]:
        # Cap n_neighbors by the physical limit of the training set (N_eff - 1).
        # Use n_samples_eff instead of the total number of rows in the dataset.
        physical_limit = max(1, n_samples - 1)
        max_k = int(min(30, physical_limit))

        return {
            "n_neighbors": t.suggest_int("n_neighbors", 1, max_k),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
            "p": t.suggest_int("p", 1, 2),
        }

    return _space


# ══════════ SVR space: epsilon depends on the spread of y ══════════
def _svr_adaptive_epsilon_entry(y: Any) -> SearchSpaceEntry:
    """Build a SearchSpaceEntry for SVR epsilon with bounds scaled by y.

    The bounds are computed from the spread of ``y`` (σ(y) by default) ONCE —
    before the Optuna loop (called from ``resolve_search_space``, not from
    ``space_fn``): otherwise every trial would have its own bounds and the
    TPE sampler would lose determinism.

    Args:
        y (Any): Target variable vector.
    Returns:
        SearchSpaceEntry: float_log distribution of epsilon within adaptive bounds.
    """
    low, high = compute_svr_epsilon_bounds(y)
    log.info("SVR adaptive epsilon bounds: [%.6g, %.6g] (from y spread)", low, high)
    return SearchSpaceEntry.model_validate([low, high, "float_log"])


def _make_svr_space_dict(y: Any) -> dict[str, Any]:
    """Build the default SVR search space dict with adaptive epsilon.

    The default space is taken from ``DEFAULT_SPACES["svr"]``, and epsilon is
    replaced by a float_log distribution within bounds derived from the spread
    of ``y_train``.

    Args:
        y (Any): Target variable vector.
    Returns:
        dict[str, Any]: A dict of ``SearchSpaceEntry`` objects.
    """
    space = DEFAULT_SPACES["svr"].copy()
    space["epsilon"] = _svr_adaptive_epsilon_entry(y)
    return space


# ═════════════════════════════ helper utilities ═════════════════════════════


def _apply_dynamic_space(trial: Trial, space_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a configuration dict into model parameters via Optuna methods.

    Args:
        trial (Trial): The current Optuna trial object.
        space_dict (dict[str, Any]): A dict of SearchSpaceEntry objects
            (with bounds and distribution types) or constant values.
    Returns:
        dict[str, Any]: The concrete hyperparameter values for this iteration.
    """
    params: dict[str, Any] = {}
    for key, value in space_dict.items():
        # If this is a SearchSpaceEntry
        # (use the low, high, dist_type, step properties)
        if hasattr(value, "dist_type"):
            low, high = value.low, value.high
            dist_type = value.dist_type
            step = value.step
            if dist_type == "int":
                low_val, high_val = int(cast(float, low)), int(cast(float, high))
                params[key] = trial.suggest_int(
                    key, low_val, high_val, step=int(step) if step is not None else 1
                )
            elif dist_type == "float":
                params[key] = trial.suggest_float(
                    key,
                    float(low),
                    float(high),
                    step=float(step) if step is not None else None,
                )
            elif dist_type == "float_log":
                FloatSpace.validate_log_low(float(low))
                params[key] = trial.suggest_float(
                    key, float(low), float(high), log=True
                )
            elif dist_type == "categorical":
                # Extract the options list from the options attribute or nested config
                options = getattr(value, "options", None)
                if options is None and hasattr(value, "config"):
                    options = getattr(value.config, "options", None)

                # If still None, fall back to low (for compatibility)
                if options is None:
                    options = low

                params[key] = trial.suggest_categorical(key, options)
        else:
            # If it is a plain value (constant), use it as is
            params[key] = value
    return params


def _validate_data(X: Any, y: Any) -> None:
    """Validate the types and sizes of the input data X and y.

    Args:
        X (Any): Feature matrix (expected np.ndarray or pd.DataFrame).
        y (Any): Target variable vector
            (expected np.ndarray, pd.Series or pd.DataFrame).
    Raises:
        InvalidDataError: If the data types are not supported
            or the sizes of X and y do not match.
    """
    ok_types = (np.ndarray, pd.DataFrame)
    if not isinstance(X, ok_types):
        raise InvalidDataError("X must be a numpy.ndarray or pandas.DataFrame")
    if not isinstance(y, ok_types + (pd.Series,)):
        raise InvalidDataError(
            "y must be a numpy.ndarray, pandas.Series, or pandas.DataFrame"
        )
    if len(X) != len(y):
        raise InvalidDataError(f"Size mismatch: X={len(X)} and y={len(y)}")


def _get_estimator(algo: str) -> Any:
    """Check the availability and validity of the given algorithm.

    Args:
        algo (str): Algorithm name (alias).
    Returns:
        Any: Returns True if the model is created successfully by the base factory.
    Raises:
        InvalidAlgorithmError: If the algorithm is not supported or a required
            dependency (library) is missing.
    """
    try:
        # Try to create a model with minimal parameters to check existence
        create_model(algo)
        return True
    except (ValueError, ImportError) as err:
        # If the algorithm is not found in models.py or the package is not
        #  installed (e.g. XGBoost)
        raise InvalidAlgorithmError(f"Algorithm '{algo}' is not supported: {err}")


def _missing_indicator_enabled(algorithm: str | None) -> bool:
    """Safely determine whether missing-value indicators are needed.

    Strictly univariate algorithms (``isotonic_regression``) are incompatible
    with missing-indicator columns (their input contract is exactly one
    feature), so ``False`` is returned for them. An empty algorithm name or an
    invalid name (unknown algorithm) must not crash the HPO phase — in that
    case the safe default ``True`` is returned.

    Args:
        algorithm: Regression algorithm name (or alias); it may be
            ``None``/an empty string before name validation.

    Returns:
        ``True`` — missing indicators are enabled; ``False`` — the algorithm
        is strictly univariate.
    """
    if not algorithm:
        return True
    try:
        return requires_missing_indicator(algorithm)
    except ValueError:
        # Unknown algorithm name: do not mask the config error, but do not
        # let it surface as an unhandled exception during preprocessor
        # assembly — name validation is a separate step.
        return True


def _build_scorer(name: str) -> Any:
    """Create a scorer object from a metric name.

    Args:
        name (str): String metric name
            (e.g. 'r2' or 'neg_mean_squared_error').
    Returns:
        Any: A metric object compatible with the sklearn API.
    Raises:
        HyperoptError: If an unknown metric name is given.
    """
    try:
        # Use the new API, which returns either a make_scorer object or a string
        return get_scorer_object(name)
    except Exception as err:
        raise HyperoptError(f"Unknown metric name: '{name}'") from err


def _can_stratify(y: Any) -> bool:
    """Determine whether stratification is possible for the y vector.

    Args:
        y (Any): Target variable vector.
    Returns:
        bool: True if the data is discrete (integers/bool) and the number of
            unique classes does not exceed 15. Otherwise — False.
    """
    # np.asarray guarantees a plain np.ndarray and np.dtype for mypy (no ExtensionArray/Dtype)
    arr = np.asarray(y)

    if arr.ndim != 1:
        return False

    uniq = np.unique(arr)
    return uniq.size <= 15 and (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)
    )


def _split_train_test(
    X: Any, y: Any, *, test_size: float = 0.2, random_state: int | None = 42
) -> tuple[Any, Any, Any, Any]:
    """Split data into train and test sets with automatic stratification.

    Args:
        X (Any): Feature matrix.
        y (Any): Target vector.
        test_size (float): Test set fraction. Defaults to 0.2.
        random_state (int | None): Random number generator seed. Defaults to 42.
    Returns:
        tuple[Any, Any, Any, Any]: Tuple of (X_train, X_test, y_train, y_test).
    """
    strat = y if _can_stratify(y) else None
    try:
        return cast(
            tuple[Any, Any, Any, Any],
            train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=True,
                random_state=random_state,
                stratify=strat,
            ),
        )
    except ValueError:
        return cast(
            tuple[Any, Any, Any, Any],
            train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_state
            ),
        )


# ═══════════════ early stopping (pruning) helpers ═══════════════════════════
_DEFAULT_PRUNING_CONFIG: dict[str, Any] = {
    "enable": False,
    "strategy": "median",
    "min_steps": 1,
    "n_startup_trials": 5,
    "reduction_factor": 3,
}


def _normalize_pruning_config(
    pruning: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize the early stopping configuration into a dict with valid fields.

    Accepts a settings dict (e.g. forwarded from ``training_engine``) or
    None. Returns ``None`` if early stopping is disabled or the config is not
    provided — in that case the tuner behaves identically to the current one.

    Args:
        pruning (dict[str, Any] | None): A dict with enable, strategy,
            min_steps, n_startup_trials, reduction_factor keys (or a subset).
    Returns:
        dict[str, Any] | None: The normalized pruner configuration, or None.
    """
    if not pruning:
        return None
    enabled = bool(pruning.get("enable", False))
    if not enabled:
        return None
    return {
        "strategy": str(pruning.get("strategy", _DEFAULT_PRUNING_CONFIG["strategy"])),
        "min_steps": int(
            pruning.get("min_steps", _DEFAULT_PRUNING_CONFIG["min_steps"])
        ),
        "n_startup_trials": int(
            pruning.get("n_startup_trials", _DEFAULT_PRUNING_CONFIG["n_startup_trials"])
        ),
        "reduction_factor": int(
            pruning.get("reduction_factor", _DEFAULT_PRUNING_CONFIG["reduction_factor"])
        ),
    }


def _build_pruner(cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    """Create an Optuna pruner object from a normalized configuration.

    Args:
        cfg (dict[str, Any]): Normalized early stopping configuration
            (the result of ``_normalize_pruning_config``).
    Returns:
        optuna.pruners.BasePruner: A configured Optuna pruner.
    Raises:
        HyperoptError: If an unknown pruning strategy is given or the
            parameter values are invalid.
    """
    strategy = cfg["strategy"]
    min_steps = cfg["min_steps"]
    if min_steps < 1:
        raise HyperoptError(
            f"pruning.min_steps must be a positive integer, got {min_steps}"
        )
    if strategy == "median":
        n_startup_trials = cfg["n_startup_trials"]
        if n_startup_trials < 1:
            raise HyperoptError(
                "pruning.n_startup_trials must be a positive integer, "
                f"got {n_startup_trials}"
            )
        # n_warmup_steps=min_steps: steps are numbered from 1, so the first
        # pruning decision is possible exactly after min_steps reports.
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=min_steps,
        )
    if strategy == "hyperband":
        reduction_factor = cfg["reduction_factor"]
        if reduction_factor < 2:
            raise HyperoptError(
                "pruning.reduction_factor must be at least 2 (Hyperband), "
                f"got {reduction_factor}"
            )
        return optuna.pruners.HyperbandPruner(
            min_resource=min_steps,
            reduction_factor=reduction_factor,
        )
    raise HyperoptError(
        f"Unknown pruning strategy: '{strategy}'. Must be 'median' or 'hyperband'."
    )


def _warn_failed_fold(fold_idx: int, err: Exception, stage: str) -> None:
    """Log the exclusion of a failed fold at the given evaluation stage.

    Shared by the per-fold evaluation loops (``_evaluate_with_intermediate_
    reports`` and ``_evaluate_pooled_cv``) so the warning text stays uniform:
    a fold whose fit/predict/scoring raised a non-fatal exception is skipped,
    its rows do not enter the (pooled) estimate, and the trial continues — the
    semantics of ``cross_val_score(error_score=np.nan)``.

    Args:
        fold_idx (int): 1-based fold number.
        err (Exception): The raised exception (the message is logged with its
            type name).
        stage (str): One of 'fit or predict' / 'score' — the operation that
            failed for this fold.
    """
    log.warning(
        "Fold %d failed to %s: %s: %s — the fold is excluded from the evaluation",
        fold_idx,
        stage,
        type(err).__name__,
        err,
    )


def _evaluate_with_intermediate_reports(
    trial: Trial,
    estimator: Any,
    X: Any,
    y: Any,
    *,
    method: str,
    n_folds: int,
    test_size: float,
    random_state: int | None,
    scorer: Any,
    pool_metric: bool = False,
    algo_name: str = "",
    phase_name: str = "",
) -> float:
    """Evaluate the estimator fold by fold, publishing intermediate results.

    Serves as the early stopping enabler: after each fold the running estimate
    of the metric is published via ``trial.report``, followed by a call to
    ``trial.should_prune()``. A hopeless trial is interrupted by an
    ``optuna.TrialPruned`` exception before the full evaluation completes.

    For non-additive metrics (``pool_metric=True``: RMSE, MSE, R², nRMSE,
    RMSLE, D², explained_variance, max error, median absolute error) the
    published intermediate value is the metric
    computed over the predictions accumulated up to the current step (the
    pooled value over the covered rows), not the cumulative mean of per-fold
    values — each row enters the calculation exactly once, in the same way as
    the final pooled estimate. Non-finite predictions of individual folds are
    masked out of the pooled value (the same contract as
    ``metrics.oof_rmse``). A fold whose fit/predict raises a NON-FATAL
    exception (e.g. ``TypeError``, ``KeyError``, ``IndexError``,
    ``ValueError``/``numpy.linalg.LinAlgError``) is skipped the same way: its
    rows do not enter the pooled value and a warning is logged — the semantics
    of the legacy ``cross_val_score(error_score=np.nan)`` (a failed fold must
    not abort the trial). Fatal exceptions (``MemoryError``, ``RuntimeError``,
    ``InvalidDataError``) are NOT swallowed — they propagate to
    ``handle_failure`` so the circuit breaker stays active. If EVERY fold
    fails, a ``ValueError`` is raised — the same
    "no valid result" contract as before. A running pooled value that is
    itself non-finite
    (e.g. R² on a single LOO pair, which is undefined) is NOT published to
    ``trial.report``: Optuna pruners cannot handle NaN and would emit
    RuntimeWarnings and compare incorrectly. The step is skipped until the
    accumulated rows admit a finite pooled estimate. Additive metrics (MAE and
    similar per-sample means, ``pool_metric=False``) keep publishing the
    cumulative fold mean (which equals the pooled value); a failed fold is
    skipped with a warning and does not contribute to the cumulative mean.

    When the pooled value is impossible FINALLY (the scorer exposes no raw
    metric function, or no finite pair remains by the end of the evaluation),
    a WARNING about the fallback to the per-fold mean is logged once per
    (``algo_name``, ``phase_name``) pair. A TRANSIENT pooled failure inside
    the fold loop (e.g. the first fold predicted all-NaN) is not a metric
    degradation — the final pooled value may still be valid — so no WARNING
    is emitted there.

    Args:
        trial (Trial): The current Optuna trial.
        estimator (Any): The estimator (possibly a pipeline) to evaluate.
        X (Any): Feature matrix.
        y (Any): Target variable vector.
        method (str): Split strategy ('k_fold' or 'loo').
        n_folds (int): Number of folds for k_fold.
        test_size (float): Test fraction (unused for k_fold/loo; passed to
            keep a uniform iter_splits call).
        random_state (int | None): Randomness seed.
        scorer (Any): sklearn metric object.
        pool_metric (bool): Whether the metric is non-additive and must be
            pooled over the accumulated OOF predictions. Defaults to False —
            the legacy cumulative-fold-mean behavior.
        algo_name (str): The algorithm being evaluated, for the fallback
            WARNING deduplication.
        phase_name (str): The pipeline phase, for the fallback WARNING
            deduplication.
    Returns:
        float: The mean metric value across all folds (for additive metrics)
            or the pooled value over all rows (for non-additive metrics).
    Raises:
        optuna.TrialPruned: If the pruner decides to prune the trial.
        ValueError: If every fold fails during evaluation (fit/predict) — the
            estimate is impossible.
        MemoryError, RuntimeError, InvalidDataError: A fatal per-fold failure —
            propagated so the circuit breaker stays active.
        Exception: A genuine failure outside a single fold (e.g. split
            iteration) propagates (fatal or non-fatal depending on the
            exception type, as before).
    """
    fold_scores: list[float] = []
    # Non-additive metrics (RMSE, MSE, R², nRMSE, ...) are pooled over the
    # rows covered by the folds processed so far; the running pooled value is
    # published as the intermediate result. The OOF rows are accumulated in
    # pre-allocated full-size buffers (filled by positional indices) and
    # pooled via ``pooled_scorer_value``, whose non-finite masking drops the
    # not-yet-covered rows — no repeated array concatenation (O(k·n) copies
    # on the old code, O(n²) for LOO).
    n_rows = len(X)
    pooled_y_buf = np.full(n_rows, np.nan, dtype=float)
    pooled_pred_buf = np.full(n_rows, np.nan, dtype=float)
    n_folds_total = 0
    n_failed_folds = 0
    for fold_idx, (X_tr, X_te, y_tr, y_te, test_idx) in enumerate(
        iter_splits(
            X,
            y,
            method=method,
            n_folds=n_folds,
            test_size=test_size,
            random_state=random_state,
            include_indices=True,
        ),
        start=1,
    ):
        n_folds_total += 1
        # Clone the estimator for every fold (as cross_val_score does) to
        # avoid state accumulating between folds. A fold whose fit/predict
        # raises a NON-FATAL exception is skipped — it does not contribute to
        # the estimate or to the intermediate reports, and a warning is logged
        # (the ``cross_val_score(error_score=np.nan)`` contract: a failed fold
        # must not abort the trial). Fatal exception types (MemoryError /
        # RuntimeError / InvalidDataError) are NOT swallowed here — they
        # propagate to ``handle_failure`` and keep the circuit breaker active.
        try:
            fold_estimator = clone(estimator)
            fold_estimator.fit(X_tr, y_tr)
        except (MemoryError, RuntimeError, InvalidDataError):
            raise
        except Exception as err:  # noqa: BLE001 — per-fold failure, cross_val_score(error_score=nan) semantics
            n_failed_folds += 1
            _warn_failed_fold(fold_idx, err, "fit or predict")
            continue
        try:
            fold_pred = np.asarray(fold_estimator.predict(X_te)).reshape(-1)
        except (MemoryError, RuntimeError, InvalidDataError):
            raise
        except Exception as err:  # noqa: BLE001 — per-fold failure, cross_val_score(error_score=nan) semantics
            n_failed_folds += 1
            _warn_failed_fold(fold_idx, err, "fit or predict")
            continue
        fold_true = np.asarray(y_te, dtype=float).reshape(-1)
        if pool_metric:
            # The per-fold scorer value is used only as the fallback when the
            # pooled value cannot be computed; for non-additive metrics a fold
            # with non-finite predictions must not abort the trial (those rows
            # are masked out of the pooled value). The scorer is called lazily
            # — only when the pooled value turns out to be impossible — so
            # that per-fold scoring noise (e.g. LOO+R², where every fold R² is
            # undefined) does not pollute the logs on the common path.
            idx = np.asarray(test_idx)
            pooled_y_buf[idx] = fold_true
            pooled_pred_buf[idx] = fold_pred
            try:
                running_value = pooled_scorer_value(
                    scorer, pooled_y_buf, pooled_pred_buf
                )
            except ValueError:
                # A transient pooled failure (e.g. the first fold predicted
                # all-NaN, so the accumulated buffer has no valid pair yet)
                # is NOT a metric degradation: later folds may still produce a
                # valid pooled value. Only the per-fold fallback value is used
                # to publish the intermediate step; no WARNING is emitted here
                # (the warning is reserved for the final return point, where a
                # real fallback decision is made).
                try:
                    fold_scores.append(float(scorer(fold_estimator, X_te, y_te)))
                except Exception:  # noqa: BLE001, S110 — only a fallback for the pooled value
                    pass
                running_value = (
                    float(np.mean(fold_scores)) if fold_scores else float("nan")
                )
            # Optuna pruners cannot compare NaN intermediate values (they
            # produce RuntimeWarnings and incorrect decisions), so a
            # non-finite pooled value (e.g. LOO+R² on the first single-row
            # step) is not published — the step is skipped until the
            # accumulated rows admit a finite estimate.
            if np.isfinite(running_value):
                trial.report(running_value, step=fold_idx)
        else:
            try:
                fold_scores.append(float(scorer(fold_estimator, X_te, y_te)))
            except (MemoryError, RuntimeError, InvalidDataError):
                raise
            except Exception as err:  # noqa: BLE001 — per-fold scoring, same cross_val_score(error_score=nan) semantics
                n_failed_folds += 1
                _warn_failed_fold(fold_idx, err, "score")
                continue
            running_value = float(np.mean(fold_scores))
            trial.report(running_value, step=fold_idx)
        if trial.should_prune():
            log.debug(
                "Trial %d pruned by early stopping at step %d",
                trial.number,
                fold_idx,
            )
            raise optuna.TrialPruned()
    if n_folds_total > 0 and n_failed_folds == n_folds_total:
        raise ValueError(
            f"All {n_folds_total} fold(s) failed during evaluation — the "
            "estimate is impossible"
        )
    if pool_metric:
        try:
            return pooled_scorer_value(scorer, pooled_y_buf, pooled_pred_buf)
        except ValueError:
            warn_pooled_fallback_once(
                algo_name=algo_name or "unknown",
                phase_name=phase_name or "untitled_phase",
                reason="the pooled value over the accumulated OOF rows "
                "cannot be computed (no raw scorer metric function or no "
                "valid pairs)",
            )
            return float(np.mean(fold_scores)) if fold_scores else float("nan")
    return float(np.mean(fold_scores)) if fold_scores else float("nan")


def _evaluate_pooled_cv(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    method: str,
    n_folds: int,
    test_size: float,
    random_state: int | None,
    scorer: Any,
    algo_name: str = "",
    phase_name: str = "",
) -> float:
    """Evaluate a trial pipeline over pooled OOF predictions (non-additive).

    Iterates the folds (the same splits as ``cross_val_score(cv=cv_obj)`` via
    ``iter_splits``), collects the out-of-fold prediction/target vectors, and
    computes the metric over the whole vector at once — the same estimate the
    final ``ModelTrainer`` and the select phase use. This is the required form
    for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D²,
    explained_variance, max error, median absolute error): the pooled value
    differs from the average of per-fold values.

    Non-finite predictions of individual folds are masked out of the pooled
    value (the same contract as ``metrics.oof_rmse``), so a fold producing NaN
    (e.g. isotonic regression on a degenerate split) does not abort the trial.
    A fold whose fit/predict (or per-fold scoring, on the non-poolable path)
    raises a NON-FATAL exception (e.g. ``TypeError``, ``KeyError``,
    ``IndexError``, ``ValueError``/``numpy.linalg.LinAlgError``) is skipped the
    same way: its rows do not enter the pooled estimate and a warning is
    logged — the semantics of the legacy ``cross_val_score(error_score=np.nan)``
    (a failed fold became NaN and was excluded from the estimate instead of
    aborting the trial). Fatal exceptions (``MemoryError``, ``RuntimeError``,
    ``InvalidDataError``) are NOT swallowed — they propagate to
    ``handle_failure`` so the circuit breaker stays active. If EVERY fold
    fails at the fit/predict stage, the evaluation is impossible and a
    ``ValueError`` is raised — the same "no valid result" contract as before
    (mirroring ``cross_val_score``, which raises ``ValueError`` when every
    fit fails, and returns an all-NaN array when the fits succeed but every
    score is impossible).

    The per-fold scorer value is computed lazily — only when the pooled value
    cannot be computed — so per-fold scoring noise (e.g. LOO+R², where every
    fold R² is undefined on a single row) does not pollute the logs on the
    common path.

    If the pooled value cannot be computed (no valid pairs, or the scorer does
    not expose a raw metric function), the mean of the available per-fold
    values is returned as a defensive fallback, and a WARNING about the
    fallback is logged once per (``algo_name``, ``phase_name``) pair.

    Args:
        estimator (Any): The estimator (possibly a pipeline) to evaluate.
        X (Any): Feature matrix.
        y (Any): Target variable vector.
        method (str): Split strategy ('k_fold' or 'loo').
        n_folds (int): Number of folds for k_fold.
        test_size (float): Test fraction (unused for k_fold/loo; passed to
            keep a uniform iter_splits call).
        random_state (int | None): Randomness seed.
        scorer (Any): sklearn metric object.
        algo_name (str): The algorithm being evaluated, for the fallback
            WARNING deduplication.
        phase_name (str): The pipeline phase, for the fallback WARNING
            deduplication.

    Returns:
        float: The pooled metric value over all out-of-fold rows.

    Raises:
        ValueError: If every fold fails during evaluation (fit/predict or the
            per-fold scoring fallback) — the pooled estimate is impossible.
        MemoryError, RuntimeError, InvalidDataError: A fatal per-fold failure —
            propagated so the circuit breaker stays active.
    """
    fold_scores: list[float] = []
    n_rows = len(X)
    # Whether the scorer exposes a raw metric function: if it does not, the
    # pooled value is impossible and the per-fold path is taken inside the
    # SAME loop (no double iteration of the splits).
    poolable = getattr(scorer, "_score_func", None) is not None
    # OOF rows are accumulated in pre-allocated full-size buffers (filled by
    # positional indices); the pooled value is computed over the buffers with
    # non-finite masking — no repeated array concatenation.
    pooled_y = np.full(n_rows, np.nan, dtype=float)
    pooled_pred = np.full(n_rows, np.nan, dtype=float)
    # Per-fold (true, pred) pairs are retained so that the per-fold fallback
    # (used when the pooled value is impossible) can be scored WITHOUT
    # re-iterating the splits or re-training the folds.
    fold_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    n_folds_total = 0
    n_failed_folds = 0
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X,
        y,
        method=method,
        n_folds=n_folds,
        test_size=test_size,
        random_state=random_state,
        include_indices=True,
    ):
        n_folds_total += 1
        # Clone the estimator for every fold (as cross_val_score does) to
        # avoid state accumulating between folds. A fold whose fit/predict
        # raises a NON-FATAL exception is skipped — its rows do not enter the
        # pooled estimate, a warning is logged (the
        # ``cross_val_score(error_score=np.nan)`` contract: a failed fold must
        # not abort the trial). Fatal exception types (MemoryError /
        # RuntimeError / InvalidDataError) are NOT swallowed here — they
        # propagate to ``handle_failure`` and keep the circuit breaker active.
        try:
            fold_estimator = clone(estimator)
            fold_estimator.fit(X_tr, y_tr)
            fold_pred = np.asarray(fold_estimator.predict(X_te)).reshape(-1)
        except (MemoryError, RuntimeError, InvalidDataError):
            raise
        except Exception as err:  # noqa: BLE001 — per-fold failure, cross_val_score(error_score=nan) semantics
            n_failed_folds += 1
            _warn_failed_fold(n_folds_total, err, "fit or predict")
            continue
        fold_true = np.asarray(y_te, dtype=float).reshape(-1)
        if not poolable:
            # The scorer does not expose a raw metric function: pooling is
            # impossible, collect the per-fold values as the fallback. A
            # non-fatal scoring failure excludes the fold, like a fit failure.
            try:
                fold_scores.append(float(scorer(fold_estimator, X_te, y_te)))
            except (MemoryError, RuntimeError, InvalidDataError):
                raise
            except Exception as err:  # noqa: BLE001 — per-fold scoring, same cross_val_score(error_score=nan) semantics
                n_failed_folds += 1
                _warn_failed_fold(n_folds_total, err, "score")
            continue
        idx = np.asarray(test_idx)
        pooled_y[idx] = fold_true
        pooled_pred[idx] = fold_pred
        fold_pairs.append((fold_true, fold_pred))
    if n_folds_total > 0 and n_failed_folds == n_folds_total:
        raise ValueError(
            f"All {n_folds_total} fold(s) failed during evaluation — the "
            "pooled estimate is impossible"
        )
    if poolable:
        try:
            return pooled_scorer_value(scorer, pooled_y, pooled_pred)
        except ValueError:
            # The pooled value is impossible (all rows masked out, or the raw
            # metric function raised on the pooled vector): fall back to the
            # mean of the per-fold values, scored lazily from the RETAINED
            # per-fold pairs. The scorer is invoked only now — not on the
            # common path — so per-fold scoring noise (e.g. LOO+R², where
            # every fold R² is undefined) does not pollute the logs, and NO
            # split is iterated or fold re-trained a second time.
            warn_pooled_fallback_once(
                algo_name=algo_name or "unknown",
                phase_name=phase_name or "untitled_phase",
                reason="the pooled value over the collected OOF rows cannot "
                "be computed (no valid pairs, or the raw metric function "
                "raised on the pooled vector)",
            )
            for yt, yp in fold_pairs:
                try:
                    fold_scores.append(pooled_scorer_value(scorer, yt, yp))
                except ValueError:
                    pass
            return float(np.mean(fold_scores)) if fold_scores else float("nan")
    warn_pooled_fallback_once(
        algo_name=algo_name or "unknown",
        phase_name=phase_name or "untitled_phase",
        reason="the scorer does not expose a raw metric function "
        "(_score_func/_sign/_kwargs)",
    )
    return float(np.mean(fold_scores)) if fold_scores else float("nan")


# ═════════════════════ PUBLIC: optimize() ═══════════════════════════════════
class _DistributionRecorder:
    """Record the canonical distributions of the phase search space.

    A lightweight stand-in for an Optuna ``Trial`` that only collects the
    distributions of every ``suggest_*`` call (``suggest_int`` /
    ``suggest_float`` / ``suggest_categorical``). Running ``space_fn`` on the
    recorder yields the distributions the current phase would produce — they
    are compared against the distributions of ``previous_trials`` BEFORE
    ``study.add_trials`` so that an incompatible history fails fast with a
    clear message instead of a silent skip (or a confusing error during the
    first trial of the new phase).
    """

    def __init__(self) -> None:
        """Initialize an empty distribution recorder."""
        self.distributions: dict[str, Any] = {}
        # Trial-compatible attributes that user space functions may read
        # (e.g. ``trial.number`` to alternate between good/bad points).
        self.number = -1

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
        """Record an ``IntDistribution`` and return ``low`` (a valid value)."""
        self.distributions[name] = optuna.distributions.IntDistribution(
            low=low, high=high, step=step
        )
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Record a ``FloatDistribution`` and return ``low`` (a valid value)."""
        self.distributions[name] = optuna.distributions.FloatDistribution(
            low=low, high=high, step=step, log=log
        )
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Record a ``CategoricalDistribution`` and return the first choice."""
        self.distributions[name] = optuna.distributions.CategoricalDistribution(
            choices=list(choices)
        )
        return choices[0]


class _OptimizeRunner:
    """A single hyperparameter optimisation run, split into steps.

    ``optimize()`` remains the public entry point with its original signature
    and delegates all the work to this class. The monolithic logic of the
    original function is split into isolated steps:

        run()
         ├─ resolve_validation()
         ├─ build_trial_pipeline()               (before the search space:
         │                                        the complexity guard needs p)
         ├─ resolve_search_space()
         ├─ handle_pruning()
         ├─ transfer_previous_trials()        (TPE history warm-start)
         ├─ evaluate_trial() / handle_failure()  (in the Optuna loop)
         ├─ select_best_trial()
         └─ _fit_best_model()  (skipped when ``fit_best_model=False``)

    TPE history transfer: when ``previous_trials`` is passed, the trials are
    added to the study with ``study.add_trials`` AFTER ``create_study`` and
    BEFORE ``enqueue_trial``/``optimize``. The boundary of the current phase's
    own trials (``phase_start_number``) is fixed right after the transfer, so
    ``select_best_trial`` counts and ranks ONLY the trials of the current
    phase — the transferred history feeds the TPE sampler but cannot mask a
    total failure of the current phase (the no-fallback refine contract).

    The run state is stored in instance attributes: each step fills in its own
    fields, and subsequent steps (as well as the Optuna objective) read them
    without long argument chains.
    """

    MAX_FATAL_FAILURES = 5
    """Limit of consecutive fatal failures before the algorithm is disqualified."""

    def __init__(
        self,
        *,
        algo_name: str,
        X: Any,
        y: Any,
        data_oversampling: bool,
        data_oversampling_multiplier: float,
        data_oversampling_algorithm: str,
        metric: str,
        val_method: ValidationStrategy | str,
        validation_strategy: ValidationStrategy | str | None,
        n_folds: int,
        n_trials: int,
        random_state: int | None,
        sampler_seed: int | None,
        train_test_split_test_size: float,
        space_overrides: dict[str, Callable[[Trial], dict[str, Any]]] | None,
        initial_params: dict[str, Any] | None,
        previous_trials: list[FrozenTrial] | None = None,
        preprocessor: Any | None,
        categorical_features: list[str] | None,
        numerical_features: list[str] | None,
        encoding: EncodingStrategy | None,
        preprocessing_override: PreprocessingOverride | dict[str, Any] | None,
        pruning: dict[str, Any] | None,
        high_cardinality_threshold: int | None,
        high_cardinality_encoding: EncodingStrategy | None,
        hashing_n_components: int,
        target_encoding_smoothing: float,
        target_encoding_fallback: float | None,
        feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None,
        complexity_guard_cfg: ComplexityGuardConfig | dict[str, Any] | None = None,
        user_hyperparameter_keys: list[str] | None = None,
        fit_best_model: bool = True,
        phase_name: str | None = None,
    ) -> None:
        """Store the parameters of a single optimisation run.

        Args:
            All arguments mirror the identically named ``optimize()``
            parameters — see its docstring for the semantics.
        """
        self.algo_name = algo_name
        self.X = X
        self.y = y
        self.metric = metric
        # Whether the metric is non-additive (RMSE, MSE, R², nRMSE) and must
        # therefore be evaluated over the pooled out-of-fold vector instead of
        # the average of per-fold values (required for LOO, applied everywhere
        # with multiple folds). MAE and similar per-sample means stay on the
        # fold-average path (their fold mean equals the pooled value).
        self.pool_metric = is_non_additive_metric(metric)
        self.val_method = val_method
        self.validation_strategy = validation_strategy
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        # The seed of the TPE sampler. Deliberately separate from
        # ``random_state`` (which drives the data splits and the preprocessor):
        # each HPO phase may use a different ``sampler_seed`` so that TPE
        # startup points do not repeat, while the folds/splits stay the same
        # (RANDOM_STATE) — otherwise every phase would re-evaluate models on
        # other folds, breaking score monotonicity and mixing incomparable
        # estimates in the transferred TPE history. ``None`` (the default for
        # direct callers) falls back to ``random_state`` to preserve the old
        # behavior.
        self.sampler_seed = sampler_seed
        self.train_test_split_test_size = train_test_split_test_size
        self.space_overrides = space_overrides
        self.initial_params = initial_params
        self.previous_trials = previous_trials
        self.preprocessor = preprocessor
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoding = encoding
        self.preprocessing_override = preprocessing_override
        self.pruning = pruning
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.hashing_n_components = hashing_n_components
        self.target_encoding_smoothing = target_encoding_smoothing
        self.target_encoding_fallback = target_encoding_fallback
        self.feature_selection_cfg = feature_selection_cfg
        # The adaptive complexity guard config (trees/boosting on small data).
        # ``None`` — the guard is disabled (defaults to the off behavior for
        # direct ``optimize`` callers); the training engine passes the config
        # from ``general.complexity_guard``.
        if complexity_guard_cfg is None:
            self.complexity_guard_cfg: ComplexityGuardConfig | None = None
        elif isinstance(complexity_guard_cfg, ComplexityGuardConfig):
            self.complexity_guard_cfg = complexity_guard_cfg
        else:
            self.complexity_guard_cfg = ComplexityGuardConfig.from_dict(
                complexity_guard_cfg
            )
        # Names of parameters set explicitly by the user
        # (``algo.hyperparameters``); the complexity guard treats them as
        # "user ranges" (intersected, with ``user_range_policy`` on conflict).
        self.user_hyperparameter_keys = list(user_hyperparameter_keys or [])
        # Number of features after preprocessing (what the model sees).
        # Filled in build_trial_pipeline (falls back to the raw column count).
        self.n_features_eff: int = 1
        # Number of ORIGINAL columns (a categorical column counts as 1,
        # before the one-hot expansion). Used only by the complexity guard
        # for the size-mode ratio ``n / p``; filled in build_trial_pipeline.
        self.n_features_size: int = 1
        # Whether to train the best model on all data after the Optuna loop
        # (``_fit_best_model``). Set to False when the caller only needs the
        # best parameters and score (e.g. the training engine's HPO phase):
        # the full-data fit is then skipped and the ``model`` field of the
        # returned ``OptimizeResult`` is None.
        self.fit_best_model = fit_best_model
        # The pipeline phase name (e.g. 'coarse'/'refine') for the fallback
        # WARNING deduplication: ``None`` for direct ``optimize()`` callers.
        self.phase_name = phase_name

        # Names of columns dropped by cleaning uninformative features.
        # Filled in build_trial_pipeline for diagnostics; empty by default.
        self.dropped_features_: list[str] = []

        # Oversampling config for pipeline assembly (single point of truth).
        self.oversampling_config: dict[str, Any] = {
            "active": data_oversampling,
            "params": {
                "multiplier": data_oversampling_multiplier,
                "algorithm": data_oversampling_algorithm,
            },
        }

        # ── Fields populated by the run() steps ───────────────────────────────
        self.algo: str = ""
        self.fs_cfg = FeatureSelectionCfg()
        self.fs_mode = self.fs_cfg.mode
        self.is_isotonic = False
        self.n_samples_eff: int = 0
        self.val_method_eff: str = ""
        self.cv_obj: Any = None
        self.auto_decision: dict[str, Any] | None = None
        self.resolved_via_auto = False
        self.effective_n_folds = self.n_folds
        # A placeholder until the search space is resolved (step 2); replaced
        # in resolve_search_space(). run() always calls resolve_search_space
        # before the Optuna loop, so the real placeholder call is impossible.
        self.space_fn: Callable[[Trial], dict[str, Any]] = lambda trial: {}
        # The resolved dict search space (parameter name -> SearchSpaceEntry or
        # constant) for dict-based spaces; empty for function/callable spaces.
        # Used to warn when the best trial lies at a range boundary.
        self.search_space_dict: dict[str, Any] = {}
        self.scorer: Any = None
        self.pruner: optuna.pruners.BasePruner | None = None
        self.pruning_active = False
        self.study: optuna.Study
        # The trial-number boundary between the transferred TPE history and the
        # current phase's own trials: fixed in run() right after
        # study.add_trials(previous_trials) and before enqueue_trial/optimize.
        self.phase_start_number = 0
        # Trials of the current phase only (number >= phase_start_number);
        # filled in run() after the Optuna loop and returned in OptimizeResult.
        self.phase_trials: list[FrozenTrial] = []
        self.consecutive_fatal_failures = 0
        self.best_params: dict[str, Any]
        self.best_score: float
        self.clean_model_params: dict[str, Any]
        self.best_apply_fs: bool
        # Log records of the adaptive complexity guard, filled in
        # resolve_search_space; travels with OptimizeResult for the report.
        self.complexity_guard_logs: list[dict[str, Any]] = []

    def run(self) -> OptimizeResult | None:
        """Run the full optimisation cycle, split into steps.

        Returns:
            OptimizeResult: The result of the phase — the final model, the
                best params, the best score, and the trials of the current
                phase ONLY (transferred ``previous_trials`` are excluded).
            None: The "no valid result" signal — ``n_trials=0`` or no
                COMPLETED trial of the current phase. Transferred history
                does NOT count toward the current phase validity.
        """
        if not self._validate_n_trials():
            return None

        self.resolve_validation()
        # The preprocessor must be built BEFORE the search space is resolved:
        # the adaptive complexity guard needs ``p`` — the number of features
        # after preprocessing (one-hot, missing-value indicators).
        self.build_trial_pipeline()
        self.resolve_search_space()
        self.handle_pruning()

        # If early stopping is disabled, explicitly disable any pruner
        # (NopPruner) so that trials are guaranteed to run to completion.
        # The TPE sampler uses ``sampler_seed`` (per-phase, so startup points
        # do not repeat between HPO phases); the data splits and the
        # preprocessor keep ``random_state`` (RANDOM_STATE from the engine),
        # so every phase evaluates models on the same folds.
        sampler_seed = (
            self.sampler_seed if self.sampler_seed is not None else self.random_state
        )
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=sampler_seed),
            pruner=self.pruner if self.pruning_active else optuna.pruners.NopPruner(),
        )
        # TPE history transfer BEFORE enqueue_trial/optimize: the transferred
        # trials feed the TPE sampler (incl. the n_startup_trials window) but
        # are NOT part of the current phase's own trial set (the boundary
        # phase_start_number is fixed right after add_trials).
        self._transfer_previous_trials()
        self.phase_start_number = len(self.study.get_trials(deepcopy=False))
        # If there are parameters from the previous phase, enqueue them as the first trial
        if self.initial_params is not None:
            self.study.enqueue_trial(self.initial_params)
            log.info(
                "Enqueued initial params from previous phase: %s", self.initial_params
            )
        try:
            self.study.optimize(self._objective, n_trials=self.n_trials)
        except InvalidAlgorithmError:
            log.warning(
                "Algorithm %s disqualified after %d consecutive fatal failures",
                self.algo,
                self.MAX_FATAL_FAILURES,
            )
            raise

        # The current phase's own trials (transferred ones have smaller
        # numbers and are excluded). Returned with the result so the caller
        # can warm-start the next phase with them.
        self.phase_trials = [
            t
            for t in self.study.get_trials(deepcopy=False)
            if t.number >= self.phase_start_number
        ]
        if not self.select_best_trial():
            return None
        self._log_best_result()
        if not self.fit_best_model:
            # HPO-only mode (e.g. the training engine's phases): the caller
            # needs only the best parameters and score — the final model is
            # trained later by its own trainer. The redundant full-data fit
            # is skipped, saving one full training per algorithm per phase;
            # the model slot of the OptimizeResult stays None.
            return OptimizeResult(
                model=None,
                params=self.best_params,
                score=self.best_score,
                trials=self.phase_trials,
                complexity_guard=self.complexity_guard_logs or None,
            )
        return OptimizeResult(
            model=self._fit_best_model(),
            params=self.best_params,
            score=self.best_score,
            trials=self.phase_trials,
            complexity_guard=self.complexity_guard_logs or None,
        )

    # ─────────────────── run steps ─────────────────────────────────────────

    def _transfer_previous_trials(self) -> None:
        """Transfer the TPE history of the previous phase into the study.

        Fails fast (``HPOHistoryError``) instead of silently skipping the
        history:
        - when some transferred trial has a parameter whose distribution does
          not match the current search space (a search space change between
          phases would break the TPE state in an obscure way during the first
          trial of the new phase);
        - when the current search space cannot be reproduced by the
          distribution pre-check (an exotic custom ``space_fn``) — the
          transfer cannot be validated, and silently dropping the history
          would degrade refine to a cold run (the issue this feature fixes);
        - an empty list is a no-op (``previous_trials=[]`` is allowed).
        """
        if not self.previous_trials:
            return
        canonical = self._canonical_distributions()
        if canonical is None:
            # The search space cannot be reproduced by the pre-check (an
            # exotic custom ``space_fn``, e.g. one using legacy ``suggest_*``
            # methods the recorder does not implement). Transferring the
            # history blind would risk a silent ``study.add_trials`` desync,
            # and skipping the transfer would silently degrade refine to a
            # cold run — both violate the fail-fast contract: raise.
            raise HPOHistoryError(
                f"Cannot transfer TPE history for algorithm '{self.algo}': "
                "the search space cannot be reproduced for the distribution "
                "compatibility check (the space function uses constructs the "
                "check does not support). TPE history transfer requires a "
                "search space that can be replayed for validation."
            )
        for trial in self.previous_trials:
            for name, dist in trial.distributions.items():
                expected = canonical.get(name)
                if expected is None or expected != dist:
                    raise HPOHistoryError(
                        f"Cannot transfer trial #{trial.number} of algorithm "
                        f"'{self.algo}': parameter '{name}' has distribution "
                        f"{dist}, incompatible with the current search space "
                        f"({expected!r}). TPE history transfer requires the "
                        f"same search space between HPO phases."
                    )
        self.study.add_trials(self.previous_trials)
        log.info(
            "Transferred %d trial(s) from the previous phase", len(self.previous_trials)
        )

    def _canonical_distributions(self) -> dict[str, Any] | None:
        """Compute the canonical distributions of the current phase space.

        Runs ``space_fn`` (plus the feature selection suggestion) on a
        ``_DistributionRecorder`` — the distributions the phase WOULD produce
        for its trials. If the space function cannot be reproduced (a custom
        exotic ``space_fn``, e.g. one using legacy ``suggest_*`` methods the
        recorder does not implement), ``None`` is returned — the caller
        (``_transfer_previous_trials``) fails fast with ``HPOHistoryError``:
        the history transfer cannot be validated, and silently dropping it
        would degrade refine to a cold run.

        Returns:
            dict[str, Any]: Parameter name → Optuna distribution; ``None`` —
                the space cannot be reproduced (the caller fails fast).
        """
        try:
            recorder = _DistributionRecorder()
            # The recorder is a Trial-compatible stand-in: it implements the
            # suggest_* methods (and the ``number`` attribute) that space_fn
            # and _resolve_apply_fs call.
            trial_like = cast(Trial, recorder)
            self.space_fn(trial_like)
            self._resolve_apply_fs(trial_like)
        except Exception as err:  # noqa: BLE001 — the pre-check is best-effort
            log.warning(
                "Cannot reproduce the search space for the history transfer "
                "check of algorithm '%s': %s. Skipping the distribution "
                "pre-check.",
                self.algo,
                err,
            )
            return None
        return recorder.distributions

    def resolve_validation(self) -> None:
        """Resolve the validation strategy and related quantities (step 1).

        Performs input normalization (the ``validation_strategy`` alias), data
        and algorithm availability checks, normalization of the feature
        selection configuration, and picks the CV strategy via ``make_cv``.
        The results are stored in instance attributes for later steps.
        """
        # the alias takes precedence
        if self.validation_strategy is not None:
            self.val_method = self.validation_strategy

        self.algo = self.algo_name.lower()
        _validate_data(self.X, self.y)

        _get_estimator(self.algo)

        # Normalize the feature selection configuration: None -> default
        # (mode='disabled'), dict -> FeatureSelectionCfg (validated at run
        # time), FeatureSelectionCfg object -> as is. Isotonic regression is
        # strictly univariate and uses its own IsotonicDataTransformer, so
        # selection is forcibly disabled for it regardless of the mode.
        fs_cfg = self.feature_selection_cfg
        if fs_cfg is None:
            self.fs_cfg = FeatureSelectionCfg()
        elif isinstance(fs_cfg, FeatureSelectionCfg):
            self.fs_cfg = fs_cfg
        elif isinstance(fs_cfg, dict):
            try:
                self.fs_cfg = FeatureSelectionCfg.model_validate(fs_cfg)
            except Exception as err:
                raise HyperoptError(f"Invalid feature_selection_cfg: {err}") from err
        else:
            raise TypeError(
                "feature_selection_cfg must be a FeatureSelectionCfg, dict or None, "
                f"got {type(fs_cfg).__name__}"
            )
        self.fs_mode = self.fs_cfg.mode
        self.is_isotonic = resolve_algorithm_name(self.algo) == "isotonic_regression"
        log.info(
            "Feature selection mode for algorithm '%s': %s",
            self.algo,
            self.fs_mode.value,
        )

        # Determine how many rows the model actually "sees" during training
        # inside CV/Split, and resolve the validation strategy (incl. 'auto').
        n_samples = len(self.y)
        n_features = self.X.shape[1]
        self.n_samples_eff = get_effective_train_size(
            n_total=n_samples,
            strategy=self.validation_strategy or self.val_method,
            n_folds=self.n_folds,
            test_size=self.train_test_split_test_size,
            n_features=n_features,
        )
        self.val_method_eff, self.cv_obj, self.auto_decision = make_cv(
            n_samples,
            val_method=self.val_method,
            n_folds=self.n_folds,
            random_state=self.random_state,
            test_size=self.train_test_split_test_size,
            n_features=n_features,
        )
        # 'auto' is resolved exactly once (here, in make_cv). The resulting
        # decision is passed to iter_splits so that the test size is not
        # recomputed and cannot diverge between n_samples_eff and the actual
        # split.
        self.resolved_via_auto = norm_val_method(self.val_method) == "auto"

        # A single fold count for the early stopping branch (k_fold). The
        # 'auto' strategy could resolve to kfold with a number of folds k that
        # differs from the original n_folds: in that case the pruning branch
        # must use the computed k, otherwise iter_splits inside
        # _evaluate_with_intermediate_reports would fall back to the default
        # n_folds and the evaluation would diverge from
        # cross_val_score(cv=cv_obj).
        if self.val_method_eff == "k_fold":
            # Take the fold count from the ready cv_obj (for k_fold it is
            # guaranteed to be non-None — make_cv always creates a KFold in
            # this branch). One formula covers auto→kfold (k from the
            # decision), the 'auto' fallback without P (clamped to max(2, k)),
            # and an explicit k_fold.
            assert self.cv_obj is not None
            self.effective_n_folds = int(self.cv_obj.get_n_splits())
        else:
            # Explicit loo / train_test_split: the fold count is not used.
            self.effective_n_folds = self.n_folds

    def _resolve_complexity_guard(
        self, space: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Apply the adaptive complexity guard to a dict search space.

        The guard depends on ``n`` (the effective training-fold rows,
        ``self.n_samples_eff``) and the feature counts: the size-mode ratio
        uses ``self.n_features_size`` — the number of ORIGINAL columns (a
        categorical column counts as 1, before one-hot), while the rules for
        ``max_features``/``colsample_bytree`` use ``self.n_features_eff`` —
        the number of features after preprocessing. Both are computed once
        per phase and are the same in every HPO phase (deterministic between
        phases) and in the subprocesses (identical inputs -> identical
        results), so the TPE history transfer between phases stays valid.

        When the guard is disabled (``complexity_guard_cfg is None``) the
        space is returned unchanged and no records are produced — the exact
        legacy behavior for direct ``optimize()`` callers.

        Args:
            space: The already physically-clipped search space.

        Returns:
            tuple[dict[str, Any], list[dict[str, Any]]]: The (possibly
                adjusted) space and the guard log records.
        """
        cfg = self.complexity_guard_cfg
        if cfg is None or cfg.mode == "off":
            return space, []
        user_keys = set(self.user_hyperparameter_keys or ())
        adjusted, logs = apply_complexity_guard(
            space,
            self.algo,
            n=self.n_samples_eff,
            p=self.n_features_eff,
            config=cfg,
            user_space_keys=user_keys,
            p_size=self.n_features_size,
        )
        return adjusted, logs

    def _log_complexity_guard(self) -> None:
        """Log the adaptive complexity guard records at INFO level.

        One record per (algorithm, phase, parameter): "was → became" with
        the reason, e.g. ``max_depth 3–20 → 2–3: n=15, p=5, mode=small,
        leaf_min=2``.
        """
        cfg_mode = (
            self.complexity_guard_cfg.mode if self.complexity_guard_cfg else "off"
        )
        cfg_policy = (
            self.complexity_guard_cfg.user_range_policy
            if self.complexity_guard_cfg
            else "user_wins"
        )
        for record in self.complexity_guard_logs:
            before = record["before"]
            after = record["after"]

            def fmt(v: Any) -> str:
                if v is None:
                    return "n/a"
                return str(v) if isinstance(v, int) else f"{v:g}"

            if record.get("applied"):
                message = (
                    f"[complexity_guard] {record['algo']} {record['param']} "
                    f"{fmt(before[0])}–{fmt(before[1])} → "
                    f"{fmt(after[0])}–{fmt(after[1])}: n={record['n']}, "
                    f"p={record['p']}, p_size={fmt(record['p_size'])}, "
                    f"mode={record['size_mode']}, "
                    f"leaf_min={fmt(record['leaf_min'])}, level={record['level']}"
                )
                if record.get("conflict"):
                    message += " (user range conflict, engine wins)"
                if record.get("fixed"):
                    message += " (parameter fixed)"
                log.info("%s: %s", message, record["reason"])
            elif record.get("conflict"):
                rule_range = record.get("rule_range") or [None, None]
                if cfg_mode == "warn":
                    # In the warn mode nothing is applied at all — report the
                    # conflict without claiming that the user range wins.
                    log.warning(
                        "[complexity_guard] %s %s user range %s–%s conflicts "
                        "with the guard range %s–%s; computed but not applied "
                        "(mode='warn', user_range_policy='%s')",
                        record["algo"],
                        record["param"],
                        fmt(before[0]),
                        fmt(before[1]),
                        fmt(rule_range[0]),
                        fmt(rule_range[1]),
                        cfg_policy,
                    )
                else:
                    log.warning(
                        "[complexity_guard] %s %s user range %s–%s conflicts "
                        "with the guard range %s–%s; keeping the user range "
                        "(user_range_policy='%s')",
                        record["algo"],
                        record["param"],
                        fmt(before[0]),
                        fmt(before[1]),
                        fmt(rule_range[0]),
                        fmt(rule_range[1]),
                        cfg_policy,
                    )
            else:
                # Computed but not applied (e.g. mode='warn'): report the
                # would-be change without claiming a conflict.
                log.info(
                    "[complexity_guard] %s %s %s–%s → %s–%s: computed but "
                    "not applied (mode=%s): %s",
                    record["algo"],
                    record["param"],
                    fmt(before[0]),
                    fmt(before[1]),
                    fmt(after[0]),
                    fmt(after[1]),
                    cfg_mode,
                    record["reason"],
                )

    def _normalize_space_dict(self, space_dict: dict[str, Any]) -> dict[str, Any]:
        """Re-resolve inferred range types in a dict search space.

        The same normalization the Config path applies (see
        ``Config._normalize_hyperparameter_types``): a range whose type was
        inferred from the boundary numbers (``[1, 50]`` in the list shorthand,
        two-element lists passed directly to ``space_overrides``) is re-typed
        against the parameter type (the default search space or the estimator
        signature). So ``space_overrides={"svr": {"C": [1, 50]}}`` searches
        ``C`` as a real range (``float_log``), not as an integer-only range —
        the exact issue #127 bug. A parameter whose default space entry is
        ``float_log`` (e.g. ``alpha_1`` of an ARD regression) is likewise
        searched on a logarithmic scale without an explicit type. Raw lists
        (not yet parsed into ``SearchSpaceEntry`` objects) are parsed first;
        explicitly typed entries and constants pass through unchanged.

        Args:
            space_dict: The parameter name → value mapping (``SearchSpaceEntry``
                objects, raw lists, or constants).

        Returns:
            dict[str, Any]: A new dict with the same structure: every numeric
                key holds a ``SearchSpaceEntry`` (raw lists are parsed),
                inferred range types re-resolved against the parameter type.
        """
        normalized: dict[str, Any] = {}
        for param, value in space_dict.items():
            if isinstance(value, SearchSpaceEntry):
                entry = value
            elif isinstance(value, list):
                # A raw list from a direct call: parse it the same way the
                # Config path does. Constants (scalars) are never lists, so a
                # list here is always a search-space shorthand.
                entry = SearchSpaceEntry.model_validate(value)
            else:
                # A constant value — no normalization possible.
                normalized[param] = value
                continue
            normalized[param] = normalize_hyperparameter_entry(self.algo, param, entry)
        return normalized

    def resolve_search_space(self) -> None:
        """Resolve the hyperparameter search space and the scorer (step 2).

        Supports the dynamic KNN space (depends on the effective sample
        size), adaptive SVR epsilon (bounds scaled by the spread of
        ``y_train``), direct function overrides, and dict-based YAML
        configurations (SearchSpaceEntry). Stores ``space_fn`` and ``scorer``
        in instance attributes.

        Raises:
            HyperoptError: If no search space is defined for the selected
                algorithm.
        """
        base_space_fn: Callable[[Trial], dict[str, Any]] | None = None
        base_space_dict: dict[str, Any] | None = None
        if self.algo == "knn":
            base_space_fn = _make_knn_space(self.n_samples_eff)
        elif resolve_algorithm_name(self.algo) == "gaussian_process_regression":
            # GPR without an external config (including a direct
            # optimize("gpr", X, y) call): the DEFAULT_SPACES search space
            # (small: kernel family only).
            base_space_dict = DEFAULT_SPACES["gaussian_process_regression"].copy()
            base_space_fn = partial(_apply_dynamic_space, space_dict=base_space_dict)
        elif resolve_algorithm_name(self.algo) == "isotonic_regression":
            # Isotonic without an external config (including a direct
            # optimize("isotonic_regression", X, y) call): a data-dependent
            # space. ``feature_index`` selects the column fed to the strictly
            # univariate model (consumed by ``IsotonicDataTransformer``) and
            # must cover the post-preprocessing feature count; ``increasing``
            # is the single IsotonicRegression constructor parameter searched.
            max_feature_index = max(
                0,
                int(
                    self.n_features_eff
                    if self.preprocessor is not None
                    else self.X.shape[1]
                )
                - 1,
            )
            base_space_dict = {
                "feature_index": SearchSpaceEntry.model_validate(
                    [0, max_feature_index, "int"]
                ),
                "increasing": SearchSpaceEntry.model_validate(
                    [[True, False], "categorical"]
                ),
            }
            base_space_fn = partial(_apply_dynamic_space, space_dict=base_space_dict)

        # Priority 1: Direct overrides (functions)
        # Priority 2: Dynamic config from YAML (dict with SearchSpaceEntry)
        external_config = (self.space_overrides or {}).get(self.algo)

        # If a function was passed (the old mechanism), use it
        if callable(external_config):
            space_fn: Callable[[Trial], dict[str, Any]] | None = external_config
        # If a dict was passed (new YAML mechanism), create a wrapper
        elif isinstance(external_config, dict):
            # Remap legacy names once (e.g. ARDRegression n_iter → max_iter)
            # BEFORE the Optuna loop: otherwise clean_hyperparameters would
            # log an identical "Remapped ..." line for every trial.
            space_dict = self._normalize_space_dict(
                remap_legacy_param_names(self.algo, external_config)
            )
            # SVR: adaptive epsilon bounds from the y_train spread, if the
            # user did not set epsilon explicitly (prepare_search_space omits
            # the default epsilon from DEFAULT_SPACES, so its absence in the
            # dict means "not overridden"). The bounds are computed once
            # BEFORE the Optuna loop.
            if self.algo == "svr" and "epsilon" not in space_dict:
                space_dict = {
                    **space_dict,
                    "epsilon": _svr_adaptive_epsilon_entry(self.y),
                }
            clipped_config = clip_search_space(space_dict, self.n_samples_eff)
            clipped_config, guard_logs = self._resolve_complexity_guard(clipped_config)
            self.complexity_guard_logs = guard_logs
            self._log_complexity_guard()
            space_fn = partial(_apply_dynamic_space, space_dict=clipped_config)
            self.search_space_dict = clipped_config
        else:
            # SVR without an external config (including a direct
            # optimize("svr", X, y) call): the default space with adaptive
            # epsilon instead of the blind [1e-3, 1.0] range from
            # DEFAULT_SPACES.
            if self.algo == "svr":
                base_space_dict = _make_svr_space_dict(self.y)
                base_space_fn = partial(
                    _apply_dynamic_space, space_dict=base_space_dict
                )
            space_fn = base_space_fn
        if space_fn is None:
            raise HyperoptError(f"No search space for algorithm '{self.algo}'")

        self.space_fn = space_fn
        if base_space_dict is not None:
            self.search_space_dict = base_space_dict
        self.scorer = _build_scorer(self.metric)

    def build_trial_pipeline(self) -> None:
        """Build the preprocessor for the HPO phase (step 3).

        A single point of preprocessor assembly: categories -> one-hot (or
        ordinal if encoding is set), numerics -> the preprocessing preset
        selected automatically for the algorithm. The same logic is used in
        the final training (trainer.ModelTrainer), so HPO and the final fit
        stay consistent. The ready pipeline factory is ``assemble_estimator``.

        Cleaning of uninformative features is applied here too — at the point
        where ``detect_feature_types`` runs, before the HPO folds and before
        the final training, so that both phases use a consistent feature set.
        Cleaning runs only in the standalone branch (``self.preprocessor is
        None``): if the user passed a ready preprocessor, it has already fixed
        its feature set, and ``self.X`` must not be silently changed. For
        ``np.ndarray`` without column names, cleaning is skipped (a
        limitation shared with ``detect_feature_types``); the list of dropped
        columns is stored in ``self.dropped_features_``.
        """
        self.dropped_features_ = []
        if self.preprocessor is None and isinstance(self.X, pd.DataFrame):
            cleaned = _clean_uninformative_features(self.X)
            self.dropped_features_ = [
                col for col in self.X.columns if col not in cleaned.columns
            ]
            if self.dropped_features_:
                self.X = cleaned
                # Explicitly passed column lists are reconciled with the
                # cleaned feature set: otherwise build_preprocessor would fail
                # on missing columns (Unknown feature name).
                if self.categorical_features is not None:
                    self.categorical_features = [
                        col
                        for col in self.categorical_features
                        if col in self.X.columns
                    ]
                if self.numerical_features is not None:
                    self.numerical_features = [
                        col for col in self.numerical_features if col in self.X.columns
                    ]
        if self.preprocessor is None:
            # Guard for an unknown algorithm: the preprocessor cannot be
            # assembled (``resolve_preprocessing_preset`` would raise). The
            # search-space resolution raises the proper ``HyperoptError``
            # afterwards; skipping the preprocessor here preserves the exact
            # error contract for unknown algorithms.
            if resolve_algorithm_name(self.algo) not in AVAILABLE_ALGORITHMS:
                self.n_features_eff = max(1, int(self.X.shape[1]))
                self.n_features_size = max(1, int(self.X.shape[1]))
                return
            encoding_strategy: EncodingStrategy = self.encoding or "one_hot"
            # Automatic preset selection + application of an explicit override.
            # The preset is resolved once before training (performance) and
            # logged for observability.
            preset = resolve_preprocessing_preset(
                self.algo, self.preprocessing_override
            )
            log.info(
                "Resolved preprocessing preset for algorithm '%s': %s",
                self.algo,
                preset,
            )
            if (
                self.categorical_features is not None
                or self.numerical_features is not None
            ):
                # Explicitly passed column lists (the main path from
                # training_engine)
                if isinstance(self.X, pd.DataFrame):
                    self.preprocessor = build_preprocessor(
                        list(self.X.columns),
                        self.categorical_features or [],
                        self.numerical_features or [],
                        encoding=encoding_strategy,
                        imputation_strategy=preset.imputation_strategy,
                        scaling=preset.scaling,
                        high_cardinality_threshold=self.high_cardinality_threshold,
                        high_cardinality_encoding=self.high_cardinality_encoding,
                        hashing_n_components=self.hashing_n_components,
                        target_encoding_smoothing=self.target_encoding_smoothing,
                        target_encoding_fallback=self.target_encoding_fallback,
                        random_state=self.random_state,
                        force_dense_output=requires_dense_input(self.algo),
                        add_indicator=_missing_indicator_enabled(self.algo),
                    )
                else:
                    log.warning(
                        "categorical/numerical_features provided, but X is not a "
                        "pandas.DataFrame — skipping automatic preprocessor."
                    )
            elif isinstance(self.X, pd.DataFrame):
                # Autodetect categories (default behavior when called outside
                # the engine). The preprocessor is built when ANY features
                # (categorical or numerical) are present: this guarantees that
                # on a purely numerical DataFrame the HPO phase applies
                # scaling just like the final ModelTrainer.
                cats, nums = detect_feature_types(self.X)
                if cats or nums:
                    self.preprocessor = build_preprocessor(
                        list(self.X.columns),
                        cats,
                        nums,
                        encoding=encoding_strategy,
                        imputation_strategy=preset.imputation_strategy,
                        scaling=preset.scaling,
                        high_cardinality_threshold=self.high_cardinality_threshold,
                        high_cardinality_encoding=self.high_cardinality_encoding,
                        hashing_n_components=self.hashing_n_components,
                        target_encoding_smoothing=self.target_encoding_smoothing,
                        target_encoding_fallback=self.target_encoding_fallback,
                        random_state=self.random_state,
                        force_dense_output=requires_dense_input(self.algo),
                        add_indicator=_missing_indicator_enabled(self.algo),
                    )
            else:
                log.warning(
                    "X is not a pandas.DataFrame and categorical_features/"
                    "numerical_features were not provided — categorical columns "
                    "(if any) will be treated as numeric."
                )

        # The number of features the model actually sees after preprocessing
        # (including one-hot and missing-value indicators). Used by the
        # adaptive complexity guard as ``p`` for the rules that depend on
        # the feature count (``max_features``, ``colsample_bytree``).
        # Computed deterministically once per phase — the same value in every
        # HPO phase and in the subprocesses.
        self.n_features_eff = self._estimate_n_features_after_preprocessing()
        # The number of ORIGINAL columns (a categorical column counts as 1,
        # before the one-hot expansion). Used by the adaptive complexity
        # guard for the size-mode ratio ``n / p`` only. Without a preprocessor
        # (np.ndarray input without explicit column lists) the model sees
        # exactly the original columns, so the size count is aligned with the
        # post-preprocessing count — this preserves the legacy behavior of
        # direct optimize() calls (no one-hot expansion, the guard sees the
        # same p it always did).
        self.n_features_size = max(
            1,
            int(self.X.shape[1])
            if self.preprocessor is not None
            else self.n_features_eff,
        )

    def _estimate_n_features_after_preprocessing(self) -> int:
        """Estimate the number of features after the preprocessor.

        The number of columns the model actually sees: numerical features
        plus their missing-value indicators (``add_indicator=True``) plus the
        encoded categorical features. The estimate is computed
        deterministically from the data and the encoding settings — the same
        value in every HPO phase and in the subprocesses (no fitting, no
        randomness). It is used by the adaptive complexity guard as ``p`` for
        the rules that depend on the feature count (``max_features``,
        ``colsample_bytree``); the size-mode ratio ``n / p`` uses the
        original column count instead (``n_features_size``).

        One-hot encoding expands each categorical column into its number of
        unique categories; ordinal/target/frequency encoding keeps one column
        per categorical column; hashing encoding expands into
        ``n_components`` columns.

        Returns:
            int: The estimated feature count (>= 1).
        """
        if self.preprocessor is not None:
            n_out = getattr(self.preprocessor, "n_features_out_", None)
            if n_out is not None and int(n_out) >= 1:
                return int(n_out)

        # Explicitly passed column lists (the engine path) — the accurate
        # source; otherwise fall back to auto-detected types.
        cats = list(self.categorical_features or [])
        nums = list(self.numerical_features or [])
        if not cats and not nums and isinstance(self.X, pd.DataFrame):
            cats, nums = detect_feature_types(self.X)

        n = 0
        if nums:
            n += len(nums)
            if self._missing_indicator_added():
                # add_indicator=True: one binary column per numerical feature
                # that has missing values in the training set.
                try:
                    n += int(self.X[nums].isna().any().sum())
                except Exception:  # noqa: BLE001 — estimate only
                    n += 0
        if cats and isinstance(self.X, pd.DataFrame):
            encoding = self.encoding or "one_hot"
            for col in cats:
                if col not in self.X.columns:
                    continue
                if encoding == "hashing":
                    n += self.hashing_n_components
                elif encoding == "one_hot":
                    try:
                        # NaN is kept as a separate category: the categorical
                        # pipeline converts columns to strings first
                        # (``_to_string_array``), so a missing value becomes
                        # the literal "nan" category of OneHotEncoder.
                        n += int(self.X[col].nunique(dropna=False))
                    except Exception:  # noqa: BLE001 — estimate only
                        n += 1
                else:
                    # ordinal / target / frequency: one column per column.
                    n += 1
        return max(1, n)

    def _missing_indicator_added(self) -> bool:
        """Whether missing-value indicators are added for numerical features."""
        return _missing_indicator_enabled(self.algo)

    def assemble_estimator(
        self,
        model_impl: Any,
        apply_fs: bool = False,
        feature_index: int | None = None,
    ) -> Any:
        """Assemble the pipeline in the order preprocessor -> [selector] -> [sampler] -> model.

        Oversampling receives already-encoded numerical features (one-hot), so
        SMOTE/ADASYN always work with numerical data. Feature selection is
        activated strictly between the preprocessor and the oversampler:
        synthetic rows are generated only for informative features (the order
        is consistent with ModelTrainer).

        For ``isotonic_regression`` an ``IsotonicDataTransformer`` step is
        inserted right after the preprocessor (before selection/oversampling):
        the algorithm is strictly univariate and the adapter reduces the
        preprocessed matrix to the single column selected via
        ``feature_index`` (0 by default).

        Args:
            model_impl: The base regressor.
            apply_fs: Flag to enable the ``feature_selector`` step.
            feature_index: The ordinal feature index for the isotonic data
                adapter (``None`` — 0).
        """
        steps: list[tuple[str, Any]] = []
        if self.preprocessor is not None:
            steps.append(("preprocessor", self.preprocessor))
        if self.is_isotonic:
            steps.append(
                (
                    "isotonic_selector",
                    IsotonicDataTransformer(
                        feature_index=feature_index if feature_index is not None else 0
                    ),
                )
            )
        # New step: feature selection is activated strictly between the
        # preprocessor and the oversampler.
        if apply_fs:
            steps.append(("feature_selector", self._fs_transformer_factory()))
        if self.oversampling_config["active"]:
            steps.append(
                ("sampler", DataOversampler(**self.oversampling_config["params"]))
            )
        steps.append(("model", model_impl))
        if len(steps) == 1:
            return model_impl
        return ImbPipeline(steps)

    def _resolve_apply_fs(self, trial: Trial) -> bool:
        """Determine whether feature selection is needed for the current trial.

        In 'auto' mode Optuna itself explores the decision space (whether
        reducing features is beneficial); in 'always'/'disabled' the decision
        strictly follows the configuration. For isotonic_regression the
        use_feature_selection category is not offered even in 'auto' mode:
        selection is forcibly disabled (a univariate algorithm with its own
        IsotonicDataTransformer), so the suggestion would be meaningless and
        the auxiliary key must not leak into best_params.

        Args:
            trial (Trial): The current Optuna trial object.
        Returns:
            bool: Whether the ``feature_selector`` step should be added to the pipeline.
        """
        if self.fs_mode == FeatureSelectionMode.auto and not self.is_isotonic:
            apply_fs = trial.suggest_categorical("use_feature_selection", [True, False])
        elif self.fs_mode == FeatureSelectionMode.always:
            apply_fs = True
        else:
            apply_fs = False
        # Isotonic regression requires exactly one feature and uses its own
        # IsotonicDataTransformer — selection is forcibly disabled (including
        # when 'always' or enqueued initial_params carried the flag).
        if self.is_isotonic:
            apply_fs = False
        return apply_fs

    def evaluate_trial(self, trial: Trial, current_estimator: Any) -> float:
        """Evaluate the assembled trial pipeline and return the target metric (step 4).

        Evaluates ``current_estimator`` with the selected validation strategy.
        A non-finite result (+inf on an nrmse error or NaN) is replaced with
        ``HPO_WORST_SCORE`` — the "worst outcome" sentinel. Hyperparameter
        generation and pipeline assembly happen in ``_objective`` BEFORE this
        method is invoked and outside try/except: errors from a user space
        function or the model constructor must propagate (trial FAIL) rather
        than turn into pruning/disqualification.

        Args:
            trial (Trial): The current Optuna trial object.
            current_estimator (Any): The pipeline/model of the current trial.
        Returns:
            float: The target metric value for the current parameter set.
        Raises:
            ValueError: Non-fatal evaluation error (the trial is pruned).
            MemoryError, RuntimeError, InvalidDataError: Fatal failures —
                counted in the ``handle_failure`` counter.
            optuna.TrialPruned: Pruning by the early stopping pruner.
        """
        # 1. evaluation according to the validation strategy
        if self.val_method_eff == "train_test_split":
            # If 'auto' was resolved in make_cv, use the already computed
            # (dataset-dependent) integer test_size so that iter_splits does
            # not recompute the decision. Otherwise — the fixed fraction.
            if self.resolved_via_auto and self.auto_decision is not None:
                split_test_size: float | int = int(self.auto_decision["test_size"])
            else:
                split_test_size = self.train_test_split_test_size
            # Use iter_splits for uniformity; since it is a generator, take
            # next()
            X_tr, X_te, y_tr, y_te = next(
                iter_splits(
                    self.X,
                    self.y,
                    method="train_test_split",
                    test_size=split_test_size,
                    random_state=self.random_state,
                )
            )
            current_estimator.fit(X_tr, y_tr)
            return float(self.scorer(current_estimator, X_te, y_te))
        if self.pruning_active:
            # Early stopping: evaluate fold by fold and publish intermediate
            # results for the Optuna pruner. For non-additive metrics the
            # published value is the pooled metric over the predictions
            # accumulated up to the current step.
            avg_score = _evaluate_with_intermediate_reports(
                trial,
                current_estimator,
                self.X,
                self.y,
                method=self.val_method_eff,
                n_folds=self.effective_n_folds,
                test_size=self.train_test_split_test_size,
                random_state=self.random_state,
                scorer=self.scorer,
                pool_metric=self.pool_metric,
                algo_name=self.algo_name,
                phase_name=self.phase_name or "",
            )
            # If we got +inf (an nrmse error) or NaN, return the worst float
            # (analog of the float32 minimum or float('-inf')).
            if not np.isfinite(avg_score):
                return HPO_WORST_SCORE
            return avg_score
        if self.pool_metric:
            # Non-additive metrics (RMSE, MSE, R², nRMSE): evaluate over the
            # pooled OOF vector at once (the same estimate the final
            # ModelTrainer and the select phase use) instead of averaging
            # per-fold values. The splits are the same as
            # ``cross_val_score(cv=cv_obj)``.
            avg_score = _evaluate_pooled_cv(
                current_estimator,
                self.X,
                self.y,
                method=self.val_method_eff,
                n_folds=self.effective_n_folds,
                test_size=self.train_test_split_test_size,
                random_state=self.random_state,
                scorer=self.scorer,
                algo_name=self.algo_name,
                phase_name=self.phase_name or "",
            )
            # If we got +inf (an nrmse error) or NaN, return the worst float
            # (analog of the float32 minimum or float('-inf')).
            if not np.isfinite(avg_score):
                return HPO_WORST_SCORE
            return avg_score
        # Additive metrics: the legacy fold-average path. cv_obj is guaranteed
        # non-None here: a non-train_test_split and non-pruning branch means
        # k_fold/loo, where make_cv always creates a cv object.
        scores = model_selection.cross_val_score(
            current_estimator,
            self.X,
            self.y,
            cv=self.cv_obj,
            scoring=self.scorer,
            n_jobs=1,
        )
        avg_score = float(np.mean(scores))
        # If we got +inf (an nrmse error) or NaN, return the worst float
        # (analog of the float32 minimum or float('-inf')).
        if not np.isfinite(avg_score):
            return HPO_WORST_SCORE
        return avg_score

    def handle_pruning(self) -> None:
        """Configure early stopping (pruning) for the selected validation (step 5).

        Normalizes the early stopping configuration and builds an Optuna
        pruner. Pruning applies only when there are natural evaluation steps
        (k_fold / loo); for hold-out (train_test_split) there are no
        intermediate steps, so early stopping is not applied (documented
        behavior).
        """
        pruning_cfg = _normalize_pruning_config(self.pruning)
        self.pruner = None
        if pruning_cfg is not None:
            self.pruner = _build_pruner(pruning_cfg)
            log.info(
                "Early stopping enabled: strategy=%s, min_steps=%d, "
                "n_startup_trials=%d, reduction_factor=%d",
                pruning_cfg["strategy"],
                pruning_cfg["min_steps"],
                pruning_cfg["n_startup_trials"],
                pruning_cfg["reduction_factor"],
            )
        self.pruning_active = (
            self.pruner is not None and self.val_method_eff != "train_test_split"
        )
        if self.pruner is not None and not self.pruning_active:
            log.warning(
                "Early stopping is enabled, but validation strategy '%s' has no "
                "intermediate steps — pruning will not be applied.",
                self.val_method_eff,
            )

    def _objective(self, trial: Trial) -> float:
        """Optuna objective function: evaluate_trial + handle_failure.

        Splits trial evaluation (``evaluate_trial``) and its failure handling
        (``handle_failure``) into separate steps. Hyperparameter generation and
        pipeline assembly run OUTSIDE try/except — as in the original
        implementation: errors from a user space function or the model
        constructor must propagate (trial FAIL and an exception from
        ``study.optimize``) rather than silently become pruning /
        disqualification. The consecutive fatal failure counter is reset on
        any non-fatal outcome: success, pruner pruning (optuna.TrialPruned
        from ``_evaluate_with_intermediate_reports``), or a non-fatal
        ValueError.

        Args:
            trial (Trial): The current Optuna trial object.
        Returns:
            float: The target metric value for the current parameter set.
        Raises:
            optuna.TrialPruned: If the trial is pruned (by an error or the pruner).
            InvalidAlgorithmError: When the fatal failure limit is exceeded.
        """
        # 1. hyperparameters, model and pipeline assembly — outside try/except
        #    (contract of the original code: their errors are not fatal
        #    evaluation failures and are not counted by the circuit breaker).
        params = self.space_fn(trial)
        model = create_model(self.algo, **params)
        current_estimator = self.assemble_estimator(
            model,
            apply_fs=self._resolve_apply_fs(trial),
            feature_index=params.get("feature_index"),
        )

        fatal_failure = False
        try:
            return self.evaluate_trial(trial, current_estimator)
        except (ValueError, MemoryError, RuntimeError, InvalidDataError) as err:
            fatal_failure = not isinstance(err, ValueError)
            self.handle_failure(trial, err)
            # Unreachable: handle_failure always raises (NoReturn). The stub
            # is only needed by static analysis so the except branch has a
            # return value.
            return 0.0  # pragma: no cover
        finally:
            # Any non-fatal trial outcome breaks the sequence of fatal
            # failures: a successful _score return, pruner pruning
            # (optuna.TrialPruned), or a non-fatal ValueError. Without this
            # reset, the counter would accumulate errors separated by normal
            # trials and falsely disqualify the algorithm.
            if not fatal_failure:
                self.consecutive_fatal_failures = 0

    def handle_failure(self, trial: Trial, err: Exception) -> NoReturn:
        """Handle a trial evaluation failure according to the outcome contract (step 6).

        A non-fatal ``ValueError`` simply prunes the trial
        (``optuna.TrialPruned``) without counting it. ``MemoryError`` /
        ``RuntimeError`` / ``InvalidDataError`` are considered fatal: they
        increment the consecutive failure counter and disqualify the algorithm
        once the limit is reached (``InvalidAlgorithmError``).

        Args:
            trial (Trial): The current Optuna trial (for the user_attr "fail_reason").
            err (Exception): The caught evaluation exception.
        Raises:
            optuna.TrialPruned: Always (non-fatal or fatal failure).
            InvalidAlgorithmError: If the fatal failure limit is reached.
        """
        trial.set_user_attr("fail_reason", str(err))
        if isinstance(err, ValueError):
            raise optuna.TrialPruned() from err
        self.consecutive_fatal_failures += 1
        if self.consecutive_fatal_failures >= self.MAX_FATAL_FAILURES:
            raise InvalidAlgorithmError(
                f"Algorithm '{self.algo}' disqualified after "
                f"{self.consecutive_fatal_failures} consecutive fatal failures"
            ) from err
        raise optuna.TrialPruned() from err

    def _natural_domain_bounds(
        self, param: str, entry: SearchSpaceEntry
    ) -> tuple[float | None, float | None]:
        """Natural (physical) domain bounds of a parameter.

        Returns ``(low, high)``; a side is ``None`` when the domain is
        unbounded there. A boundary that coincides with a natural bound is not
        actionable: the value cannot be pushed beyond it even by widening the
        search range, so the "widen it" hint would be a false signal. Bounds
        that depend on the data (leaf sizes, neighbor counts) are derived from
        ``n_samples_eff`` / ``n_features_eff`` — these are the same physical
        limits that ``clip_search_space`` enforces.

        Args:
            param: The hyperparameter name.
            entry: The search-space entry (its distribution type decides the
                fraction-vs-count interpretation of ``min_samples_leaf`` and
                ``max_features``).

        Returns:
            tuple[float | None, float | None]: Natural ``(low, high)`` bounds.
        """
        low: float | None = None
        high: float | None = None
        dist_type = entry.dist_type
        if param in {"subsample", "colsample_bytree"}:
            # A row/feature sampling fraction: (0, 1].
            high = 1.0
        elif param == "min_samples_leaf":
            if (
                dist_type == "float"
                and float(entry.low) < 1.0
                and float(entry.high) <= 1.0
            ):
                # A fraction of the sample size: (0, 1].
                high = 1.0
            else:
                # An absolute row count: at least 1 and at most n.
                low = 1.0
                high = float(max(1, int(getattr(self, "n_samples_eff", 0))))
        elif param == "min_samples_split":
            low = 2.0
            high = float(max(2, int(getattr(self, "n_samples_eff", 0))))
        elif param == "n_neighbors":
            low = 1.0
            high = float(max(1, int(getattr(self, "n_samples_eff", 0)) - 1))
        elif param == "max_features":
            if dist_type == "float":
                # A fraction of the features: (0, 1].
                high = 1.0
            else:
                # An absolute feature count.
                low = 1.0
                high = float(max(1, int(getattr(self, "n_features_eff", 1))))
        elif param == "l1_ratio":
            low = 0.0
            high = 1.0
        elif param == "max_depth":
            low = 1.0
        elif param in {"min_child_weight", "gamma"}:
            low = 0.0
        elif param == "n_estimators":
            low = 1.0
        elif param == "p":
            # The Minkowski power of the KNN space.
            low = 1.0
        elif param == "feature_index":
            # The ordinal column index consumed by IsotonicDataTransformer:
            # the natural domain spans [0, n_features - 1] — neither side can
            # be pushed beyond it even by widening the search range.
            low = 0.0
            n_features = int(
                self.n_features_eff
                if self.preprocessor is not None
                else getattr(self.X, "shape", (0, 1))[1]
            )
            high = float(max(0, n_features - 1))
        return low, high

    def _is_guard_boundary(self, param: str, value: float, tolerance: float) -> bool:
        """Whether ``value`` lies on a bound introduced by the complexity guard.

        The guard records the "before → after" bounds of every range it
        actually narrowed. A bound is guard-introduced when the guard moved
        that side of the range: the "after" bound differs from "before" (the
        other side may be the user's or the default bound and stays a normal
        "widen it" candidate). An optimum on a guard-moved bound is expected —
        the range was tightened deliberately (e.g. ``max_depth`` 2–3 on small
        data) — so the "widen it" warning would be a false signal.

        Args:
            param: The hyperparameter name.
            value: The boundary value of the best trial.
            tolerance: The same absolute tolerance the boundary check uses.

        Returns:
            bool: ``True`` when the value is a guard-introduced bound.
        """
        algo = resolve_algorithm_name(self.algo)
        for record in getattr(self, "complexity_guard_logs", []):
            if record.get("param") != param:
                continue
            if resolve_algorithm_name(str(record.get("algo", ""))) != algo:
                continue
            if not record.get("applied"):
                continue
            before = record.get("before")
            after = record.get("after")
            if not isinstance(before, list) or len(before) != 2:
                continue
            if not isinstance(after, list) or len(after) != 2:
                continue
            try:
                before = [float(before[0]), float(before[1])]
                after = [float(after[0]), float(after[1])]
            except (TypeError, ValueError):
                continue
            # A side introduced by the guard: the after bound differs from
            # the before one on that side.
            moved_low = not math.isclose(
                after[0], before[0], rel_tol=0.0, abs_tol=tolerance
            )
            moved_high = not math.isclose(
                after[1], before[1], rel_tol=0.0, abs_tol=tolerance
            )
            if (
                moved_low
                and math.isclose(value, after[0], rel_tol=0.0, abs_tol=tolerance)
            ) or (
                moved_high
                and math.isclose(value, after[1], rel_tol=0.0, abs_tol=tolerance)
            ):
                return True
        return False

    def _warn_if_optimum_at_boundary(self, best_params: dict[str, Any]) -> None:
        """Warn when the best trial's value lies at a range boundary.

        An optimum that coincides with the lower or upper bound of the search
        range means the search was cut off: the true optimum may lie beyond
        the range, and the user should widen it. Only dict-based numeric
        search spaces (with ``SearchSpaceEntry`` bounds) are checked;
        categorical parameters and constant values are skipped.

        The warning is suppressed when the boundary is not actionable:
        * a fixed parameter (``low == high``) — e.g. a user
          ``min_samples_leaf [1, 1]`` or an XGBoost ``max_depth`` collapsed
          to 3–3 — the optimum sits at the only possible value;
        * a bound introduced by the adaptive complexity guard — the range
          was narrowed deliberately (e.g. ``max_depth`` 2–3) and the optimum
          is expected on it (an INFO line is logged instead);
        * a natural (physical) domain boundary of the parameter —
          ``subsample=1.0``, ``colsample_bytree=1.0``, ``min_samples_leaf=1``,
          ``l1_ratio=0/1`` — the value cannot be pushed beyond it even by
          widening the range.

        Args:
            best_params: The parameters of the best trial.
        """
        if not self.search_space_dict:
            return
        for name, entry in self.search_space_dict.items():
            if not hasattr(entry, "dist_type"):
                continue
            dist_type = entry.dist_type
            if dist_type not in ("int", "float", "float_log"):
                continue
            if name not in best_params:
                continue
            value = best_params[name]
            try:
                low = float(entry.low)
                high = float(entry.high)
                num = float(value)
            except (TypeError, ValueError):
                continue
            if low == high:
                # A fixed parameter: the optimum is at the only possible
                # value, the "widen it" hint is meaningless.
                continue
            # Tolerance: 1e-6 of the range width (at least 1e-6 for tiny ranges).
            # Exact equality is not enough because Optuna may return 1.0 vs
            # a boundary 1.0000001. A relative tolerance is deliberately not
            # used: on very wide ranges (e.g. [0, 1e9]) a rel_tol of 1e-9
            # would already accept values ~1.0 away from the boundary — too
            # coarse for a "widen the range" hint.
            tolerance = 1e-6 * max(1.0, abs(high - low))
            at_low = math.isclose(num, low, rel_tol=0.0, abs_tol=tolerance)
            at_high = math.isclose(num, high, rel_tol=0.0, abs_tol=tolerance)
            if not (at_low or at_high):
                continue
            boundary = "low" if at_low else "high"
            bound_value = low if at_low else high

            # A guard-introduced bound: the range was narrowed deliberately,
            # the optimum on it is expected — INFO instead of a warning.
            if self._is_guard_boundary(name, bound_value, tolerance):
                log.info(
                    "Algorithm '%s': the optimum of '%s'=%.6g lies at the %s "
                    "boundary of [%.6g, %.6g], set by the complexity_guard — "
                    "expected, the range is narrowed deliberately, no 'widen "
                    "it' warning",
                    self.algo,
                    name,
                    num,
                    boundary,
                    low,
                    high,
                )
                continue

            # A natural (physical) domain boundary of the parameter: the value
            # cannot be pushed beyond it even by widening the range.
            natural_low, natural_high = self._natural_domain_bounds(name, entry)
            natural_bound = natural_low if at_low else natural_high
            if natural_bound is not None and math.isclose(
                bound_value, natural_bound, rel_tol=0.0, abs_tol=tolerance
            ):
                log.info(
                    "Algorithm '%s': the optimum of '%s'=%.6g lies at the %s "
                    "boundary of [%.6g, %.6g] — the natural domain boundary "
                    "of the parameter, no 'widen it' warning",
                    self.algo,
                    name,
                    num,
                    boundary,
                    low,
                    high,
                )
                continue

            log.warning(
                "Algorithm '%s': the optimum of '%s'=%.6g lies at the %s "
                "boundary of the search range [%.6g, %.6g] — the true "
                "optimum may lie beyond the range, widen it",
                self.algo,
                name,
                num,
                boundary,
                low,
                high,
            )

    def select_best_trial(self) -> bool:
        """Select the best completed trial of the CURRENT phase (step 7).

        Explicit trial state counting: the presence of a valid result is
        determined by the number of completed (COMPLETED) trials of the
        current phase (``number >= phase_start_number``) rather than by the
        internal behavior of ``study.best_params``. Transferred trials
        (``previous_trials``) feed the TPE sampler but are NEVER counted or
        ranked here: they must not mask a total failure of the current phase
        (otherwise a refine phase that wiped out all its own trials would
        silently fall back to the previous phase's parameters — the forbidden
        no-fallback contract). Counters are logged for observability ("no
        valid result" must be explainable from the logs).

        The best trial is selected EXPLICITLY by its value among the current
        phase's COMPLETED trials; ``study.best_params`` is deliberately not
        used — it would rank the transferred trials too.

        Returns:
            bool: True if there is a completed trial of the current phase;
                False — the "no valid result" contract (all OWN trials
                PRUNED/FAILED or an empty list).
        """
        assert self.study is not None
        own_trials = self.phase_trials or [
            t
            for t in self.study.get_trials(deepcopy=False)
            if t.number >= self.phase_start_number
        ]
        n_completed = sum(
            1 for t in own_trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        n_pruned = sum(
            1 for t in own_trials if t.state == optuna.trial.TrialState.PRUNED
        )
        n_failed = sum(1 for t in own_trials if t.state == optuna.trial.TrialState.FAIL)
        if self.pruning_active:
            log.info(
                "Early stopping: pruned %d of %d trials (algo=%s)",
                n_pruned,
                len(own_trials),
                self.algo,
            )
        log.info(
            "Trial states for algorithm '%s': completed=%d, pruned=%d, failed=%d",
            self.algo,
            n_completed,
            n_pruned,
            n_failed,
        )
        if n_completed == 0:
            # The "no valid result" contract: all OWN trials are pruned
            # (PRUNED), failed (FAILED), or the trial list is empty. Transferred
            # history is irrelevant here. Return False so the caller (_run_hpo)
            # can tell a failure from a valid result and exclude the algorithm
            # from the candidates.
            log.warning(
                "Algorithm '%s' produced no completed trials "
                "(completed=%d, pruned=%d, failed=%d); returning no result",
                self.algo,
                n_completed,
                n_pruned,
                n_failed,
            )
            return False

        best_trial = max(
            (t for t in own_trials if t.state == optuna.trial.TrialState.COMPLETE),
            key=lambda t: cast(float, t.value),
        )
        self.best_score = float(cast(float, best_trial.value))
        best_params = dict(best_trial.params)

        self._warn_if_optimum_at_boundary(best_params)

        # The feature selection decision comes from the best trial's params
        # (mode='auto': the "use_feature_selection" key is already fixed by
        # Optuna) or from the configuration mode ('always' -> True, 'disabled'
        # -> False). For isotonic_regression selection is forcibly ignored.
        best_apply_fs = bool(
            best_params.get(
                "use_feature_selection", self.fs_mode == FeatureSelectionMode.always
            )
        )
        if self.is_isotonic:
            best_apply_fs = False
            # For isotonic_regression, the auxiliary key is also excluded from
            # the returned best_params (a guard against initial_params carried
            # through enqueue_trial from the previous phase with the
            # use_feature_selection key).
            best_params = {
                k: v for k, v in best_params.items() if k != "use_feature_selection"
            }
        self.best_params = best_params
        # The base model constructor parameters are cleaned of the tuner's
        # auxiliary key ("use_feature_selection" is not a model
        # hyperparameter). They are built from the already filtered
        # self.best_params (for isotonic_regression the key was excluded
        # above) so the source is single; the best trial's params are
        # not mutated and the key is guaranteed not to reach the model
        # constructor.
        self.clean_model_params = dict(self.best_params)
        self.clean_model_params.pop("use_feature_selection", None)
        self.best_apply_fs = best_apply_fs
        return True

    def _log_best_result(self) -> None:
        """Log the outcome of the best trial (shared by both return paths)."""
        log.info(
            "Hyperopt: algo=%s | val=%s | score=%.5f | params=%s",
            self.algo,
            self.val_method_eff,
            self.best_score,
            self.best_params,
        )

    def _fit_best_model(self) -> Any:
        """Train the final model on all provided data (step 8).

        Assembles the pipeline with the best parameters (taking the feature
        selection decision from ``select_best_trial`` into account) and trains
        it on the full dataset. If oversampling was enabled, the final model
        goes through it as well. Skipped entirely when ``fit_best_model`` is
        False (HPO-only callers). The trained model is stored in
        ``self.best_model``.

        Returns:
            Any: The trained best model (``self.best_model``).
        """
        best_model = self.assemble_estimator(
            create_model(self.algo, **self.clean_model_params),
            apply_fs=self.best_apply_fs,
            feature_index=self.clean_model_params.get("feature_index"),
        )
        best_model.fit(self.X, self.y)

        self.best_model = best_model
        return best_model

    # ─────────────────── private helpers ───────────────────────────────────

    def _validate_n_trials(self) -> bool:
        """Validate ``n_trials`` and the "no valid result" contract.

        Returns:
            bool: True if Optuna should run; False — return
                ``None`` (the "no valid result" signal) without running.
        Raises:
            ValueError: If ``n_trials`` is negative or not an integer.
        """
        if not isinstance(self.n_trials, int) or self.n_trials < 0:
            raise ValueError(
                f"n_trials must be a non-negative integer, got {self.n_trials}"
            )
        if self.n_trials == 0:
            # Empty trial list — the "no valid result" contract: return a
            # plain None without running Optuna and without exceptions
            # (including for algorithms without a search space).
            log.warning(
                "Algorithm '%s' requested with n_trials=0; no trials to run — "
                "returning no result",
                self.algo_name,
            )
            return False
        return True

    def _fs_transformer_factory(self) -> FeatureSelector:
        """Create a fresh feature selector instance for a pipeline step.

        Each call returns a new ``FeatureSelector`` with a deterministic
        ``random_state`` (consistent with the final ModelTrainer training),
        which guarantees reproducible selection across trials and threads.
        With ``random_state=None`` (a fully non-deterministic Optuna run),
        selection still uses the fixed seed 42: otherwise each factory call
        would create a selector with a new random seed and selection would be
        irreproducible even with all other pipeline stages deterministic.
        """
        return FeatureSelector(
            method=self.fs_cfg.method.value,
            percentile=self.fs_cfg.percentile,
            min_features=self.fs_cfg.min_features,
            variance_threshold=self.fs_cfg.variance_threshold,
            n_estimators=self.fs_cfg.n_estimators,
            random_state=self.random_state if self.random_state is not None else 42,
        )


def optimize(
    algo_name: str,
    X: Any,
    y: Any,
    *,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
    metric: str = "r2",
    # keep the old argument for backward compatibility
    val_method: ValidationStrategy | str = "k_fold",
    # alias sent by training_engine.component
    validation_strategy: ValidationStrategy | str | None = None,
    n_folds: int = 5,
    n_trials: int = 50,
    random_state: int | None = 42,
    sampler_seed: int | None = None,
    train_test_split_test_size: float = 0.2,
    space_overrides: dict[str, Callable[[Trial], dict[str, Any]]] | None = None,
    initial_params: dict[str, Any] | None = None,
    previous_trials: list[FrozenTrial] | None = None,
    preprocessor: Any | None = None,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    encoding: EncodingStrategy | None = None,
    preprocessing_override: PreprocessingOverride | dict[str, Any] | None = None,
    pruning: dict[str, Any] | None = None,
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: EncodingStrategy | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
    complexity_guard_cfg: ComplexityGuardConfig | dict[str, Any] | None = None,
    user_hyperparameter_keys: list[str] | None = None,
    fit_best_model: bool = True,
    phase_name: str | None = None,
) -> OptimizeResult | None:
    """Run the model hyperparameter optimisation process using Optuna.

    The function automatically selects the validation strategy, sets up the
    parameter search space and trains the final model on all provided data
    (unless ``fit_best_model=False`` is passed).

    Args:
        algo_name (str): Algorithm name to optimise.
        X (Any): Input features.
        y (Any): Target variable.
        data_oversampling (bool): Flag to enable class balancing.
            Defaults to False.
        data_oversampling_multiplier (float): Scaling factor
            (for oversampling).
        data_oversampling_algorithm (str): Balancing algorithm name
            (e.g. 'random', 'smote').
        metric (str): Metric name to maximise. Defaults to 'r2'.
        val_method (ValidationStrategy | str): Validation method
            ('k_fold', 'leave_one_out', 'train_test_split').
        validation_strategy (ValidationStrategy | str | None): Alias for
            ``val_method`` (takes precedence).
        n_folds (int): Number of cross-validation folds. Defaults to 5.
        n_trials (int): Number of search iterations (trials). Defaults to 50.
        random_state (int | None): Randomness state for the data splits and
            the preprocessor (CV folds, train_test_split, pruning folds,
            hashing / target-encoding cross-fit). Defaults to 42. With
            ``None``, the data splits are not fixed; however, the feature
            selection step (``FeatureSelector``) always uses the fixed seed 42
            so that selection stays reproducible across runs.
        sampler_seed (int | None): Seed of the TPE sampler, separated from
            ``random_state`` so that each HPO phase can use a different
            sampler seed (e.g. ``RANDOM_STATE + phase_ordinal``) while the
            data splits stay fixed across phases. This keeps CV evaluations
            comparable between phases (score monotonicity of
            ``enqueue_trial``, consistent TPE history). ``None`` (default) —
            the sampler falls back to ``random_state`` (backward-compatible
            behavior for direct callers).
        train_test_split_test_size (float): Test size for split validation.
            Defaults to 0.2.
        space_overrides (dict | None): Dict for overriding search spaces.
            Each value may be a callable (a custom ``space_fn``), a dict of
            ``SearchSpaceEntry`` objects, or a dict whose values are the same
            shorthands accepted by the Config path: a raw range list
            (``[low, high]`` / ``[low, high, "int"|"float"|"float_log"]`` /
            ``[options, "categorical"]``) is parsed and a range whose type was
            not written explicitly is re-resolved against the parameter's
            type (the default search space or the estimator signature), the
            same way ``Config`` does it — so
            ``space_overrides={"svr": {"C": [1, 50]}}`` searches ``C`` as a
            real range, not an integer-only one.
        initial_params (dict[str, Any] | None): Hyperparameters from the
            previous phase for enqueue_trial. Keeps the improvement monotonic
            between HPO phases ("monotonicity anchor").
        previous_trials (list[FrozenTrial] | None): Optuna trials of the
            previous HPO phase to be transferred into the new study via
            ``study.add_trials`` — the TPE history warm-start. The transfer
            happens BEFORE ``enqueue_trial``/``optimize``; the transferred
            trials feed the TPE sampler (including the ``n_startup_trials``
            window) but never count as the current phase's own trials. An
            empty list is a no-op. Trials with distributions incompatible with
            the current search space raise ``HPOHistoryError`` (fail fast).
        preprocessor (Any | None): A ready ``ColumnTransformer`` for feature
            preprocessing (categories -> one-hot, numerics -> StandardScaler).
            If passed, it takes precedence over
            ``categorical_features``/``numerical_features``.
        categorical_features (list[str] | None): Names of categorical columns.
            If ``preprocessor`` is not passed, a preprocessor is built from them.
        numerical_features (list[str] | None): Names of numerical columns.
            Used together with ``categorical_features``.
        encoding (EncodingStrategy | None): Category encoding strategy
            ('one_hot', 'ordinal', 'target', 'frequency' or 'hashing').
            Defaults to ``None`` — 'one_hot' is used. Applied when building
            the preprocessor if ``preprocessor`` is not passed.
        preprocessing_override (PreprocessingOverride | dict | None): Explicit
            override of the feature preprocessing preset. Set partially or
            fully; takes precedence over the automatic selection and is
            applied consistently with the final training.
        pruning (dict[str, Any] | None): Early stopping (pruning) settings.
            A dict with fields: ``enable`` (bool, False by default),
            ``strategy`` ('median' | 'hyperband', 'median' by default),
            ``min_steps`` (int >= 1, 1 by default), ``n_startup_trials``
            (int >= 1, only for 'median', 5 by default), ``reduction_factor``
            (int >= 2, only for 'hyperband', 3 by default). With
            ``enable=False`` or ``None`` the behavior is identical to the
            current one: every trial runs to completion. For validation
            strategies without natural steps (train_test_split) the pruner is
            not applied.
        high_cardinality_threshold (int | None): Cardinality threshold for the
            automatic mode (>= 0). Columns whose number of unique values is
            strictly greater than the threshold are encoded with
            ``high_cardinality_encoding``, the rest with ``encoding``.
            ``None`` — the mode is disabled.
        high_cardinality_encoding (EncodingStrategy | None): Encoding strategy
            for high-cardinality columns. Set together with
            ``high_cardinality_threshold``.
        hashing_n_components (int): Number of binary columns per categorical
            column with 'hashing' encoding (>= 1).
        target_encoding_smoothing (float): Smoothing parameter of the target
            encoding (>= 0).
        target_encoding_fallback (float | None): Fallback value of target
            encoding for unknown categories (None — the global mean).
        feature_selection_cfg (FeatureSelectionCfg | dict | None): Feature
            selection settings. ``None`` — the ``'disabled'`` mode (selection
            is not applied, the default behavior). In ``'always'`` mode the
            ``feature_selector`` step is present in every trial pipeline; in
            ``'disabled'`` mode it is absent; in ``'auto'`` mode Optuna itself
            chooses ``use_feature_selection`` (True/False) on validation
            splits in competition with model hyperparameters. For
            ``isotonic_regression`` selection is forcibly disabled (a
            univariate algorithm).
        complexity_guard_cfg (ComplexityGuardConfig | dict | None): Settings
            of the adaptive complexity guard for tree/boosting search spaces
            (from ``general.complexity_guard``). ``None`` — the guard is
            disabled (the exact legacy behavior for direct callers). When
            enabled, the dict-based search spaces of trees/boostings are
            tightened depending on ``n`` (effective training-fold rows) and
            ``p`` (features after preprocessing).
        user_hyperparameter_keys (list[str] | None): Names of parameters set
            explicitly by the user (``algo.hyperparameters``); the guard
            intersects its ranges with these user ranges and applies
            ``user_range_policy`` on an empty intersection. ``None`` — no
            user-specified parameters.
        fit_best_model (bool): Whether to train the final model on all
            provided data after the search and return it in the ``model``
            field of the result. Defaults to True (the original behavior).
            Set to False when only the best parameters and score are needed
            (e.g. the training engine's HPO phases — the final model is
            trained by its own trainer): the redundant full-data fit is
            skipped and the ``model`` field of the result is None.
        phase_name (str | None): The name of the pipeline phase (e.g.
            'coarse'/'refine') used to deduplicate the WARNING about the
            fallback from the pooled metric to the per-fold mean (emitted
            once per algorithm and phase). ``None`` for direct callers — the
            warning falls back to a generic phase label.

    Returns:
        OptimizeResult: The result of the phase — ``model`` (the trained
            model with the best parameters; ``None`` when
            ``fit_best_model=False`` — the fit is skipped), ``params`` (the
            found optimal hyperparameters; in ``'auto'`` mode it contains
            the auxiliary ``use_feature_selection`` key with the selection
            decision), ``score`` (the best raw "optimizer" metric value on
            validation, higher is better), and ``trials`` — the frozen
            trials of the current phase ONLY (transferred
            ``previous_trials`` are excluded).
        None: The "no valid result" signal — ``n_trials=0`` (an empty trial
            list) or no trial of the current phase completed (e.g., every
            trial was turned into ``optuna.TrialPruned`` by a non-fatal
            error, or every trial failed). Transferred history never counts
            toward the current phase validity; the caller (``_run_hpo``)
            treats ``None`` as the failure signal and excludes the algorithm
            from candidates (no fallback to previous-phase parameters).

    Raises:
        ValueError: If ``n_trials`` is a negative integer (or not an
            integer). The value ``n_trials=0`` is allowed and leads to an
            empty trial list → ``None``.
        HyperoptError: If no search space is defined for the selected
            algorithm, or an invalid ``feature_selection_cfg`` is passed.
        HPOHistoryError: If ``previous_trials`` contain distributions
            incompatible with the current search space (fail fast).
    """
    # Thin wrapper: all run logic is moved into the _OptimizeRunner steps so
    # that optimize() stays a readable entry point.
    return _OptimizeRunner(
        algo_name=algo_name,
        X=X,
        y=y,
        data_oversampling=data_oversampling,
        data_oversampling_multiplier=data_oversampling_multiplier,
        data_oversampling_algorithm=data_oversampling_algorithm,
        metric=metric,
        val_method=val_method,
        validation_strategy=validation_strategy,
        n_folds=n_folds,
        n_trials=n_trials,
        random_state=random_state,
        sampler_seed=sampler_seed,
        train_test_split_test_size=train_test_split_test_size,
        space_overrides=space_overrides,
        initial_params=initial_params,
        previous_trials=previous_trials,
        preprocessor=preprocessor,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        encoding=encoding,
        preprocessing_override=preprocessing_override,
        pruning=pruning,
        high_cardinality_threshold=high_cardinality_threshold,
        high_cardinality_encoding=high_cardinality_encoding,
        hashing_n_components=hashing_n_components,
        target_encoding_smoothing=target_encoding_smoothing,
        target_encoding_fallback=target_encoding_fallback,
        feature_selection_cfg=feature_selection_cfg,
        complexity_guard_cfg=complexity_guard_cfg,
        user_hyperparameter_keys=user_hyperparameter_keys,
        fit_best_model=fit_best_model,
        phase_name=phase_name,
    ).run()

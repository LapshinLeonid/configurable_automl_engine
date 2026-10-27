"""Feature selection decisions of the single-model trainer.

Validates the feature selection configuration of ``ModelTrainer`` and
resolves whether the ``feature_selector`` step is active in ``fit()``: the
isotonic bypass, the explicit HPO flag, the configuration mode and the
standalone auto-check (mode ``auto``) that compares two control pipelines on
a hold-out split. The functions take explicit inputs; the control pipelines
come from a ``make_steps`` callable passed in. The module must not import
``configurable_automl_engine.trainer``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from configurable_automl_engine.training_engine.metrics import (
    direction_label,
    get_scorer_object,
    to_user_value,
)

from .models import resolve_algorithm_name
from .trainer_errors import TrainingError
from .trainer_scoring import StepsFactory

# The records keep the logger name of ``ModelTrainer`` (its ``logger``).
_LOG = logging.getLogger("configurable_automl_engine.trainer")

#: Minimum number of rows for the standalone feature selection check (mode
#: ``auto``). The threshold ensures that both parts of the hold-out split (80/20)
#: contain at least 2 samples: 10 * 0.8 = 8 and 10 * 0.2 = 2.
_MIN_AUTO_CHECK_SAMPLES = 10


def _validate_feature_selection_cfg(
    cfg: FeatureSelectionCfg | dict[str, Any] | None,
) -> FeatureSelectionCfg:
    """Validate and normalize the feature selection configuration.

    Logic follows :meth:`_validate_preprocessing_override`:
    ``None`` → default ``FeatureSelectionCfg()`` (mode='disabled');
    ``dict`` → ``FeatureSelectionCfg.model_validate(...)``, error → TrainingError;
    ``FeatureSelectionCfg`` → used as is;
    any other type → TrainingError.

    Args:
        cfg: A dict or a :class:`FeatureSelectionCfg` model.

    Returns:
        The normalized feature selection configuration model.

    Raises:
        TrainingError: If an object of an unsupported type or a dict with
            invalid values is passed.
    """
    if cfg is None:
        return FeatureSelectionCfg()
    if isinstance(cfg, FeatureSelectionCfg):
        return cfg
    try:
        if isinstance(cfg, dict):
            return FeatureSelectionCfg.model_validate(cfg)
    except Exception as e:
        raise TrainingError(f"Invalid feature_selection_cfg: {e}") from e
    raise TrainingError(
        "feature_selection_cfg must be a dict or FeatureSelectionCfg, "
        f"got {type(cfg).__name__}"
    )


def _resolve_feature_selection_active(
    X_train: Any,
    y_train: Any,
    make_steps: StepsFactory,
    *,
    algorithm: str,
    explicit_flag: bool | None,
    fs_cfg: FeatureSelectionCfg,
    metric: str,
    random_state: int | None,
) -> bool:
    """Resolve the actual feature selection activity for fit().

    Priority (strictly in this order):
    1. IsotonicRegression — selection is forcibly disabled (the algorithm
       is strictly univariate and uses its own ``IsotonicDataTransformer``);
    2. the explicit ``feature_selection_active`` flag (from HPO via
       training_engine.artifact) takes precedence over the configuration;
    3. the mode from ``fs_cfg.mode``: ``always`` → True, ``disabled`` →
       False, ``auto`` → Standalone Auto-Check.

    Args:
        X_train: Training feature matrix (for the ``auto`` mode).
        y_train: Target variable (for the ``auto`` mode).
        make_steps: Builds fresh pipeline steps (for the ``auto`` mode).
        algorithm: The algorithm name.
        explicit_flag: The explicit activity flag (``None`` — decide by the
            configuration).
        fs_cfg: The feature selection configuration.
        metric: The primary metric (for the ``auto`` mode).
        random_state: The trainer seed (for the ``auto`` mode).

    Returns:
        ``True`` if the selection step should be present in the pipeline.
    """
    # 1. Isotonic bypass: a univariate algorithm is incompatible with feature selection.
    if resolve_algorithm_name(algorithm) == "isotonic_regression":
        _LOG.debug(
            "Feature selection is forcibly disabled for IsotonicRegression "
            "(univariate algorithm with its own data transformer)."
        )
        return False

    # 2. The explicit flag from HPO (training_engine.artifact) takes precedence over the config.
    if explicit_flag is not None:
        return bool(explicit_flag)

    # 3. Decision based on the configuration mode.
    mode = fs_cfg.mode
    if mode == FeatureSelectionMode.always:
        return True
    if mode == FeatureSelectionMode.disabled:
        return False
    # mode == auto: standalone feasibility check for feature selection.
    return _run_feature_selection_auto_check(
        X_train, y_train, make_steps, metric=metric, random_state=random_state
    )


def _run_feature_selection_auto_check(
    X_train: Any,
    y_train: Any,
    make_steps: StepsFactory,
    *,
    metric: str,
    random_state: int | None,
) -> bool:
    """Run the Standalone Auto-Check of feature selection usefulness.

    The ``auto`` mode is used when ``ModelTrainer.fit()`` is called
    directly without HPO. Logic:
    1. Hold-out split 80/20 with a fixed ``random_state``.
    2. Two control pipelines (without and with the selector) are trained
       on the train part; scoring is done on hold-out via
       ``get_scorer_object(metric, global_y=y_train)``.
    3. Directional comparison of "raw" scorer values: any scorer (from the
       registry or sklearn) is designed so that a larger value is better
       (neg_ metrics are already inverted for maximization), so no heuristic
       based on the metric name is needed and it does not break for
       neg_* metrics. Values are logged in user-facing semantics
       (``to_user_value``).
    4. Fail-safe: small data, split/scoring errors, or None/nan scores
       produce a WARNING and return ``False`` — the check does not crash fit().

    Cost: each ``fit()`` in ``auto`` mode additionally trains two full
    pipelines on 80% of the data — for heavy algorithms this is roughly
    2-3x the regular training time. The mode targets direct calls without
    HPO; within an HPO loop, selection activity should be passed via the
    explicit ``feature_selection_active`` flag so the check is not run for
    every trial.

    Args:
        X_train: Training feature matrix.
        y_train: Target variable.
        make_steps: Builds fresh pipeline steps (clones of the preprocessor
            and the model) for a control pipeline.
        metric: The primary metric.
        random_state: The trainer seed (``None`` — the fixed fallback 42).

    Returns:
        ``True`` if the quality with selection is strictly higher, otherwise ``False``.
    """
    n_samples = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
    if n_samples < _MIN_AUTO_CHECK_SAMPLES:
        _LOG.warning(
            "Standalone feature selection auto-check skipped: only %d "
            "samples available, need at least %d. Feature selection "
            "disabled.",
            n_samples,
            _MIN_AUTO_CHECK_SAMPLES,
        )
        return False
    try:
        # Deterministic hold-out split: with random_state=None (non-fixed
        # main training) the split must still be reproducible, otherwise the
        # selection decision would change between fit() calls. Use a fixed
        # fallback seed without touching the main training configuration.
        check_random_state = random_state if random_state is not None else 42
        X_sub, X_hold, y_sub, y_hold = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=check_random_state,
        )
        # Control pipelines are isolated from each other and from the final
        # training: ``make_steps`` returns fresh clones every call (sklearn
        # Pipeline does not clone the passed steps).
        pipe_full = Pipeline(make_steps(False, None))
        pipe_reduced = Pipeline(make_steps(True, None))
        pipe_full.fit(X_sub, y_sub)
        pipe_reduced.fit(X_sub, y_sub)

        # global_y=y_train: the global NRMSE is normalized by the full
        # target (as in HPO and final validation scoring).
        scorer = cast(Callable[..., Any], get_scorer_object(metric, global_y=y_train))
        raw_full = scorer(pipe_full, X_hold, y_hold)
        raw_reduced = scorer(pipe_reduced, X_hold, y_hold)
        if (
            raw_full is None
            or raw_reduced is None
            or not np.isfinite(float(raw_full))
            or not np.isfinite(float(raw_reduced))
        ):
            _LOG.warning(
                "Standalone feature selection auto-check produced "
                "non-finite scores (full=%r, reduced=%r). Feature "
                "selection disabled.",
                raw_full,
                raw_reduced,
            )
            return False

        # The comparison direction is determined by the scorer object
        # itself: any scorer (from the registry or sklearn) is designed so
        # that a larger "raw" value is better — neg_ metrics are already
        # inverted for maximization. No heuristic based on the metric name
        # is used here, so the direction does not break for neg_* metrics.
        active = raw_reduced > raw_full
        # Log values in user-facing semantics (positive for errors), while
        # the decision is made on the "raw" values. The direction explicitly
        # accompanies the values ("min better" for errors) — the optimizer
        # greater_is_better flag is not used.
        score_full = to_user_value(metric, raw_full)
        score_reduced = to_user_value(metric, raw_reduced)
        _LOG.info(
            "Standalone feature selection auto-check: score_full=%.4f, "
            "score_reduced=%.4f (%s) -> active=%s",
            score_full,
            score_reduced,
            direction_label(metric),
            active,
        )
        return active
    except Exception as e:  # noqa: BLE001
        _LOG.warning(
            "Standalone feature selection auto-check failed: %s. "
            "Feature selection disabled.",
            e,
        )
        return False

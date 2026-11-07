"""Data preparation step of the training pipeline (Training Engine).

Loads and validates the run configuration (``load_config``), splits the input
into X/y, applies the input-data diagnostics, reserves the outer hold-out,
resolves the ``auto`` target transformation (on the TRAIN part of the split,
never on the hold-out), the validation strategy and the column types exactly
once (``prepare_dataset``); it also registers the datetime expansion on a
saved trainer. It may import ``state`` only and must not import the other
pipeline step modules (``hpo_phases``, ``finalists``, ``select_steps``,
``artifact``, ``component``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configurable_automl_engine.common.datetime_features import (
    datetime_expansion_map,
)
from configurable_automl_engine.common.definitions import TargetTransform
from configurable_automl_engine.common.log_setup import setup_logging
from configurable_automl_engine.common.metrics import to_sklearn_name
from configurable_automl_engine.common.validation_utils import (
    check_target_constant,
    check_target_exists,
    prepare_X_y,
    validate_df_not_empty,
    validate_features,
)
from configurable_automl_engine.preprocessing import (
    _clean_uninformative_features,
    detect_feature_types,
    resolve_target_transform,
)
from configurable_automl_engine.training_engine.config_parser import (
    AUTO_HOLDOUT_FRACTION,
    AUTO_HOLDOUT_MIN_ROWS,
    Config,
    ValidationStrategy,
    read_config,
)
from configurable_automl_engine.validation import make_cv

from .state import PreparedData

if TYPE_CHECKING:
    from configurable_automl_engine.trainer import ModelTrainer

_LOG = logging.getLogger(__name__)


def _register_datetime_expansion(
    trainer: ModelTrainer,
    expansion: dict[str, list[str]] | None,
) -> None:
    """Register an upstream datetime expansion on a trainer before saving."""
    if expansion:
        trainer.register_datetime_expansion(expansion)


# --------------------------------------------------------------------------- #
#  Orchestration helpers                                                      #
#  Pipeline steps extracted from the monolithic ``train_best_model``:         #
#  load_config → prepare_dataset → execute_phases → select_winner →          #
#  persist_artifact.                                                          #
# --------------------------------------------------------------------------- #


def load_config(
    config: str | Path | Config | dict[str, Any],
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[Config, str]:
    """Load and validate the run configuration.

    Pipeline step 1. Validates the input data and the target column, coerces
    the configuration into a ``Config`` object (from a file, a dict, or an
    already-built object), and sets up file logging when required.

    Args:
        config: The training
            configuration.
        df: The input data.
        target: The target column name. Defaults to 'target'.

    Returns:
        Tuple[Config, str]: A tuple (``Config`` object, target column name).

    Raises:
        TypeError: If a config of an unsupported type is passed.
        ValueError: If the DataFrame is empty or the target column is missing.
    """
    # Centralized validation of the input data before initializing heavy
    # resources
    validate_df_not_empty(df)
    target_col = target or "target"
    check_target_exists(df, target_col)

    # If a string or Path is passed, read the file.
    # If a Config object or dict is passed, process it.
    if isinstance(config, Config):
        cfg = config
    elif isinstance(config, dict):
        _LOG.debug("CONFIG TYPE: %s", type(config))
        _LOG.debug(
            "ALGORITHMS: %s",
            (config.get("algorithms") if isinstance(config, dict) else "N/A"),
        )
        cfg = Config.model_validate(config)
    elif isinstance(config, (str, Path)):
        cfg = read_config(config)
    else:
        raise TypeError(
            f"Unsupported config type: {type(config)}. "
            f"Expected Config, dict, str, or Path."
        )

    # If a log-file path is set in the config, set up logging
    if cfg.general.log_to_file:
        setup_logging(cfg.general.log_to_file)
    return cfg, target_col


def _resolve_holdout_fraction(
    configured: float | Literal["auto"] | None, n_rows: int
) -> float | None:
    """Resolve ``general.holdout_fraction`` into a concrete fraction.

    ``'auto'`` reserves ``AUTO_HOLDOUT_FRACTION`` of the rows when the input
    has at least ``AUTO_HOLDOUT_MIN_ROWS`` rows. On smaller data no hold-out
    is reserved (it would be too small to be informative and would take rows
    away from HPO): the honest estimate is then the BBC-CV
    ``result["bbc_score"]``, and a WARNING says so. An explicit fraction
    or ``None`` is returned as is.

    Args:
        configured: The configured value.
        n_rows: The number of rows in the input data.


    Returns:
        float | None: The fraction of rows to reserve, or ``None`` when no
            outer hold-out is reserved.
    """
    if configured != "auto":
        return configured
    if n_rows >= AUTO_HOLDOUT_MIN_ROWS:
        return AUTO_HOLDOUT_FRACTION
    _LOG.warning(
        "general.holdout_fraction='auto': no outer hold-out is reserved on "
        "small data (%d rows < %d). result['score'] and result['oof_score'] "
        "are computed on the same folds that selected the algorithm and "
        "hyperparameters and are therefore optimistic; the honest estimate "
        "is the bias-corrected result['bbc_score'].",
        n_rows,
        AUTO_HOLDOUT_MIN_ROWS,
    )
    return None


def _resolve_auto_target_transform(cfg: Config, y: pd.Series) -> None:
    """Resolve ``general.target_transform: 'auto'`` into a concrete method.

    Runs once in ``prepare_dataset`` — after the outer hold-out is split off
    (the decision sample is the TRAIN part of the split, so the hold-out
    target values never influence the decision) and before the HPO phases
    and the pre-HPO GLM target-domain guard. The decision
    (``resolve_target_transform``) reads the target skewness: ``|skew(y)| >``
    ``general.target_transform_skew_threshold`` selects ``log1p`` when
    ``min(y) > -1``, otherwise ``yeo-johnson``; a more symmetric target
    keeps ``none``. The resolved method is written back into
    ``cfg.general.target_transform``, so every later consumer of the config
    (the HPO spec, the finalist trainers, the artifact trainer, the
    ``ExperimentResult`` assembly) sees the concrete method and ``auto``
    never leaks into the pipeline or the artifacts.

    Args:
        cfg: The configuration object (``cfg.general.target_transform`` is
            always replaced with the resolved value; for a non-``auto``
            configured method the resolved value is the method itself).
        y: The raw target vector (the decision sample; the TRAIN part of the
            outer split when a hold-out is reserved, the full target
            otherwise).
    """
    cfg.general.target_transform = resolve_target_transform(
        y,
        cfg.general.target_transform,
        skew_threshold=cfg.general.target_transform_skew_threshold,
    )


def _fallback_transform_for_full_y(
    resolved: TargetTransform, full_y: pd.Series
) -> TargetTransform:
    """Guard the resolved transform against the FULL data domain.

    The decision sample for ``target_transform: 'auto'`` is the train part of
    the outer split, but the final artifact is refit on ALL data (train +
    hold-out). A method with a restricted target domain (``log1p`` requires
    ``min(y) > -1``, ``boxcox`` requires ``min(y) > 0``) can violate that
    domain on the hold-out rows: the train part may resolve (or be
    configured) to ``log1p`` while the hold-out contains a value ≤ -1.
    Training would then fail in ``validate_target_for_transform`` AFTER the
    HPO budget was spent.

    When the method's domain is violated by the full target, the method is
    downgraded to ``yeo-johnson`` (any real ``y``) with a WARNING. The
    downgrade applies to the auto-resolved decision AND to an explicitly
    configured restricted method — the alternative is a guaranteed crash of
    the final fit, and the WARNING reports the override explicitly. ``none``
    and ``yeo-johnson`` accept any real ``y`` and are returned unchanged.

    The ``log1p``/``boxcox`` domain bounds themselves are undefined for
    non-finite or non-numeric ``full_y``: ``np.min`` of a vector with NaN
    returns NaN, so the comparison would silently report a violation and
    downgrade the method. To keep the helper predictable for any direct
    caller, such input raises ``ValueError`` with a message reusing the
    target-validation wording. In ``prepare_dataset`` this branch is
    unreachable — the target is validated before this point
    (``prepare_X_y`` → ``check_target_numeric``) — so the pipeline behavior
    is unchanged.

    Args:
        resolved: The resolved (or explicitly configured) method.
        full_y: The FULL target vector (train + hold-out) — the data the
            final artifact is refit on; must be numeric and finite.

    Returns:
        TargetTransform: The method to run the pipeline with — the input
            method, or ``yeo-johnson`` as the domain-safe fallback.

    Raises:
        ValueError: If ``full_y`` is non-numeric (a string vector etc.) or
            contains NaN/±inf — the domain decision is undefined for such
            data (``prepare_dataset`` rejects them earlier anyway).
    """
    if resolved not in (TargetTransform.log1p, TargetTransform.boxcox):
        return resolved
    try:
        arr = np.asarray(full_y, dtype=float).reshape(-1)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "target_transform domain guard requires a numeric full target "
            "vector (train + hold-out); cannot convert it to float."
        ) from err
    # The WHOLE vector must be finite, not just its minimum: min() of a
    # vector with +inf stays finite, so the domain decision would be made
    # on corrupted data. NaN (and ±inf) make min(y) undefined for the
    # domain check — fail fast instead of a silent downgrade.
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "target_transform domain guard requires a finite full target "
            "vector (train + hold-out); found NaN or infinite value(s), so "
            "the min(y) domain decision is undefined."
        )
    minimum = float(np.min(arr))
    lower_bound = -1.0 if resolved is TargetTransform.log1p else 0.0
    if minimum > lower_bound:
        return resolved
    _LOG.warning(
        "target_transform='%s' is violated by the FULL target "
        "(min(y)=%s <= %s, an extreme value in the reserved outer hold-out): "
        "the final artifact is refit on all data, so the method is "
        "downgraded to 'yeo-johnson' (valid for any real y). The decision "
        "was made on the train part of the split only.",
        resolved.value,
        minimum,
        lower_bound,
    )
    return TargetTransform.yeo_johnson


def prepare_dataset(cfg: Config, df: pd.DataFrame, target_col: str) -> PreparedData:
    """Prepare the data and resolve the settings for the HPO phases.

    Pipeline step 2. Splits the DataFrame into X/y, reserves the outer
    hold-out, resolves the validation strategy ('auto' → a concrete
    method/folds/test_size), resolves the 'auto' target transformation from
    the TRAIN-part target skewness (never from the hold-out rows) and
    detects column types exactly once — a guarantee of consistent
    preprocessing between HPO and the final training.

    Args:
        cfg: The configuration object.
        df: The input data.
        target_col: The target column name.

    Returns:
        PreparedData: The prepared container of data and settings.
    """
    metric_user = cfg.general.comparison_metric
    metric_sklearn = to_sklearn_name(metric_user)

    # Centralized splitting
    X, y = prepare_X_y(df, target_col)

    # Centralized input-data diagnostics: duplicate columns, ±inf in numeric
    # features, datetime/timedelta columns and ID-like columns are detected
    # and (depending on the configured policies) repaired or rejected here —
    # before the HPO budget is spent on data that would fail with an unclear
    # error or silently degrade the search.
    X_raw = X
    X, generated = validate_features(
        X,
        inf_policy=cfg.general.inf_policy,
        datetime_policy=cfg.general.datetime_policy,
        drop_id_like_columns=cfg.general.drop_id_like_columns,
    )
    # Recorded for inference: the saved trainer only ever sees the expanded
    # data, so it cannot derive the mapping itself and the artifact must
    # accept the raw schema it was trained on.
    datetime_expansion = datetime_expansion_map(X_raw, generated)
    check_target_constant(y)
    # Uninformative-feature cleaning at the same point where it runs in
    # ModelTrainer.fit / hpo.trial_pipeline.plan_preprocessing: fully empty and strictly
    # constant columns (including datetime-derived components such as
    # ts_year that are constant on the sample) are removed BEFORE type
    # detection, so the column lists in PreparedData match the feature set
    # actually used by HPO and the final fit — no drift between the
    # detected lists and the cleaned data.
    X = _clean_uninformative_features(X)

    # The engine seed: drives the data splits (and, through the trainer, the
    # model seeds). An explicit ``general.random_state=None`` means a
    # non-reproducible run — the splits are not fixed; the WARNING is logged
    # here (the pipeline entry point) once per run.
    random_state = cfg.general.random_state
    if random_state is None:
        _LOG.warning(
            "general.random_state=None: the run is NOT reproducible — "
            "data splits, model seeds and the TPE sampler are not fixed."
        )

    # Reserved outer hold-out (general.holdout_fraction): the rows are split
    # OFF before HPO and are never used for model selection. The honest
    # estimate on these rows is reported as result["holdout_score"]; the final
    # model is still trained on ALL data (train + hold-out). The split is
    # shuffled with the engine seed (general.random_state); a deterministic
    # stratified split cannot be applied to a continuous target, so a plain
    # random split is used.
    holdout_fraction = _resolve_holdout_fraction(cfg.general.holdout_fraction, len(X))
    holdout_X: pd.DataFrame | None = None
    holdout_y: pd.Series | None = None
    if holdout_fraction is not None:
        if len(X) < 2:
            raise ValueError(
                f"general.holdout_fraction requires at least 2 rows for the "
                f"outer split; the input has {len(X)} row(s)."
            )
        # Guard against a degenerate split BEFORE the sklearn call: an extreme
        # fraction (e.g. 0.999 on a tiny dataset) would otherwise surface as a
        # raw sklearn ValueError about an empty training set. sklearn computes
        # n_test = ceil(fraction * n), so n_train = n - ceil(fraction * n).
        n_hold = int(np.ceil(holdout_fraction * len(X)))
        if len(X) - n_hold < 1:
            raise ValueError(
                f"general.holdout_fraction={holdout_fraction:.6f} leaves an "
                f"EMPTY training part ({len(X)} row(s) in the input); lower "
                f"the fraction so that at least 1 row remains for HPO."
            )
        X_hold, holdout_X, y_hold, holdout_y = train_test_split(
            X,
            y,
            test_size=holdout_fraction,
            shuffle=True,
            random_state=random_state,
        )
        X, y = X_hold, y_hold
        _LOG.info(
            "Reserving outer hold-out: %.1f%% of the rows "
            "(%d train, %d hold-out); HPO runs on the train part only, the "
            "final model is trained on all rows.",
            holdout_fraction * 100.0,
            len(X),
            len(holdout_X),
        )
        if len(holdout_X) <= 2:
            # A degenerate tiny hold-out (1-2 rows) yields an estimate with
            # extremely high variance: warn explicitly so the user can raise
            # the fraction (or disable the hold-out) instead of trusting a
            # noise-level number.
            _LOG.warning(
                "Outer hold-out is very small (%d row(s)): the hold-out score "
                "will have extremely high variance. Increase "
                "general.holdout_fraction or set it to null.",
                len(holdout_X),
            )

    # Target-transform resolution ('auto'): AFTER the outer hold-out is split
    # OFF and BEFORE the HPO budget is spent (and before the pre-HPO GLM
    # target-domain guard, which must check the domains against the target
    # the trials will actually train on). The decision sample is the TRAIN
    # part of the outer split (or the full target when no hold-out is
    # reserved): the hold-out must model the unseen data, so its target
    # values must not influence how the pipeline itself is built — including
    # the choice of the target transformation (an extreme value that lands
    # in the hold-out must not flip the skewness decision, e.g. to log1p,
    # for the pipeline that is later evaluated on that hold-out). The
    # resolved concrete method is written back into cfg.general, so every
    # later consumer (the HPO spec, the finalists, the artifact trainer,
    # the run-result assembly) reads the resolved value — 'auto' itself
    # never reaches them, and the same single decision is reused for both
    # the HPO trials and the final trainer.
    _resolve_auto_target_transform(cfg, y)
    # Domain guard of the resolved method against the FULL target (train +
    # hold-out): the final artifact is refit on all data, so a restricted
    # method (log1p / boxcox) — whether auto-resolved or explicitly
    # configured — can be violated by an extreme hold-out value — that
    # would crash the fit AFTER the HPO budget was spent. Downgraded to
    # yeo-johnson with a WARNING.
    full_y = pd.concat([y, holdout_y], axis=0) if holdout_y is not None else y
    cfg.general.target_transform = _fallback_transform_for_full_y(
        cfg.general.target_transform, full_y
    )

    # Single resolution of the validation strategy: 'auto' is resolved exactly
    # once on the raw X/y — before the HPO phases. The resolved values (method
    # + number of folds / hold-out size) are passed both to HPO and to the
    # final fit, so the model comparison estimate and the final val_score do
    # not diverge (e.g. auto → train_test_split with an integer test_size from
    # choose_validation_method, not the fixed 0.2).
    val_method_eff, cv_obj, auto_decision = make_cv(
        n_samples=len(X),
        val_method=cfg.general.validation_strategy,
        n_folds=cfg.general.n_folds,
        random_state=random_state,
        test_size=0.2,
        n_features=X.shape[1],
    )
    resolved_validation = ValidationStrategy(val_method_eff)
    if val_method_eff == "k_fold":
        # The fold count is taken from the ready cv_obj: for 'auto'→kfold it
        # holds the k from the choose_validation_method decision, which may
        # differ from cfg.general.n_folds.
        assert cv_obj is not None
        resolved_n_folds = int(cv_obj.get_n_splits())
    else:
        resolved_n_folds = cfg.general.n_folds
    # For auto → train_test_split, choose_validation_method returns an integer
    # number of rows; for an explicit strategy — the fixed fraction 0.2.
    if val_method_eff == "train_test_split" and auto_decision is not None:
        resolved_test_size: float | int = int(auto_decision["test_size"])
    else:
        resolved_test_size = 0.2

    # Detect column types exactly once on the input DataFrame and pass them to
    # the HPO phase. This guarantees the same preprocessing (one-hot) between
    # HPO and the final ModelTrainer training.
    categorical_features, numerical_features = detect_feature_types(X)

    return PreparedData(
        X=X,
        y=y,
        metric_user=metric_user,
        metric_sklearn=metric_sklearn,
        resolved_validation=resolved_validation,
        resolved_n_folds=resolved_n_folds,
        resolved_test_size=resolved_test_size,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        random_state=random_state,
        holdout_X=holdout_X,
        holdout_y=holdout_y,
        holdout_fraction=holdout_fraction,
        datetime_expansion=datetime_expansion,
    )

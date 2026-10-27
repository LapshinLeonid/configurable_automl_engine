"""Data preparation step of the training pipeline (Training Engine).

Loads and validates the run configuration (``load_config``), splits the input
into X/y, applies the input-data diagnostics, reserves the outer hold-out and
resolves the validation strategy and the column types exactly once
(``prepare_dataset``); it also registers the datetime expansion on a saved
trainer. It may import ``state`` only and must not import the other pipeline
step modules (``hpo_phases``, ``finalists``, ``select_steps``, ``artifact``,
``component``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configurable_automl_engine.common.log_setup import setup_logging
from configurable_automl_engine.common.metrics import to_sklearn_name
from configurable_automl_engine.common.validation_utils import (
    check_target_constant,
    check_target_exists,
    datetime_expansion_map,
    prepare_X_y,
    validate_df_not_empty,
    validate_features,
)
from configurable_automl_engine.preprocessing import (
    _clean_uninformative_features,
    detect_feature_types,
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
    """Register an upstream datetime expansion on a trainer before saving.

    Args:
        trainer (ModelTrainer): The trained trainer instance.
        expansion (dict[str, list[str]] | None): Source column → generated
            component columns; nothing is done when empty.
    """
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
        config (Union[str, Path, Config, Dict[str, Any]]): The training
            configuration.
        df (pd.DataFrame): The input data.
        target (str | None): The target column name. Defaults to 'target'.

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
        configured (float | Literal["auto"] | None): The configured value.
        n_rows (int): The number of rows in the input data.

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


def prepare_dataset(cfg: Config, df: pd.DataFrame, target_col: str) -> PreparedData:
    """Prepare the data and resolve the settings for the HPO phases.

    Pipeline step 2. Splits the DataFrame into X/y, resolves the validation
    strategy ('auto' → a concrete method/folds/test_size), and detects column
    types exactly once — a guarantee of consistent preprocessing between HPO
    and the final training.

    Args:
        cfg (Config): The configuration object.
        df (pd.DataFrame): The input data.
        target_col (str): The target column name.

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
    # data, so it cannot derive the mapping itself (B3 — the artifact must
    # accept the raw schema it was trained on).
    datetime_expansion = datetime_expansion_map(X_raw, generated)
    check_target_constant(y)
    # Uninformative-feature cleaning at the same point where it runs in
    # ModelTrainer.fit / hpo.trial_pipeline.plan_preprocessing: fully empty and strictly
    # constant columns (including datetime-derived components such as
    # ts_year that are constant on the sample) are removed BEFORE type
    # detection, so the column lists in PreparedData match the feature set
    # actually used by HPO and the final fit (S3 — no drift between the
    # detected lists and the cleaned data).
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

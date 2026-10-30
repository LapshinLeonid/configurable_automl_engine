"""The legacy functional training API: the body of ``train_model``.

Holds :func:`run_train_model`, the logic of the ``train_model`` facade
(``trainer.train_model`` keeps the documented signature and passes
``ModelTrainer`` in). The module must not import ``trainer``: the facade is
re-exported from there, and ``trainer`` imports this module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.common.log_setup import setup_logging
from configurable_automl_engine.common.metrics import is_error_metric
from configurable_automl_engine.feature_selection import FeatureSelectionCfg


def run_train_model(
    trainer_cls: Callable[..., Any],
    cfg_or_algo: dict[str, Any] | str,
    metric_or_testsize: str | float,
    params_or_metric: dict[str, Any] | str,
    X: Any = None,
    y: Any = None,
    enable_logging: bool = False,
    random_state: int | None = 42,
    log_path: str | Path | None = None,
    *,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
    feature_selection_active: bool | None = None,
) -> float:
    """Run the legacy ``train_model`` facade with the given trainer class.

    ``trainer.train_model`` calls this with ``ModelTrainer`` (passing the
    class keeps this module independent of ``trainer``, which re-exports the
    facade). The rest of this docstring is the one of ``train_model``.

    Provide backward compatibility with the legacy model training API.
    A facade function that accepts a configuration or a set of positional
    arguments, instantiates a ModelTrainer, and returns the metric value
    computed on held-out data (hold-out or the CV estimate — the fold mean
    for additive metrics, the pooled OOF value for non-additive metrics such
    as RMSE/MSE/R²/nRMSE) — using the
    same validation method used to assess quality in ``ModelTrainer.fit()``
    (not a train score).

    The feature selection parameters (``feature_selection_cfg`` /
    ``feature_selection_active``) are keyword-only: in the "config dict"
    branch, the same-named dict keys take precedence over the explicit
    arguments (similarly to ``data_oversampling``). A key present in the config
    with the value ``None`` is treated as "value not set": in that case, as
    when the key is absent, the explicit function argument is used (fallback).
    In the simple API branch, the function arguments are used directly.

    Attributes:
        trainer_cls: The trainer class to instantiate (``ModelTrainer``).
        cfg_or_algo (dict | str): Configuration dict or algorithm name.
        metric_or_testsize (str | float): Metric or test sample size.
        params_or_metric (dict | str): Hyperparameters or metric name.
        X (Any): Feature matrix.
        y (Any): Target variable vector.
        enable_logging (bool): Flag enabling logging.
        random_state (int | None): Random seed.
        log_path (str | Path | None): Path to the log file. Used only when
            ``enable_logging`` is True: file logging is initialized with
            ``setup_logging(logfile=...)`` so the "Training finished" message
            and the library logs are written to this file. ``None`` — no file
            is created here and messages go to the already configured handlers
            (e.g. the console). The file handler is attached once per process:
            if one is already registered, the first path wins and later
            ``log_path`` values do not re-route the output. In the "config
            dict" branch, the ``log_path`` key takes precedence over this
            argument, except when the key is present with the value ``None`` —
            then this argument is used (fallback).
        feature_selection_cfg (FeatureSelectionCfg | dict | None):
            Feature selection configuration (method, percentile, min_features,
            etc.). In the "config dict" branch, the ``feature_selection_cfg``
            key takes precedence over this argument, except when the key is
            present with the value ``None`` — then this argument is used
            (fallback). ``None`` — ``FeatureSelectionCfg()`` (mode='disabled').
        feature_selection_active (bool | None): Explicit boolean flag of
            feature selection activity (takes precedence over the mode from
            ``feature_selection_cfg``). In the "config dict" branch, the
            ``feature_selection_active`` key takes precedence over this
            argument, except when the key is present with the value ``None`` —
            then this argument is used (fallback). Non-boolean values are
            rejected by the ``ModelTrainer`` constructor (strict typing).
            ``None`` — the decision is based on the configuration.
    """
    algo: str = ""
    metric: str = ""
    hyperparams: dict[str, Any] = {}
    rs: int | None = random_state

    fs_cfg: FeatureSelectionCfg | dict[str, Any] | None
    fs_active: bool | None

    # The "config dict" case
    if isinstance(cfg_or_algo, dict):
        cfg: dict[str, Any] = cfg_or_algo
        algo = str(cfg.get("algorithm", ""))
        metric = str(cfg.get("metric", ""))
        hyperparams = cast(dict[str, Any], cfg.get("hyperparams", {}))
        rs = cast(int | None, cfg.get("random_state", 42))
        enable_logging = bool(cfg.get("enable_logging", False))
        # ``log_path`` follows the same precedence rule as
        # ``feature_selection_cfg``: the config key takes precedence over the
        # explicit argument, except when the key is present with the value
        # None ("value not set") — then the explicit argument is used
        # (fallback).
        if "log_path" in cfg and cfg["log_path"] is not None:
            log_path = cast(str | Path | None, cfg["log_path"])
        data_os = bool(cfg.get("data_oversampling", False))
        data_os_mult = float(cfg.get("data_oversampling_multiplier", 1.0))
        data_os_alg = str(cfg.get("data_oversampling_algorithm", "random"))
        # Config keys take precedence over the explicit arguments (similarly to
        # ``data_oversampling``); a key with the value None is treated as
        # "value not set" and falls back to the function argument (see docstring).
        if "feature_selection_cfg" in cfg and cfg["feature_selection_cfg"] is not None:
            fs_cfg = cast(
                FeatureSelectionCfg | dict[str, Any] | None,
                cfg["feature_selection_cfg"],
            )
        else:
            fs_cfg = feature_selection_cfg
        if (
            "feature_selection_active" in cfg
            and cfg["feature_selection_active"] is not None
        ):
            fs_active = cast(bool | None, cfg["feature_selection_active"])
        else:
            fs_active = feature_selection_active
    else:
        # Simple API
        # If None is passed, we do NOT turn it into the string "None" right
        # away to preserve the original validation logic for the tests.
        algo = cfg_or_algo if cfg_or_algo is not None else ""
        metric = str(metric_or_testsize)
        hyperparams = cast(dict[str, Any], params_or_metric)
        rs = random_state
        data_os = False
        data_os_mult = 1.0
        data_os_alg = "random"
        fs_cfg = feature_selection_cfg
        fs_active = feature_selection_active

    # Validate the algorithm
    if not isinstance(cfg_or_algo, (str, dict)):
        raise TrainingError("Invalid algorithm")
    algo_key = algo.lower()

    if not hyperparams and algo_key not in ["isotonic", "isotonic_regression"]:
        raise TrainingError("Model parameters are not specified")

    # Create and train the ModelTrainer
    trainer = trainer_cls(
        algorithm=algo,
        hyperparams=hyperparams,
        metric=metric,
        random_state=rs,
        data_oversampling=data_os,
        data_oversampling_multiplier=data_os_mult,
        data_oversampling_algorithm=data_os_alg,
        feature_selection_cfg=fs_cfg,
        feature_selection_active=fs_active,
    )
    trainer.fit(X, y)
    val_score = trainer.val_score
    if val_score is None:
        raise TrainingError("Model did not return a metric value")

    # Logging (if enable_logging=True)
    if enable_logging:
        # Initialize file logging when an explicit log path is provided.
        # ``setup_logging`` attaches a file handler only once per process:
        # when one is already registered, the first path wins and this
        # argument does not re-route existing output (see the docstring).
        if log_path is not None:
            setup_logging(logfile=Path(log_path))
        # Get the logger and write the message
        logger = logging.getLogger(__name__)
        # Metric type follows the user-facing representation: errors (RMSE/MAE,
        # including neg_ metrics whose value is inverted back) are labeled as
        # "Error (Natural)", while score metrics (R² and similar) are labeled
        # as "Score".
        metric_type = "Error (Natural)" if is_error_metric(metric) else "Score"

        logger.info(
            f"Training finished: Algorithm={algo_key}, "
            f"Metric={metric.upper()} ({metric_type}), "
            f"Value={val_score:.4f} "
            f"(validation={trainer.validation_strategy})"
        )

    return float(val_score)

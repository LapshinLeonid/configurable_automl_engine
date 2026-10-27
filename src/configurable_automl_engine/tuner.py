"""Hyperparameter Optimisation Module:
An engine for automated hyperparameter search.

The module provides a high-level wrapper around Optuna for automatic model
configuration tuning with support for dynamic search spaces, cross-validation
and integrated oversampling.

Key features:
    1. Hybrid Model Zoo: Full support for the AutoML engine's base model pool
       with extensions for the SGD, GaussianProcess, Isotonic and GLM families.
    2. Adaptive Validation: Intelligent switching between k-fold,
       Leave-One-Out and Train-Test Split (80/20) depending on the sample size
       (fallback when n_samples < 2k).
    3. Integrated Oversampling: Seamless class-balancing integration through an
       Imbalance-Pipeline right inside the optimisation process.
    4. Dynamic Search Spaces: Support for both hard-coded spaces (e.g. the
       adaptive KNN space) and external configurations via YAML/SearchSpaceEntry.
    5. Metric Agnostic: Optimisation against any standard or custom Sklearn
       metric (R² by default).

This module is the public façade: ``optimize()`` keeps its signature,
collects its arguments into an ``hpo.request.HpoRequest`` and delegates the
run to ``hpo.runner.run_optimization``; the exceptions,
``OptimizeResult`` and the sentinel constants are re-exported from
``hpo.types``. The implementation lives in the ``hpo`` package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from optuna.trial import FrozenTrial, Trial

from configurable_automl_engine.common.complexity_guard import (
    ComplexityGuardConfig,
)
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
)
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
)
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
)

from .common.settings import EncodingSettings, InputPolicies, OversamplingSettings
from .hpo.request import HpoRequest, as_complexity_guard_config
from .hpo.runner import run_optimization
from .hpo.types import (
    HPO_WORST_SCORE,
    WORST_SCORE_THRESHOLD,
    HPOHistoryError,
    HyperoptError,
    InvalidAlgorithmError,
    InvalidDataError,
    OptimizeResult,
)

__all__ = [
    "HPO_WORST_SCORE",
    "WORST_SCORE_THRESHOLD",
    "HPOHistoryError",
    "HyperoptError",
    "InvalidAlgorithmError",
    "InvalidDataError",
    "OptimizeResult",
    "optimize",
]


# ═════════════════════ PUBLIC: optimize() ═══════════════════════════════════
def optimize(
    algo_name: str,
    X: Any,
    y: Any,
    *,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
    data_oversampling_target_bins: int = 8,
    data_oversampling_add_noise: bool = False,
    data_oversampling_noise_level: float = 0.01,
    metric: str = "r2",
    # keep the old argument for backward compatibility
    val_method: ValidationStrategy | str = "auto",
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
    high_cardinality_threshold: int | None = 50,
    high_cardinality_encoding: EncodingStrategy | None = "frequency",
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
    complexity_guard_cfg: ComplexityGuardConfig | dict[str, Any] | None = None,
    user_hyperparameter_keys: list[str] | None = None,
    limit_hyperparameters: bool = False,
    fit_best_model: bool = True,
    phase_name: str | None = None,
    inf_policy: str = "as_missing",
    datetime_policy: str = "expand",
    drop_id_like_columns: bool = False,
    collect_oof: bool = False,
) -> OptimizeResult | None:
    """Run the model hyperparameter optimisation process using Optuna.

    The function automatically selects the validation strategy, sets up the
    parameter search space and trains the final model on all provided data
    (unless ``fit_best_model=False`` is passed).

    The defaults of the parameters that have a ``general.*`` counterpart in
    the engine config (validation strategy, high-cardinality mode, complexity
    guard, ±inf / datetime / ID-like policies, encoding settings) are the
    same as in the config, so a direct call and a config run with the same
    settings behave identically.

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
        data_oversampling_target_bins (int): Number of quantile bins for the
            regression oversampling strategy. Defaults to 8.
        data_oversampling_add_noise (bool): Whether to add Gaussian noise to
            the numeric features of the oversampled training fold.
            Defaults to False.
        data_oversampling_noise_level (float): Noise intensity relative to
            the column standard deviation. Defaults to 0.01.
        metric (str): Metric name to maximise. Defaults to 'r2'.
        val_method (ValidationStrategy | str): Validation method
            ('auto', 'k_fold', 'leave_one_out', 'train_test_split').
            Defaults to 'auto', as ``general.validation_strategy``.
        validation_strategy (ValidationStrategy | str | None): Alias for
            ``val_method`` (takes precedence).
        n_folds (int): Number of cross-validation folds. Defaults to 5.
        n_trials (int): Number of search iterations (trials). Defaults to 50.
        random_state (int | None): Randomness state for the data splits, the
            preprocessor and the model seeds (CV folds, train_test_split,
            pruning folds, hashing / target-encoding cross-fit, stochastic
            estimator seeding). Defaults to 42. With ``None``, the data splits
            are not fixed and stochastic models draw a fresh RNG at fit time
            (a fully non-deterministic run); however, the feature selection
            step (``FeatureSelector``) always uses the fixed seed 42 so that
            selection stays reproducible across runs.
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
        space_overrides (dict | None): Search space overrides keyed by the
            algorithm name. ``None`` (or no entry for ``algo_name``) — the
            default space of the algorithm (``DEFAULT_SPACES``), the same
            one a config run without ``hyperparameters`` searches. Each
            value may be:

            * a callable (a custom ``space_fn``) — used as is;
            * a dict of ``SearchSpaceEntry`` objects;
            * a dict whose values are the same shorthands accepted by the
              Config path: a raw range list (``[low, high]`` /
              ``[low, high, "int"|"float"|"float_log"]`` /
              ``[options, "categorical"]``) is parsed and a range whose type
              was not written explicitly is re-resolved against the
              parameter's type (the default search space or the estimator
              signature), the same way ``Config`` does it — so
              ``space_overrides={"svr": {"C": [1, 50]}}`` searches ``C`` as
              a real range, not an integer-only one.

            A dict is merged over ``DEFAULT_SPACES`` (a partial override
            searches the listed ranges plus every other default range),
            unless ``limit_hyperparameters=True``, which searches only the
            listed parameters. Both forms go through the same physical
            clipping and complexity guard as a config run. An algorithm
            without a default space needs an override, otherwise
            ``HyperoptError`` is raised.
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
            Defaults to 50, as ``general.high_cardinality_threshold``.
            ``None`` (together with ``high_cardinality_encoding=None``) —
            the mode is disabled.
        high_cardinality_encoding (EncodingStrategy | None): Encoding strategy
            for high-cardinality columns. Set together with
            ``high_cardinality_threshold``. Defaults to 'frequency', as
            ``general.high_cardinality_encoding``.
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
            (from ``general.complexity_guard``). ``None`` — the defaults of
            ``general.complexity_guard`` (mode 'apply'); ``{"mode": "off"}``
            disables the guard. When enabled, the dict-based search spaces of trees/boostings are
            tightened depending on ``n`` (effective training-fold rows) and
            ``p`` (features after preprocessing).
        user_hyperparameter_keys (list[str] | None): Names of parameters set
            explicitly by the user (``algo.hyperparameters``); the guard
            intersects its ranges with these user ranges and applies
            ``user_range_policy`` on an empty intersection. ``None`` — the
            keys of the dict ``space_overrides`` entry of the algorithm (no
            dict override — no user ranges).
        limit_hyperparameters (bool): Restrict the search to ONLY the
            parameters present in ``space_overrides`` for this algorithm
            (``AlgoCfg.limit_hyperparameters: true``): no ``DEFAULT_SPACES``
            entries are added and the SVR adaptive epsilon is not injected.
            Defaults to False — user ranges override and are merged into
            ``DEFAULT_SPACES``.
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
        inf_policy (str): Policy for ±inf values in numeric features during
            the data validation: ``"as_missing"`` (default, as
            ``general.inf_policy``) — replace with NaN and warn; ``"error"`` —
            reject the data.
        datetime_policy (str): Policy for datetime/timedelta columns during
            the data validation: ``"expand"`` (default) — replace with
            derived components; ``"drop"`` — drop; ``"error"`` — reject.
        drop_id_like_columns (bool): Whether ID-like columns are dropped
            (default ``False`` — they are only reported with a WARNING).

            The three policies above are applied once, on entry, and both the
            HPO trials and the final best-model fit work on the repaired
            matrix. The defaults are the same as in ``ModelTrainer`` and the
            engine config; the engine forwards the configured ``general.*``
            values to both. The returned ``model`` is a plain
            pipeline: unlike ``ModelTrainer.predict`` it does NOT re-apply
            the datetime expansion, so it expects the repaired schema.
        collect_oof (bool): Whether to keep the out-of-fold predictions of
            every COMPLETED trial with a valid score and return them in
            ``OptimizeResult.oof_predictions`` (the input of the BBC-CV
            estimate). Costs ``n_trials × len(X)`` floats of memory.
            Defaults to False.

    Returns:
        OptimizeResult: The result of the phase — ``model`` (the trained
            model with the best parameters; ``None`` when
            ``fit_best_model=False`` — the fit is skipped), ``params`` (the
            found optimal hyperparameters; in ``'auto'`` mode it contains
            the auxiliary ``use_feature_selection`` key with the selection
            decision), ``score`` (the best raw "optimizer" metric value on
            validation, higher is better), ``trials`` — the frozen
            trials of the current phase ONLY (transferred
            ``previous_trials`` are excluded), and
            ``n_trials_evaluated`` — the number of trials actually
            evaluated by the phase (own COMPLETED trials; transferred
            history never counts).
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
    # Thin wrapper: the keyword arguments are collected into one request
    # object, and the run itself lives in ``hpo.runner``.
    request = HpoRequest(
        algo_name=algo_name,
        metric=metric,
        n_trials=n_trials,
        # the alias takes precedence
        val_method=validation_strategy
        if validation_strategy is not None
        else val_method,
        n_folds=n_folds,
        train_test_split_test_size=train_test_split_test_size,
        random_state=random_state,
        sampler_seed=sampler_seed,
        space_overrides=space_overrides,
        user_hyperparameter_keys=(
            None if user_hyperparameter_keys is None else list(user_hyperparameter_keys)
        ),
        limit_hyperparameters=limit_hyperparameters,
        complexity_guard=as_complexity_guard_config(complexity_guard_cfg),
        initial_params=initial_params,
        previous_trials=previous_trials,
        preprocessor=preprocessor,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        encoding=EncodingSettings(
            encoding=encoding,
            high_cardinality_threshold=high_cardinality_threshold,
            high_cardinality_encoding=high_cardinality_encoding,
            hashing_n_components=hashing_n_components,
            target_encoding_smoothing=target_encoding_smoothing,
            target_encoding_fallback=target_encoding_fallback,
        ),
        preprocessing_override=preprocessing_override,
        feature_selection_cfg=feature_selection_cfg,
        oversampling=OversamplingSettings(
            enable=data_oversampling,
            multiplier=data_oversampling_multiplier,
            algorithm=data_oversampling_algorithm,
            target_bins=data_oversampling_target_bins,
            add_noise=data_oversampling_add_noise,
            noise_level=data_oversampling_noise_level,
        ),
        pruning=pruning,
        policies=InputPolicies(
            inf_policy=inf_policy,
            datetime_policy=datetime_policy,
            drop_id_like_columns=drop_id_like_columns,
        ),
        fit_best_model=fit_best_model,
        phase_name=phase_name,
        collect_oof=collect_oof,
    )
    return run_optimization(X, y, request)

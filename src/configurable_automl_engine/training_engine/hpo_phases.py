"""HPO phase execution of the training pipeline (Training Engine).

Runs a hyperparameter search for one algorithm through a pluggable tuner
(``_run_hpo``), prepares the per-algorithm search space, executes an HPO phase
for every candidate (``_hpo_phase_worker`` — a module-level function so that
it can be pickled in the "processes" mode) and merges the worker results into
the phase context. It also holds the target-domain guard, the BBC-CV memory
budget check and the complexity guard record merging. It may import ``state``
only and must not import ``data_prep``, ``finalists``, ``select_steps``,
``artifact`` or ``component``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Iterable
from types import ModuleType
from typing import Any

import pandas as pd
from optuna.trial import FrozenTrial

from configurable_automl_engine.common.complexity_guard import ComplexityGuardConfig
from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
from configurable_automl_engine.common.target_domain import check_target_domain
from configurable_automl_engine.models import remap_legacy_param_names
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    FeatureSelectionCfg,
    FeatureSelectionMode,
    ValidationStrategy,
)

# ───────────────────────── canonical IAE ─────────────────────── #
from ..tuner import HPOHistoryError, OptimizeResult
from ..tuner import InvalidAlgorithmError as _CanonicalIAE
from .metrics import BBC_MAX_OOF_CELLS, direction_label, to_user_value
from .state import (
    PhaseContext,
    PhaseResult,
    PreparedData,
    _HpoPhaseSpec,
    _HPOWorkerResult,
)

# The records keep the logger name of the pre-split ``component`` module: the
# logger name is part of the observable behavior (handlers, filters and
# ``caplog`` assertions select records by it).
_LOG = logging.getLogger("configurable_automl_engine.training_engine.component")


def _algorithms_as_dict(algorithms_cfg: Any) -> dict[str, AlgoCfg]:
    """Convert an AlgorithmsConfig into a plain dictionary {name: AlgoCfg}."""
    # model_fields via an instance also works, but it is deprecated
    # (PydanticDeprecatedSince211, removal in V3.0) — use the class
    return {
        name: algo_cfg
        for name in type(algorithms_cfg).model_fields
        if (algo_cfg := getattr(algorithms_cfg, name)) is not None
    }


def _feature_selection_mode(
    cfg: FeatureSelectionCfg | dict[str, Any] | None,
) -> str | None:
    """Extract the string value of ``mode`` from a feature selection config.

    Args:
        cfg: A ``FeatureSelectionCfg`` object, a config dict, or ``None``.

    Returns:
        str | None: The mode ('disabled'/'always'/'auto') or ``None`` if the
            configuration is missing or contains no mode.
    """
    if isinstance(cfg, FeatureSelectionCfg):
        return cfg.mode.value
    if isinstance(cfg, dict):
        mode = cfg.get("mode")
        if isinstance(mode, FeatureSelectionMode):
            return mode.value
        return mode if isinstance(mode, str) else None
    return None


# --------------------------------------------------------------------------- #
#  Dyn-import helper                                                          #
# --------------------------------------------------------------------------- #
def _load_module(path: str) -> ModuleType:
    """Import a module by the given path.
    Dynamically loads a Python module using dotted-path notation.
    On failure, raises an informative exception.
    Args:
        path (str): The module path in 'package.module' format.
    Returns:
        ModuleType: The imported module object.
    Raises:
        ImportError: If the module is not found at the given path.
    """
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(f"Can't import module '{path}'") from exc


# --------------------------------------------------------------------------- #
#  HPO wrapper                                                                #
# --------------------------------------------------------------------------- #
def _run_hpo(  # noqa: C901
    *,
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    metric_name_sklearn: str,
    n_trials: int,
    validation_strategy: ValidationStrategy,
    n_folds: int | None = None,
    train_test_split_test_size: float = 0.2,
    search_space_override: dict[str, Any] | None = None,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
    data_oversampling_target_bins: int = 8,
    data_oversampling_add_noise: bool = False,
    data_oversampling_noise_level: float = 0.01,
    initial_params: dict[str, Any] | None = None,
    previous_trials: list[FrozenTrial] | None = None,
    random_state: int | None = None,
    sampler_seed: int | None = None,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    encoding: str = "one_hot",
    pruning: dict[str, Any] | None = None,
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: str | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
    complexity_guard_cfg: ComplexityGuardConfig | dict[str, Any] | None = None,
    user_hyperparameter_keys: list[str] | None = None,
    limit_hyperparameters: bool = False,
    phase_name: str | None = None,
    inf_policy: str | None = None,
    datetime_policy: str | None = None,
    drop_id_like_columns: bool | None = None,
    collect_oof: bool = False,
) -> PhaseResult | None:
    """Run the hyperparameter search for an algorithm.
    Workflow:
    1. Tuner loading: the module specified in `algo_cfg.tuner` is imported.
    2. Signature check: the tuner's `optimize` method is inspected for support
       of specific parameters (n_folds, oversampling, space_overrides,
       previous_trials, random_state).
    3. Execution: the optimization is run. If the algorithm is deemed invalid
       via `InvalidAlgorithmError`, the canonical exception is raised. The
       phase's own trials are returned inside the ``PhaseResult``.
    The HPO phase consumes only the best parameters and score: the final
    full-data fit of the best model is skipped (``fit_best_model=False`` for
    tuners that support it) and the model is discarded — the engine trains
    the final model itself with ``ModelTrainer``.
    Args:
        algo_name (str): The algorithm name.
        algo_cfg (AlgoCfg): The algorithm configuration with tuner paths.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        metric_name_sklearn (str): The metric name in sklearn format.
        n_trials (int): The number of search iterations.
        validation_strategy (ValidationStrategy): The validation strategy
            (k_fold, loo, etc.).
        n_folds (int | None): The number of folds for cross-validation.
        train_test_split_test_size (float): The hold-out size for the
            'train_test_split' strategy (a fraction in (0, 1) or an integer
            number of rows after 'auto' resolution). Passed to tuners that
            support the parameter so that the HPO estimate does not diverge
            from the final val_score.
        search_space_override (Dict[str, Any] | None):
            The overridden search space.
        data_oversampling (bool): The oversampling enable flag.
        data_oversampling_multiplier (float): The sample size multiplier.
        data_oversampling_algorithm (str): The oversampling algorithm name.
        data_oversampling_target_bins (int): Number of quantile bins for the
            regression oversampling strategy.
        data_oversampling_add_noise (bool): Whether to add Gaussian noise to
            the oversampled numeric features.
        data_oversampling_noise_level (float): Noise intensity relative to the
            column standard deviation.
        initial_params (dict[str, Any] | None): Hyperparameters from the
            previous HPO phase for enqueue_trial (monotonic improvement between
            phases).
        previous_trials (list[FrozenTrial] | None): The TPE history of the
            previous phase, forwarded to tuners that accept the argument
            (``study.add_trials`` warm-start). A tuner WITHOUT the
            ``previous_trials`` argument that receives non-empty history is a
            configuration error — ``HPOHistoryError`` is raised (fail fast),
            refine must not silently degrade to a cold run. An empty list is a
            no-op.
        random_state (int | None): The random seed of the data splits and the
            preprocessor (CV folds, train_test_split, pruning folds, hashing /
            target-encoding cross-fit). Always forwarded to tuners that accept
            the argument — for the bundled
            ``configurable_automl_engine.tuner`` it is the
            ``general.random_state`` engine seed so that every HPO phase (and
            the final ``ModelTrainer`` fit) evaluates models on the SAME
            folds; ``None`` — non-reproducible splits.
        sampler_seed (int | None): The seed of the TPE sampler, separated from
            ``random_state``: it changes per HPO phase
            (``general.random_state + hpo_phase_ordinal``) so that TPE
            startup points do not repeat, while the data splits stay fixed.
            ``None`` — the sampler is not seeded. Forwarded to tuners that
            accept the ``sampler_seed`` argument (the bundled tuner); custom
            tuners without the argument are unaffected.
        categorical_features (list[str] | None): Categorical column names,
            detected once in ``train_best_model`` and passed to the tuner.
        numerical_features (list[str] | None): Numerical column names.
        encoding (str): The categorical encoding strategy ('one_hot' or
            'ordinal'), passed to the tuner for preprocessing consistent with
            the final fit.
        pruning (dict[str, Any] | None): Early stopping (pruning) settings
            from the ``general.pruning`` section. Passed only to tuners that
            support the ``pruning`` argument; custom tuners are unaffected.
        feature_selection_cfg (FeatureSelectionCfg | dict | None): The feature
            selection config from the root ``Config.general.feature_selection``.
            Passed only to tuners that support the ``feature_selection_cfg``
            argument or accept ``**kwargs`` (the signature is checked via
            ``inspect.signature``); custom tuners are unaffected.
        phase_name (str | None): The name of the HPO phase (e.g. 'coarse').
        inf_policy (str | None): ``general.inf_policy``, forwarded to tuners
            that accept it (``None`` — not forwarded, the tuner default).
        datetime_policy (str | None): ``general.datetime_policy``, forwarded
            the same way.
        drop_id_like_columns (bool | None): ``general.drop_id_like_columns``,
            forwarded the same way.
            Forwarded to tuners that accept the argument and used to
            deduplicate the WARNING about the pooled-metric fallback (once
            per algorithm and phase).
        limit_hyperparameters (bool): Restrict the search to ONLY the
            user-listed parameters (``AlgoCfg.limit_hyperparameters: true``):
            the prepared search space contains no ``DEFAULT_SPACES`` entries,
            and the tuner must not add any parameter beyond it (e.g. the SVR
            adaptive epsilon). Forwarded to tuners that accept the argument.
    Returns:
        PhaseResult | None: The phase result — score, params and the phase's
            own Optuna trials; or ``None`` if an error occurred during HPO
            (the algorithm is excluded from candidates; NO fallback to the
            previous phase's parameters).
    Raises:
        _CanonicalIAE: If the tuner reports that the algorithm is incompatible
            with the data.
        HPOHistoryError: If ``previous_trials`` are passed to a tuner that
            does not accept the argument, or the transferred history is
            incompatible (fail fast — the pipeline fails with a clear
            message).
        AttributeError: If the tuner module lacks an `optimize` function.
    """
    if algo_cfg.tuner is None:
        raise ValueError("Tuner path is not configured")
    tuner = _load_module(algo_cfg.tuner)
    if not hasattr(tuner, "optimize"):
        raise AttributeError(f"Module {algo_cfg.tuner} lacks `optimize`")

    sig = inspect.signature(tuner.optimize)
    # A signature with **kwargs is considered compatible (same as
    # _fit_and_save): the tuner itself decides what to do with extra keys, so
    # the feature-selection helper arguments can be passed safely.
    tuner_accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    # TPE history transfer (refine): a tuner without `previous_trials` that
    # receives non-empty history is a configuration error — fail fast with the
    # tuner name instead of silently degrading refine to a cold run. A
    # ``**kwargs`` signature is considered compatible (the tuner itself
    # decides what to do with the extra key), same as for the other helper
    # arguments below.
    if (
        previous_trials
        and "previous_trials" not in sig.parameters
        and not tuner_accepts_var_kwargs
    ):
        raise HPOHistoryError(
            f"Tuner '{algo_cfg.tuner}' does not accept the `previous_trials` "
            f"argument, but the pipeline needs to transfer TPE history for "
            f"algorithm '{algo_name}'. Update the tuner to accept "
            f"`previous_trials` (list of optuna FrozenTrial) or drop the "
            f"refine phase for this algorithm."
        )

    kwargs: dict[str, Any] = {
        "algo_name": algo_name,
        "X": X,
        "y": y,
        "metric": metric_name_sklearn,
        "n_trials": n_trials,
        "validation_strategy": validation_strategy,
    }

    # pass the data-split seed if the tuner supports it (mandatory for the
    # bundled tuner: every HPO phase and the final fit must use the same
    # folds/preprocessor — general.random_state)
    if "random_state" in sig.parameters:
        kwargs["random_state"] = random_state

    # pass the per-phase TPE sampler seed if the tuner supports it (mandatory
    # for the bundled tuner: each HPO phase uses a different seed so that TPE
    # startup points do not repeat, while the folds stay fixed)
    if "sampler_seed" in sig.parameters:
        kwargs["sampler_seed"] = sampler_seed

    # pass n_folds if the module supports it
    if (
        validation_strategy == ValidationStrategy.k_fold
        and n_folds is not None
        and "n_folds" in sig.parameters
    ):
        kwargs["n_folds"] = n_folds

    # pass the resolved hold-out size: for 'auto' → 'train_test_split'
    # choose_validation_method returns an integer number of rows, which must
    # match both in HPO and in the final fit.
    if "train_test_split_test_size" in sig.parameters:
        kwargs["train_test_split_test_size"] = train_test_split_test_size

    # pass oversampling parameters if the tuner supports them
    if "data_oversampling" in sig.parameters:
        kwargs["data_oversampling"] = data_oversampling

    if "data_oversampling_multiplier" in sig.parameters:
        kwargs["data_oversampling_multiplier"] = data_oversampling_multiplier

    if "data_oversampling_algorithm" in sig.parameters:
        kwargs["data_oversampling_algorithm"] = data_oversampling_algorithm

    if "data_oversampling_target_bins" in sig.parameters:
        kwargs["data_oversampling_target_bins"] = data_oversampling_target_bins

    if "data_oversampling_add_noise" in sig.parameters:
        kwargs["data_oversampling_add_noise"] = data_oversampling_add_noise

    if "data_oversampling_noise_level" in sig.parameters:
        kwargs["data_oversampling_noise_level"] = data_oversampling_noise_level

    # pass the custom search space if provided
    if search_space_override is not None:
        # We pass a dict {algo_name: hyperparameters_dict}
        overrides = {algo_name: search_space_override}
        if "space_overrides" in sig.parameters:
            kwargs["space_overrides"] = overrides

    # pass initial_params if available (for the refine phase)
    if initial_params is not None and "initial_params" in sig.parameters:
        kwargs["initial_params"] = initial_params

    # pass the TPE history of the previous phase (refine) if available;
    # a ``**kwargs`` tuner receives the key as well (it decides what to do)
    if previous_trials and (
        "previous_trials" in sig.parameters or tuner_accepts_var_kwargs
    ):
        kwargs["previous_trials"] = previous_trials

    # pass the detected categorical/numerical columns so that the HPO phase
    # builds its preprocessor consistently with the final training
    if "categorical_features" in sig.parameters:
        kwargs["categorical_features"] = categorical_features

    if "numerical_features" in sig.parameters:
        kwargs["numerical_features"] = numerical_features

    # pass the encoding strategy if the tuner supports it
    if "encoding" in sig.parameters:
        kwargs["encoding"] = encoding
    else:
        # The custom tuner does not accept encoding: the strategy is silently
        # ignored during HPO but applied in the final training -> a mismatch in
        # the feature representation. Warn so the user can update the tuner to
        # one supporting the `encoding` argument.
        _LOG.warning(
            "Tuner %s does not accept `encoding`; categorical_encoding=%r "
            "will NOT be applied during HPO (final fit may diverge).",
            algo_cfg.tuner,
            encoding,
        )

    # pass the preprocessing preset override: an explicit user setting from the
    # configuration applies both in HPO and in the final training
    preprocessing_override = getattr(algo_cfg, "preprocessing", None)
    if "preprocessing_override" in sig.parameters:
        kwargs["preprocessing_override"] = preprocessing_override

    # pass the early stopping (pruning) settings if the tuner supports them;
    # custom tuners without the `pruning` argument are unaffected
    if pruning is not None and "pruning" in sig.parameters:
        kwargs["pruning"] = pruning

    # pass the high-cardinality encoding parameters and the new strategies if
    # the tuner supports them; consistent with the final training
    if "high_cardinality_threshold" in sig.parameters:
        kwargs["high_cardinality_threshold"] = high_cardinality_threshold

    if "high_cardinality_encoding" in sig.parameters:
        kwargs["high_cardinality_encoding"] = high_cardinality_encoding

    if "hashing_n_components" in sig.parameters:
        kwargs["hashing_n_components"] = hashing_n_components

    if "target_encoding_smoothing" in sig.parameters:
        kwargs["target_encoding_smoothing"] = target_encoding_smoothing

    if "target_encoding_fallback" in sig.parameters:
        kwargs["target_encoding_fallback"] = target_encoding_fallback

    # pass the feature selection config if the tuner supports the
    # `feature_selection_cfg` argument or accepts **kwargs; custom tuners
    # without support are unaffected. When feature selection is active, warn
    # about the HPO ↔ final fit mismatch (as with encoding): the configuration
    # will not be applied during the search phase.
    if "feature_selection_cfg" in sig.parameters or tuner_accepts_var_kwargs:
        kwargs["feature_selection_cfg"] = feature_selection_cfg
    elif _feature_selection_mode(feature_selection_cfg) not in (None, "disabled"):
        _LOG.warning(
            "Tuner %s does not accept `feature_selection_cfg`; feature "
            "selection mode=%r will NOT be applied during HPO "
            "(final fit may diverge).",
            algo_cfg.tuner,
            _feature_selection_mode(feature_selection_cfg),
        )

    # pass the adaptive complexity guard settings if the tuner supports them;
    # custom tuners without the arguments are unaffected
    if complexity_guard_cfg is not None and "complexity_guard_cfg" in sig.parameters:
        kwargs["complexity_guard_cfg"] = complexity_guard_cfg

    if user_hyperparameter_keys and "user_hyperparameter_keys" in sig.parameters:
        kwargs["user_hyperparameter_keys"] = user_hyperparameter_keys

    if limit_hyperparameters and "limit_hyperparameters" in sig.parameters:
        kwargs["limit_hyperparameters"] = limit_hyperparameters

    # The HPO phase needs only the best parameters and score — the final model
    # is trained later by the engine's own trainer. Ask a tuner that supports
    # skipping the final full-data fit to do so: otherwise one redundant full
    # training per algorithm per phase is wasted (the model is discarded
    # below).
    if "fit_best_model" in sig.parameters:
        kwargs["fit_best_model"] = False

    # pass the HPO phase name if the tuner supports it: the bundled tuner
    # uses it to deduplicate the WARNING about the pooled-metric fallback
    # (emitted once per algorithm and phase).
    if phase_name is not None and "phase_name" in sig.parameters:
        kwargs["phase_name"] = phase_name

    # Input-data diagnostics policies: the same values as for the final
    # ModelTrainer (_build_trainer), so the tuner's own data validation follows
    # the configuration instead of its own defaults. On the engine path the data is already repaired in
    # prepare_dataset, so re-validation is a no-op; the forwarding keeps
    # custom tuners consistent with the final fit.
    for key, value in (
        ("inf_policy", inf_policy),
        ("datetime_policy", datetime_policy),
        ("drop_id_like_columns", drop_id_like_columns),
    ):
        if value is not None and key in sig.parameters:
            kwargs[key] = value

    # BBC-CV input: the out-of-fold predictions of every trial. Only the
    # bundled tuner (or a custom one with the same argument) supports it; for
    # the others the BBC-CV estimate is simply not reported.
    if collect_oof and "collect_oof" in sig.parameters:
        kwargs["collect_oof"] = True

    try:
        result = tuner.optimize(**kwargs)
        if result is None:
            # Tuner couldn't get a valid result: either all trials failed
            # (optuna.TrialPruned) or it failed otherwise. Return None so the
            # main loop knows the algorithm didn't pass: the dummy candidate
            # with score -3.4e38 and params=None should no longer end up in
            # phase_results.
            _LOG.error(
                "Algorithm %s failed during HPO: no valid result (tuner returned None)",
                algo_name,
            )
            return None
        if isinstance(result, OptimizeResult):
            # The bundled tuner: a self-documenting result with the phase's
            # own trials (transferred history excluded).
            best_score, best_params, trials = (
                result.score,
                result.params,
                list(result.trials),
            )
            complexity_guard_logs: list[dict[str, Any]] = list(
                result.complexity_guard or []
            )
            # The honest number of trials the best score was selected from
            # (own COMPLETED trials of the phase).
            n_trials_evaluated = int(result.n_trials_evaluated or 0)
            oof_predictions = result.oof_predictions
        else:
            # A legacy custom tuner returning the (model, params, score)
            # triple; the trials are unavailable (empty history).
            _, best_params, best_score = result
            trials = []
            complexity_guard_logs = []
            n_trials_evaluated = 0
            oof_predictions = None
        if best_score is None or best_params is None:
            # Tuner couldn't get a valid result: either all trials failed
            # (optuna.TrialPruned) or it failed otherwise. Return None so the
            # main loop knows the algorithm didn't pass: the dummy candidate
            # with score -3.4e38 and params=None should no longer end up in
            # phase_results.
            _LOG.error(
                "Algorithm %s failed during HPO: no valid result "
                "(best_score=%r, best_params=%r)",
                algo_name,
                best_score,
                best_params,
            )
            return None
        return PhaseResult(
            score=best_score,
            params=best_params,
            trials=trials,
            complexity_guard=complexity_guard_logs,
            n_trials_evaluated=n_trials_evaluated,
            oof_predictions=oof_predictions,
        )
    except Exception as err:
        if err.__class__.__name__ == "InvalidAlgorithmError":
            raise _CanonicalIAE(str(err)) from err
        if isinstance(err, HPOHistoryError):
            # TPE history transfer problems are configuration errors: fail
            # fast with a clear message instead of silently excluding the
            # algorithm.
            raise
        _LOG.error("Algorithm %s failed during HPO: %s", algo_name, err, exc_info=True)
        # Return None so the main loop knows the algorithm didn't pass.
        return None


def _prepare_search_space(
    algo_name: str,
    user_overrides: dict[str, Any] | None,
    limit: bool = False,
) -> dict[str, Any]:
    """Prepare the hyperparameter search space.

    Merges the system defaults with the user overrides from the
    configuration. A module-level function (not a closure) so that the
    ``_hpo_phase_worker`` worker is serializable in the "processes" mode.

    When ``limit`` is True the search is restricted to ONLY the parameters
    listed in ``user_overrides`` (``AlgoCfg.hyperparameters``): no
    ``DEFAULT_SPACES`` entries are added, and every other parameter stays at
    the estimator default. When ``limit`` is False (the default) the user
    ranges override and are merged into ``DEFAULT_SPACES`` — the current
    behavior.

    A limited search must not be empty: the caller passes the already
    validated ``AlgoCfg.hyperparameters``, whose keys are guaranteed to be
    allowed for the algorithm (the config validator rejects unknown names).

    Legacy hyperparameter names (e.g. ``n_iter`` → ``max_iter`` for
    ARDRegression) are remapped ONCE here, before the Optuna loop — not at
    every model instantiation (which would log an identical "Remapped ..."
    line for ~100 trials).

    Args:
        algo_name (str): The algorithm name used to look up defaults.
        user_overrides (dict[str, Any] | None): A dict of parameters to override.
        limit (bool): Restrict the search to the user-listed parameters only
            (no ``DEFAULT_SPACES`` entries). Defaults to False.

    Returns:
        dict[str, Any]: The resulting search space.
    """
    if limit:
        if not user_overrides:
            return {}
        # Only the user-listed parameters are searched; everything else
        # stays at the estimator default. Legacy names are remapped once.
        return remap_legacy_param_names(algo_name, user_overrides)

    space = DEFAULT_SPACES.get(algo_name, {}).copy()
    if user_overrides:
        # Remap legacy names in the user overrides BEFORE the merge so that
        # an explicit user value (e.g. ARDRegression n_iter → max_iter) wins
        # over the system default of the modern name.
        space.update(remap_legacy_param_names(algo_name, user_overrides))

    # Adaptive epsilon for SVR: the epsilon search bounds are scaled by the
    # y_train spread at runtime (tuner.resolve_search_space), so the
    # default blind range [1e-3, 1.0] from DEFAULT_SPACES is dropped. If the
    # user set epsilon explicitly — it is already in space
    # (space.update(user_overrides) above), and the user's priority is
    # preserved: the adaptive logic does not interfere.
    if algo_name == "svr" and not (user_overrides and "epsilon" in user_overrides):
        space.pop("epsilon", None)

    # Remap legacy names once (e.g. ARDRegression n_iter → max_iter)
    # so that per-trial clean_hyperparameters has nothing to remap/log.
    return remap_legacy_param_names(algo_name, space)


def _execute_hpo_phase(
    phase_name: str,
    algo: str,
    a_cfg: AlgoCfg,
    n_trials: int,
    spec: _HpoPhaseSpec,
    X: pd.DataFrame,
    search_space: dict[str, Any] | None = None,
    initial_params: dict[str, Any] | None = None,
    previous_trials: list[FrozenTrial] | None = None,
    random_state: int | None = None,
    sampler_seed: int | None = None,
    user_hyperparameter_keys: list[str] | None = None,
    limit_hyperparameters: bool = False,
) -> PhaseResult | None:
    """Run a specific HPO phase for an algorithm (module-level function).

    Provides phase logging and oversampling result processing. The function
    is module-level (not a closure): all phase data is passed through
    ``spec`` so that the ``_hpo_phase_worker`` worker is serializable in
    the "processes" mode. The feature matrix ``X`` is passed as a separate
    argument (not inside ``spec``): in the processes mode ``run_parallel``
    transfers it through shared memory / disk, so it is not pickled for
    every algorithm.

    Args:
        phase_name (str): The current phase name (e.g. 'coarse').
        algo (str): The algorithm name.
        a_cfg (AlgoCfg): The algorithm configuration.
        n_trials (int): The number of iterations in this phase.
        spec (_HpoPhaseSpec): Snapshot of the HPO phase parameters.
        X (pd.DataFrame): The feature matrix.
        search_space (dict[str, Any] | None): The search space for the phase.
        initial_params (dict[str, Any] | None): Warm-start parameters (refine).
        previous_trials (list[FrozenTrial] | None): The TPE history of the
            previous phase (refine warm-start), forwarded to ``_run_hpo``.
        random_state (int | None): The random seed of the data splits and the
            preprocessor (the ``general.random_state`` engine seed), forwarded
            to ``_run_hpo``; ``None`` — non-reproducible splits.
        sampler_seed (int | None): The per-phase TPE sampler seed
            (``general.random_state + hpo_phase_ordinal``), forwarded to
            ``_run_hpo``; ``None`` — the sampler is not seeded.
        limit_hyperparameters (bool): Restrict the search to ONLY the
            user-listed parameters (``AlgoCfg.limit_hyperparameters: true``);
            forwarded to ``_run_hpo``.

    Returns:
        PhaseResult | None: The phase result (score, params, trials), or
            ``None`` when HPO did not return a valid result (the algorithm
            is excluded from candidates).

    Raises:
        Exception: Exceptions from ``_run_hpo`` are logged and re-raised
            (including ``InvalidAlgorithmError`` and ``HPOHistoryError``).
    """
    _LOG.info(f"=== {phase_name} phase: {algo} ({n_trials} tries) ===")

    try:
        result = _run_hpo(
            algo_name=algo,
            algo_cfg=a_cfg,
            X=X,
            y=spec.y,
            metric_name_sklearn=spec.metric_sklearn,
            n_trials=n_trials,
            validation_strategy=spec.resolved_validation,
            n_folds=spec.resolved_n_folds,
            train_test_split_test_size=spec.resolved_test_size,
            search_space_override=search_space,
            data_oversampling=spec.oversampling_enabled,
            data_oversampling_multiplier=spec.oversampling_multiplier,
            data_oversampling_algorithm=spec.oversampling_algorithm,
            data_oversampling_target_bins=spec.oversampling_target_bins,
            data_oversampling_add_noise=spec.oversampling_add_noise,
            data_oversampling_noise_level=spec.oversampling_noise_level,
            initial_params=initial_params,
            previous_trials=previous_trials,
            random_state=random_state,
            sampler_seed=sampler_seed,
            categorical_features=spec.categorical_features,
            numerical_features=spec.numerical_features,
            encoding=spec.encoding,
            pruning=spec.pruning_cfg,
            high_cardinality_threshold=spec.high_cardinality_threshold,
            high_cardinality_encoding=spec.high_cardinality_encoding,
            hashing_n_components=spec.hashing_n_components,
            target_encoding_smoothing=spec.target_encoding_smoothing,
            target_encoding_fallback=spec.target_encoding_fallback,
            feature_selection_cfg=spec.feature_selection_cfg,
            complexity_guard_cfg=spec.complexity_guard_cfg,
            user_hyperparameter_keys=user_hyperparameter_keys,
            limit_hyperparameters=limit_hyperparameters,
            phase_name=phase_name,
            inf_policy=spec.inf_policy,
            datetime_policy=spec.datetime_policy,
            drop_id_like_columns=spec.drop_id_like_columns,
            collect_oof=spec.collect_oof,
        )

        if result is None:
            return None

        # Log the value in user semantics: for neg_*-metrics (e.g.
        # neg_root_mean_squared_error) the raw scorer value is inverted, so
        # the natural (positive) value is shown to the user. The direction
        # accompanies the value explicitly ("min better" for errors): the
        # optimizer's greater_is_better flag is not used.
        disp = to_user_value(spec.metric_sklearn, result.score)
        _LOG.info(
            f"{phase_name} {algo:15} | score {disp:.5f} "
            f"({direction_label(spec.metric_sklearn)}) | params {result.params}"
        )
        return result
    except Exception as err:
        _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
        raise


def _hpo_phase_worker(
    algo_name: str,
    algo_cfg: AlgoCfg,
    spec: _HpoPhaseSpec,
    X: pd.DataFrame,
) -> _HPOWorkerResult:
    """Worker for parallel or sequential execution of an HPO task.

    A module-level function (not a closure): serializable in the "processes"
    mode. Does not mutate the shared context: the disqualification is
    returned in the result's ``disqualify_reason`` and applied in the main
    process (otherwise a write from another process would be lost).

    Args:
        algo_name (str): The algorithm name.
        algo_cfg (AlgoCfg): The algorithm configuration.
        spec (_HpoPhaseSpec): Snapshot of the HPO phase parameters.
        X (pd.DataFrame): The feature matrix (transferred through shared
            memory / disk by ``run_parallel`` in the "processes" mode instead
            of being pickled inside ``spec`` for every algorithm).

    Returns:
        _HPOWorkerResult: The task result: score/params/trials on success,
            ``disqualify_reason`` on a circuit breaker disqualification,
            otherwise a result without a score and reason.
    """
    # Determine initial_params for refine: each candidate
    # warm-starts from its OWN previous-phase parameters (enqueue_trial in the
    # tuner guarantees score monotonicity). The prev_results snapshot lives in
    # the immutable spec (frozen at phase definition time), so the worker uses
    # the state at phase start. Workers only READ this dict; writes to
    # phase_results happen after all phase workers finish, and at the end of
    # the phase phase_results is re-bound to the filtered valid_results
    # (objects are never mutated) — therefore the snapshot is stable in both
    # parallel and sequential modes.
    # Determine the warm-start for refine: each candidate warm-starts from
    # its OWN previous-phase record — initial_params (the enqueue_trial
    # anchor, which guarantees score monotonicity) AND previous_trials (the
    # TPE history transferred via study.add_trials). The prev_results
    # snapshot lives in the immutable spec (frozen at phase definition time),
    # so the worker uses the state at phase start. Workers only READ this
    # dict; writes to phase_results happen after all phase workers finish,
    # and at the end of the phase phase_results is re-bound to the filtered
    # valid_results (objects are never mutated) — therefore the snapshot is
    # stable in both parallel and sequential modes.
    init_params = None
    previous_trials: list[FrozenTrial] | None = None
    if spec.phase.action == "refine" and algo_name in spec.prev_results:
        prev = spec.prev_results[algo_name]
        init_params = prev.params
        previous_trials = prev.trials or None

    # 1. Take the system defaults and overlay what is in AlgoCfg (from YAML/JSON).
    #    In the limited mode (limit_hyperparameters: true) ONLY the user-listed
    #    parameters are searched — no DEFAULT_SPACES entries.
    limit = bool(getattr(algo_cfg, "limit_hyperparameters", False))
    full_search_space = _prepare_search_space(
        algo_name,
        algo_cfg.hyperparameters,  # this is the dict from config_parser
        limit=limit,
    )
    # The names of parameters the user set explicitly (algo.hyperparameters):
    # the complexity guard treats them as "user ranges" (intersected, with
    # user_range_policy on an empty intersection).
    user_keys = list(algo_cfg.hyperparameters or {})
    try:
        result = _execute_hpo_phase(
            spec.phase.name,
            algo_name,
            algo_cfg,
            spec.phase.n_trials,
            spec,
            X,
            full_search_space,
            initial_params=init_params,
            previous_trials=previous_trials,
            random_state=spec.random_state,
            sampler_seed=spec.sampler_seed,
            user_hyperparameter_keys=user_keys,
            limit_hyperparameters=limit,
        )

        if result is None:
            return _HPOWorkerResult(name=algo_name)

        return _HPOWorkerResult(
            name=algo_name,
            score=result.score,
            params=result.params,
            trials=list(result.trials),
            complexity_guard=list(result.complexity_guard or []),
            n_trials_evaluated=int(result.n_trials_evaluated or 0),
            oof_predictions=result.oof_predictions,
        )
    except _CanonicalIAE as err:
        # Circuit breaker: the algorithm is disqualified after
        # MAX_FATAL_FAILURES consecutive fatal errors. This is NOT an abort
        # of the whole run — the algorithm is excluded from the candidates,
        # the rest continue training (same as a None result).
        _LOG.warning(
            "Algorithm %s disqualified in phase %s: %s",
            algo_name,
            spec.phase.name,
            err,
        )
        return _HPOWorkerResult(name=algo_name, disqualify_reason=str(err))
    except HPOHistoryError as err:
        # TPE history transfer problems are configuration errors: fail fast
        # with a clear message instead of silently excluding the algorithm
        # (the pipeline must not continue with a degraded refine). The error
        # is carried in the result (a string, serializable in the "processes"
        # mode) and re-raised in the main process by
        # ``_consume_worker_results``: raising here would be swallowed by
        # ``run_parallel``, which converts worker exceptions to ``None``.
        _LOG.error(
            "Algorithm %s cannot transfer TPE history in phase %s: %s",
            algo_name,
            spec.phase.name,
            err,
        )
        return _HPOWorkerResult(name=algo_name, history_error=str(err))
    except Exception as e:  # noqa: BLE001
        _LOG.warning(f"Algorithm {algo_name} failed in phase {spec.phase.name}: {e}")
        return _HPOWorkerResult(name=algo_name)


def _consume_worker_results(
    results: Iterable[_HPOWorkerResult | None],
    phase_results: dict[str, PhaseResult],
    ctx: PhaseContext,
) -> None:
    """Distribute worker results into the working set and disqualifications.

    In the "processes" mode the worker cannot write to ``ctx`` directly
    (changes in the child process would be lost), so disqualifications are
    returned in the results and applied here, in the main process.
    ``None`` results (a task timeout/failure in the pool) are ignored.
    A ``history_error`` result re-raises ``HPOHistoryError`` in the main
    process: TPE history transfer problems are configuration errors and must
    fail the run even in the parallel mode (where ``run_parallel`` swallows
    worker exceptions).

    Args:
        results (Iterable[_HPOWorkerResult | None]): Worker results.
        phase_results (dict[str, PhaseResult]): The working set of the
            current phase — valid results land here.
        ctx (PhaseContext): The phase context — disqualifications merge here.

    Raises:
        HPOHistoryError: If a worker reported a TPE history transfer error
            (fail fast — the pipeline must not continue with a degraded
            refine).
    """
    for res in results:
        if res is None:
            continue
        if res.history_error is not None:
            raise HPOHistoryError(res.history_error)
        if res.disqualify_reason is not None:
            ctx.disqualified_algorithms[res.name] = res.disqualify_reason
            continue
        if res.score is None or res.params is None:
            continue
        # Every evaluated configuration was a candidate of the selection, so
        # its OOF predictions enter the BBC-CV pool even if the algorithm is
        # later dropped by a select step.
        if res.oof_predictions is not None:
            ctx.oof_pool.append(res.oof_predictions)
        phase_results[res.name] = PhaseResult(
            score=res.score,
            params=res.params,
            trials=list(res.trials or []),
            complexity_guard=list(res.complexity_guard or []),
            n_trials_evaluated=int(res.n_trials_evaluated or 0),
            oof_predictions=res.oof_predictions,
        )


def _apply_target_domain_guard(
    all_algorithms: dict[str, AlgoCfg],
    y: pd.Series,
    ctx: PhaseContext,
) -> dict[str, AlgoCfg]:
    """Exclude or restrict GLMs whose loss domain does not fit the target.

    Runs once, before the first HPO phase. ``gammaregressor`` (``y > 0``) and
    ``poissonregressor`` (``y >= 0``) on a target outside their domain — and
    ``tweedieregressor`` / ``glm`` whose whole ``power`` search range is
    invalid for the target — would fail on every fold of every trial, so they
    are put into the circuit breaker (``ctx.disqualified_algorithms``) with a
    reason and a single WARNING, and no trial is ever run for them. A Tweedie
    ``power`` range that is only partly invalid is clipped to its valid part
    instead (the algorithm stays a candidate).

    Args:
        all_algorithms (dict[str, AlgoCfg]): The configured algorithms.
        y (pd.Series): The target vector.
        ctx (PhaseContext): The phase context — disqualifications land here.

    Returns:
        dict[str, AlgoCfg]: The algorithms with clipped search spaces applied
            (unchanged configs are returned as is; the input is not mutated).
    """
    guarded: dict[str, AlgoCfg] = {}
    for name, a_cfg in all_algorithms.items():
        if not a_cfg.enable or name in ctx.disqualified_algorithms:
            guarded[name] = a_cfg
            continue
        space = _prepare_search_space(
            name,
            a_cfg.hyperparameters,
            limit=bool(a_cfg.limit_hyperparameters),
        )
        check = check_target_domain(name, space, y)
        if check.disqualify_reason is not None:
            _LOG.warning(
                "Algorithm %s disqualified before HPO: %s",
                name,
                check.disqualify_reason,
            )
            ctx.disqualified_algorithms[name] = check.disqualify_reason
            guarded[name] = a_cfg
            continue
        if check.clipped_entries:
            for message in check.messages:
                _LOG.warning("Target domain guard: %s", message)
            hyperparameters = {
                **(a_cfg.hyperparameters or {}),
                **check.clipped_entries,
            }
            a_cfg = a_cfg.model_copy(update={"hyperparameters": hyperparameters})
        guarded[name] = a_cfg
    return guarded


def _bbc_enabled(
    cfg: Config, prepared: PreparedData, algorithms: dict[str, AlgoCfg]
) -> bool:
    """Decide whether the OOF predictions of every trial are kept for BBC-CV.

    BBC-CV is the honest estimate when no outer hold-out is reserved. The
    memory cost is known before HPO: ``n_rows`` floats per evaluated trial,
    at most the sum of ``n_trials`` over the HPO phases times the number of
    enabled algorithms. Above ``BBC_MAX_OOF_CELLS`` the estimate is skipped
    with a WARNING.

    Args:
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data (hold-out already split).
        algorithms (dict[str, AlgoCfg]): The configured algorithms.

    Returns:
        bool: True when the OOF predictions must be collected.
    """
    if prepared.holdout_X is not None:
        return False
    n_algorithms = sum(1 for a in algorithms.values() if a.enable)
    n_trials = sum(p.n_trials for p in cfg.general.phases if p.action != "select")
    cells = n_algorithms * n_trials * len(prepared.y)
    if cells > BBC_MAX_OOF_CELLS:
        _LOG.warning(
            "BBC-CV estimate skipped: keeping the OOF predictions of up to %d "
            "trials x %d rows exceeds the budget of %d values. Reserve an "
            "outer hold-out (general.holdout_fraction) for an honest estimate.",
            n_algorithms * n_trials,
            len(prepared.y),
            BBC_MAX_OOF_CELLS,
        )
        return False
    return True


def _merge_complexity_guard_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge per-(algo, param, level) guard records into one record per parameter.

    The guard applies several rules to the same parameter (level A then
    level B), producing one record per rule, and the same restrictions repeat
    in every HPO phase (the guard is deterministic between phases). The report
    documents "one record per restricted parameter", so the raw per-rule/per
    -phase list is merged:

    * the ``after`` range of a later rule (the final state of the parameter)
      becomes the record's ``after``;
    * the ``before`` of the first record and the final ``after`` span the
      whole adaptation;
    * the ``levels`` field lists all applied rule levels (e.g. ``["A", "B"]``);
    * a record produced in an earlier phase is not duplicated by a later one
      (identical (algo, param, before, after) collapses).

    Args:
        records: Raw guard records (per rule and per phase).

    Returns:
        list[dict[str, Any]]: One record per (algo, param) with ``levels``
            aggregating the applied rule levels.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        key = (rec.get("algo", "?"), rec.get("param", "?"))
        cur = merged.get(key)
        if cur is None:
            merged[key] = dict(rec)
            cur = merged[key]
            cur["levels"] = [rec["level"]] if rec.get("level") else []
            continue
        # Same parameter in a later phase or a later rule of the same phase:
        # the final range of the parameter is the latest ``after``.
        cur["after"] = rec["after"]
        if rec.get("level") and rec["level"] not in cur["levels"]:
            cur["levels"].append(rec["level"])
        # The aggregate "before" is the earliest one (the starting range
        # before any rule of the guard was applied).
        if cur.get("before") == cur.get("after") and rec.get("before") != rec.get(
            "after"
        ):
            cur["before"] = rec["before"]
        # The guard restriction is "applied" if any rule actually applied it.
        cur["applied"] = bool(cur.get("applied") or rec.get("applied"))
        cur["fixed"] = bool(cur.get("fixed") or rec.get("fixed"))
        if rec.get("conflict"):
            cur["conflict"] = True
    return list(merged.values())

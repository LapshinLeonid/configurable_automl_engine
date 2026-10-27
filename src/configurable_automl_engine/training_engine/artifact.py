"""Final training, artifact persistence and the report (Training Engine).

Assembles the ``ModelTrainer`` of an algorithm from the configuration
(``_build_trainer`` — the single assembly point shared with the finalists),
trains and saves the winner (``_fit_and_save``), computes the reported
estimates (OOF score, outer hold-out score, BBC-CV) and builds the result
dict (``persist_artifact``). It may import ``state``, ``data_prep`` and
``hpo_phases`` and must not import ``finalists``, ``select_steps`` or
``component``.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    ValidationStrategy,
)

from .data_prep import _register_datetime_expansion
from .hpo_phases import _algorithms_as_dict
from .metrics import (
    bbc_cv_score,
    get_scorer_object,
    pooled_scorer_value,
    to_sklearn_name,
    to_user_value,
)
from .state import PreparedData

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Final fit & save                                                           #
# --------------------------------------------------------------------------- #
def _build_trainer(
    algo_name: str,
    algo_cfg: AlgoCfg,
    best_params: dict[str, Any],
    cfg: Config,
    metric_name_sklearn: str = "r2",
    *,
    validation_strategy: ValidationStrategy | str = "auto",
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> ModelTrainer:
    """Construct a ``ModelTrainer`` for an algorithm from its configuration.

    The single assembly point for the trainer, used both by the final training
    of the winner (``_fit_and_save``) and by the training of finalists on 100%
    of the data (``_fit_finalist``): it guarantees that both paths pass the
    same preprocessing, oversampling, encoding, feature selection, and
    validation settings to the trainer.

    Args:
        algo_name (str): The name of the selected algorithm.
        algo_cfg (AlgoCfg): The algorithm configuration.
        best_params (Dict[str, Any]): The found optimal hyperparameters.
        cfg (Config): The common configuration object.
        metric_name_sklearn (str): The main metric name in sklearn format.
        validation_strategy (ValidationStrategy | str): The resolved
            validation strategy (the engine passes the method resolved once
            in ``prepare_dataset``, so the final ``val_score`` is computed
            with the same method as the model comparison estimate in HPO).
        n_folds (int): The number of folds (for 'k_fold').
        test_size (float | int): The hold-out size.
        random_state (int | None): The engine seed from
            ``general.random_state``; drives the ModelTrainer's data splits
            and model seeds. ``None`` — non-reproducible final training.

    Returns:
        ModelTrainer: The constructed (but not yet trained) trainer.

    Raises:
        ValueError: If ``best_params`` is ``None`` (a signal of a complete HPO
            failure).
    """
    if best_params is None:
        raise ValueError(
            f"Algorithm '{algo_name}' produced no hyperparameters "
            "(best_params is None): HPO phase failed completely. The algorithm "
            "should have been excluded before the final fit."
        )
    # The HPO winner's feature selection decision: in 'auto' mode the
    # "use_feature_selection" key is fixed in the Optuna best_params; in
    # 'always'/'disabled' modes the key is absent -> None, and ModelTrainer
    # resolves the activity itself from the configuration. The helper key is
    # popped from a copy: the original best_params (also returned in
    # result["params"]) is not mutated, and "use_feature_selection" is
    # guaranteed not to leak into the ModelTrainer hyperparameters.
    # ``isotonic_feature`` is the informational name of the numerical feature
    # selected by the isotonic HPO (reported in the params, not a constructor
    # or data-adapter hyperparameter) — it is popped for the same reason.
    trainer_params = dict(best_params)
    fs_active = trainer_params.pop("use_feature_selection", None)
    trainer_params.pop("isotonic_feature", None)

    general = cfg.general
    oversampling = cfg.oversampling
    return ModelTrainer(
        algorithm=algo_name,
        hyperparams=trainer_params,
        metric=metric_name_sklearn,
        # The engine seed: the ModelTrainer's data splits and model seeds
        # follow ``general.random_state`` (None — a non-reproducible fit).
        random_state=random_state,
        data_oversampling=oversampling.enable,
        data_oversampling_multiplier=oversampling.multiplier,
        data_oversampling_algorithm=oversampling.algorithm,
        data_oversampling_target_bins=oversampling.target_bins,
        data_oversampling_add_noise=oversampling.add_noise,
        data_oversampling_noise_level=oversampling.noise_level,
        serialization_format=general.serialization_format,
        encoding_strategy=general.categorical_encoding,
        additional_metrics=general.additional_metrics,
        # Explicit preprocessing preset override from the config: applied
        # consistently with the HPO phase.
        preprocessing_override=algo_cfg.preprocessing,
        high_cardinality_threshold=general.high_cardinality_threshold,
        high_cardinality_encoding=general.high_cardinality_encoding,
        hashing_n_components=general.hashing_n_components,
        target_encoding_smoothing=general.target_encoding_smoothing,
        target_encoding_fallback=general.target_encoding_fallback,
        # Feature selection: the root configuration plus the HPO winner's
        # decision.
        feature_selection_cfg=general.feature_selection,
        feature_selection_active=fs_active,
        validation_strategy=validation_strategy,
        n_folds=n_folds,
        test_size=test_size,
        # Input-data diagnostics policies: the final fit treats ±inf /
        # datetime / ID-like columns exactly as the pipeline entry point
        # (prepare_dataset has already applied them; re-validation is a
        # no-op on the already-repaired data). The ID-like WARNING is
        # suppressed: prepare_dataset already reported it once.
        inf_policy=general.inf_policy,
        datetime_policy=general.datetime_policy,
        drop_id_like_columns=general.drop_id_like_columns,
        warn_id_like=False,
    )


def _fit_and_save(
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    model_path: Path,
    cfg: Config,
    metric_name_sklearn: str = "r2",
    *,
    validation_strategy: ValidationStrategy | str = "auto",
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int | None = None,
    datetime_expansion: dict[str, list[str]] | None = None,
) -> ModelTrainer:
    """Train the final model and save the result to disk.
    Args:
        algo_name (str): The name of the selected algorithm.
        algo_cfg (AlgoCfg): The algorithm configuration with the trainer path.
        X (pd.DataFrame): The full feature matrix for training.
        y (pd.Series): The full target vector.
        best_params (Dict[str, Any]): The found optimal hyperparameters.
        model_path (Path): The path to save the model file to.
        cfg (Config): The common configuration object used to obtain the
            oversampling settings.
        metric_name_sklearn (str): The main metric name in sklearn format.
        validation_strategy (ValidationStrategy | str): The resolved
            validation strategy of the final model.
        n_folds (int): The number of folds (for 'k_fold').
        test_size (float | int): The hold-out size (a fraction or a number
            of rows after 'auto' resolution).
        random_state (int | None): The engine seed (``general.random_state``)
            for the final model's splits and model seeds.
        datetime_expansion (dict[str, list[str]] | None): The datetime
            expansion applied upstream (``PreparedData.datetime_expansion``);
            registered on the trainer before saving so that the artifact
            accepts the raw input schema.
    Returns:
        ModelTrainer: The trainer after training and saving (used to access
            the additional metric values).
    Raises:
        ValueError: If ``best_params`` is ``None`` — a sign that HPO returned
            no valid hyperparameters (protection against the TypeError
            'NoneType' object is not iterable).
    """
    trainer = _build_trainer(
        algo_name,
        algo_cfg,
        best_params,
        cfg,
        metric_name_sklearn=metric_name_sklearn,
        validation_strategy=validation_strategy,
        n_folds=n_folds,
        test_size=test_size,
        random_state=random_state,
    )
    trainer.fit(X, y)
    _register_datetime_expansion(trainer, datetime_expansion)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(model_path)
    return trainer


def _oof_score_in_user_metric(
    metric_user: str,
    trainer: Any,
    y: pd.Series,
) -> float | None:
    """Compute the comparison metric on the final pipeline's OOF vector.

    The OOF vector is built on the same splits that HPO used to select the
    winner and its hyperparameters, so the value carries the same best-of-N
    selection bias as ``result["score"]`` (on ``k_fold``/``loo`` without an
    outer hold-out it equals the best-trial score for pooled metrics). It is
    a diagnostic, not an independent estimate of generalization: the honest
    estimate is ``result["holdout_score"]``.

    The comparison metric is recomputed on the final pipeline's pooled OOF
    predictions (``trainer.oof_predictions_``) against the true target —
    the same values the trainer's OOF channel uses, expressed in the
    comparison metric semantics. The raw scorer value over the aligned
    vector is obtained via ``metrics.pooled_scorer_value`` (the same masking
    contract as the trainer's OOF channel: non-finite pairs are dropped) and
    converted to user semantics (``to_user_value``) — so error metrics are
    reported positive and score metrics (R² and similar) keep their natural
    direction.

    Args:
        metric_user (str): The user-facing comparison metric name.
        trainer (Any): The trained ``ModelTrainer`` (OOF attributes populated).
        y (pd.Series): The full target vector (for the OOF rows alignment).

    Returns:
        float | None: The OOF estimate in the comparison metric semantics,
            or ``None`` when the OOF vector is unavailable (no valid pairs).
    """
    oof_pred = getattr(trainer, "oof_predictions_", None)
    if oof_pred is None:
        return None
    try:
        oof_true = np.asarray(y, dtype=float).reshape(-1)
        oof_pred_arr = np.asarray(oof_pred, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        # The OOF vector is not array-convertible (e.g. a mocked trainer
        # without a real fit): the estimate is unavailable.
        return None
    if oof_true.shape != oof_pred_arr.shape:
        # A shape mismatch (e.g. a mocked vector) means no aligned pairs.
        return None
    mask = np.isfinite(oof_true) & np.isfinite(oof_pred_arr)
    if not np.any(mask):
        return None
    try:
        scorer = get_scorer_object(
            metric_user, global_y=np.asarray(y, dtype=float).reshape(-1)
        )
    except Exception as err:  # noqa: BLE001 — an unknown metric must not crash the report
        _LOG.warning(
            "OOF score could not be computed for metric '%s': %s", metric_user, err
        )
        return None
    try:
        raw = pooled_scorer_value(scorer, oof_true[mask], oof_pred_arr[mask])
    except (ValueError, TypeError):
        # The pooled value is impossible (the scorer exposes no raw metric
        # function): fall back to the RMSE_oof value already computed by the
        # trainer. The fallback is exact ONLY when the comparison metric IS
        # RMSE (the trainer's OOF channel always computes RMSE_oof); for any
        # other error metric (MAE, MSE, NRMSE, ...) RMSE_oof is a DIFFERENT
        # metric, so silently substituting it would mislabel the report — the
        # key is omitted instead. For score metrics (R² and similar) RMSE_oof
        # is not an estimate of the score metric either — omitted as well.
        if metric_user.lower().strip().startswith("neg_"):
            metric_base = metric_user.lower().strip()[4:]
        else:
            metric_base = metric_user.lower().strip()
        if metric_base != "rmse":
            return None
        oof_score = getattr(trainer, "oof_score_", None)
        if oof_score is None:
            return None
        return to_user_value(metric_user, -float(oof_score))
    if not np.isfinite(raw):
        return None
    return to_user_value(metric_user, raw)


def _compute_holdout_score(
    *,
    cfg: Config,
    winner_cfg: AlgoCfg,
    prepared: PreparedData,
    winner_algo: str,
    final_params: dict[str, Any],
) -> float:
    """Compute the honest hold-out score of the winner.

    Trains the winning algorithm with its best parameters on the train part
    of the reserved outer hold-out (the rows that were split off before HPO),
    predicts the hold-out rows, and returns the comparison metric
    (``general.comparison_metric``) in user semantics. The final artifact is
    still trained on ALL data by ``persist_artifact``; this estimate is a
    true out-of-sample quality of the chosen pipeline configuration.

    Args:
        cfg (Config): The configuration object.
        winner_cfg (AlgoCfg): The configuration of the winning algorithm.
        prepared (PreparedData): The prepared data and settings (X/y are the
            TRAIN part of the outer split; ``holdout_X``/``holdout_y`` are
            the reserved rows).
        winner_algo (str): The winning algorithm name.
        final_params (Dict[str, Any]): The winner's best hyperparameters.

    Returns:
        float: The comparison metric value on the hold-out rows in user
            semantics. ``float("nan")`` when the estimate cannot be computed.
    """
    try:
        # The trainer is assembled through the same single point as the final
        # fit (``_build_trainer``): the feature-selection helper key
        # (``use_feature_selection``) is popped from a copy by the same
        # contract. The validation parameters are forwarded so the pipeline
        # matches the engine's fit.
        holdout_trainer = _build_trainer(
            winner_algo,
            winner_cfg,
            final_params,
            cfg,
            metric_name_sklearn=to_sklearn_name(prepared.metric_user),
            validation_strategy=prepared.resolved_validation,
            n_folds=prepared.resolved_n_folds,
            test_size=prepared.resolved_test_size,
            random_state=prepared.random_state,
        )
        # Train on the train part of the outer split only: the hold-out rows
        # never participate in the fit, so the estimate on them is honest.
        holdout_trainer.fit(prepared.X, prepared.y)
        pred = np.asarray(
            holdout_trainer.predict(prepared.holdout_X), dtype=float
        ).reshape(-1)
        true = np.asarray(prepared.holdout_y, dtype=float).reshape(-1)
        mask = np.isfinite(true) & np.isfinite(pred)
        if not np.any(mask):
            return float("nan")
        scorer = get_scorer_object(
            prepared.metric_user,
            global_y=np.asarray(prepared.y, dtype=float).reshape(-1),
        )
        raw = pooled_scorer_value(scorer, true[mask], pred[mask])
        if not np.isfinite(raw):
            return float("nan")
        return to_user_value(prepared.metric_user, raw)
    except Exception as err:  # noqa: BLE001 — a hold-out estimate failure is non-fatal
        _LOG.warning(
            "Hold-out score could not be computed for winner '%s': %s",
            winner_algo,
            err,
        )
        return float("nan")


def persist_artifact(
    *,
    cfg: Config,
    prepared: PreparedData,
    winner_algo: str,
    final_score: float,
    final_params: dict[str, Any],
    model_path_override: str | Path | None,
    disqualified_algorithms: dict[str, str],
    pre_trained_trainer: Any | None = None,
    sanity_gate_sections: dict[str, dict[str, Any]] | None = None,
    complexity_guard_sections: dict[str, Any] | None = None,
    n_trials_evaluated: int | None = None,
    oof_pool: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Final training of the winner, artifact persistence, and report assembly.

    Pipeline step 5. Trains the winning algorithm's model on the full dataset
    via ``_fit_and_save`` (or reuses the already-trained finalist of the
    terminal ``sanity_gate`` select step — unless ``general.holdout_fraction``
    is set, see below), saves the artifact to disk, and
    builds the result dict: ``algorithm``, ``score`` (in user semantics),
    ``metric``, ``params``, ``model_path``; when present it also adds
    ``disqualified_algorithms`` (the circuit breaker, including Sanity Gate
    select-step disqualifications), ``additional_metrics``, ``sanity_gate``
    — a dict of the report sections of the ``sanity_gate`` select steps (key —
    the phase name), ``oof_score`` — the comparison metric computed on the
    final pipeline's pooled out-of-fold predictions (on the same splits that
    selected the winner, so it is optimistic like ``score``; omitted when the
    OOF estimate is impossible), ``n_trials_evaluated`` — the number of trials
    actually evaluated by the winning algorithm's HPO phase (omitted when
    unknown), ``holdout_score`` — the honest comparison metric of the
    winner on the reserved outer hold-out (``general.holdout_fraction``;
    present only when a hold-out is reserved), and ``bbc_score`` — the
    bootstrap bias-corrected (BBC-CV) estimate of the selection procedure
    from the OOF predictions of all evaluated trials (present when no
    hold-out is reserved and the OOF pool is available).

    Args:
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        winner_algo (str): The name of the winning algorithm (from the terminal
            select step context).
        final_score (float): The raw winner score (already passed the
            ``is_valid_winner_score`` invariant).
        final_params (Dict[str, Any]): The winner's best hyperparameters.
        model_path_override (str | Path | None): An alternative model save
            path.
        disqualified_algorithms (Dict[str, str]): Disqualified algorithms
            (name → reason); added to the result when present.
        pre_trained_trainer (Any | None): The winner's already-trained
            ``ModelTrainer`` on 100% of the data (terminal ``sanity_gate``
            select step). If given, retraining is NOT performed — the artifact
            is saved from it (finalists are trained once). Exception: when
            ``general.holdout_fraction`` is set, the pre-trained finalist
            covered only the train part of the outer split, so it is NOT
            reused — the artifact is refit on ALL data (train + hold-out
            concatenated in ``full_X``/``full_y``) via ``_fit_and_save``.
        sanity_gate_sections (Dict[str, Dict[str, Any]] | None): Report
            sections of the ``sanity_gate`` select steps (key — the phase
            name). The ``sanity_gate`` key is added to the result only when at
            least one section is present.
        complexity_guard_sections (Dict[str, Any] | None): Aggregated report
            section of the adaptive complexity guard: key ``"parameters"`` —
            the list of "before → after" records (with the reason, size mode,
            ``n``/``p``, ``leaf_min``, level, and the applied flag). The
            ``complexity_guard`` key is added to the result only when the
            guard produced at least one record.
        n_trials_evaluated (int | None): The number of trials actually
            evaluated by the winner's HPO phase (own COMPLETED trials;
            transferred history never counts). ``None`` or 0 (unknown) — the
            key is omitted from the result.
        oof_pool (list[np.ndarray] | None): The OOF prediction matrices of
            all evaluated trials (``PhaseContext.oof_pool``); empty or
            ``None`` — ``bbc_score`` is not reported.

    Returns:
        Dict[str, Any]: A dict with the training results.

    Raises:
        Exception: The final training/saving error is propagated (logged
            before being re-raised).
    """
    model_path = Path(model_path_override or cfg.general.path_to_model)
    winner_cfg = _algorithms_as_dict(cfg.algorithms)[winner_algo]

    # Reserved outer hold-out (general.holdout_fraction): the final model is
    # refit on ALL data (train + hold-out) so that the production artifact
    # does not silently lose the reserved fraction. The hold-out rows are
    # concatenated back in the deterministic order produced by
    # ``prepare_dataset`` (train part first, then the hold-out part).
    full_X = prepared.X
    full_y = prepared.y
    if prepared.holdout_X is not None and prepared.holdout_y is not None:
        full_X = pd.concat([prepared.X, prepared.holdout_X], axis=0)
        full_y = pd.concat([prepared.y, prepared.holdout_y], axis=0)
        _LOG.info(
            "Refitting the final model on ALL data (%d train + %d hold-out rows); "
            "the hold-out score remains an honest estimate of the winner on "
            "the reserved rows.",
            len(prepared.X),
            len(prepared.holdout_X),
        )

    try:
        if pre_trained_trainer is not None and prepared.holdout_X is None:
            trainer = pre_trained_trainer
            _register_datetime_expansion(trainer, prepared.datetime_expansion)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(model_path)
        else:
            # When a hold-out is configured, the pre-trained finalist (trained
            # on the train part only) is NOT reused: the artifact must be
            # refit on all data, so the winner is retrained with its best
            # parameters via the standard fit-and-save path.
            trainer = _fit_and_save(
                winner_algo,
                winner_cfg,
                full_X,
                full_y,
                final_params,
                model_path,
                cfg,
                metric_name_sklearn=prepared.metric_sklearn,
                validation_strategy=prepared.resolved_validation,
                n_folds=prepared.resolved_n_folds,
                test_size=prepared.resolved_test_size,
                random_state=cfg.general.random_state,
                datetime_expansion=prepared.datetime_expansion,
            )
        _LOG.info("Model saved to %s", model_path.resolve())
    except Exception as e:
        _LOG.error(f"Failed to save final model: {e}")
        raise

    result: dict[str, Any] = {
        "algorithm": winner_algo,
        # Consistent user semantics: score always reflects the natural metric
        # value (positive RMSE/MAE, plain R²), and metric is the user-facing
        # comparison metric name from the configuration. The raw (inverted for
        # neg_*-scorers) value remains an internal optimizer detail and never
        # reaches the user.
        "score": to_user_value(prepared.metric_user, final_score),
        "metric": prepared.metric_user,
        "params": final_params,
        "model_path": str(model_path),
        # The engine seed of the run, recorded in the report so that a
        # reproducible rerun is possible (None — the run was not reproducible).
        "random_state": cfg.general.random_state,
    }
    # The comparison metric on the FINAL pipeline's pooled OOF predictions.
    # The trainer's OOF channel uses the same validation method/splits as HPO,
    # so this value is NOT independent of model selection: like ``score`` it
    # is optimistic (without an outer hold-out on k_fold/loo it equals the
    # best-trial score for pooled metrics). The honest estimate is
    # ``holdout_score``. Omitted when no valid OOF pair exists. With a
    # reserved outer hold-out the OOF vector is aligned to the FULL data
    # (train + hold-out), so the target is the concatenated vector.
    oof_score = _oof_score_in_user_metric(prepared.metric_user, trainer, full_y)
    if oof_score is not None:
        result["oof_score"] = oof_score
    # Honest evaluation count of the winner's HPO phase: reported next to the
    # best-trial score as the denominator of the "best of N" maximum. Omitted
    # when the count is unknown (None or 0).
    if n_trials_evaluated is not None and int(n_trials_evaluated) > 0:
        result["n_trials_evaluated"] = int(n_trials_evaluated)
    # Circuit breaker diagnostics: which algorithms were disqualified and why.
    # The key is added only when there are disqualifications, otherwise the
    # results are identical to the previous behavior. The sources are mixed
    # deliberately: both HPO failures and Sanity Gate select-step
    # disqualifications land here; the detailed audit lives in the
    # ``sanity_gate`` sections.
    if disqualified_algorithms:
        result["disqualified_algorithms"] = dict(disqualified_algorithms)
    # Report sections of the select steps: the ``sanity_gate`` key is a dict of
    # "select phase name → section". Added only when there are select steps
    # with strategy: sanity_gate. Deep-copied: sections contain nested
    # lists/dicts, and source mutations must not affect the report.
    if sanity_gate_sections:
        result["sanity_gate"] = copy.deepcopy(sanity_gate_sections)
    # Report section of the adaptive complexity guard: the key is added only
    # when the guard produced at least one record (the "before → after"
    # parameter table with reasons). Deep-copied so that source mutations do
    # not affect the report.
    if complexity_guard_sections:
        result["complexity_guard"] = copy.deepcopy(complexity_guard_sections)
    # Additional informational metrics of the final model. The key is added
    # only when metrics are present (after deduplication and exclusion of the
    # main comparison metric), otherwise the results are identical to the
    # previous behavior.
    if cfg.general.additional_metrics:
        result["additional_metrics"] = dict(trainer.additional_scores)
    # Reserved outer hold-out (general.holdout_fraction): the honest estimate
    # of the WINNER on the rows that were split off before HPO and never
    # participated in model selection. The hold-out model is trained with the
    # winner's best parameters on the train part of the outer split; the final
    # artifact is trained on ALL data (the issue requirement: "the final model
    # is still refit on all data"), so the hold-out score is a true
    # out-of-sample estimate of the chosen pipeline configuration. The key is
    # added only when the hold-out is configured.
    if prepared.holdout_X is not None and prepared.holdout_y is not None:
        holdout_score = _compute_holdout_score(
            cfg=cfg,
            winner_cfg=winner_cfg,
            prepared=prepared,
            winner_algo=winner_algo,
            final_params=final_params,
        )
        result["holdout_score"] = holdout_score
    # BBC-CV: the honest estimate when no outer hold-out is reserved. It
    # corrects the best-of-N bias of ``score`` using the OOF predictions of
    # every evaluated configuration, without retraining.
    if oof_pool and prepared.holdout_X is None:
        bbc_score = _bbc_score_in_user_metric(
            prepared.metric_user, oof_pool, prepared.y, prepared.random_state
        )
        if bbc_score is not None:
            result["bbc_score"] = bbc_score
    return result


def _bbc_score_in_user_metric(
    metric_user: str,
    oof_pool: list[np.ndarray],
    y: pd.Series,
    random_state: int | None,
) -> float | None:
    """Compute the BBC-CV estimate in the comparison metric semantics.

    Args:
        metric_user (str): The user-facing comparison metric name.
        oof_pool (list[np.ndarray]): The OOF prediction matrices of all
            evaluated trials (``(n_trials, n_rows)`` each).
        y (pd.Series): The target vector the OOF rows are aligned to.
        random_state (int | None): The bootstrap seed (the engine seed).

    Returns:
        float | None: The estimate in user semantics, or ``None`` when it
            cannot be computed (a WARNING is logged).
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    matrices = [m for m in oof_pool if m.ndim == 2 and m.shape[1] == len(y_arr)]
    if len(matrices) != len(oof_pool):
        _LOG.warning(
            "BBC-CV: %d OOF matrix(es) are not aligned to the %d target rows "
            "and are ignored.",
            len(oof_pool) - len(matrices),
            len(y_arr),
        )
    if not matrices:
        return None
    try:
        scorer = get_scorer_object(metric_user, global_y=y_arr)
        raw = bbc_cv_score(
            scorer, y_arr, np.vstack(matrices), random_state=random_state
        )
    except Exception as err:  # noqa: BLE001 — a BBC-CV failure must not crash the report
        _LOG.warning("BBC-CV estimate could not be computed: %s", err)
        return None
    return to_user_value(metric_user, raw)

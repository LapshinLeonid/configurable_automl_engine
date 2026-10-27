"""Shared state of the training pipeline (Training Engine).

The dataclasses exchanged between the pipeline steps: the per-algorithm HPO
record (``PhaseResult``), the prepared data (``PreparedData``), the phase
context (``PhaseContext``), the finalists and the robust winner result
(``FinalistCandidate`` / ``RobustWinnerResult``) and the HPO worker payloads
(``_HPOWorkerResult`` / ``_HpoPhaseSpec``). The module holds data only, so
the step modules can share these types without importing each other. It must
not import any other pipeline step module (``data_prep``, ``hpo_phases``,
``finalists``, ``select_steps``, ``artifact``, ``component``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from optuna.trial import FrozenTrial

from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    HPOPhaseCfg,
    SanityGateMode,
    ValidationStrategy,
)

from .sanity_gate import SanityCheckResult


@dataclass(frozen=True)
class PhaseResult:
    """Result of an HPO phase for a single algorithm.

    The single record of the pipeline working set
    (``PhaseContext.working_set: dict[str, PhaseResult]``): replaces the
    legacy ``(score, params)`` tuple plus the parallel trials dictionary.
    The select steps narrow the working set by rebuilding the dict — the
    ``trials`` travel with the record automatically (no separate
    synchronization).

    Attributes:
        score (float): The raw CV score of the phase (optimizer semantics,
            higher is better). Already passes the ``is_valid_winner_score``
            invariant.
        params (dict[str, Any]): The best found hyperparameters.
        trials (list[FrozenTrial]): The Optuna trials of the current phase
            ONLY (transferred TPE history is excluded). Empty for custom
            tuners that do not expose trials; passed as ``previous_trials``
            to the next HPO phase (refine warm-start).
        complexity_guard (list[dict[str, Any]]): Log records of the adaptive
            complexity guard for the phase (one per restricted parameter,
            "before → after" with the reason). Empty for custom tuners or
            when the guard produced no records.
        n_trials_evaluated (int): The number of trials actually evaluated by
            the phase (own COMPLETED trials; transferred history never
            counts). Reported in ``result["n_trials_evaluated"]`` next to the
            best-trial ``score``: the score is the maximum over exactly these
            trials, so a larger count implies a more optimistic score.
            Defaults to 0 for custom tuners and legacy code paths that do not
            report the count.
        oof_predictions (np.ndarray | None): The out-of-fold predictions of
            the phase's completed trials, shape ``(n_trials, n_rows)`` (the
            BBC-CV input); ``None`` when OOF collection is off or the tuner
            does not support it.
    """

    score: float
    params: dict[str, Any]
    trials: list[FrozenTrial] = field(default_factory=list)
    complexity_guard: list[dict[str, Any]] = field(default_factory=list)
    n_trials_evaluated: int = 0
    oof_predictions: np.ndarray | None = None


@dataclass(frozen=True)
class FinalistCandidate:
    """A two-stage selection finalist: a model trained on 100% of the data.

    Assembled before ``select_robust_winner`` (finalists are trained on the
    full data exactly once; the result is reused for ``RMSE_full``, the audit,
    and the report — no duplicated training).

    Attributes:
        algo_name (str): The algorithm name.
        model (Any): The pipeline trained on 100% of the data
            (``trainer.pipeline``); used by the audit (circuit B) and for
            ``y_pred_full`` predictions.
        params (dict[str, Any]): The best hyperparameters from HPO
            (``phase_results[algo_name].params``).
        cv_score (float): The raw score from the HPO phase (optimizer
            semantics); used by the ``cv_best`` select step and the report.
        rmse_oof (float | None): The finalist's ``RMSE_oof``
            (``trainer.oof_score_``); ``None`` if an OOF estimate is impossible.
        y_pred_oof (Any): The OOF prediction vector
            (``trainer.oof_predictions_``), aligned to the rows of X.
        y_pred_full (Any): The full-fit model's predictions on the same rows
            (``trainer.predict(X)``); used by circuit D (RMSE_full) and the
            report.
        rmse_full (float | None): ``RMSE_full = RMSE(y, y_pred_full)``;
            ``None`` if it cannot be computed.
        trainer (Any | None): The finalist's ``ModelTrainer`` itself; passed to
            ``persist_artifact`` so the artifact can be saved WITHOUT retraining
            the winner.
    """

    algo_name: str
    model: Any
    params: dict[str, Any]
    cv_score: float
    rmse_oof: float | None
    y_pred_oof: Any
    y_pred_full: Any
    rmse_full: float | None = None
    trainer: Any | None = None


@dataclass(frozen=True)
class RobustWinnerResult:
    """Result of the robust winner selection.

    Attributes:
        winner_algo (str): The name of the winning algorithm.
        mode (SanityGateMode): The gate operation mode ('off'/'active').
            Select steps with ``strategy: sanity_gate`` run in ``active`` mode;
            ``off`` remains the legacy mode of direct calls (CV leader without
            audit).
        pool (list[str]): The finalists pool order (deterministic tie-break on
            equal RMSE_oof).
        audit (dict[str, SanityCheckResult]): The audit of each finalist
            (name → result). Filled when mode != 'off'.
        disqualified (dict[str, list[str]]): Disqualified finalists
            (name → reasons). In ``active`` mode the reasons are applied.
        fallback_used (bool): True when all candidates were rejected and the
            winner was chosen by the fallback criterion (the "least
            problematic" model).
        winner_rmse_oof (float | None): The winner's ``RMSE_oof``.
    """

    winner_algo: str
    mode: SanityGateMode
    pool: list[str]
    audit: dict[str, SanityCheckResult] = field(default_factory=dict)
    disqualified: dict[str, list[str]] = field(default_factory=dict)
    fallback_used: bool = False
    winner_rmse_oof: float | None = None


@dataclass(frozen=True)
class PreparedData:
    """Prepared data and resolved settings for the HPO phases.

    The single container that ``prepare_dataset`` creates exactly once and that
    ``execute_phases`` / ``persist_artifact`` consume. Guarantees that all
    pipeline stages use the same feature representation, metric, and validation
    strategy without duplicating computations.

    Attributes:
        X (pd.DataFrame): The feature matrix without the target column.
        y (pd.Series): The target vector.
        metric_user (str): The user-facing comparison metric name
            (``general.comparison_metric``).
        metric_sklearn (str): The metric name in sklearn semantics (e.g.
            ``neg_root_mean_squared_error``).
        resolved_validation (ValidationStrategy): The resolved validation
            strategy: 'auto' is already expanded into a concrete method
            ('k_fold'/'train_test_split'/'loo').
        resolved_n_folds (int): The number of folds (relevant for 'k_fold').
        resolved_test_size (float | int): The hold-out size — a fraction or an
            integer number of rows after 'auto' resolution.
        categorical_features (list[str]): The detected categorical columns.
        numerical_features (list[str]): The detected numerical columns.
        random_state (int | None): The engine seed from
            ``general.random_state``; ``None`` — a non-reproducible run.
        holdout_X (pd.DataFrame | None): The feature matrix of the reserved
            outer hold-out (``general.holdout_fraction``); ``None`` when the
            hold-out is not configured.
        holdout_y (pd.Series | None): The target vector of the reserved outer
            hold-out; ``None`` when the hold-out is not configured.
        holdout_fraction (float | None): The resolved
            ``general.holdout_fraction`` value (the fraction of rows reserved
            before HPO; ``'auto'`` already expanded); ``None`` when no
            hold-out is reserved.
        datetime_expansion (dict[str, list[str]]): The datetime expansion
            applied by the input-data diagnostics (``general.datetime_policy:
            'expand'``): source column → generated component columns. It is
            registered on the saved trainer so that the artifact can predict
            on the raw input schema.
    """

    X: pd.DataFrame
    y: pd.Series
    metric_user: str
    metric_sklearn: str
    resolved_validation: ValidationStrategy
    resolved_n_folds: int
    resolved_test_size: float | int
    categorical_features: list[str]
    numerical_features: list[str]
    random_state: int | None = None
    holdout_X: pd.DataFrame | None = None
    holdout_y: pd.Series | None = None
    holdout_fraction: float | None = None
    datetime_expansion: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class PhaseContext:
    """Explicit context shared between pipeline phases.

    Replaces the local ``phase_results`` variable: the context is created by
    ``execute_phases`` once and is mutated sequentially by the phases. The
    select step output REPLACES the working set (subsequent ``refine`` /
    ``select`` steps operate on it).

    Attributes:
        working_set (dict[str, PhaseResult]): The working set
            "algorithm → PhaseResult". Rebuilt by each HPO phase and
            replaced by the output of each select step.
        disqualified_algorithms (dict[str, str]): Circuit breaker:
            "algorithm → reason". Populated by HPO failures and Sanity Gate
            select-step disqualifications — the sources are mixed deliberately;
            the detailed gate audit lives in ``sanity_gate_sections``.
        full_fit_trainers (dict[str, Any]): The ``ModelTrainer`` objects of the
            last ``sanity_gate`` select step, trained on 100% of the data.
            Cleared by each HPO phase (params are no longer current) and
            rebuilt by each sanity-select (old full-fit trainers are not
            reused).
        sanity_gate_sections (dict[str, dict[str, Any]]): Report sections of
            the ``sanity_gate`` select steps (key — the phase name).
        winner_algo (str | None): The winner of the terminal select step
            (``output: winner``); ``None`` until it runs.
        pre_trained_trainer (Any | None): The winner's trainer of the terminal
            ``sanity_gate`` select step, trained on 100% of the data (no
            retraining of the winner required); ``None`` otherwise.
        complexity_guard_records (list[dict[str, Any]]): The accumulated
            "before → after" records of the adaptive complexity guard across
            all HPO phases (one per restricted parameter), for the report.
        oof_pool (list[np.ndarray]): The OOF prediction matrices of every
            algorithm in every HPO phase (``(n_trials, n_rows)`` each) — all
            configurations the selection chose from; the BBC-CV input. Empty
            when OOF collection is off.
    """

    working_set: dict[str, PhaseResult] = field(default_factory=dict)
    disqualified_algorithms: dict[str, str] = field(default_factory=dict)
    full_fit_trainers: dict[str, Any] = field(default_factory=dict)
    sanity_gate_sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    winner_algo: str | None = None
    pre_trained_trainer: Any | None = None
    complexity_guard_records: list[dict[str, Any]] = field(default_factory=list)
    oof_pool: list[np.ndarray] = field(default_factory=list)


@dataclass
class _HPOWorkerResult:
    """Result of a single HPO worker (serialized between processes).

    Attributes:
        name (str): The algorithm name.
        score (float | None): A valid "raw" score; ``None`` on failure.
        params (dict[str, Any] | None): The best parameters; ``None`` on
            failure.
        trials (list[FrozenTrial] | None): The phase's own Optuna trials;
            ``None`` on failure or for tuners that do not expose trials.
        disqualify_reason (str | None): Circuit breaker disqualification
            reason; ``None`` — the algorithm is not disqualified.
        history_error (str | None): A TPE history transfer error
            (``HPOHistoryError``) raised inside the worker. Carried as a
            string (not the exception object) so the result stays
            serializable in the "processes" mode; the consumer re-raises the
            error in the main process — fail fast even in the parallel mode,
            where ``run_parallel`` otherwise swallows worker exceptions.
        complexity_guard (list[dict[str, Any]]): The adaptive complexity
            guard "before → after" records of the phase (serializable in the
            "processes" mode).
        n_trials_evaluated (int): The number of trials actually evaluated by
            the phase (own COMPLETED trials; transferred history never
            counts); 0 for custom tuners that do not report the count.
        oof_predictions (np.ndarray | None): The OOF predictions of the
            phase's completed trials (``(n_trials, n_rows)``); ``None`` when
            OOF collection is off.
    """

    name: str
    score: float | None = None
    params: dict[str, Any] | None = None
    trials: list[FrozenTrial] | None = None
    disqualify_reason: str | None = None
    history_error: str | None = None
    complexity_guard: list[dict[str, Any]] = field(default_factory=list)
    n_trials_evaluated: int = 0
    oof_predictions: np.ndarray | None = None


@dataclass
class _HpoPhaseSpec:
    """An immutable snapshot of the HPO phase parameters passed to the worker.

    Built in ``execute_phases`` once per phase and passed to
    ``_hpo_phase_worker`` as an explicit argument. This keeps the worker a
    module-level function (not a closure), which can be serialized in the
    "processes" mode: a closure inside ``execute_phases`` cannot be
    serialized. The feature matrix ``X`` is NOT part of the snapshot: in the
    "processes" mode it is transferred to the workers through ``run_parallel``
    shared memory / disk, so it is not pickled for every algorithm.

    Attributes:
        phase (HPOPhaseCfg): Configuration of the current phase.
        prev_results (dict[str, PhaseResult]): Snapshot of the previous phase
            results (for warm-start refine: params + TPE trials).
        y (pd.Series): The target vector.
        metric_sklearn (str): Metric name in sklearn semantics.
        resolved_validation (ValidationStrategy): The resolved validation
            strategy.
        resolved_n_folds (int): Number of folds for 'k_fold'.
        resolved_test_size (float | int): Size of the hold-out part.
        categorical_features (list[str]): Detected categorical columns.
        numerical_features (list[str]): Detected numeric columns.
        oversampling_enabled (bool): Whether oversampling is enabled.
        oversampling_multiplier (float): Sample increase coefficient.
        oversampling_algorithm (str): Oversampling algorithm name.
        oversampling_target_bins (int): Number of quantile bins for the
            regression oversampling strategy.
        oversampling_add_noise (bool): Whether to add Gaussian noise to the
            oversampled numeric features.
        oversampling_noise_level (float): Noise intensity relative to the
            column standard deviation.
        pruning_cfg (dict[str, Any] | None): Early stopping settings.
        encoding (str): Categorical encoding strategy.
        high_cardinality_threshold (int | None): High-cardinality threshold.
        high_cardinality_encoding (str | None): High-cardinality strategy.
        hashing_n_components (int): Number of hashing components.
        target_encoding_smoothing (float): Target encoding smoothing.
        target_encoding_fallback (float | None): Target encoding fallback.
        feature_selection_cfg (FeatureSelectionCfg | dict | None): Feature
            selection configuration.
        random_state (int | None): The random seed of the data splits and the
            preprocessor — the ``general.random_state`` engine seed for every
            HPO phase (so folds stay comparable between phases and with the
            final ``ModelTrainer`` fit); ``None`` — the splits are not fixed
            (non-reproducible run).
        sampler_seed (int | None): The per-phase TPE sampler seed
            (``general.random_state + hpo_phase_ordinal`` — a different seed
            per HPO phase, so TPE random startup points do not repeat between
            phases); ``None`` — the sampler is not seeded.
        inf_policy (str | None): ``general.inf_policy`` for the tuner's data
            validation (``None`` — the tuner default).
        datetime_policy (str | None): ``general.datetime_policy`` for the
            tuner's data validation.
        drop_id_like_columns (bool | None): ``general.drop_id_like_columns``
            for the tuner's data validation.
        collect_oof (bool): Whether the tuner keeps the OOF predictions of
            every trial for the BBC-CV estimate.
    """

    phase: HPOPhaseCfg
    prev_results: dict[str, PhaseResult]
    y: pd.Series
    metric_sklearn: str
    resolved_validation: ValidationStrategy
    resolved_n_folds: int
    resolved_test_size: float | int
    categorical_features: list[str]
    numerical_features: list[str]
    oversampling_enabled: bool
    oversampling_multiplier: float
    oversampling_algorithm: str
    oversampling_target_bins: int
    oversampling_add_noise: bool
    oversampling_noise_level: float
    pruning_cfg: dict[str, Any] | None
    encoding: str
    high_cardinality_threshold: int | None
    high_cardinality_encoding: str | None
    hashing_n_components: int
    target_encoding_smoothing: float
    target_encoding_fallback: float | None
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None
    complexity_guard_cfg: Any = None
    user_hyperparameter_keys: list[str] | None = None
    random_state: int | None = None
    sampler_seed: int | None = None
    inf_policy: str | None = None
    datetime_policy: str | None = None
    drop_id_like_columns: bool | None = None
    collect_oof: bool = False

"""Module managing the model training lifecycle (Training Engine).
Provides an automated pipeline from input data validation to saving the final
model. Supports multi-phase hyperparameter search (HPO), dynamic algorithm
loading, and parallel computation.
The public training loop is decomposed into steps:
    - load_config: Load and validate the configuration and input data.
    - prepare_dataset: Prepare X/y, resolve validation strategy and column types.
    - execute_phases: Multi-phase hyperparameter search (HPO).
    - select_winner: Deterministically select the winning algorithm.
    - persist_artifact: Final training, model persistence, and report assembly.
Two-stage winner selection is performed by an explicit pipeline select step:
    - select with strategy: sanity_gate — a finalists pool, training each
      finalist on 100% of the data, a ModelSanityGate audit, and ranking of the
      audited candidates by RMSE_oof (active mode);
    - select with strategy: cv_best — deterministic ranking by CV score without
      audit (the equivalent of the legacy path with mode: off).
This module only orchestrates the steps and is the only one that mutates the
``PhaseContext``: the steps are pure functions that return outcomes, applied
here. The steps live in dedicated modules (import them from there):
    - state: Shared dataclasses (PhaseResult, PreparedData, PhaseContext, ...).
    - data_prep: load_config, prepare_dataset.
    - hpo_phases: HPO phase execution (_run_hpo).
    - finalists: select_winner, select_finalists, select_robust_winner.
    - select_steps: The explicit select steps (cv_best / sanity_gate).
    - artifact: Final model training on the full dataset, persist_artifact.
Main components:
    - train_best_model: Public interface — orchestrates the steps above.
    - execute_phases: The phase loop (HPO phases and select steps).
Usage example:
    results = train_best_model(
        config="path/to/config.yaml",
        df=my_dataframe,
        target="target_col",
        model_path_override="models/best.pkl"
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from configurable_automl_engine import oversampling as _oversampling
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    HPOPhaseCfg,
)

from .artifact import persist_artifact
from .data_prep import load_config, prepare_dataset
from .finalists import is_valid_winner_score
from .hpo_phases import (
    _algorithms_as_dict,
    _apply_target_domain_guard,
    _bbc_enabled,
    _consume_worker_results,
    _hpo_phase_worker,
    _merge_complexity_guard_records,
)
from .select_steps import select_step_outcome
from .state import (
    PhaseContext,
    PhaseResult,
    PreparedData,
    SelectOutcome,
    WorkerOutcome,
    _HpoPhaseSpec,
)
from .thread_pool import run_parallel

__all__ = ["execute_phases", "train_best_model"]

_LOG = logging.getLogger(__name__)


def execute_phases(
    cfg: Config,
    prepared: PreparedData,
) -> PhaseContext:
    """Run all pipeline phases and return the final phase context.

    Pipeline step 3. Contains the phase loop, parallel HPO execution,
    synchronous select steps, circuit breaker disqualification, and filtering
    of invalid results. The HPO worker ``_hpo_phase_worker`` and the helper
    ``_execute_hpo_phase`` are module-level functions: the phase data is passed through ``_HpoPhaseSpec`` so that the
    worker is serializable in the "processes" mode. The feature matrix ``X``
    is passed as a separate argument and transferred via ``run_parallel``
    shared memory / disk in that mode.

    Multi-phase semantics: the working set is not accumulated — each HPO phase
    rebuilds it from scratch, so an algorithm that failed in the current phase
    (complete HPO failure: ``None`` or an invalid score) is fully excluded and
    does not keep a stale record from a previous phase. The select step output
    replaces the pipeline working set; the final winner is determined by the
    terminal select step.

    Refine step: ``action='refine'`` performs ``n_trials`` warm-started HPO
    attempts for **every** candidate of the incoming working set (the results
    of the last HPO phase, or the output of the previous select step). The
    incoming records are used completely: ``params`` become ``initial_params``
    for ``enqueue_trial`` (the monotonicity anchor) and ``trials`` become
    ``previous_trials`` for ``study.add_trials`` (the TPE history warm-start);
    incoming scores are ignored. The output keeps the same set of names with
    updated records and replaces the working set. There is no narrowing in
    refine — narrowing is the job of explicit select steps. As the first
    phase, refine runs over the full set of enabled algorithms without
    warm-start. A per-algorithm failure excludes only that algorithm (circuit
    breaker / invalid-result filters); if refine wipes out the whole working
    set, ``RuntimeError`` is raised — a silent fallback to pre-refine
    parameters would hide the regression (deliberate design decision).

    Per-phase sampler seeds: each HPO phase (``all_algorithms``/``refine``)
    runs the TPE sampler with its own seed derived from the engine seed,
    ``general.random_state + hpo_phase_ordinal`` (the first HPO phase keeps
    the engine seed — 42 by default, the second = 43, etc.; select steps do
    not count). Together with the ``previous_trials`` transfer this
    guarantees that TPE startup points do not repeat between phases. The
    data splits and the preprocessor, in contrast, keep the SAME engine seed
    for every phase: every phase (and the final ``ModelTrainer`` fit)
    evaluates models on the same folds, so scores stay comparable and the
    monotonicity of ``enqueue_trial`` holds. With
    ``general.random_state=None`` the run is non-reproducible: neither the
    splits nor the sampler are seeded (a WARNING is logged in
    ``prepare_dataset``).

    Args:
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and resolved settings.

    Returns:
        PhaseContext: The final phase context. For a valid config the last
            phase is a terminal select (``output: winner``), so ``winner_algo``
            points to the winner and the working set holds a single record. The
            context is empty if no phases ran (``general.phases`` is empty) —
            the protection against an empty phase list stays in
            ``train_best_model``.

    Raises:
        RuntimeError: If an HPO phase produced no valid score, a select step
            received an empty working set, or no finalist was trained on the
            full dataset.
        HPOHistoryError: If the transferred TPE history of a refine phase is
            incompatible with the current search space (fail fast).
    """
    ctx = PhaseContext()

    # Initial candidate list (all enabled algorithms)
    all_algorithms = _algorithms_as_dict(cfg.algorithms)
    # Target-domain guard (before any trial): GLMs whose loss domain does not
    # fit y are disqualified up front, a partly invalid Tweedie power range
    # is clipped to its valid part.
    guard = _apply_target_domain_guard(
        all_algorithms, prepared.y, ctx.disqualified_algorithms
    )
    all_algorithms = guard.algorithms
    ctx.disqualified_algorithms.update(guard.disqualified)
    # Disqualified algorithms (circuit breaker): name -> reason.
    # Disqualification does not abort the run: the algorithm is excluded from
    # candidates and results, the rest continue training.
    # living in ctx: populated by HPO failures and select steps.
    # The engine seed from ``general.random_state``: drives the data splits /
    # preprocessor (passed to the HPO workers as ``random_state``) and the TPE
    # sampler base seed (``random_state + hpo_phase_ordinal``; with None the
    # sampler and the splits are non-deterministic). The default value of
    # ``general.random_state`` is 42 (the ``validation.RANDOM_STATE`` default),
    # so configs without an explicit seed behave exactly as before.
    engine_random_state = cfg.general.random_state
    # BBC-CV: without an outer hold-out the OOF predictions of every trial
    # are kept, so persist_artifact can report the bias-corrected estimate.
    collect_oof = _bbc_enabled(cfg, prepared, all_algorithms)
    # HPO phase ordinal: incremented only for all_algorithms/refine phases
    # (select steps are not counted). The per-phase SAMPLER seed is derived
    # from the engine seed: the first HPO phase keeps the documented engine
    # seed (42 by default), every next HPO phase gets a different seed,
    # so TPE random startup points never repeat between phases. When the
    # engine seed is None the sampler is not seeded at all. The data
    # splits / preprocessor seed stays fixed for every phase (see the
    # _HpoPhaseSpec construction in _build_phase_spec).
    hpo_phase_ordinal = 0
    for phase in cfg.general.phases:
        if phase.action == "select":
            # The select step runs synchronously (outside run_parallel): it
            # operates on already-finished HPO results.
            _apply_select_outcome(
                ctx, select_step_outcome(phase, cfg, prepared, ctx), phase.name
            )
            continue

        phase_seed = (
            engine_random_state + hpo_phase_ordinal
            if engine_random_state is not None
            else None
        )
        hpo_phase_ordinal += 1
        _execute_hpo_step(
            phase,
            cfg=cfg,
            prepared=prepared,
            ctx=ctx,
            all_algorithms=all_algorithms,
            phase_seed=phase_seed,
            collect_oof=collect_oof,
        )
    return ctx


def _execute_hpo_step(
    phase: HPOPhaseCfg,
    *,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
    all_algorithms: dict[str, AlgoCfg],
    phase_seed: int | None,
    collect_oof: bool,
) -> None:
    """Run one HPO phase (``all_algorithms`` / ``refine``) of the pipeline.

    Args:
        phase (HPOPhaseCfg): The HPO phase configuration.
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and resolved settings.
        ctx (PhaseContext): The pipeline phase context (mutated).
        all_algorithms (dict[str, AlgoCfg]): The configured algorithms (after
            the target-domain guard).
        phase_seed (int | None): The per-phase TPE sampler seed.
        collect_oof (bool): Whether the OOF predictions are kept for BBC-CV.

    Raises:
        RuntimeError: If the phase produced no valid score.
    """
    _LOG.info(
        f"--- Starting Phase: {phase.name} ({phase.n_trials}"
        f" trials, action: {phase.action}, sampler_seed: {phase_seed}) ---"
    )

    # Snapshot of the incoming working set. Used ONLY to pass
    # initial_params to the worker of a refine phase. Records from
    # previous phases are NOT carried into the current one: the working
    # set is rebuilt from the current phase only, so an algorithm that
    # failed in it (returned None or an invalid score) does not keep a
    # stale record from a previous phase and does not compete with real
    # results. Consequence: the final winner is chosen by the terminal
    # select step.
    prev_results = ctx.working_set

    # The HPO phase rebuilds the working set from scratch, so the full-fit
    # trainers and the terminal select result of previous phases no longer
    # match the working set parameters.
    ctx.full_fit_trainers.clear()
    ctx.winner_algo = None
    ctx.pre_trained_trainer = None

    current_candidates = _phase_candidates(phase, all_algorithms, prev_results, ctx)

    # Each phase starts with an empty working set: it is rebuilt from the
    # current phase only, so an algorithm that fails in this phase does not
    # keep a stale record from a previous phase.
    phase_results: dict[str, PhaseResult] = {}

    # Candidates participating in the current phase (needed for the
    # diagnostics in the "no valid results" message)
    phase_candidates = list(current_candidates.keys())

    spec = _build_phase_spec(
        phase,
        cfg=cfg,
        prepared=prepared,
        prev_results=prev_results,
        phase_seed=phase_seed,
        collect_oof=collect_oof,
    )
    _run_phase_workers(cfg, current_candidates, spec, prepared.X, phase_results, ctx)

    # Disqualified algorithms (circuit breaker) are removed from the results and
    # from the candidates of the following phases: they must not win and must
    # not waste compute again.
    # working_set.pop is a safety net for disqualification in the same
    # phase (a no-op for the already-rebuilt dict).
    for name in list(ctx.disqualified_algorithms):
        phase_results.pop(name, None)
        current_candidates.pop(name, None)

    valid_results = _valid_phase_results(phase_results)

    # Keep only the valid results of the current phase in the working set:
    # entries with an invalid score (HPO_WORST_SCORE etc.) are removed.
    # Since the phase results started empty, the working set is guaranteed
    # to contain only this phase's results.
    ctx.working_set = valid_results

    # Accumulate the adaptive complexity guard records of the phase into
    # the report section (one "before → after" record per restricted
    # parameter across all algorithms and phases: the A/B rules of the
    # same parameter are merged, and identical records of repeated
    # phases are not duplicated).
    for record in phase_results.values():
        ctx.complexity_guard_records.extend(record.complexity_guard)
    ctx.complexity_guard_records = _merge_complexity_guard_records(
        ctx.complexity_guard_records
    )

    if not valid_results:
        failed_algos = [n for n in phase_candidates if n not in valid_results]
        message = (
            f"No algorithms produced valid scores in phase '{phase.name}'. "
            f"Failed algorithms: {failed_algos}"
        )
        if ctx.disqualified_algorithms:
            message += f". Disqualified algorithms: {dict(ctx.disqualified_algorithms)}"
        raise RuntimeError(message)


def _phase_candidates(
    phase: HPOPhaseCfg,
    all_algorithms: dict[str, AlgoCfg],
    prev_results: dict[str, PhaseResult],
    ctx: PhaseContext,
) -> dict[str, AlgoCfg]:
    """Select the candidates of an HPO phase.

    Args:
        phase (HPOPhaseCfg): The HPO phase configuration.
        all_algorithms (dict[str, AlgoCfg]): The configured algorithms.
        prev_results (dict[str, PhaseResult]): The incoming working set.
        ctx (PhaseContext): The pipeline phase context.

    Returns:
        dict[str, AlgoCfg]: The algorithms that run in the phase.
    """
    if phase.action == "refine":
        # Refine: warm-start HPO for the ENTIRE incoming working set. The
        # input is the last HPO phase results or the output of the previous
        # select step; params become initial_params for enqueue_trial (the
        # monotonicity anchor) and trials become previous_trials for
        # study.add_trials (the TPE history warm-start); incoming scores
        # are ignored. There is no narrowing: no winners, top-K, or
        # corridors — narrowing is done by explicit select steps. The
        # output replaces the working set: the same names with updated
        # records.
        if prev_results:
            return {
                n: all_algorithms[n]
                for n in prev_results
                if n in all_algorithms and n not in ctx.disqualified_algorithms
            }
        # refine as the first phase: the incoming working set is the
        # full set of enabled algorithms, no warm-start is passed
        # (there are no previous results).
        return {
            n: a
            for n, a in all_algorithms.items()
            if a.enable and n not in ctx.disqualified_algorithms
        }
    # all_algorithms phase: every enabled algorithm, results from
    # previous phases are not accumulated. Algorithms disqualified by
    # the circuit breaker do not return to the phase.
    return {
        n: a
        for n, a in all_algorithms.items()
        if a.enable and n not in ctx.disqualified_algorithms
    }


def _build_phase_spec(
    phase: HPOPhaseCfg,
    *,
    cfg: Config,
    prepared: PreparedData,
    prev_results: dict[str, PhaseResult],
    phase_seed: int | None,
    collect_oof: bool,
) -> _HpoPhaseSpec:
    """Build the immutable snapshot of the HPO phase parameters for the worker.

    The worker is a module-level function and all phase data is passed to it
    through ``spec`` explicitly so that the worker is serializable in the
    "processes" mode (a closure inside ``execute_phases`` cannot be
    serialized). The feature matrix ``X`` is NOT put into the spec: in the
    processes mode it is transferred to the workers through ``run_parallel``
    shared memory / disk, so it is not pickled for every algorithm. The
    target vector ``y`` (a single column) stays in the spec.

    Args:
        phase (HPOPhaseCfg): The HPO phase configuration.
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and resolved settings.
        prev_results (dict[str, PhaseResult]): The incoming working set.
        phase_seed (int | None): The per-phase TPE sampler seed.
        collect_oof (bool): Whether the OOF predictions are kept for BBC-CV.

    Returns:
        _HpoPhaseSpec: The phase snapshot.
    """
    ovr = cfg.oversampling
    pruning_cfg: dict[str, Any] | None = None
    if cfg.general.pruning.enable:
        pruning_cfg = {
            "enable": cfg.general.pruning.enable,
            "strategy": cfg.general.pruning.strategy.value,
            "min_steps": cfg.general.pruning.min_steps,
            "n_startup_trials": cfg.general.pruning.n_startup_trials,
            "reduction_factor": cfg.general.pruning.reduction_factor,
        }
    return _HpoPhaseSpec(
        phase=phase,
        prev_results=prev_results,
        y=prepared.y,
        metric_sklearn=prepared.metric_sklearn,
        resolved_validation=prepared.resolved_validation,
        resolved_n_folds=prepared.resolved_n_folds,
        resolved_test_size=prepared.resolved_test_size,
        categorical_features=prepared.categorical_features,
        numerical_features=prepared.numerical_features,
        oversampling_enabled=ovr.enable,
        oversampling_multiplier=ovr.multiplier,
        oversampling_algorithm=ovr.algorithm.value,
        oversampling_target_bins=ovr.target_bins,
        oversampling_add_noise=ovr.add_noise,
        oversampling_noise_level=ovr.noise_level,
        pruning_cfg=pruning_cfg,
        encoding=cfg.general.categorical_encoding,
        high_cardinality_threshold=cfg.general.high_cardinality_threshold,
        high_cardinality_encoding=cfg.general.high_cardinality_encoding,
        hashing_n_components=cfg.general.hashing_n_components,
        target_encoding_smoothing=cfg.general.target_encoding_smoothing,
        target_encoding_fallback=cfg.general.target_encoding_fallback,
        feature_selection_cfg=cfg.general.feature_selection,
        complexity_guard_cfg=cfg.general.complexity_guard,
        # Data splits and preprocessor: the engine seed (general.
        # random_state), so every phase (and the final ModelTrainer fit)
        # uses the same folds; None — non-reproducible splits.
        random_state=cfg.general.random_state,
        # TPE sampler: a per-phase seed derived from the engine seed, so
        # startup points do not repeat between phases; None — unseeded.
        sampler_seed=phase_seed,
        inf_policy=cfg.general.inf_policy,
        datetime_policy=cfg.general.datetime_policy,
        drop_id_like_columns=cfg.general.drop_id_like_columns,
        collect_oof=collect_oof,
    )


def _run_phase_workers(
    cfg: Config,
    current_candidates: dict[str, AlgoCfg],
    spec: _HpoPhaseSpec,
    X: pd.DataFrame,
    phase_results: dict[str, PhaseResult],
    ctx: PhaseContext,
) -> None:
    """Run the HPO workers of a phase (parallel or sequential).

    Args:
        cfg (Config): The configuration object.
        current_candidates (dict[str, AlgoCfg]): The algorithms of the phase.
        spec (_HpoPhaseSpec): The phase snapshot.
        X (pd.DataFrame): The feature matrix.
        phase_results (dict[str, PhaseResult]): The phase results (filled).
        ctx (PhaseContext): The pipeline phase context (disqualifications).
    """
    # Execution (parallel or sequential)
    if cfg.general.parallel_strategy == "algorithms" and len(current_candidates) > 1:
        results = run_parallel(
            _hpo_phase_worker,
            args_seq=[(n, a, spec, X) for n, a in current_candidates.items()],
            max_workers=cfg.general.max_workers,
            mode=cfg.general.parallel_mode,
            # X is transferred through shared memory (or the disk cache)
            # instead of being pickled inside the spec for every algorithm.
            shared_args_indices=(
                [3] if cfg.general.parallel_mode == "processes" else None
            ),
            # Timeouts are supported only in processes mode.
            # In threads mode config validation forbids them, and
            # run_parallel ignores them with a warning on direct calls.
            # Here the threads path does not pass the fallback `or 3600`:
            # otherwise, on a default threads config
            # (phase_timeout=None -> 3600) the ignore-warning would fire
            # on every phase.
            timeout=(
                None
                if cfg.general.parallel_mode == "threads"
                else (cfg.general.phase_timeout or 3600)
            ),
            task_timeout=(
                None
                if cfg.general.parallel_mode == "threads"
                else cfg.general.task_timeout
            ),
        )
        _apply_worker_outcome(ctx, phase_results, _consume_worker_results(results))
    else:
        for n, a in current_candidates.items():
            res = _hpo_phase_worker(n, a, spec, X)
            _apply_worker_outcome(ctx, phase_results, _consume_worker_results([res]))


def _apply_worker_outcome(
    ctx: PhaseContext,
    phase_results: dict[str, PhaseResult],
    outcome: WorkerOutcome,
) -> None:
    """Merge the HPO worker outcome into the phase results and the context.

    Args:
        ctx (PhaseContext): The pipeline phase context (disqualifications and
            the BBC-CV pool are extended).
        phase_results (dict[str, PhaseResult]): The phase results (extended).
        outcome (WorkerOutcome): The sorted worker results.
    """
    ctx.disqualified_algorithms.update(outcome.disqualified)
    ctx.oof_pool.extend(outcome.oof_pool)
    phase_results.update(outcome.results)


def _apply_select_outcome(
    ctx: PhaseContext, outcome: SelectOutcome, phase_name: str
) -> None:
    """Replace the pipeline state with the outcome of a select step.

    Args:
        ctx (PhaseContext): The pipeline phase context (mutated).
        outcome (SelectOutcome): The new state returned by the select step.
        phase_name (str): The select phase name (the key of its gate report
            section).
    """
    ctx.working_set = outcome.working_set
    ctx.winner_algo = outcome.winner_algo
    ctx.pre_trained_trainer = outcome.pre_trained_trainer
    ctx.full_fit_trainers = outcome.full_fit_trainers
    ctx.disqualified_algorithms = outcome.disqualified_algorithms
    if outcome.sanity_gate_section is not None:
        ctx.sanity_gate_sections[phase_name] = outcome.sanity_gate_section


def _valid_phase_results(
    phase_results: dict[str, PhaseResult],
) -> dict[str, PhaseResult]:
    """Filter out invalid phase results.

    None/NaN/±inf and the worst-score sentinel class are excluded. The
    sentinel class is defined as any score at or below the exact float32
    minimum (WORST_SCORE_THRESHOLD): it covers both the HPO_WORST_SCORE
    constant (returned by the tuner when a trial metric is non-finite) and
    the raw float(np.finfo(np.float32).min) value (possible from numpy-cast
    metrics), which the old math.isclose(rel_tol=1e-9)
    check missed (relative difference ≈ 9.88e-9). Real metrics of such
    magnitude are practically impossible, so the threshold cannot drop a
    legitimate winner. params must be a dict (the tuner failure signal is
    params=None); garbage score types (None, strings, etc.) are rejected
    by is_valid_winner_score.

    Args:
        phase_results (dict[str, PhaseResult]): The raw phase results.

    Returns:
        dict[str, PhaseResult]: The valid results (insertion order kept).
    """
    valid_results: dict[str, PhaseResult] = {}
    for name, record in phase_results.items():
        params = record.params
        score = record.score
        if not isinstance(params, dict):
            _LOG.warning(
                "Algorithm %s produced invalid params %r (expected dict); "
                "excluding from phase candidates",
                name,
                params,
            )
            continue
        if not is_valid_winner_score(score):
            _LOG.warning(
                "Algorithm %s produced invalid score %r (non-finite or "
                "worst-score sentinel); excluding from phase candidates",
                name,
                score,
            )
            continue
        valid_results[name] = record
    return valid_results


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #


def train_best_model(
    config: str | Path | Config | dict[str, Any],
    df: pd.DataFrame,
    target: str | None = None,
    model_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Main interface for training the best model.

    Orchestrates the decomposed pipeline (``load_config → prepare_dataset →
    execute_phases → persist_artifact``). The logic of each step is extracted
    into separate module functions; here only the winner-score invariant and
    the hand-off of the winner/trainer from the terminal select step context
    to ``persist_artifact`` remain.

    Two-stage winner selection is performed by an explicit pipeline select step
    with ``strategy: sanity_gate``: a finalists pool (``select_finalists``),
    one-time training of the finalists on 100% of the data, a
    ``ModelSanityGate`` audit, and ranking by ``RMSE_oof``
    (``select_robust_winner``, ``active`` mode). The terminal select
    (``output: winner``) hands the winner's already-trained-on-100%-of-the-data
    trainer to ``persist_artifact`` via ``pre_trained_trainer`` — no retraining
    of the winner is required. Exception: when ``general.holdout_fraction`` is
    set, the saved artifact is refit on ALL data (train + hold-out rows are
    concatenated back), because the sanity-gate finalists were trained on the
    train part of the outer split only — the pre-trained trainer is then NOT
    reused. There is no global
    ``general.sanity_gate.mode`` switch anymore: selection is performed ONLY
    by an explicit select step, and the ``general.sanity_gate`` section is the
    source of thresholds.

    Multi-phase semantics: the working set is not accumulated — each HPO phase
    rebuilds it from scratch, and the select step output replaces it. The last
    phase of a valid config is a terminal select (``output: winner``), whose
    result determines the artifact.
    Args:
        config (Union[str, Path, Config, Dict[str, Any]]): The training
            configuration. Can be a file path, a dict, or a Config object.
        df (pd.DataFrame): The input data.
        target (str | None): The target column name. Defaults to 'target'.
        model_path_override (str | Path | None): An alternative model save path.
    Returns:
        Dict[str, Any]: A dict with the results: the algorithm name, ``score``
            — the metric value in user semantics (positive RMSE/MAE, plain R²;
            for neg_-scorers the value is inverted back), ``metric`` — the
            user-facing comparison metric name, the parameters, and the file
            path. If any algorithms were disqualified during the run by the
            circuit breaker (consecutive fatal HPO errors or a Sanity Gate
            select-step audit), the ``disqualified_algorithms`` key is added —
            a dict {algorithm name: disqualification reason}. Each select step
            with ``strategy: sanity_gate`` adds its section to
            ``result["sanity_gate"]`` (key — the phase name).
    Raises:
        TypeError: If a config of an unsupported type is passed.
        RuntimeError: If no algorithm completed an HPO phase successfully, a
            select step received an empty working set, or no finalist was
            trained on the full dataset. The disqualification of a single
            algorithm (``InvalidAlgorithmError``) does NOT abort the run: the
            algorithm is excluded from the candidates and the rest continue
            training.
    """
    cfg, target_col = load_config(config, df, target)
    # Evidence gate: surface the experimental status once per run at
    # config validation time; the sampler itself warns once per process per
    # (algorithm, multiplier) instead of once per fold per trial.
    if cfg.oversampling.enable and _oversampling.OVERSAMPLING_EXPERIMENTAL:
        _LOG.warning(_oversampling.EXPERIMENTAL_WARNING_MESSAGE)
    prepared = prepare_dataset(cfg, df, target_col)
    ctx = execute_phases(cfg, prepared)

    # The winner is taken from the terminal select step context: the last phase
    # of a valid config must be a select with output: winner (cross-validated
    # by the schema), so ctx.winner_algo is already set.
    if ctx.winner_algo is None or not ctx.working_set:
        # Protection against an empty phase list (the config allows phases: []):
        # the loop in execute_phases did not run — raise a clear RuntimeError.
        raise RuntimeError("No valid results after HPO phases; cannot select a winner.")

    winner_algo = ctx.winner_algo
    winner_record = ctx.working_set[winner_algo]
    final_score = winner_record.score
    final_params = winner_record.params
    # The honest evaluation count of the winner's HPO phase: the number of
    # trials actually evaluated (own COMPLETED trials; transferred history
    # never counts). 0 means "unknown" — persist_artifact then omits the
    # key.
    n_trials_evaluated = int(winner_record.n_trials_evaluated or 0)
    # Final invariant: the winner's score must be finite and strictly above the
    # worst-score sentinel class before entering result["score"]. Otherwise the
    # sentinel could leak into the report — for neg_-metrics to_user_value
    # would invert it into the absurd +3.4e38.
    if not is_valid_winner_score(final_score):
        raise RuntimeError(
            f"Winner '{winner_algo}' produced an invalid final score "
            f"{final_score!r} (non-finite or worst-score sentinel); "
            f"refusing to report it."
        )
    return persist_artifact(
        cfg=cfg,
        prepared=prepared,
        winner_algo=winner_algo,
        final_score=final_score,
        final_params=final_params,
        model_path_override=model_path_override,
        disqualified_algorithms=ctx.disqualified_algorithms,
        pre_trained_trainer=ctx.pre_trained_trainer,
        sanity_gate_sections=ctx.sanity_gate_sections or None,
        complexity_guard_sections=(
            {"parameters": ctx.complexity_guard_records}
            if ctx.complexity_guard_records
            else None
        ),
        n_trials_evaluated=n_trials_evaluated,
        oof_pool=ctx.oof_pool,
    )

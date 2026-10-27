"""Finalists pool and robust winner selection (Training Engine).

Selects the winner by the raw CV score (``select_winner``), builds the
corridor-based finalists pool (``select_finalists``), trains every finalist on
100% of the data exactly once and runs the Sanity Gate audit with the
fallback ranking (``select_robust_winner``). It may import ``state``,
``hpo_phases`` and ``artifact`` (the shared trainer assembly) and must not
import ``select_steps`` or ``component``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.common.metrics import (
    is_error_metric,
    oof_rmse,
    to_user_value,
    user_direction,
)
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    CorridorMode,
    SanityGateMode,
    ValidationStrategy,
)

from ..tuner import WORST_SCORE_THRESHOLD
from .artifact import _build_trainer
from .hpo_phases import _algorithms_as_dict
from .sanity_gate import ModelSanityGate, SanityCheckResult
from .state import FinalistCandidate, PhaseResult, PreparedData, RobustWinnerResult

_LOG = logging.getLogger(__name__)


def is_valid_winner_score(score: Any) -> bool:
    """Check that a "raw" score value can qualify as a winning score.

    A valid winner score must be:
    - numeric (int/float/np.floating): None, strings (including numeric ones
      such as ``"0.5"``) and other "garbage" types are rejected;
    - finite (NaN and ±inf signal the absence of a valid metric);
    - strictly above the worst-score sentinel class: any score at or below the
      exact float32 minimum (``WORST_SCORE_THRESHOLD``) is treated as a
      sentinel. The threshold covers both the ``HPO_WORST_SCORE`` constant
      (returned by the tuner for non-finite metrics) and the raw
      ``float(np.finfo(np.float32).min)`` value (possible from numpy-cast
      metrics), which the old ``math.isclose(rel_tol=1e-9)``
      filter let through (relative difference ≈ 9.88e-9 > 1e-9). Real metrics
      of such magnitude are practically impossible, so false exclusion is
      ruled out.

    Args:
        score (Any): The raw score value from the HPO phase.

    Returns:
        bool: True if the score may end up in ``result["score"]``.
    """
    if not isinstance(score, (int, float, np.floating)):
        return False
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > WORST_SCORE_THRESHOLD


def select_winner(results: dict[str, PhaseResult]) -> str:
    """Deterministically select the winning algorithm by its "raw" score.

    The algorithm with the maximum score wins (optimizer semantics — "higher
    is better"). Ties are resolved by iteration order over ``results``: ``max``
    is stable, and the dict preserves insertion order — for the sequential path
    this is the configuration order, so on equal scores the first configured
    algorithm wins (documented tie-break).

    Args:
        results (dict[str, PhaseResult]): The "algorithm → PhaseResult" map of
            the current phase. Must not be empty.

    Returns:
        str: The name of the winning algorithm.

    Raises:
        ValueError: If ``results`` is empty — no winner exists.
    """
    if not results:
        raise ValueError("Cannot select a winner from empty results")
    return max(results.items(), key=lambda kv: kv[1].score)[0]


# --------------------------------------------------------------------------- #
#  Finalists pool Top-K                                                        #
#  Builds the finalists pool: a δ corridor from the leader in user semantics  #
#  plus a hard cap of top_k_candidates. The ``CorridorMode`` type is defined  #
#  in ``config_parser`` (also used by the ``SanityGateCfg`` config).          #
# --------------------------------------------------------------------------- #
FAMILY_DIVERSITY_MULTIPLIER_DEFAULT = 1.5


# Small relative tolerance for comparisons at the corridor boundary: floating
# point artifacts (e.g. 0.55 - 0.5 = 0.050000000000000044) must not exclude a
# candidate that is mathematically exactly at the boundary (the "boundary
# counts as included" rule, a documented tie-break). The tolerance scales with
# the operand magnitude (float64 relative precision), so for scores on the
# order of 1e-300 it does not "widen" the corridor to 1e-12.
_CORRIDOR_EPS = 1e-12


def _tol_scale(value: float, limit: float) -> float:
    """Tolerance scale: the maximum operand magnitude (0 when both are zero)."""
    scale = max(abs(value), abs(limit))
    return _CORRIDOR_EPS * scale if scale > 0.0 else 0.0


def _le_tol(value: float, limit: float) -> bool:
    """``value <= limit`` with a small relative tolerance (boundary included)."""
    return value <= limit + _tol_scale(value, limit)


def _ge_tol(value: float, limit: float) -> bool:
    """``value >= limit`` with a small relative tolerance (boundary included)."""
    return value >= limit - _tol_scale(value, limit)


def _family_of(algo_name: str, algorithm_families: Mapping[str, str] | None) -> str:
    """Resolve the algorithm family (unmapped algorithms are their own family).

    Args:
        algo_name (str): Algorithm name.
        algorithm_families (Mapping[str, str] | None): Algorithm → family map,
            or None when no family classification is provided.

    Returns:
        str: The family name (the algorithm name itself when unmapped).
    """
    if algorithm_families is None:
        return algo_name
    return algorithm_families.get(algo_name, algo_name)


def select_finalists(
    results: dict[str, PhaseResult],
    *,
    metric_user: str,
    top_k_candidates: int = 3,
    corridor_delta: float = 0.15,
    corridor_mode: CorridorMode = "auto",
    enforce_family_diversity: bool = False,
    algorithm_families: Mapping[str, str] | None = None,
    allowed_families: set[str] | None = None,
    family_diversity_multiplier: float = FAMILY_DIVERSITY_MULTIPLIER_DEFAULT,
    disqualified: Iterable[str] = (),
) -> list[str]:
    """Build the deterministic finalists pool.

    Replaces the "blind" CV-first-place selection with a pool of candidates for
    the second phase: candidates whose CV score falls inside the δ corridor from
    the leader, intersected with the hard ``top_k_candidates`` cap. The corridor
    is expressed in **user** metric semantics (see ``metrics.user_direction``
    and ``metrics.to_user_value``): raw optimizer scores from ``results`` are
    converted back to natural values first (the direction comes from
    ``user_direction``, not from the optimizer's ``greater_is_better``).

    Corridor forms (``corridor_mode``):
    - ``multiplicative`` — min-better metrics (errors):
      ``Score_CV(m) <= Score_CV_best * (1 + δ)``; max-better metrics (R² etc.):
      ``Score_CV(m) >= Score_CV_best * (1 - δ)``.
    - ``additive`` — for metrics where the multiplicative form is incorrect
      (zero/negative values: R² < 0, MAE ≈ 0):
      ``|Score_CV(m) - Score_CV_best| <= δ * (max_score - min_score)``, where
      the range is computed over all valid candidates.
    - ``auto`` (default) — chosen by the metric type: error → multiplicative,
      score metric with a possible sign → additive. For error metrics with a
      practically zero best score (MAE ≈ 0) the multiplicative form would
      collapse the pool to the leader, so the additive form is used instead.
    When the multiplicative corridor would exclude the leader itself (e.g. a
    max-better metric with a negative best score such as R² < 0), the additive
    form is used instead (documented edge case).

    Combination of filters (explicit): pool = (candidates inside the corridor)
    ∩ (first ``top_k_candidates`` by CV score). The corridor is the main
    filter; if it yields fewer candidates than ``top_k_candidates`` the pool
    contains exactly those — no topping up beyond the corridor, except the
    optional family-diversity rule below.

    Optional soft family top-up (``enforce_family_diversity``, default False):
    when the pool represents exactly one algorithm family (linear / ensembles /
    kernel-GPR) **and that family is permitted by ``allowed_families``**, the
    best representative of each missing family is added from the *extended*
    corridor (``family_diversity_multiplier * δ``), never exceeding
    ``top_k_candidates``. The added families are also restricted to
    ``allowed_families``; when the pool's single family is not permitted, no
    top-up happens at all.

    Algorithms disqualified by the circuit breaker never enter the
    pool.

    Determinism: the pool is ordered by the raw score descending (optimizer
    semantics — the same ordering as ``select_winner``); ties keep the
    insertion order of ``results`` (the configuration order), which is the
    documented tie-break.

    Args:
        results (dict[str, PhaseResult]): HPO phase results "algorithm →
            PhaseResult". Must not be empty after removing the disqualified
            algorithms.
        metric_user (str): User-facing metric name (``general.comparison_metric``);
            used to convert raw scores to user semantics and to pick the
            corridor direction/form.
        top_k_candidates (int): Hard cap on the pool size (>= 1, default 3).
        corridor_delta (float): Relative corridor width δ (>= 0, default 0.15).
        corridor_mode (CorridorMode): 'multiplicative' | 'additive' | 'auto'
            (default 'auto').
        enforce_family_diversity (bool): Enable the optional soft family
            top-up (default False).
        algorithm_families (Mapping[str, str] | None): Algorithm → family map
            (e.g. 'linear', 'ensemble', 'kernel'). Algorithms absent from the
            map are treated as their own family.
        allowed_families (set[str] | None): Families participating in the
            top-up rule: the pool's single family must be permitted AND only
            these families may be added; None — all families are permitted.
        family_diversity_multiplier (float): Width multiplier of the extended
            corridor for the top-up (default 1.5, must be > 0).
        disqualified (Iterable[str]): Algorithms disqualified by the circuit
            breaker; they are excluded from the pool.

    Returns:
        list[str]: Ordered finalist names (best first), length in
            [1, top_k_candidates].

    Raises:
        RuntimeError: If ``results`` is empty (after removing disqualified
            algorithms) — no finalists exist.
        ValueError: If parameters are out of bounds (``top_k_candidates`` < 1,
            ``corridor_delta`` < 0, ``family_diversity_multiplier`` <= 0,
            unknown ``corridor_mode``) or the metric name cannot be resolved.
    """
    _validate_finalists_args(
        results,
        top_k_candidates=top_k_candidates,
        corridor_delta=corridor_delta,
        corridor_mode=corridor_mode,
        family_diversity_multiplier=family_diversity_multiplier,
    )

    # Circuit breaker: disqualified algorithms never enter the pool.
    # The iterable is materialized BEFORE the dict-comprehension: a generator
    # passed as ``disqualified`` would otherwise be consumed by ``set(...)`` on
    # the first comprehension iteration, letting disqualified algorithms "leak"
    # into the pool.
    disqualified_set = set(disqualified)
    candidates = {
        name: value for name, value in results.items() if name not in disqualified_set
    }
    if not candidates:
        raise RuntimeError(
            "Cannot build a finalists pool: no valid results after removing "
            "disqualified algorithms"
        )

    # User semantics: raw scores are converted into natural values, and the
    # direction is taken from user_direction (for errors this is "minimize",
    # even though the optimizer maximizes their inversions).
    user_scores = {
        name: to_user_value(metric_user, cand.score)
        for name, cand in candidates.items()
    }
    direction = user_direction(metric_user)

    # Leader by user semantics; ties go to the first in insertion order (the
    # same tie-break as in select_winner).
    if direction == "minimize":
        best_name = min(user_scores, key=lambda n: user_scores[n])
    else:
        best_name = max(user_scores, key=lambda n: user_scores[n])
    best_score = user_scores[best_name]

    # Score spread over valid candidates — computed once (for the additive
    # corridor form) rather than inside _in_corridor for each candidate (O(n²)).
    score_spread = max(user_scores.values()) - min(user_scores.values())

    effective_mode = _effective_corridor_mode(corridor_mode, metric_user, best_score)

    def _in_corridor(name: str, mode: CorridorMode, delta: float) -> bool:
        """Check whether a candidate falls inside the corridor of the given form
        and width.

        Boundary equality counts as inclusion (documented tie-break); the small
        relative tolerance ``_CORRIDOR_EPS`` neutralizes floating point
        artifacts.
        """
        score = user_scores[name]
        if mode == "additive":
            limit = delta * score_spread
            return _le_tol(score, best_score + limit) and _ge_tol(
                score, best_score - limit
            )
        if direction == "minimize":
            return _le_tol(score, best_score * (1.0 + delta))
        return _ge_tol(score, best_score * (1.0 - delta))

    base_members = [
        name
        for name in candidates
        if _in_corridor(name, effective_mode, corridor_delta)
    ]
    if best_name not in base_members:
        # The corridor is empty for a max-better metric with a negative best
        # score (e.g. R² < 0): the multiplicative form excludes the leader
        # itself (best*(1-δ) > best when best < 0). Switch to the additive
        # form (documented edge case).
        _LOG.warning(
            "Finalists corridor (mode=%s, direction=%s) excludes the leader "
            "%s (best %s); falling back to the additive corridor form",
            effective_mode,
            direction,
            best_name,
            best_score,
        )
        effective_mode = "additive"
        base_members = [
            name
            for name in candidates
            if _in_corridor(name, effective_mode, corridor_delta)
        ]

    # Deterministic ordering: raw score descending (optimizer semantics, as in
    # select_winner); ties — insertion order (configuration order). The sort is
    # stable.
    ordered = sorted(candidates, key=lambda n: candidates[n].score, reverse=True)

    # Explicit combination (intersection): (candidates inside the corridor) ∩
    # (first top_k by CV score). No topping up beyond the corridor.
    base_set = set(base_members)
    pool = [name for name in ordered if name in base_set][:top_k_candidates]

    # Optional soft family top-up: if the pool represents exactly one family
    # AND that family is permitted by the config (allowed_families), add the
    # best representative of each missing family from the extended corridor
    # (family_diversity_multiplier × δ), never exceeding top_k_candidates. The
    # added families are also restricted to allowed_families. Behavior matches
    # the documentation.
    if enforce_family_diversity and len(pool) < top_k_candidates:
        _family_diversity_topup(
            pool,
            candidates=list(candidates),
            ordered=ordered,
            top_k_candidates=top_k_candidates,
            extended_delta=corridor_delta * family_diversity_multiplier,
            algorithm_families=algorithm_families,
            allowed_families=allowed_families,
            in_corridor=lambda name, delta: _in_corridor(name, effective_mode, delta),
        )

    # Final stable ordering: added families are weaker than base pool members,
    # but the re-sort makes the invariant explicit.
    pool.sort(key=lambda n: candidates[n].score, reverse=True)
    return pool


def _validate_finalists_args(
    results: dict[str, PhaseResult],
    *,
    top_k_candidates: int,
    corridor_delta: float,
    corridor_mode: CorridorMode,
    family_diversity_multiplier: float,
) -> None:
    """Validate the arguments of ``select_finalists``.

    Raises:
        RuntimeError: If ``results`` is empty — no finalists exist.
        ValueError: If a parameter is out of bounds.
    """
    if not results:
        raise RuntimeError("Cannot build a finalists pool from empty results")
    if top_k_candidates < 1:
        raise ValueError("top_k_candidates must be >= 1")
    if corridor_delta < 0:
        raise ValueError("corridor_delta must be >= 0")
    if corridor_mode not in ("multiplicative", "additive", "auto"):
        raise ValueError(
            f"corridor_mode must be 'multiplicative', 'additive' or 'auto'; "
            f"got {corridor_mode!r}"
        )
    if family_diversity_multiplier <= 0:
        raise ValueError("family_diversity_multiplier must be > 0")


def _effective_corridor_mode(
    corridor_mode: CorridorMode, metric_user: str, best_score: float
) -> CorridorMode:
    """Resolve the ``auto`` corridor form by the metric type.

    Args:
        corridor_mode (CorridorMode): The configured corridor form.
        metric_user (str): The user-facing metric name.
        best_score (float): The leader's score in user semantics.

    Returns:
        CorridorMode: ``'multiplicative'`` or ``'additive'``.
    """
    if corridor_mode == "auto":
        if is_error_metric(metric_user):
            # Error → multiplicative, except the degenerate case best≈0
            # (MAE≈0): the multiplicative form yields a threshold ≈0 and
            # collapses the pool to the leader — switch to the additive form.
            effective_mode: CorridorMode = (
                "additive" if abs(best_score) <= _CORRIDOR_EPS else "multiplicative"
            )
        else:
            effective_mode = "additive"
    else:
        effective_mode = corridor_mode
    return effective_mode


def _family_diversity_topup(
    pool: list[str],
    *,
    candidates: list[str],
    ordered: list[str],
    top_k_candidates: int,
    extended_delta: float,
    algorithm_families: Mapping[str, str] | None,
    allowed_families: set[str] | None,
    in_corridor: Callable[[str, float], bool],
) -> None:
    """Soft family top-up of a single-family finalists pool (in place).

    Args:
        pool (list[str]): The finalists pool (extended in place).
        candidates (list[str]): The valid candidates in insertion order.
        ordered (list[str]): The candidates ordered by the raw score.
        top_k_candidates (int): The hard cap on the pool size.
        extended_delta (float): The width of the extended corridor.
        algorithm_families (Mapping[str, str] | None): Algorithm → family map.
        allowed_families (set[str] | None): Families permitted by the config.
        in_corridor (Callable[[str, float], bool]): Whether a candidate is
            inside the corridor of the given width.
    """
    pool_families = {_family_of(name, algorithm_families) for name in pool}
    single_pool_family = next(iter(pool_families)) if len(pool_families) == 1 else None
    if single_pool_family is not None and (
        allowed_families is None or single_pool_family in allowed_families
    ):
        extended_members = [
            name for name in candidates if in_corridor(name, extended_delta)
        ]
        extended_set = set(extended_members)
        missing_families = {
            _family_of(name, algorithm_families)
            for name in candidates
            if _family_of(name, algorithm_families) != single_pool_family
        }
        if allowed_families is not None:
            missing_families &= allowed_families

        pool_names = set(pool)
        remaining = set(missing_families)
        while len(pool) < top_k_candidates and remaining:
            picked: str | None = None
            for name in ordered:
                family = _family_of(name, algorithm_families)
                if (
                    name not in pool_names
                    and name in extended_set
                    and family in remaining
                ):
                    picked = name
                    break
            if picked is None:
                break
            pool.append(picked)
            pool_names.add(picked)
            remaining.remove(_family_of(picked, algorithm_families))


# --------------------------------------------------------------------------- #
#  Robust winner selection                                                     #
#  Two-stage winner selection: finalists pool → training each finalist on     #
#  100% of the data ONCE → ModelSanityGate audit → ranking of the audited     #
#  candidates by RMSE_oof.                                                    #
# --------------------------------------------------------------------------- #


def _fit_finalist(
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    cfg: Config,
    metric_name_sklearn: str,
    *,
    validation_strategy: ValidationStrategy | str,
    n_folds: int,
    test_size: float,
    random_state: int | None,
) -> Any:
    """Train a finalist on 100% of the data and return its ``ModelTrainer``.

    Performs exactly one training pass (``ModelTrainer.fit``): validation
    scoring with the OOF channel plus a final fit on the full dataset. The
    result is reused downstream for ``RMSE_full``, the audit, and the report —
    no duplicated training.

    Args:
        algo_name (str): The algorithm name.
        algo_cfg (AlgoCfg): The algorithm configuration.
        X (pd.DataFrame): The full feature matrix.
        y (pd.Series): The full target vector.
        best_params (dict[str, Any]): The best hyperparameters from HPO.
        cfg (Config): The root config.
        metric_name_sklearn (str): The metric name in sklearn semantics.
        validation_strategy (ValidationStrategy | str): The resolved
            validation strategy.
        n_folds (int): The number of folds (for 'k_fold').
        test_size (float | int): The hold-out size.
        random_state (int | None): The engine seed (``general.random_state``)
            for the finalist's splits and model seeds; ``None`` — a
            non-reproducible fit.

    Returns:
        Any: The trained ``ModelTrainer`` (pipeline on 100% of the data, OOF
            attributes).

    Raises:
        Exception: Any training error is propagated upward — the calling code
            (``_train_finalists``) catches it and excludes the finalist.
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
    return trainer


def _to_candidate(
    algo_name: str,
    trainer: Any,
    params: dict[str, Any],
    cv_score: float,
    X: pd.DataFrame,
    y: pd.Series,
) -> FinalistCandidate:
    """Assemble a ``FinalistCandidate`` from a trained ``ModelTrainer``.

    Computes the derived quantities (``y_pred_full``, ``rmse_full``) exactly
    once from the already-trained trainer; the OOF vector and ``RMSE_oof`` come
    from the trainer's attributes.

    Args:
        algo_name (str): The algorithm name.
        trainer (Any): The trained ``ModelTrainer``.
        params (dict[str, Any]): The best hyperparameters.
        cv_score (float): The raw CV score from HPO.
        X (pd.DataFrame): The full feature matrix.
        y (pd.Series): The full target vector.

    Returns:
        FinalistCandidate: A candidate with reusable full-fit vectors.
    """
    y_pred_full = np.asarray(trainer.predict(X), dtype=float).reshape(-1)
    rmse_full: float | None = None
    try:
        rmse_full = oof_rmse(y, y_pred_full)
    except ValueError:
        rmse_full = None
    return FinalistCandidate(
        algo_name=algo_name,
        model=trainer.pipeline,
        params=params,
        cv_score=cv_score,
        rmse_oof=trainer.oof_score_,
        y_pred_oof=trainer.oof_predictions_,
        y_pred_full=y_pred_full,
        rmse_full=rmse_full,
        trainer=trainer,
    )


def _train_finalists(
    pool: list[str],
    phase_results: dict[str, PhaseResult],
    cfg: Config,
    prepared: PreparedData,
    random_state: int | None,
) -> dict[str, FinalistCandidate]:
    """Train every finalist in the pool on 100% of the data exactly once.

    A training failure of a single finalist does not abort the run: the
    exception is caught, the reason is logged (WARNING), and the candidate is
    skipped (moving on to the next one).

    Args:
        pool (list[str]): The ordered finalists pool (from ``select_finalists``).
        phase_results (dict[str, PhaseResult]): HPO results
            "algorithm → PhaseResult".
        cfg (Config): The root config.
        prepared (PreparedData): The prepared data and settings.
        random_state (int | None): The engine seed (``general.random_state``)
            forwarded to every finalist's ``ModelTrainer``.

    Returns:
        dict[str, FinalistCandidate]: Trained candidates (key — the algorithm
            name), in pool order (the dict preserves insertion order).
    """
    algos = _algorithms_as_dict(cfg.algorithms)
    candidates: dict[str, FinalistCandidate] = {}
    for name in pool:
        if name not in phase_results:
            continue
        record = phase_results[name]
        params = record.params
        score = record.score
        try:
            trainer = _fit_finalist(
                name,
                algos[name],
                prepared.X,
                prepared.y,
                params,
                cfg,
                prepared.metric_sklearn,
                validation_strategy=prepared.resolved_validation,
                n_folds=prepared.resolved_n_folds,
                test_size=prepared.resolved_test_size,
                random_state=random_state,
            )
        except Exception as err:  # noqa: BLE001 — the finalist is excluded
            _LOG.warning(
                "Finalist %s failed to train on the full dataset: %s. "
                "The candidate is skipped.",
                name,
                err,
                exc_info=True,
            )
            continue
        candidates[name] = _to_candidate(
            name, trainer, params, score, prepared.X, prepared.y
        )
    return candidates


# Circuit significance for the fallback criterion: D > A > B > C.
# The higher the rank, the "heavier" the violation. Ranks are fixed in code.
# Note: in the warn mode of circuit D no violation (neither a finite gap
# above the threshold nor an infinite gap from division by zero) produces a
# reason — everything goes to soft_warnings and does not take part in the
# fallback ranking; the D rank (3) applies only to the reject mode.
_CIRCUIT_RANK_BY_LETTER = {
    "A": 2,  # A — diversity
    "B": 1,  # B — unique
    "C": 0,  # C — dead features
    "D": 3,  # D — generalization gap (the most severe)
}


def _circuit_rank(reason: str) -> int:
    """Significance rank of a circuit from the reason string (D > A > B > C).

    Parses the ``Circuit <letter>`` prefix in reason (``ModelSanityGate``
    format). An unknown format is treated as the least significant circuit
    (rank 0) — a defensive fallback so the fallback ranking never fails on
    custom reasons.

    Args:
        reason (str): The disqualification reason text.

    Returns:
        int: The significance rank (0..3); 3 — circuit D, 2 — A, 1 — B,
            0 — C/unknown.
    """
    match = re.search(r"Circuit\s+([A-D])", reason)
    if match is None:
        return 0
    return _CIRCUIT_RANK_BY_LETTER.get(match.group(1), 0)


def _severity_signature(reasons: list[str]) -> tuple[int, ...]:
    """Violation severity signature for the fallback criterion.

    A tuple of circuit significance ranks sorted in descending order (the most
    severe violation first). A lexicographically smaller signature means a less
    problematic model given the same number of violated circuits.

    Args:
        reasons (list[str]): The candidate's disqualification reasons.

    Returns:
        tuple[int, ...]: The severity signature sorted in descending order.
    """
    return tuple(sorted((_circuit_rank(r) for r in reasons), reverse=True))


def _best_by_rmse_oof(
    names: list[str], candidates: Mapping[str, FinalistCandidate]
) -> str:
    """Select the candidate with the best (minimal) ``RMSE_oof``.

    Tie-break: on equal ``RMSE_oof`` the first in list order wins (the pool
    order is deterministic). A candidate with unavailable ``RMSE_oof`` (None)
    is ranked last.

    Args:
        names (list[str]): Candidates in deterministic order.
        candidates (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        str: The winner's name.
    """

    def _key(name: str) -> tuple[int, float]:
        rmse = candidates[name].rmse_oof
        return (rmse is None, rmse if rmse is not None else float("inf"))

    return min(names, key=_key)


def _fallback_sort_key(
    name: str,
    audits: Mapping[str, SanityCheckResult],
    candidate_models: Mapping[str, FinalistCandidate],
) -> tuple[int, tuple[int, ...], bool, float]:
    """Fallback ranking key (see ``_select_active_winner``).

    Minimum number of violated circuits → minimum severity (D > A > B > C) →
    best ``RMSE_oof``. Hoisted to module level so the select step ranking
    (``output: top_k``/``pool``) matches the winner selection.

    Args:
        name (str): The candidate name.
        audits (Mapping[str, SanityCheckResult]): The audit of each finalist.
        candidate_models (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        tuple[int, tuple[int, ...], bool, float]: The sort key.
    """
    reasons = audits[name].reasons
    rmse = candidate_models[name].rmse_oof
    return (
        len(reasons),
        _severity_signature(reasons),
        rmse is None,
        rmse if rmse is not None else float("inf"),
    )


def _select_active_winner(
    pool_order: list[str],
    audits: Mapping[str, SanityCheckResult],
    candidate_models: Mapping[str, FinalistCandidate],
) -> str:
    """Select the winner in ``active`` mode.

    First — the best ``RMSE_oof`` among the candidates that passed the audit
    (CV and ``RMSE_full`` do not participate in the ranking). If there are no
    valid candidates (total pool failure) — the formal fallback criterion of
    the "least problematic" model: minimum number of violated circuits →
    minimum severity (circuit significance is fixed in code: D > A > B > C,
    see ``_circuit_rank``) → best ``RMSE_oof``. The run never fails.

    Args:
        pool_order (list[str]): The ordered finalists pool.
        audits (Mapping[str, SanityCheckResult]): The audit of each finalist.
        candidate_models (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        str: The winner's name.
    """
    valid = [n for n in pool_order if audits[n].is_valid]
    if valid:
        return _best_by_rmse_oof(valid, candidate_models)
    return min(
        pool_order,
        key=lambda n: _fallback_sort_key(n, audits, candidate_models),
    )


def _fmt_audit_value(value: float) -> str:
    """Format a numeric audit metric for logs.

    NaN/±inf (circuit disabled or not computable) are rendered as ``"n/a"`` so
    that the winner's audit summary log stays compact and readable.

    Note: the formatter is used in the INFO log of the audit summary of the
    ``sanity_gate`` select step; it is also used by the report mechanisms (see
    also ``select_finalists`` / ``_train_finalists`` / ``select_robust_winner``).

    Args:
        value (float): The audit metric value (diversity/unique/gap).

    Returns:
        str: ``"%.4f"`` for finite values, ``"n/a"`` otherwise.
    """
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


class _AuditSkipReason(str, Enum):
    """Reasons for skipping the Sanity Gate audit.

    An audit skip differs from a failure: the check did not run because of a
    resource limit of the phase (``audit_time_budget_seconds``), not because
    of the model quality. A candidate with such a reason cannot win the
    current select step (there is no audit), but must NOT be disqualified
    forever: the limit of one phase does not throw the algorithm out of all
    subsequent ones.
    """

    BUDGET_EXCEEDED = "Sanity audit skipped: audit time budget"


def _audit_budget_skip_reason(budget_seconds: float) -> str:
    """Build the reason text for a time-budget audit skip.

    The single source of the format: used by ``select_robust_winner`` when
    building the ``SanityCheckResult`` and by tests — the string is not
    duplicated anywhere.

    Args:
        budget_seconds (float): The phase's ``audit_time_budget_seconds`` value.

    Returns:
        str: A reason like "Sanity audit skipped: audit time budget (Ns) exceeded".
    """
    return f"{_AuditSkipReason.BUDGET_EXCEEDED.value} ({budget_seconds:g}s) exceeded"


# Full match of the audit-skip reason to the "{prefix} (Ns) exceeded" format,
# where N is a number in the %g format (integer/decimal/exponent). The
# fullmatch check rules out false positives on reasons that only start with
# the prefix.
_AUDIT_BUDGET_SKIP_RE = re.compile(
    rf"{re.escape(_AuditSkipReason.BUDGET_EXCEEDED.value)} "
    r"\([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?s\) exceeded"
)


def _is_audit_budget_skip(reasons: list[str]) -> bool:
    """True if ALL candidate reasons are a time-budget audit skip.

    Used in ``select_steps._gate_disqualifications`` so that such candidates do not
    enter the circuit breaker (``disqualified_algorithms``) — the phase
    budget must not throw the algorithm out of all subsequent phases.

    Args:
        reasons (list[str]): The candidate's audit invalidity reasons.

    Returns:
        bool: True if the reasons reduce to a skip caused by an exhausted
            ``audit_time_budget_seconds`` (at least one reason exists).
    """
    return bool(reasons) and all(
        _AUDIT_BUDGET_SKIP_RE.fullmatch(reason) is not None for reason in reasons
    )


def select_robust_winner(
    candidate_models: Mapping[str, FinalistCandidate],
    X: pd.DataFrame,
    y: pd.Series,
    sanity_gate: ModelSanityGate,
    *,
    mode: SanityGateMode = "off",
    audit_time_budget_seconds: float = 0,
) -> RobustWinnerResult:
    """Two-stage winner selection with a Sanity Gate audit.

    Iterates over the candidate pool (already trained on 100% of the data),
    runs each candidate through ``ModelSanityGate.check`` (diversity/unique are
    computed on the OOF vector; circuit D uses full-fit predictions) and ranks
    the candidates that passed the audit by ``RMSE_oof`` (CV and ``RMSE_full``
    do NOT participate in the ranking — the composite score was removed after
    the v2 review).

    Modes (``mode``):
    - ``off``: legacy ``select_winner`` behavior — winner = best ``cv_score``
      (no audit; ``audit`` stays empty);
    - ``active``: disqualifications are applied; winner = best ``RMSE_oof``
      among those that passed all checks. Pipeline select steps with
      ``strategy: sanity_gate`` run exactly this mode. The ``warn_only`` mode
      was removed along with the ``sanity_gate_warn_only``/``winner_if_active``
      report keys.

    Audit time budget (``audit_time_budget_seconds``): the total time of the
    audit phase is limited; when the budget is exhausted the remaining
    candidates are not audited and receive the explicit reason "audit skipped:
    budget exceeded" (in ``active`` mode they cannot win). ``0`` — no limit.

    Fallback (formalized): if all candidates are rejected, the "least
    problematic" model is chosen — minimum number of violated circuits, on ties
    — minimum violation severity (circuit significance is fixed in code:
    D > A > B > C, see ``_circuit_rank``), on further ties — best ``RMSE_oof``.
    The run never fails; a WARNING with the reasons is logged.

    Args:
        candidate_models (Mapping[str, FinalistCandidate]): The candidate pool
            (trained on 100% of the data, OOF vectors available). Insertion
            order is the deterministic pool order.
        X (pd.DataFrame): The full feature matrix.
        y (pd.Series): The full target vector.
        sanity_gate (ModelSanityGate): The configured gate instance.
        mode (SanityGateMode): The operation mode ('off'/'active').
        audit_time_budget_seconds (float): The total audit phase time budget in
            seconds; 0 — no limit.

    Returns:
        RobustWinnerResult: The winner, the audit of each candidate, the
            applied disqualifications, and the fallback flag.

    Raises:
        RuntimeError: If the candidate pool is empty — no winner exists.
        ValueError: If ``mode`` is unknown.
    """
    if not candidate_models:
        raise RuntimeError("Cannot select a robust winner from an empty candidate pool")

    pool_order = list(candidate_models)

    def _cv_winner() -> str:
        """Legacy select_winner behavior: best raw score, ties — order."""
        return max(pool_order, key=lambda n: candidate_models[n].cv_score)

    if mode == "off":
        winner = _cv_winner()
        return RobustWinnerResult(
            winner_algo=winner,
            mode=mode,
            pool=pool_order,
            winner_rmse_oof=candidate_models[winner].rmse_oof,
        )

    if mode not in ("off", "active"):
        raise ValueError(f"mode must be 'off' or 'active'; got {mode!r}")

    # Audit each finalist: diversity/unique on the OOF vector, circuit D on the
    # full-fit predictions. The total audit phase time is limited by the
    # audit_time_budget_seconds budget (0 — no limit).
    audits: dict[str, SanityCheckResult] = {}
    audit_started = time.monotonic()

    def _audit_budget_exhausted() -> bool:
        return (
            audit_time_budget_seconds > 0
            and time.monotonic() - audit_started >= audit_time_budget_seconds
        )

    for name in pool_order:
        if _audit_budget_exhausted():
            _LOG.warning(
                "[sanity_gate] audit time budget (%.3fs) exceeded; "
                "skipping audit of candidate %s",
                audit_time_budget_seconds,
                name,
            )
            audits[name] = SanityCheckResult(
                is_valid=False,
                reasons=[_audit_budget_skip_reason(audit_time_budget_seconds)],
                severity=0,
            )
            continue
        cand = candidate_models[name]
        try:
            audits[name] = sanity_gate.check(
                y=y,
                y_pred_oof=cand.y_pred_oof,
                y_pred_full=cand.y_pred_full,
                X=X,
                model=cand.model,
                candidate_name=name,
            )
        except Exception as err:  # noqa: BLE001 — audit failure = disqualification
            _LOG.warning("Sanity audit failed for candidate %s: %s", name, err)
            audits[name] = SanityCheckResult(
                is_valid=False,
                reasons=[f"Sanity audit failed: {err}"],
                severity=4,
            )

    disqualified = {
        name: list(audits[name].reasons) for name in pool_order if audits[name].reasons
    }

    # For each rejected candidate — a WARNING listing the disqualification
    # reasons (active mode applies to winner selection).
    for name, reasons in disqualified.items():
        _LOG.warning(
            "[sanity_gate active] Candidate %s disqualified by the sanity gate: %s",
            name,
            reasons,
        )

    # mode == "active": apply disqualifications, rank by RMSE_oof.
    winner = _select_active_winner(pool_order, audits, candidate_models)
    all_failed = not any(audits[n].is_valid for n in pool_order)
    if all_failed:
        # Fallback: all candidates were rejected — pick the "least
        # problematic" model. The formal criterion is stated explicitly in the
        # log.
        _LOG.warning(
            "[sanity_gate active] All %d candidate(s) failed the sanity audit; "
            "falling back to the least problematic model %s. Formal criterion: "
            "min violated circuits -> min severity (D > A > B > C) -> "
            "best RMSE_oof. Reasons: %s",
            len(pool_order),
            winner,
            {n: audits[n].reasons for n in pool_order},
        )
    return RobustWinnerResult(
        winner_algo=winner,
        mode=mode,
        pool=pool_order,
        audit=audits,
        disqualified=disqualified,
        fallback_used=all_failed,
        winner_rmse_oof=candidate_models[winner].rmse_oof,
    )

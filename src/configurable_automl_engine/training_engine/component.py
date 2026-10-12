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
Main components:
    - train_best_model: Public interface — orchestrates the steps above.
    - PhaseContext: Explicit context shared between phases (working set,
      gate context, circuit breaker).
    - _run_hpo: Wrapper for hyperparameter search through external tuners.
    - _fit_and_save: Final model training on the full dataset.
    - _load_module: Helper for dynamic module import by path.
Usage example:
    results = train_best_model(
        config="path/to/config.yaml",
        df=my_dataframe,
        target="target_col",
        model_path_override="models/best.pkl"
    )
"""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
from configurable_automl_engine.common.validation_utils import (
    check_target_exists,
    prepare_X_y,
    validate_df_not_empty,
)
from configurable_automl_engine.preprocessing import detect_feature_types
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    CorridorMode,
    FeatureSelectionCfg,
    FeatureSelectionMode,
    HPOPhaseCfg,
    SanityGateCfg,
    SanityGateMode,
    ValidationStrategy,
    read_config,
)
from configurable_automl_engine.validation import RANDOM_STATE, make_cv

# ───────────────────────── canonical IAE ─────────────────────── #
from ..tuner import WORST_SCORE_THRESHOLD
from ..tuner import InvalidAlgorithmError as _CanonicalIAE
from .logger import setup_logging
from .metrics import (
    direction_label,
    is_error_metric,
    oof_rmse,
    to_sklearn_name,
    to_user_value,
    user_direction,
)
from .sanity_gate import ModelSanityGate, SanityCheckResult
from .thread_pool import run_parallel

_LOG = logging.getLogger("training_engine")


def is_valid_winner_score(score: Any) -> bool:
    """Check that a "raw" score value can qualify as a winning score.

    A valid winner score must be:
    - numeric (int/float/np.floating): None, strings (including numeric ones
      such as ``"0.5"``) and other "garbage" types from custom tuners are
      rejected;
    - finite (NaN and ±inf signal the absence of a valid metric);
    - strictly above the worst-score sentinel class: any score at or below the
      exact float32 minimum (``WORST_SCORE_THRESHOLD``) is treated as a
      sentinel. The threshold covers both the ``HPO_WORST_SCORE`` constant
      (returned by the tuner for non-finite metrics) and the raw
      ``float(np.finfo(np.float32).min)`` value (possible from numpy-cast
      metrics or custom tuners), which the old ``math.isclose(rel_tol=1e-9)``
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


def select_winner(results: dict[str, tuple[float, dict[str, Any]]]) -> str:
    """Deterministically select the winning algorithm by its "raw" score.

    The algorithm with the maximum score wins (optimizer semantics — "higher
    is better"). Ties are resolved by iteration order over ``results``: ``max``
    is stable, and the dict preserves insertion order — for the sequential path
    this is the configuration order, so on equal scores the first configured
    algorithm wins (documented tie-break).

    Args:
        results (dict[str, tuple[float, dict[str, Any]]]): The "algorithm →
            (score, params)" map of the current phase. Must not be empty.

    Returns:
        str: The name of the winning algorithm.

    Raises:
        ValueError: If ``results`` is empty — no winner exists.
    """
    if not results:
        raise ValueError("Cannot select a winner from empty results")
    return max(results.items(), key=lambda kv: kv[1][0])[0]


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
    results: dict[str, tuple[float, dict[str, Any]]],
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
        results (dict[str, tuple[float, dict[str, Any]]]): HPO phase results
            "algorithm → (raw score, params)". Must not be empty after removing
            the disqualified algorithms.
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
        name: to_user_value(metric_user, score)
        for name, (score, _) in candidates.items()
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
    ordered = sorted(candidates, key=lambda n: candidates[n][0], reverse=True)

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
        pool_families = {_family_of(name, algorithm_families) for name in pool}
        single_pool_family = (
            next(iter(pool_families)) if len(pool_families) == 1 else None
        )
        if single_pool_family is not None and (
            allowed_families is None or single_pool_family in allowed_families
        ):
            extended_delta = corridor_delta * family_diversity_multiplier
            extended_members = [
                name
                for name in candidates
                if _in_corridor(name, effective_mode, extended_delta)
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

    # Final stable ordering: added families are weaker than base pool members,
    # but the re-sort makes the invariant explicit.
    pool.sort(key=lambda n: candidates[n][0], reverse=True)
    return pool


# --------------------------------------------------------------------------- #
#  Robust winner selection                                                     #
#  Two-stage winner selection: finalists pool → training each finalist on     #
#  100% of the data ONCE → ModelSanityGate audit → ranking of the audited     #
#  candidates by RMSE_oof.                                                    #
# --------------------------------------------------------------------------- #
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
            (``phase_results[algo_name][1]``).
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


def _fit_finalist(
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    cfg: Config,
    metric_name_sklearn: str,
    *,
    validation_strategy: ValidationStrategy | str | None,
    n_folds: int | None,
    test_size: float | None,
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
        validation_strategy (ValidationStrategy | str | None): The resolved
            validation strategy.
        n_folds (int | None): The number of folds (for 'k_fold').
        test_size (float | int | None): The hold-out size.

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
    phase_results: dict[str, tuple[float, dict[str, Any]]],
    cfg: Config,
    prepared: PreparedData,
) -> dict[str, FinalistCandidate]:
    """Train every finalist in the pool on 100% of the data exactly once.

    A training failure of a single finalist does not abort the run: the
    exception is caught, the reason is logged (WARNING), and the candidate is
    skipped (moving on to the next one).

    Args:
        pool (list[str]): The ordered finalists pool (from ``select_finalists``).
        phase_results (dict[str, tuple[float, dict[str, Any]]]): HPO results
            "algorithm → (score, params)".
        cfg (Config): The root config.
        prepared (PreparedData): The prepared data and settings.

    Returns:
        dict[str, FinalistCandidate]: Trained candidates (key — the algorithm
            name), in pool order (the dict preserves insertion order).
    """
    algos = _algorithms_as_dict(cfg.algorithms)
    candidates: dict[str, FinalistCandidate] = {}
    for name in pool:
        if name not in phase_results:
            continue
        score, params = phase_results[name]
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

    Used in ``_execute_select_sanity_gate`` so that such candidates do not
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
def _run_hpo(
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
    initial_params: dict[str, Any] | None = None,
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
) -> tuple[float, dict[str, Any]] | None:
    """Run the hyperparameter search for an algorithm.
    Workflow:
    1. Tuner loading: the module specified in `algo_cfg.tuner` is imported.
    2. Signature check: the tuner's `optimize` method is inspected for support
       of specific parameters (n_folds, oversampling, space_overrides).
    3. Execution: the optimization is run. If the algorithm is deemed invalid
       via `InvalidAlgorithmError`, the canonical exception is raised.
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
        initial_params (dict[str, Any] | None): Hyperparameters from the
            previous HPO phase for enqueue_trial (monotonic improvement between
            phases).
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
    Returns:
        Optional[Tuple[float, Dict[str, Any]]]:
            The tuple (best metric, best params) or None if an error occurred
            during HPO.
    Raises:
        _CanonicalIAE: If the tuner reports that the algorithm is incompatible
            with the data.
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
    kwargs: dict[str, Any] = {
        "algo_name": algo_name,
        "X": X,
        "y": y,
        "metric": metric_name_sklearn,
        "n_trials": n_trials,
        "validation_strategy": validation_strategy,
    }

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

    # pass the custom search space if provided
    if search_space_override is not None:
        # We pass a dict {algo_name: hyperparameters_dict}
        overrides = {algo_name: search_space_override}
        if "space_overrides" in sig.parameters:
            kwargs["space_overrides"] = overrides

    # pass initial_params if available (for the refine phase)
    if initial_params is not None and "initial_params" in sig.parameters:
        kwargs["initial_params"] = initial_params

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

    # The HPO phase needs only the best parameters and score — the final model
    # is trained later by the engine's own trainer. Ask a tuner that supports
    # skipping the final full-data fit to do so: otherwise one redundant full
    # training per algorithm per phase is wasted (the model is discarded
    # below).
    if "fit_best_model" in sig.parameters:
        kwargs["fit_best_model"] = False

    try:
        _, best_params, best_score = tuner.optimize(**kwargs)
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
        return best_score, best_params
    except Exception as err:
        if err.__class__.__name__ == "InvalidAlgorithmError":
            raise _CanonicalIAE(str(err)) from err
        _LOG.error("Algorithm %s failed during HPO: %s", algo_name, err, exc_info=True)
        # Return None so the main loop knows the algorithm didn't pass.
        return None


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
    validation_strategy: ValidationStrategy | str | None = None,
    n_folds: int | None = None,
    test_size: float | None = None,
) -> Any:
    """Construct a ``ModelTrainer`` for an algorithm from its configuration.

    The single assembly point for the trainer, used both by the final training
    of the winner (``_fit_and_save``) and by the training of finalists on 100%
    of the data (``_fit_finalist``): it guarantees that both paths pass the
    same preprocessing, oversampling, encoding, feature selection, and
    validation settings to the trainer.

    Args:
        algo_name (str): The name of the selected algorithm.
        algo_cfg (AlgoCfg): The algorithm configuration with the trainer path.
        best_params (Dict[str, Any]): The found optimal hyperparameters.
        cfg (Config): The common configuration object.
        metric_name_sklearn (str): The main metric name in sklearn format.
        validation_strategy (ValidationStrategy | str | None): The resolved
            validation strategy. ``None`` — the ``ModelTrainer`` default
            ('train_test_split').
        n_folds (int | None): The number of folds (for 'k_fold').
        test_size (float | int | None): The hold-out size.

    Returns:
        Any: The constructed (but not yet trained) ``ModelTrainer``.

    Raises:
        ValueError: If ``algo_cfg.trainer_module`` is not set or ``best_params``
            is ``None`` (a signal of a complete HPO failure).
        AttributeError: If the trainer module lacks a ``ModelTrainer`` class.
    """
    if algo_cfg.trainer_module is None:
        raise ValueError("Trainer module path is not configured")
    if best_params is None:
        raise ValueError(
            f"Algorithm '{algo_name}' produced no hyperparameters "
            "(best_params is None): HPO phase failed completely. The algorithm "
            "should have been excluded before the final fit."
        )
    trainer_module = _load_module(algo_cfg.trainer_module)
    if not hasattr(trainer_module, "ModelTrainer"):
        raise AttributeError(
            f"Module {algo_cfg.trainer_module} lacks `ModelTrainer` class"
        )

    # The HPO winner's feature selection decision: in 'auto' mode the
    # "use_feature_selection" key is fixed in the Optuna best_params; in
    # 'always'/'disabled' modes the key is absent -> None, and ModelTrainer
    # resolves the activity itself from the configuration. The helper key is
    # popped from a copy: the original best_params (also returned in
    # result["params"]) is not mutated, and "use_feature_selection" is
    # guaranteed not to leak into the ModelTrainer hyperparameters (previously
    # it was dropped only implicitly through clean_hyperparameters).
    trainer_params = dict(best_params)
    fs_active = trainer_params.pop("use_feature_selection", None)

    # Check the ModelTrainer signature (same as for tuners in _run_hpo):
    # custom trainers without feature selection support must not receive new
    # arguments (otherwise a TypeError). A signature with **kwargs is
    # considered compatible — the trainer decides what to do with extra keys.
    trainer_sig = inspect.signature(trainer_module.ModelTrainer)
    trainer_accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_sig.parameters.values()
    )
    trainer_has_fs_cfg = "feature_selection_cfg" in trainer_sig.parameters
    trainer_has_fs_active = "feature_selection_active" in trainer_sig.parameters

    trainer_kwargs: dict[str, Any] = {
        "algorithm": algo_name,
        "hyperparams": trainer_params,
        "metric": metric_name_sklearn,
        # Pass the oversampling settings from the config to the trainer
        "data_oversampling": cfg.oversampling.enable,
        "data_oversampling_multiplier": cfg.oversampling.multiplier,
        "data_oversampling_algorithm": cfg.oversampling.algorithm,
        "serialization_format": cfg.general.serialization_format,
        "encoding_strategy": cfg.general.categorical_encoding,
        "additional_metrics": cfg.general.additional_metrics,
        # Explicit preprocessing preset override from the config: applied
        # consistently with the HPO phase.
        "preprocessing_override": getattr(algo_cfg, "preprocessing", None),
        # High-cardinality encoding parameters and the new strategies
        "high_cardinality_threshold": cfg.general.high_cardinality_threshold,
        "high_cardinality_encoding": cfg.general.high_cardinality_encoding,
        "hashing_n_components": cfg.general.hashing_n_components,
        "target_encoding_smoothing": cfg.general.target_encoding_smoothing,
        "target_encoding_fallback": cfg.general.target_encoding_fallback,
    }
    # Feature selection: the root Config configuration plus the HPO winner's
    # decision. Arguments are passed only to trainers that support them; when
    # feature selection is active we warn about silent ignoring.
    if trainer_has_fs_cfg or trainer_accepts_var_kwargs:
        trainer_kwargs["feature_selection_cfg"] = cfg.general.feature_selection
    elif _feature_selection_mode(cfg.general.feature_selection) not in (
        None,
        "disabled",
    ):
        _LOG.warning(
            "ModelTrainer %s does not accept `feature_selection_cfg`; "
            "feature selection mode=%r will NOT be applied in the final fit.",
            algo_cfg.trainer_module,
            _feature_selection_mode(cfg.general.feature_selection),
        )

    if trainer_has_fs_active or trainer_accepts_var_kwargs:
        trainer_kwargs["feature_selection_active"] = fs_active
    elif fs_active is not None:
        _LOG.warning(
            "ModelTrainer %s does not accept `feature_selection_active`; "
            "the HPO decision use_feature_selection=%r will NOT be applied "
            "in the final fit.",
            algo_cfg.trainer_module,
            fs_active,
        )

    # Validation parameters: the resolved values (method + number of folds /
    # hold-out size) are passed to the final fit instead of the 'auto' string,
    # so the final val_score is computed with the same method as the model
    # comparison estimate in HPO. Custom trainers without support for the
    # arguments do not receive the new keys (the signature is checked as for
    # feature selection); the ModelTrainer default behavior then matches.
    if validation_strategy is not None:
        if (
            "validation_strategy" in trainer_sig.parameters
            or trainer_accepts_var_kwargs
        ):
            trainer_kwargs["validation_strategy"] = validation_strategy
        else:
            _LOG.warning(
                "ModelTrainer %s does not accept `validation_strategy`; "
                "the resolved strategy %r will NOT be applied in the final fit.",
                algo_cfg.trainer_module,
                validation_strategy,
            )
    if n_folds is not None and (
        "n_folds" in trainer_sig.parameters or trainer_accepts_var_kwargs
    ):
        trainer_kwargs["n_folds"] = n_folds
    if test_size is not None and (
        "test_size" in trainer_sig.parameters or trainer_accepts_var_kwargs
    ):
        trainer_kwargs["test_size"] = test_size

    return trainer_module.ModelTrainer(**trainer_kwargs)


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
    validation_strategy: ValidationStrategy | str | None = None,
    n_folds: int | None = None,
    test_size: float | None = None,
) -> Any:
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
        validation_strategy (ValidationStrategy | str | None): The resolved
            validation strategy of the final model. ``None`` — the
            ``ModelTrainer`` default ('train_test_split').
        n_folds (int | None): The number of folds (for 'k_fold').
        test_size (float | int | None): The hold-out size (a fraction or a
            number of rows after 'auto' resolution).
    Returns:
        Any: The ``ModelTrainer`` instance after training and saving (used to
            access the additional metric values).
    Raises:
        AttributeError: If the trainer module lacks a `ModelTrainer` class.
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
    )
    trainer.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(model_path)
    return trainer


# --------------------------------------------------------------------------- #
#  Orchestration helpers                                                      #
#  Pipeline steps extracted from the monolithic ``train_best_model``:         #
#  load_config → prepare_dataset → execute_phases → select_winner →          #
#  persist_artifact.                                                          #
# --------------------------------------------------------------------------- #


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
        random_state=RANDOM_STATE,
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
    )


@dataclass
class PhaseContext:
    """Explicit context shared between pipeline phases.

    Replaces the local ``phase_results`` variable: the context is created by
    ``execute_phases`` once and is mutated sequentially by the phases. The
    select step output REPLACES the working set (subsequent ``refine`` /
    ``select`` steps operate on it).

    Attributes:
        working_set (dict[str, tuple[float, dict[str, Any]]]): The working set
            "algorithm → (raw CV score, params)". Rebuilt by each HPO phase and
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
    """

    working_set: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict)
    disqualified_algorithms: dict[str, str] = field(default_factory=dict)
    full_fit_trainers: dict[str, Any] = field(default_factory=dict)
    sanity_gate_sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    winner_algo: str | None = None
    pre_trained_trainer: Any | None = None


@dataclass
class _HPOWorkerResult:
    """Result of a single HPO worker (serialized between processes).

    Attributes:
        name (str): The algorithm name.
        score (float | None): A valid "raw" score; ``None`` on failure.
        params (dict[str, Any] | None): The best parameters; ``None`` on failure.
        disqualify_reason (str | None): Circuit breaker disqualification
            reason; ``None`` — the algorithm is not disqualified.
    """

    name: str
    score: float | None = None
    params: dict[str, Any] | None = None
    disqualify_reason: str | None = None


@dataclass
class _HpoPhaseSpec:
    """An immutable snapshot of the HPO phase parameters passed to the worker.

    Built in ``execute_phases`` once per phase and passed to
    ``_hpo_phase_worker`` as an explicit argument. This keeps the worker a
    module-level function (not a closure), which can be serialized in the
    "processes" mode: a closure inside ``execute_phases`` cannot be
    serialized.

    Attributes:
        phase (HPOPhaseCfg): Configuration of the current phase.
        prev_results (dict[str, tuple[float, dict[str, Any]]]): Snapshot of
            the previous phase results (for warm-start refine).
        X (pd.DataFrame): The feature matrix.
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
        pruning_cfg (dict[str, Any] | None): Early stopping settings.
        encoding (str): Categorical encoding strategy.
        high_cardinality_threshold (int | None): High-cardinality threshold.
        high_cardinality_encoding (str | None): High-cardinality strategy.
        hashing_n_components (int): Number of hashing components.
        target_encoding_smoothing (float): Target encoding smoothing.
        target_encoding_fallback (float | None): Target encoding fallback.
        feature_selection_cfg (FeatureSelectionCfg | dict | None): Feature
            selection configuration.
    """

    phase: HPOPhaseCfg
    prev_results: dict[str, tuple[float, dict[str, Any]]]
    X: pd.DataFrame
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
    pruning_cfg: dict[str, Any] | None
    encoding: str
    high_cardinality_threshold: int | None
    high_cardinality_encoding: str | None
    hashing_n_components: int
    target_encoding_smoothing: float
    target_encoding_fallback: float | None
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None


def _prepare_search_space(
    algo_name: str, user_overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Prepare the hyperparameter search space.

    Merges the system defaults with the user overrides from the
    configuration. A module-level function (not a closure) so that the
    ``_hpo_phase_worker`` worker is serializable in the "processes" mode.

    Args:
        algo_name (str): The algorithm name used to look up defaults.
        user_overrides (dict[str, Any] | None): A dict of parameters to override.

    Returns:
        dict[str, Any]: The resulting search space.
    """
    space = DEFAULT_SPACES.get(algo_name, {}).copy()
    if user_overrides:
        space.update(user_overrides)

    # Adaptive epsilon for SVR: the epsilon search bounds are scaled by the
    # y_train spread at runtime (tuner.resolve_search_space), so the
    # default blind range [1e-3, 1.0] from DEFAULT_SPACES is dropped. If the
    # user set epsilon explicitly — it is already in space
    # (space.update(user_overrides) above), and the user's priority is
    # preserved: the adaptive logic does not interfere.
    if algo_name == "svr" and not (user_overrides and "epsilon" in user_overrides):
        space.pop("epsilon", None)

    return space


def _execute_hpo_phase(
    phase_name: str,
    algo: str,
    a_cfg: AlgoCfg,
    n_trials: int,
    spec: _HpoPhaseSpec,
    search_space: dict[str, Any] | None = None,
    initial_params: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]] | None:
    """Run a specific HPO phase for an algorithm (module-level function).

    Provides phase logging and oversampling result processing. The function
    is module-level (not a closure): all phase data is passed through
    ``spec`` so that the ``_hpo_phase_worker`` worker is serializable in
    the "processes" mode.

    Args:
        phase_name (str): The current phase name (e.g. 'coarse').
        algo (str): The algorithm name.
        a_cfg (AlgoCfg): The algorithm configuration.
        n_trials (int): The number of iterations in this phase.
        spec (_HpoPhaseSpec): Snapshot of the HPO phase parameters.
        search_space (dict[str, Any] | None): The search space for the phase.
        initial_params (dict[str, Any] | None): Warm-start parameters (refine).

    Returns:
        tuple[float, dict[str, Any]] | None: The (score, params) tuple, or
            ``None`` when HPO did not return a valid result (the algorithm
            is excluded from candidates).

    Raises:
        Exception: Exceptions from ``_run_hpo`` are logged and re-raised
            (including ``InvalidAlgorithmError``).
    """
    _LOG.info(f"=== {phase_name} phase: {algo} ({n_trials} tries) ===")

    try:
        result = _run_hpo(
            algo_name=algo,
            algo_cfg=a_cfg,
            X=spec.X,
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
            initial_params=initial_params,
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
        )

        if result is None:
            return None

        score, params = result

        # Log the value in user semantics: for neg_*-metrics (e.g.
        # neg_root_mean_squared_error) the raw scorer value is inverted, so
        # the natural (positive) value is shown to the user. The direction
        # accompanies the value explicitly ("min better" for errors): the
        # optimizer's greater_is_better flag is not used.
        disp = to_user_value(spec.metric_sklearn, score)
        _LOG.info(
            f"{phase_name} {algo:15} | score {disp:.5f} "
            f"({direction_label(spec.metric_sklearn)}) | params {params}"
        )
        return score, params
    except Exception as err:
        _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
        raise


def _hpo_phase_worker(
    algo_name: str,
    algo_cfg: AlgoCfg,
    spec: _HpoPhaseSpec,
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

    Returns:
        _HPOWorkerResult: The task result: ``(score, params)`` on success,
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
    init_params = None
    if spec.phase.action == "refine" and algo_name in spec.prev_results:
        _, prev_params = spec.prev_results[algo_name]
        init_params = prev_params

    # 1. Take the system defaults and overlay what is in AlgoCfg (from YAML/JSON)
    full_search_space = _prepare_search_space(
        algo_name,
        algo_cfg.hyperparameters,  # this is the dict from config_parser
    )
    try:
        result = _execute_hpo_phase(
            spec.phase.name,
            algo_name,
            algo_cfg,
            spec.phase.n_trials,
            spec,
            full_search_space,
            initial_params=init_params,
        )

        if result is None:
            return _HPOWorkerResult(name=algo_name)

        score, params = result

        return _HPOWorkerResult(name=algo_name, score=score, params=params)
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
    except Exception as e:  # noqa: BLE001
        _LOG.warning(f"Algorithm {algo_name} failed in phase {spec.phase.name}: {e}")
        return _HPOWorkerResult(name=algo_name)


def _consume_worker_results(
    results: Iterable[_HPOWorkerResult | None],
    phase_results: dict[str, tuple[float, dict[str, Any]]],
    ctx: PhaseContext,
) -> None:
    """Distribute worker results into the working set and disqualifications.

    In the "processes" mode the worker cannot write to ``ctx`` directly
    (changes in the child process would be lost), so disqualifications are
    returned in the results and applied here, in the main process.
    ``None`` results (a task timeout/failure in the pool) are ignored.

    Args:
        results (Iterable[_HPOWorkerResult | None]): Worker results.
        phase_results (dict[str, tuple[float, dict[str, Any]]]): The working
            set of the current phase — valid results land here.
        ctx (PhaseContext): The phase context — disqualifications merge here.
    """
    for res in results:
        if res is None:
            continue
        if res.disqualify_reason is not None:
            ctx.disqualified_algorithms[res.name] = res.disqualify_reason
            continue
        if res.score is None or res.params is None:
            continue
        phase_results[res.name] = (res.score, res.params)


def execute_phases(
    cfg: Config,
    prepared: PreparedData,
) -> PhaseContext:
    """Run all pipeline phases and return the final phase context.

    Pipeline step 3. Contains the phase loop, parallel HPO execution,
    synchronous select steps, circuit breaker disqualification, and filtering
    of invalid results. The HPO worker ``_hpo_phase_worker`` and the helpers
    ``_prepare_search_space`` / ``_execute_hpo_phase`` are module-level
    functions: all phase data is passed through ``_HpoPhaseSpec`` so that the
    worker is serializable in the "processes" mode.

    Multi-phase semantics: the working set is not accumulated — each HPO phase
    rebuilds it from scratch, so an algorithm that failed in the current phase
    (complete HPO failure: ``None`` or an invalid score) is fully excluded and
    does not keep a stale record from a previous phase. The select step output
    replaces the pipeline working set; the final winner is determined by the
    terminal select step.

    Refine step: ``action='refine'`` performs ``n_trials`` warm-started HPO
    attempts for **every** candidate of the incoming working set (the results
    of the last HPO phase, or the output of the previous select step). Only
    ``params`` of the incoming records are used (they become ``initial_params``
    for ``enqueue_trial`` in the tuner); incoming scores are ignored. The
    output keeps the same set of names with updated ``(score', params')``
    records and replaces the working set. There is no narrowing in refine —
    narrowing is the job of explicit select steps. As the first phase, refine
    runs over the full set of enabled algorithms without warm-start. A
    per-algorithm failure excludes only that algorithm (circuit breaker /
    invalid-result filters); if refine wipes out the whole working set,
    ``RuntimeError`` is raised — a silent fallback to pre-refine parameters
    would hide the regression (deliberate design decision).

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
        NotImplementedError: For ``refine`` phases — execution is not yet
            implemented.
        RuntimeError: If an HPO phase produced no valid score, a select step
            received an empty working set, or no finalist was trained on the
            full dataset.
    """
    X = prepared.X
    y = prepared.y
    metric_sklearn = prepared.metric_sklearn
    resolved_validation = prepared.resolved_validation
    resolved_n_folds = prepared.resolved_n_folds
    resolved_test_size = prepared.resolved_test_size
    categorical_features = prepared.categorical_features
    numerical_features = prepared.numerical_features

    ctx = PhaseContext()

    # Initial candidate list (all enabled algorithms)
    all_algorithms = _algorithms_as_dict(cfg.algorithms)
    # Disqualified algorithms (circuit breaker): name -> reason.
    # Disqualification does not abort the run: the algorithm is excluded from
    # candidates and results, the rest continue training.
    # living in ctx: populated by HPO failures and select steps.
    for phase in cfg.general.phases:
        if phase.action == "select":
            # The select step runs synchronously (outside run_parallel): it
            # operates on already-finished HPO results.
            _execute_select_phase(phase=phase, cfg=cfg, prepared=prepared, ctx=ctx)
            continue

        _LOG.info(
            f"--- Starting Phase: {phase.name} ({phase.n_trials}"
            f" trials, action: {phase.action}) ---"
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

        if phase.action == "refine":
            # Refine: warm-start HPO for the ENTIRE incoming working set. The
            # input is the last HPO phase results or the output of the previous
            # select step; only params are used (initial_params for
            # enqueue_trial), incoming scores are ignored. There is no
            # narrowing: no winners, top-K, or corridors — narrowing is done by
            # explicit select steps. The output replaces the working set: the
            # same names with updated (score', params').
            if prev_results:
                current_candidates = {
                    n: all_algorithms[n]
                    for n in prev_results
                    if n in all_algorithms and n not in ctx.disqualified_algorithms
                }
            else:
                # refine as the first phase: the incoming working set is the
                # full set of enabled algorithms, no warm-start is passed
                # (there are no previous results).
                current_candidates = {
                    n: a
                    for n, a in all_algorithms.items()
                    if a.enable and n not in ctx.disqualified_algorithms
                }
        else:
            # all_algorithms phase: every enabled algorithm, results from
            # previous phases are not accumulated. Algorithms disqualified by
            # the circuit breaker do not return to the phase.
            current_candidates = {
                n: a
                for n, a in all_algorithms.items()
                if a.enable and n not in ctx.disqualified_algorithms
            }

        # Each phase starts with an empty working set: it is rebuilt from the
        # current phase only, so an algorithm that fails in this phase does not
        # keep a stale record from a previous phase.
        phase_results: dict[str, tuple[float, dict[str, Any]]] = {}

        # Candidates participating in the current phase (needed for the
        # diagnostics in the "no valid results" message)
        phase_candidates = list(current_candidates.keys())

        # An immutable snapshot of the phase parameters for the worker: the
        # worker is a module-level function and all phase data is passed to it
        # through ``spec`` explicitly so that the worker is serializable in
        # the "processes" mode (a closure inside ``execute_phases`` cannot be
        # serialized).
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
        spec = _HpoPhaseSpec(
            phase=phase,
            prev_results=prev_results,
            X=X,
            y=y,
            metric_sklearn=metric_sklearn,
            resolved_validation=resolved_validation,
            resolved_n_folds=resolved_n_folds,
            resolved_test_size=resolved_test_size,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            oversampling_enabled=ovr.enable,
            oversampling_multiplier=ovr.multiplier,
            oversampling_algorithm=ovr.algorithm.value,
            pruning_cfg=pruning_cfg,
            encoding=cfg.general.categorical_encoding,
            high_cardinality_threshold=cfg.general.high_cardinality_threshold,
            high_cardinality_encoding=cfg.general.high_cardinality_encoding,
            hashing_n_components=cfg.general.hashing_n_components,
            target_encoding_smoothing=cfg.general.target_encoding_smoothing,
            target_encoding_fallback=cfg.general.target_encoding_fallback,
            feature_selection_cfg=cfg.general.feature_selection,
        )

        # Execution (parallel or sequential)
        if (
            cfg.general.parallel_strategy == "algorithms"
            and len(current_candidates) > 1
        ):
            results = run_parallel(
                _hpo_phase_worker,
                args_seq=[(n, a, spec) for n, a in current_candidates.items()],
                max_workers=cfg.general.max_workers,
                mode=cfg.general.parallel_mode,
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
            _consume_worker_results(results, phase_results, ctx)
        else:
            for n, a in current_candidates.items():
                res = _hpo_phase_worker(n, a, spec)
                _consume_worker_results([res], phase_results, ctx)

        # Disqualified algorithms (circuit breaker) are removed from the results and
        # from the candidates of the following phases: they must not win and must
        # not waste compute again.
        # working_set.pop is a safety net for disqualification in the same
        # phase (a no-op for the already-rebuilt dict).
        for name in list(ctx.disqualified_algorithms):
            phase_results.pop(name, None)
            current_candidates.pop(name, None)

        # Filter out invalid phase results: None/NaN/±inf and the worst-score
        # sentinel class. The sentinel class is defined as any score at or below
        # the exact float32 minimum (WORST_SCORE_THRESHOLD): it covers both the
        # HPO_WORST_SCORE constant (returned by the tuner when a trial metric is
        # non-finite) and the raw float(np.finfo(np.float32).min) value (possible
        # from numpy-cast metrics or custom tuners), which the old
        # math.isclose(rel_tol=1e-9) check missed (relative difference ≈ 9.88e-9).
        # Real metrics of such magnitude are practically impossible, so the
        # threshold cannot drop a legitimate winner. params must be a dict (the
        # tuner failure signal is params=None); garbage score types from custom
        # tuners (None, strings, etc.) are rejected by is_valid_winner_score.
        valid_results: dict[str, tuple[float, dict[str, Any]]] = {}
        for name, (score, params) in phase_results.items():
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
            valid_results[name] = (score, params)

        # Keep only the valid results of the current phase in the working set:
        # entries with an invalid score (HPO_WORST_SCORE etc.) are removed.
        # Since the phase results started empty, the working set is guaranteed
        # to contain only this phase's results.
        ctx.working_set = valid_results

        if not valid_results:
            failed_algos = [n for n in phase_candidates if n not in valid_results]
            raise RuntimeError(
                f"No algorithms produced valid scores in phase '{phase.name}'. "
                f"Failed algorithms: {failed_algos}"
            )
    return ctx


# --------------------------------------------------------------------------- #
#  Select steps                                                                #
#  Explicit selection steps: strategy cv_best (ranking by CV score without    #
#  audit) and sanity_gate (finalists pool → training on 100% of the data →   #
#  audit → ranking by RMSE_oof). The select step output replaces the          #
#  pipeline working set.                                                       #
# --------------------------------------------------------------------------- #
def _rank_cv_best(
    working_set: dict[str, tuple[float, dict[str, Any]]],
) -> list[str]:
    """Rank the working set by the raw CV score in descending order.

    The optimizer semantics are the same as in ``select_winner``/
    ``select_finalists``: ties are resolved by insertion order (the
    configuration order, a documented tie-break).

    Args:
        working_set (dict[str, tuple[float, dict[str, Any]]]): The working set
            "algorithm → (score, params)".

    Returns:
        list[str]: Algorithm names, best CV score first.
    """
    return sorted(working_set, key=lambda n: working_set[n][0], reverse=True)


def _rank_sanity_candidates(
    result: RobustWinnerResult,
    candidate_models: Mapping[str, FinalistCandidate],
) -> list[str]:
    """Deterministic ranking of the ``sanity_gate`` select step.

    The strategy's ranking metric is ``RMSE_oof``: candidates that passed the
    audit are sorted by ascending ``RMSE_oof`` (None — last, ties — pool
    order), followed by the rejected ones per the fallback criterion. The
    first element of the ranking always matches ``result.winner_algo``.

    Args:
        result (RobustWinnerResult): The result of ``select_robust_winner``.
        candidate_models (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        list[str]: Candidate names in the strategy ranking order.
    """

    def _rmse_key(name: str) -> tuple[bool, float]:
        rmse = candidate_models[name].rmse_oof
        return (rmse is None, rmse if rmse is not None else float("inf"))

    valid = [n for n in result.pool if result.audit[n].is_valid]
    invalid = [n for n in result.pool if not result.audit[n].is_valid]
    return sorted(valid, key=_rmse_key) + sorted(
        invalid, key=lambda n: _fallback_sort_key(n, result.audit, candidate_models)
    )


def _serialize_audit_result(
    audit: SanityCheckResult, generalization_gap_mode: str
) -> dict[str, Any]:
    """Serialize a ``SanityCheckResult`` into a JSON-compatible dict.

    NaN statistics of disabled/non-computable circuits are replaced with
        ``None``. An infinite ``generalization_gap`` (division by zero, circuit D)
        is serialized with an explicit ``generalization_gap_infinite`` flag: in
        the ``warn`` mode an infinite gap is a soft warning, and without the flag
        the reader could not tell it apart from "circuit disabled or not
        computable". The applied circuit D mode (``generalization_gap_mode``) is
        included in the serialization of each candidate: in warn mode the reader
        sees next to a Circuit D warning that the violation did NOT disqualify
        the model, and in reject mode — that it became a reason.

        Args:
            audit (SanityCheckResult): The audit result of a single finalist.
            generalization_gap_mode (str): The applied circuit D mode:
                'warn' | 'reject'.

        Returns:
            dict[str, Any]: A dict for the select step's report section.
    """

    def _num(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "is_valid": bool(audit.is_valid),
        "reasons": list(audit.reasons),
        "soft_warnings": list(audit.soft_warnings),
        "diversity_ratio": _num(audit.diversity_ratio),
        "unique_ratio": _num(audit.unique_ratio),
        "dead_features_count": int(audit.dead_features_count),
        "generalization_gap": _num(audit.generalization_gap),
        "generalization_gap_infinite": bool(audit.generalization_gap == float("inf")),
        "generalization_gap_mode": generalization_gap_mode,
        "severity": int(audit.severity),
    }


def _build_sanity_gate(sg_cfg: SanityGateCfg) -> ModelSanityGate:
    """Build a ``ModelSanityGate`` from the ``general.sanity_gate`` section.

    The config section is the source of thresholds: all circuit thresholds and
    enable flags are passed to the gate constructor unchanged, including the
    circuit D mode ``generalization_gap_mode``: ``"warn"`` (default) —
    a gap violation (finite or infinite) goes to ``soft_warnings``;
    ``"reject"`` — strict disqualification (the old behavior). The gate's
    ``warn_only`` mode is not used: the ``sanity_gate`` select step applies
    disqualifications.

    Args:
        sg_cfg (SanityGateCfg): The ``general.sanity_gate`` section.

    Returns:
        ModelSanityGate: The configured gate instance.
    """
    return ModelSanityGate(
        min_prediction_diversity=sg_cfg.min_prediction_diversity,
        adaptive_diversity=sg_cfg.adaptive_diversity,
        min_unique_count=sg_cfg.min_unique_count,
        min_unique_ratio=sg_cfg.min_unique_ratio,
        max_dead_feature_ratio=sg_cfg.max_dead_feature_ratio,
        dead_features_require_low_diversity=sg_cfg.dead_features_require_low_diversity,
        zero_coef_tolerance=sg_cfg.zero_coef_tolerance,
        max_generalization_gap=sg_cfg.max_generalization_gap,
        generalization_gap_mode=sg_cfg.generalization_gap_mode,
        check_permutation_sensitivity=sg_cfg.check_permutation_sensitivity,
        permutation_max_rows=sg_cfg.permutation_max_rows,
        permutation_max_features=sg_cfg.permutation_max_features,
        permutation_repeats=sg_cfg.permutation_repeats,
        permutation_seed=sg_cfg.permutation_seed,
        permutation_tolerance=sg_cfg.permutation_tolerance,
        check_diversity=sg_cfg.check_diversity,
        check_unique=sg_cfg.check_unique,
        check_dead_features=sg_cfg.check_dead_features,
        check_generalization_gap=sg_cfg.check_generalization_gap,
    )


def _build_finalists_pool(
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> list[str]:
    """Build the finalists pool from the current working set.

    The thresholds (corridor δ, ``top_k_candidates``, family top-up) come from
    ``general.sanity_gate``; algorithms disqualified by the circuit breaker
    (HPO failures and previous select steps) do not enter the pool.

    Args:
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        ctx (PhaseContext): The pipeline phase context.

    Returns:
        list[str]: The ordered finalists pool (best first).
    """
    sg_cfg = cfg.general.sanity_gate
    return select_finalists(
        ctx.working_set,
        metric_user=prepared.metric_user,
        top_k_candidates=sg_cfg.top_k_candidates,
        corridor_delta=sg_cfg.corridor_delta,
        corridor_mode=sg_cfg.corridor_mode,
        enforce_family_diversity=sg_cfg.enforce_family_diversity,
        algorithm_families=sg_cfg.algorithm_families,
        allowed_families=(
            set(sg_cfg.allowed_families)
            if sg_cfg.allowed_families is not None
            else None
        ),
        family_diversity_multiplier=sg_cfg.family_diversity_multiplier,
        disqualified=tuple(ctx.disqualified_algorithms),
    )


def _execute_select_cv_best(
    phase: HPOPhaseCfg,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> None:
    """Handle a select step with the ``cv_best`` strategy.

    Ranks the incoming set by the raw CV score (the equivalent of the legacy
    path with ``sanity_gate.mode: off``): no audit, no gate report section is
    created. Output forms:

    - ``winner`` — the single winner (terminal step);
    - ``top_k`` — the first ``k`` candidates of the ranking;
    - ``pool`` — the corridor-based finalists pool (corridor δ ∩
      ``top_k_candidates``, thresholds from ``general.sanity_gate``), ranked by
      CV score, without a ``ModelSanityGate`` audit.

    The output replaces the pipeline working set.

    Args:
        phase (HPOPhaseCfg): The select phase configuration.
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        ctx (PhaseContext): The pipeline phase context (mutated).
    """
    if phase.output == "winner":
        winner_algo = select_winner(ctx.working_set)
        ctx.working_set = {winner_algo: ctx.working_set[winner_algo]}
        ctx.winner_algo = winner_algo
        ctx.pre_trained_trainer = None
        # Consistent user semantics: like the HPO phase log, the select step
        # prints the natural metric value (for errors — positive RMSE/MAE),
        # not the raw inverted optimizer scorer value.
        _LOG.info(
            "Select phase '%s' (strategy: cv_best): winner %s (score %.5f)",
            phase.name,
            winner_algo,
            to_user_value(prepared.metric_user, ctx.working_set[winner_algo][0]),
        )
        return

    if phase.output == "top_k":
        # The ranking is built lazily: not needed for winner (the source of
        # truth is select_winner), and the pool is built separately.
        kept = _rank_cv_best(ctx.working_set)[: phase.k]
        ctx.working_set = {name: ctx.working_set[name] for name in kept}
        _LOG.info(
            "Select phase '%s' (strategy: cv_best): top_k=%d kept %s",
            phase.name,
            phase.k,
            kept,
        )
        return

    # output == "pool": corridor-based finalists pool by CV score (no audit).
    pool = _build_finalists_pool(cfg, prepared, ctx)
    ctx.working_set = {name: ctx.working_set[name] for name in pool}
    _LOG.info(
        "Select phase '%s' (strategy: cv_best): corridor pool %s", phase.name, pool
    )


def _drop_kept_from_circuit_breaker(
    ctx: PhaseContext, returned_by_fallback: Iterable[str]
) -> None:
    """Un-disqualify candidates returned by the select step output.

    A select step with ``strategy: sanity_gate`` (``output: top_k``/``pool``)
    deliberately returns candidates that did not pass the audit (fallback) to
    the working set — otherwise the pipeline would end up with an empty set.
    Keeping them in ``disqualified_algorithms`` is not allowed: the next
    phase would filter them out and fail (refine — RuntimeError "no valid
    results", the next sanity-select — ``select_finalists``).

    Only the names passed via ``returned_by_fallback`` are removed: failures
    disqualified by this step and returned into its output.
    Pre-existing disqualifications (the HPO circuit breaker and others) are
    not touched — other phases are responsible for them.

    Args:
        ctx (PhaseContext): The phase context (mutated).
        returned_by_fallback (Iterable[str]): Names returned into the step
            output and disqualified by the same step.
    """
    kept = set(returned_by_fallback)
    for name in list(ctx.disqualified_algorithms):
        if name in kept:
            del ctx.disqualified_algorithms[name]


def _execute_select_sanity_gate(
    phase: HPOPhaseCfg,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> None:
    """Handle a select step with the ``sanity_gate`` strategy.

        Reuses the two-stage selection:

        1. Finalists pool (corridor δ ∩ ``top_k_candidates`` from
           ``general.sanity_gate``);
        2. ``_train_finalists`` — train each finalist on 100% of the data exactly
           once (a repeated select after HPO phases trains the finalists anew —
           old full-fit trainers are not reused);
        3. ``ModelSanityGate`` audit;
        4. ``select_robust_winner`` (``active`` mode): rank the candidates that
           passed the audit by ``RMSE_oof``, with a fallback criterion on total
           disqualification.

    Gate disqualifications enter the circuit breaker
        (``disqualified_algorithms``) on par with HPO failures; the exception is
        candidates skipped because of an exhausted ``audit_time_budget_seconds``:
        they were not audited rather than rejected, so they are not disqualified
        forever. A report section is added to ``ctx.sanity_gate_sections`` under
        the phase name. Output forms:

        - ``winner`` — the single winner (terminal step); its full-fit trainer is
          passed to ``persist_artifact`` via ``pre_trained_trainer``;
        - ``top_k`` — the first ``k`` candidates of the ranking (by ``RMSE_oof``);
        - ``pool`` — all candidates that passed the selection (audit); on total
          disqualification — the full ranking per the fallback criterion (the
          pipeline does not end up with an empty set).

        Output invariant: the working set does not intersect
        ``disqualified_algorithms`` — failures disqualified by this step and
        returned by the fallback into the ``top_k``/``pool`` output are removed
        from the circuit breaker (otherwise the next phase would filter them out
        and fail). Pre-existing disqualifications (the HPO circuit breaker and
        others) are not removed.

        Args:
            phase (HPOPhaseCfg): The select phase configuration.
            cfg (Config): The configuration object.
            prepared (PreparedData): The prepared data and settings.
            ctx (PhaseContext): The pipeline phase context (mutated).

        Raises:
            RuntimeError: If no finalist of the pool was trained on 100% of the
                data.
    """
    # 1. Finalists pool.
    pool = _build_finalists_pool(cfg, prepared, ctx)
    _LOG.info(
        "Select phase '%s' (strategy: sanity_gate): finalists pool %s",
        phase.name,
        pool,
    )

    # 2. Train the finalists on 100% of the data (exactly once; a repeated
    # select after HPO phases trains them anew).
    gate_started = time.monotonic()
    candidates = _train_finalists(pool, ctx.working_set, cfg, prepared)
    if not candidates:
        raise RuntimeError(
            f"No finalist could be trained on the full dataset in select "
            f"phase '{phase.name}' (pool: {pool})."
        )
    ctx.full_fit_trainers = {
        name: cand.trainer
        for name, cand in candidates.items()
        if cand.trainer is not None
    }

    # 3-4. Audit and ranking by RMSE_oof (active mode).
    result = select_robust_winner(
        candidates,
        prepared.X,
        prepared.y,
        _build_sanity_gate(cfg.general.sanity_gate),
        mode="active",
        audit_time_budget_seconds=cfg.general.sanity_gate.audit_time_budget_seconds,
    )
    overhead_seconds = time.monotonic() - gate_started

    # Gate disqualifications → circuit breaker (on par with HPO failures): the
    # algorithm will not return to the pools of subsequent select steps.
    # The exception — candidates skipped because of an exhausted
    # audit_time_budget: they are NOT audited rather than rejected, so they
    # must not be disqualified forever from all subsequent phases.
    for name, reasons in result.disqualified.items():
        if _is_audit_budget_skip(reasons):
            _LOG.info(
                "[sanity_gate] Candidate %s was not audited (audit time "
                "budget exceeded); not permanently disqualified",
                name,
            )
            continue
        ctx.disqualified_algorithms[name] = "; ".join(reasons)

    # Failures disqualified by THIS step exactly (without budget skips).
    # Only their output can return to the working set, removing them from the
    # circuit breaker; pre-existing disqualifications are not touched.
    disqualified_this_step = {
        name
        for name, reasons in result.disqualified.items()
        if not _is_audit_budget_skip(reasons)
    }

    ranked = _rank_sanity_candidates(result, candidates)
    _LOG.info(
        "[sanity_gate] Winner %s audit summary: diversity_ratio=%s, "
        "unique_ratio=%s, dead_features_count=%s, generalization_gap=%s",
        result.winner_algo,
        _fmt_audit_value(result.audit[result.winner_algo].diversity_ratio),
        _fmt_audit_value(result.audit[result.winner_algo].unique_ratio),
        result.audit[result.winner_algo].dead_features_count,
        _fmt_audit_value(result.audit[result.winner_algo].generalization_gap),
    )

    # Report section: the key is the select phase name. The applied
    # circuit D mode is included in the report together with the audit so
    # that the reader sees that a Circuit D warning did not disqualify the
    # model.
    generalization_gap_mode = cfg.general.sanity_gate.generalization_gap_mode
    ctx.sanity_gate_sections[phase.name] = {
        "strategy": "sanity_gate",
        "generalization_gap_mode": generalization_gap_mode,
        "pool": list(result.pool),
        "audit": {
            name: _serialize_audit_result(res, generalization_gap_mode)
            for name, res in result.audit.items()
        },
        "disqualified": {
            name: list(reasons) for name, reasons in result.disqualified.items()
        },
        "fallback_used": result.fallback_used,
        "winner_rmse_oof": result.winner_rmse_oof,
        "overhead_seconds": overhead_seconds,
    }

    # Output form: replace the working set.
    if phase.output == "winner":
        winner_algo = result.winner_algo
        ctx.working_set = {winner_algo: ctx.working_set[winner_algo]}
        ctx.winner_algo = winner_algo
        ctx.pre_trained_trainer = candidates[winner_algo].trainer
        # Only the winner's trainer is needed for the artifact: the other
        # finalists' trainers are not kept in memory until the end of
        # train_best_model.
        ctx.full_fit_trainers = {
            winner_algo: candidates[winner_algo].trainer,
        }
        return

    if phase.output == "top_k":
        kept = ranked[: phase.k]
        ctx.working_set = {name: ctx.working_set[name] for name in kept}
        ctx.full_fit_trainers = {
            name: tr for name, tr in ctx.full_fit_trainers.items() if name in kept
        }
        # Output invariant: the returned fallback candidates are removed
        # from the circuit breaker, otherwise the next phase would filter them
        # out and fail with RuntimeError. Only this step's failures are removed.
        _drop_kept_from_circuit_breaker(ctx, set(kept) & disqualified_this_step)
        _LOG.info(
            "Select phase '%s' (strategy: sanity_gate): top_k=%d kept %s",
            phase.name,
            phase.k,
            kept,
        )
        return

    # output == "pool": all candidates that passed the selection; on total
    # disqualification — the fallback ranking (otherwise the pipeline would
    # end up with an empty set).
    passed = [n for n in result.pool if result.audit[n].is_valid]
    if not passed:
        passed = ranked
        _LOG.warning(
            "Select phase '%s' (strategy: sanity_gate): all %d finalist(s) "
            "failed the audit; output 'pool' falls back to the full ranked "
            "pool %s (fallback criterion)",
            phase.name,
            len(result.pool),
            passed,
        )
    ctx.working_set = {name: ctx.working_set[name] for name in passed}
    ctx.full_fit_trainers = {
        name: tr for name, tr in ctx.full_fit_trainers.items() if name in passed
    }
    # Output invariant: see the comment for the top_k branch.
    _drop_kept_from_circuit_breaker(ctx, set(passed) & disqualified_this_step)
    _LOG.info(
        "Select phase '%s' (strategy: sanity_gate): pool of passed candidates %s",
        phase.name,
        passed,
    )


def _execute_select_phase(
    phase: HPOPhaseCfg,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> None:
    """Select-step dispatcher by strategy.

    Select runs synchronously (outside ``run_parallel``). An empty incoming
    set is an error: there is nothing to select from.

    Args:
        phase (HPOPhaseCfg): The select phase configuration.
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        ctx (PhaseContext): The pipeline phase context (mutated).

    Raises:
        RuntimeError: If the incoming working set is empty.
        ValueError: If the phase strategy is unknown (protection; the config
            schema allows only 'cv_best'/'sanity_gate').
    """
    if not ctx.working_set:
        raise RuntimeError(
            f"Select phase '{phase.name}' received an empty working set: "
            "nothing to select from. Check that a preceding HPO phase "
            "produced valid results."
        )
    # Common protection for all strategies: top_k without k makes no sense
    # (the config schema already requires k >= 1; the check is duplicated
    # explicitly).
    if phase.output == "top_k" and phase.k is None:
        raise ValueError(
            f"Select phase '{phase.name}': output 'top_k' requires "
            "the integer parameter k (>= 1)"
        )
    _LOG.info(
        f"--- Select Phase: {phase.name} (strategy: {phase.strategy}, "
        f"output: {phase.output}) on {len(ctx.working_set)} candidate(s) ---"
    )
    if phase.strategy == "cv_best":
        _execute_select_cv_best(phase, cfg, prepared, ctx)
    elif phase.strategy == "sanity_gate":
        _execute_select_sanity_gate(phase, cfg, prepared, ctx)
    else:  # pragma: no cover — protected by the config schema
        raise ValueError(
            f"Select phase '{phase.name}': unknown strategy {phase.strategy!r}"
        )


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
) -> dict[str, Any]:
    """Final training of the winner, artifact persistence, and report assembly.

    Pipeline step 5. Trains the winning algorithm's model on the full dataset
    via ``_fit_and_save`` (or reuses the already-trained finalist of the
    terminal ``sanity_gate`` select step), saves the artifact to disk, and
    builds the result dict: ``algorithm``, ``score`` (in user semantics),
    ``metric``, ``params``, ``model_path``; when present it also adds
    ``disqualified_algorithms`` (the circuit breaker, including Sanity Gate
    select-step disqualifications), ``additional_metrics``, and ``sanity_gate``
    — a dict of the report sections of the ``sanity_gate`` select steps (key —
    the phase name).

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
            is saved from it (finalists are trained once).
        sanity_gate_sections (Dict[str, Dict[str, Any]] | None): Report
            sections of the ``sanity_gate`` select steps (key — the phase
            name). The ``sanity_gate`` key is added to the result only when at
            least one section is present.

    Returns:
        Dict[str, Any]: A dict with the training results.

    Raises:
        Exception: The final training/saving error is propagated (logged
            before being re-raised).
    """
    model_path = Path(model_path_override or cfg.general.path_to_model)
    winner_cfg = _algorithms_as_dict(cfg.algorithms)[winner_algo]

    try:
        if pre_trained_trainer is not None:
            trainer = pre_trained_trainer
            model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(model_path)
        else:
            trainer = _fit_and_save(
                winner_algo,
                winner_cfg,
                prepared.X,
                prepared.y,
                final_params,
                model_path,
                cfg,
                metric_name_sklearn=prepared.metric_sklearn,
                validation_strategy=prepared.resolved_validation,
                n_folds=prepared.resolved_n_folds,
                test_size=prepared.resolved_test_size,
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
    }
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
    # Additional informational metrics of the final model. The key is added
    # only when metrics are present (after deduplication and exclusion of the
    # main comparison metric), otherwise the results are identical to the
    # previous behavior.
    if cfg.general.additional_metrics:
        result["additional_metrics"] = dict(trainer.additional_scores)
    return result


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
    of the winner is required. There is no global
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
    final_score, final_params = ctx.working_set[winner_algo]
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
    )

"""Building the experimental top-K ensemble (Training Engine, T10).

The step runs after the terminal select step, when
``general.ensemble.enable`` is true: the top-K algorithms of the run's
leaderboard (``PhaseContext.leaderboard_snapshot`` — the full last-HPO
phase leaderboard, the same records the result leaderboard is built
from) are retrained on 100% of the data exactly once each
(``finalists._fit_finalist`` — the shared assembly point), their
validation scores in user semantics are converted into normalized
weights (``experiment.ensemble.compute_weights`` — direction-aware) and
the predictions are averaged (``experiment.ensemble.EnsembleModel``).

The module imports the pure ensemble core from the ``experiment``
package (the building-block layer) and the finalists trainer from the
engine; it must not import ``component`` or ``select_steps``. The
ensemble is an ADDITIONAL experimental output: the winner, the winner
artifact and the terminal selection are not affected. A failure of the
ensemble step never aborts the run: the reason is logged and the result
simply carries no ``ensemble`` section.
"""

from __future__ import annotations

import logging
from typing import Any

from configurable_automl_engine.common.metrics import to_user_value, user_direction
from configurable_automl_engine.experiment.ensemble import (
    EnsembleMember,
    EnsembleModel,
    compute_weights,
    select_top_k,
)

from .finalists import _fit_finalist
from .hpo_phases import _algorithms_as_dict
from .state import PhaseContext, PreparedData

_LOG = logging.getLogger(__name__)

__all__ = ["build_ensemble_from_context", "top_k_from_leaderboard"]


def top_k_from_leaderboard(
    ctx: PhaseContext,
    metric_user: str,
    k: int,
) -> tuple[list[str], dict[str, float]]:
    """Select the top-K leaderboard members and their user-semantic scores.

    Reuses the documented select-step top-K semantics
    (``experiment.ensemble.select_top_k``) over the user-semantic scores
    of the leaderboard snapshot: raw optimizer scores are converted with
    ``to_user_value`` (positive RMSE/MAE, plain R²), so the direction
    handling lives in one place. The winner of the terminal select step
    is always kept in the selection: the ensemble must include the best
    model even when the snapshot ordering and the terminal select disagree
    on ties.

    Args:
        ctx: The final phase context (the leaderboard snapshot and the
            terminal winner are read).
        metric_user: The user-facing comparison metric name.
        k: The requested ensemble size (≥ 1).

    Returns:
        tuple[list[str], dict[str, float]]: The selected member names in
            rank order (best first) and their user-semantic scores.

    Raises:
        ValueError: The leaderboard snapshot is empty (no valid scores —
            "no trained models" for the ensemble) or a score fails the
            winner-score invariant.
    """
    if not ctx.leaderboard_snapshot:
        raise ValueError(
            "Cannot build the ensemble: no trained models available "
            "(no algorithm produced a valid score in the run)"
        )
    user_scores = {
        name: to_user_value(metric_user, record.score)
        for name, record in ctx.leaderboard_snapshot.items()
    }
    selected = select_top_k(user_scores, k, direction=metric_user)
    if ctx.winner_algo and ctx.winner_algo not in selected:
        # K=1 keeps the single best member; otherwise the terminal winner
        # is appended as the last member (its rank follows the selection).
        if len(selected) == 1:
            selected = [ctx.winner_algo]
        else:
            selected.append(ctx.winner_algo)
    return selected, user_scores


def build_ensemble_from_context(
    *,
    cfg: Any,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> tuple[EnsembleModel, dict[str, Any]] | None:
    """Build the weighted-average ensemble of the top-K leaderboard models.

    The step is best-effort: any failure (a member failed to retrain, no
    trained models, an invalid score) is logged as a WARNING and the run
    continues without the ensemble section — the ensemble is an
    experimental additional output and must never mask the winner result.

    Args:
        cfg: The resolved root config (``Config``); the
            ``general.ensemble`` section drives the size and the weight
            method.
        prepared: The prepared data and resolved settings (X/y are the
            data the members are refit on).
        ctx: The final phase context (the leaderboard snapshot, the
            terminal winner).

    Returns:
        tuple[EnsembleModel, dict[str, Any]] | None: The built ensemble
            and its report section (without the artifact path — the
            caller saves the model and adds it), or ``None`` when the
            ensemble could not be built (the failure is logged).
    """
    ensemble_cfg = cfg.general.ensemble
    try:
        selected, user_scores = top_k_from_leaderboard(
            ctx, prepared.metric_user, ensemble_cfg.k
        )
    except (ValueError, RuntimeError) as err:
        _LOG.warning("Ensemble step skipped: %s", err)
        return None
    _LOG.info(
        "Ensemble: top-K selection (k=%d, weight_method=%s) selected %s",
        ensemble_cfg.k,
        ensemble_cfg.weight_method,
        selected,
    )

    # Weights from the validation scores (direction-aware; the selected
    # names are guaranteed to be in user_scores).
    try:
        weights = compute_weights(
            {name: user_scores[name] for name in selected},
            direction=prepared.metric_user,
            method=ensemble_cfg.weight_method,
        )
    except (ValueError, RuntimeError) as err:
        _LOG.warning("Ensemble step skipped: %s", err)
        return None

    # Retrain every member on 100% of the data exactly once (the shared
    # finalist assembly point; a member that fails to retrain is fatal for
    # the ensemble — a silently degraded ensemble is not allowed — and the
    # whole step degrades to "no ensemble" with a warning).
    algos = _algorithms_as_dict(cfg.algorithms)
    predictors: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for name in selected:
        if name not in ctx.leaderboard_snapshot:
            _LOG.warning("Ensemble member %s has no leaderboard record", name)
            return None
        record = ctx.leaderboard_snapshot[name]
        if name not in algos:
            _LOG.warning(
                "Ensemble member %s is missing from the algorithms config", name
            )
            return None
        try:
            trainer = _fit_finalist(
                name,
                algos[name],
                prepared.X,
                prepared.y,
                record.params,
                cfg,
                prepared.metric_sklearn,
                validation_strategy=prepared.resolved_validation,
                n_folds=prepared.resolved_n_folds,
                test_size=prepared.resolved_test_size,
                random_state=cfg.general.random_state,
                datetime_expansion=prepared.datetime_expansion,
            )
        except Exception as err:  # noqa: BLE001 — the ensemble step is best-effort
            _LOG.warning(
                "Ensemble member %s failed to retrain on the full dataset: %s; "
                "the ensemble step is skipped",
                name,
                err,
            )
            return None
        predictors[name] = trainer
        scores[name] = user_scores[name]

    members = [
        EnsembleMember(
            name=name,
            score=scores[name],
            weight=weights[name],
            rank=idx,
        )
        for idx, name in enumerate(selected, start=1)
    ]
    try:
        ensemble = EnsembleModel(
            metric_user=prepared.metric_user,
            metric_direction=user_direction(prepared.metric_user),
            members=members,
            predictors=predictors,
        )
    except (ValueError, TypeError) as err:
        _LOG.warning("Ensemble step skipped: %s", err)
        return None
    section = ensemble.report_section(
        k_requested=ensemble_cfg.k, weight_method=ensemble_cfg.weight_method
    )
    _LOG.info(
        "Ensemble built: %d member(s), weights %s",
        len(ensemble),
        {m.name: round(m.weight, 6) for m in ensemble.members},
    )
    return ensemble, section

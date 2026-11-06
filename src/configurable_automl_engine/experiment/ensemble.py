"""Ensemble: weighted averaging of the top-K leaderboard models (the T10
component).

The module is the pure, engine-independent core of the experimental
ensembling branch of the experiment epic:

* :func:`select_top_k` reuses the documented select-step top-K semantics
  (the best ``k`` candidates of the ranking, ties by configuration order)
  over user-semantic scores — a new selection mechanism is NOT invented;
* :func:`compute_weights` converts validation scores into ensemble
  weights, respecting the metric direction in user semantics
  (``minimize`` for RMSE/MAE — smaller is better, ``maximize`` for R² —
  larger is better); every score must pass the winner-score invariant
  (finite and above the worst-score sentinel threshold), so invalid or
  sentinel-class scores are rejected before they can poison the weights;
* :class:`EnsembleModel` averages the predictions of the top-K members
  with the computed weights, saves/loads itself through
  ``common.serialization_utils`` (pickle/joblib round-trip) and renders
  the JSON-safe report section consumed by ``ExperimentResult.ensemble``.

Documented contract and limitations:

* regression only today: the members must predict continuous targets;
  classification support stays future work (the engine's model factory
  is regression-only as well);
* members trained on the SAME target (the engine trains every member on
  the full training data) share the prediction scale, so plain weighted
  averaging is well-defined; predictions on different scales are NOT
  normalized here — that is a documented limitation, and per-member
  standardization is left as a future extension;
* simple stacking (a meta-model on out-of-fold predictions, the raw
  material ``OptimizeResult.oof_predictions`` /
  ``PhaseContext.oof_pool`` already exists in the engine with
  ``collect_oof=True``) is explicitly OUT of the MVP scope and left as a
  future extension of this module;
* the model holds live references to the member predictors (one fitted
  pipeline per member): keeping K members in memory simultaneously is
  the documented memory cost of ensembling;
* a missing predictor for a selected member is an error (``ValueError``):
  a silently degraded ensemble would report numbers it cannot produce.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from configurable_automl_engine.common.definitions import (
    WORST_SCORE_THRESHOLD,
    MetricDirection,
    SerializationFormat,
)
from configurable_automl_engine.common.metrics import user_direction
from configurable_automl_engine.common.serialization_utils import (
    load_artifact,
    resolve_load_formats,
    save_artifact,
)

__all__ = [
    "ENSEMBLE_ARTIFACT_KEY",
    "ENSEMBLE_ARTIFACT_VERSION",
    "EnsembleMember",
    "EnsembleModel",
    "EnsembleModelError",
    "compute_weights",
    "select_top_k",
    "validate_ensemble_score",
]

_LOG = logging.getLogger(__name__)

#: The marker key of a serialized ensemble artifact: loading a file that
#: does not carry it raises a clear error instead of returning garbage.
ENSEMBLE_ARTIFACT_KEY = "__ensemble_model__"

#: The payload schema version of the serialized ensemble artifact.
ENSEMBLE_ARTIFACT_VERSION = 1

#: The weight floor of the inverse transformation: a perfect error score
#: (``0.0``) would make ``1/score`` infinite; the score is floored at
#: this epsilon so the weight stays finite and the normalization keeps
#: the relative ordering of the members.
_MIN_SCORE_EPS = 1e-12


class EnsembleModelError(Exception):
    """Base error of the ensemble module."""


def validate_ensemble_score(score: Any) -> float:
    """Validate one member score and return it as a plain ``float``.

    The same invariant the engine applies to winner scores
    (``is_valid_winner_score``): the value must be a real number, finite,
    and strictly above the worst-score sentinel threshold
    (``WORST_SCORE_THRESHOLD``). ``None``, strings, booleans, non-finite
    values and the sentinel class signal the absence of a real metric and
    must never reach the ensemble weights.

    Args:
        score: The raw score value to validate.

    Returns:
        float: The validated score as a plain ``float``.

    Raises:
        ValueError: The score is not a real number, is non-finite, or
            falls into the worst-score sentinel class.
    """
    if isinstance(score, bool) or not isinstance(
        score, (int, float, np.integer, np.floating)
    ):
        # ValueError (not TypeError): the value flows through the weight /
        # top-K validators, where a ValueError surfaces as a clear
        # "invalid member score" message instead of an unrelated
        # "error converting" text.
        raise ValueError(  # noqa: TRY004
            f"Ensemble member score must be a real number, got "
            f"{type(score).__name__}: {score!r}"
        )
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"Ensemble member score must be finite, got {value!r}")
    if value <= WORST_SCORE_THRESHOLD:
        raise ValueError(
            f"Ensemble member score {value!r} falls into the worst-score "
            f"sentinel class (<= {WORST_SCORE_THRESHOLD!r}) and must never "
            "reach the ensemble weights"
        )
    return value


def _normalize(raw: Mapping[str, float]) -> dict[str, float]:
    """Normalize a positive weight mapping so the weights sum to 1."""
    total = sum(raw.values())
    return {name: weight / total for name, weight in raw.items()}


def _resolve_direction(direction: MetricDirection | str) -> MetricDirection:
    """Resolve a direction argument: an enum value, or a metric name.

    A plain string is accepted in two forms: the direction value itself
    (``"minimize"`` / ``"maximize"``) or the user-facing metric name
    (``"rmse"``, ``"r2"``, ...), resolved through
    ``common.metrics.user_direction`` — so the engine can pass
    ``prepared.metric_user`` without a pre-resolution step.

    Raises:
        ValueError: The string is neither a direction value nor a
            resolvable metric name.
    """
    if isinstance(direction, MetricDirection):
        return direction
    text = str(direction).strip().lower()
    if text == MetricDirection.minimize.value:
        return MetricDirection.minimize
    if text == MetricDirection.maximize.value:
        return MetricDirection.maximize
    try:
        return MetricDirection(user_direction(text))
    except ValueError:
        raise ValueError(
            f"Invalid ensemble metric direction {direction!r}: expected "
            "'minimize'/'maximize' or a resolvable metric name"
        ) from None


def compute_weights(
    scores: Mapping[str, Any],
    direction: MetricDirection | str,
    *,
    method: str = "score_weighted",
) -> dict[str, float]:
    """Convert member validation scores into normalized ensemble weights.

    The conversion respects the metric direction in user semantics:

    * ``minimize`` (RMSE, MAE and every other error metric): the score is
      a natural positive error value, so smaller is better and the weight
      is inversely proportional to the score (``1 / max(score, eps)``);
    * ``maximize`` (R² and similar): larger is better and the weight is
      proportional to the score (negative scores are clipped to zero —
      a negative R² member must not get a negative weight). When every
      clipped score is zero (all members worse than the mean predictor),
      the weights fall back to uniform with a warning: a zero-weight
      ensemble is meaningless.

    The ``"equal"`` method ignores the scores (they are still validated)
    and assigns the uniform weight — the baseline the score-weighted
    variant must beat.

    Args:
        scores: The member scores in user semantics, "name → score".
            Every score must pass :func:`validate_ensemble_score`.
        direction: The metric direction in user semantics (``minimize``
            or ``maximize``; a plain string is accepted).
        method: The conversion method: ``"score_weighted"`` (default,
            direction-aware) or ``"equal"``.

    Returns:
        dict[str, float]: The normalized weights, "name → weight"; the
            weights are positive and sum to 1.

    Raises:
        ValueError: A score is invalid, the method is unknown, the score
            mapping is empty, or (``maximize``) every score is
            non-positive.
    """
    if not scores:
        raise ValueError("Ensemble weights require at least one member score")
    if method not in ("score_weighted", "equal"):
        raise ValueError(
            f"Unknown ensemble weight method {method!r}: "
            "expected 'score_weighted' or 'equal'"
        )
    validated = {name: validate_ensemble_score(score) for name, score in scores.items()}
    if method == "equal":
        uniform = 1.0 / len(validated)
        return {name: uniform for name in validated}
    resolved = _resolve_direction(direction)
    if resolved is MetricDirection.minimize:
        # Inverse weighting on the natural error values: the best (smallest)
        # score gets the largest weight. abs() guards against a negative
        # error value (never expected in user semantics), the epsilon floor
        # keeps a perfect 0.0 score finite.
        return _normalize(
            {
                name: 1.0 / max(abs(value), _MIN_SCORE_EPS)
                for name, value in validated.items()
            }
        )
    proportional = {name: max(value, 0.0) for name, value in validated.items()}
    if all(weight <= 0.0 for weight in proportional.values()):
        warnings.warn(
            "All ensemble member scores are non-positive for the 'maximize' "
            f"metric; falling back to uniform weights (scores: {validated})",
            RuntimeWarning,
            stacklevel=2,
        )
        return _normalize({name: 1.0 for name in validated})
    return _normalize(proportional)


def select_top_k(
    scores: Mapping[str, Any],
    k: int,
    direction: MetricDirection | str,
) -> list[str]:
    """Select the top-K members by score (the select-step top-K semantics).

    Reuses the documented ``output: top_k`` selection semantics over
    user-semantic scores: the candidates are ranked by their score in the
    metric direction (``minimize`` — ascending: the smallest error first;
    ``maximize`` — descending: the largest score first) and the first
    ``k`` names are returned in rank order. Ties keep the mapping
    insertion order (the configuration order of the algorithms — the same
    deterministic tie-break as ``select_winner`` and the leaderboard).

    When ``k`` exceeds the number of available candidates, ``min(k, n)``
    candidates are used and a ``RuntimeWarning`` is emitted (the caller
    sees the effective size both in the warning and in the returned list
    length).

    Args:
        scores: The candidate scores in user semantics, "name → score".
            Every score must pass :func:`validate_ensemble_score`.
        k: The requested ensemble size (≥ 1).
        direction: The metric direction in user semantics.

    Returns:
        list[str]: The selected member names in rank order (best first),
            ``min(k, n)`` of them.

    Raises:
        ValueError: The score mapping is empty, ``k`` is not a positive
            integer, or a score is invalid.
    """
    if not scores:
        raise ValueError("Top-K selection requires at least one candidate score")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"Ensemble size k must be a positive integer, got {k!r}")
    validated = {name: validate_ensemble_score(score) for name, score in scores.items()}
    n = len(validated)
    if k > n:
        warnings.warn(
            f"Ensemble size k={k} exceeds the number of available "
            f"candidates ({n}); using min(k, n) = {n} member(s)",
            RuntimeWarning,
            stacklevel=2,
        )
    resolved = _resolve_direction(direction)
    reverse = resolved is MetricDirection.maximize
    # sorted() is stable: equal scores keep the mapping insertion order
    # (the configuration order — the documented deterministic tie-break).
    ranked = sorted(validated, key=lambda name: validated[name], reverse=reverse)
    return ranked[: min(k, n)]


@dataclass(frozen=True)
class EnsembleMember:
    """One member of an ensemble.

    Attributes:
        name: The member's algorithm name (as configured in ``algorithms``).
        score: The member's validation score in user semantics (positive
            RMSE/MAE, plain R²).
        weight: The member's normalized weight (positive; the weights of
            all members sum to 1).
        rank: The member's 1-based rank in the top-K selection.
    """

    name: str
    score: float
    weight: float
    rank: int


class EnsembleModel:
    """A weighted-average ensemble of the top-K trained models.

    The model holds the live member predictors (any object exposing a
    sklearn-style ``predict(X)`` — in the engine, a fitted
    ``ModelTrainer``) and averages their predictions with the member
    weights: ``prediction = Σᵢ weightᵢ · predictᵢ(X)``. ``K = 1`` is the
    valid boundary case: the ensemble with a single member of weight 1
    predicts exactly like that member.

    Args:
        metric_user: The user-facing comparison metric name the member
            scores are expressed in.
        metric_direction: The metric direction in user semantics
            (``minimize`` for errors, ``maximize`` for score metrics).
        members: The ensemble members in rank order (best first); the
            weights must be positive and are normalized to sum to 1.
        predictors: The fitted member models, "member name → predictor".
            Every member must have a predictor: a missing one is an error,
            not a silent degradation.

    Raises:
        ValueError: No members, a non-positive/non-finite weight, or a
            member without a predictor.
    """

    def __init__(
        self,
        *,
        metric_user: str,
        metric_direction: MetricDirection | str,
        members: Sequence[EnsembleMember],
        predictors: Mapping[str, Any],
    ) -> None:
        if not members:
            raise ValueError(
                "An ensemble requires at least one member "
                "(K=1 — the single-model boundary case — is valid)"
            )
        names = [member.name for member in members]
        if len(set(names)) != len(names):
            raise ValueError(f"Ensemble member names must be unique, got {names}")
        # Per-member weight validation BEFORE the sum check: the sum is
        # insensitive to the sign structure (weights like [2.0, -1.0] sum
        # to 1.0), so a negative or non-finite member weight could slip
        # into the members, the report section and the serialized
        # artifact unnoticed.
        bad = [
            (member.name, member.weight)
            for member in members
            if not math.isfinite(member.weight) or member.weight <= 0.0
        ]
        if bad:
            raise ValueError(
                f"Ensemble member weights must be positive and finite, got {bad}"
            )
        total = sum(member.weight for member in members)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"Ensemble member weights must be positive and finite (sum={total!r})"
            )
        missing = [member.name for member in members if member.name not in predictors]
        if missing:
            raise ValueError(
                "No trained model for ensemble member(s) "
                f"{missing}: a failed member cannot be silently dropped "
                "from the ensemble — build the ensemble from members with "
                "available trained models"
            )
        self.metric_user = metric_user
        self.metric_direction = _resolve_direction(metric_direction)
        # Normalized weights: the caller may pass unnormalized positive
        # weights; the stored members always sum to 1.
        self.members: tuple[EnsembleMember, ...] = tuple(
            EnsembleMember(
                name=member.name,
                score=member.score,
                weight=member.weight / total,
                rank=idx,
            )
            for idx, member in enumerate(members, start=1)
        )
        # Live fitted predictors of the members (the documented memory
        # cost: K models are held simultaneously).
        self.predictors: dict[str, Any] = dict(predictors)

    def __len__(self) -> int:
        """The number of members (the effective ensemble size ``K``)."""
        return len(self.members)

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the weighted average of the member predictions.

        Args:
            X: The feature frame every member predictor accepts (the raw
                input schema — each member applies its own preprocessing).

        Returns:
            np.ndarray: The weighted-average prediction vector, shape
                ``(n_rows,)``.

        Raises:
            ValueError: The members produced differently shaped
                predictions for the same input.
        """
        result: np.ndarray | None = None
        for member in self.members:
            pred = np.asarray(
                self.predictors[member.name].predict(X), dtype=float
            ).reshape(-1)
            if result is None:
                result = member.weight * pred
            elif result.shape != pred.shape:
                raise ValueError(
                    "Ensemble members produced differently shaped predictions "
                    f"for the same input: expected {result.shape[0]} rows, "
                    f"member '{member.name}' produced {pred.shape[0]}"
                )
            else:
                result = result + member.weight * pred
        assert result is not None  # narrow: members is non-empty by invariant
        return result

    # ─────────────────────── report & persistence ─────────────────────── #

    def report_section(
        self,
        *,
        artifact_path: str | Path | None = None,
        k_requested: int | None = None,
        weight_method: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON-safe report section of the ensemble.

        The section is the value of ``ExperimentResult.ensemble``: the
        reporters render it, the JSON reporter serializes it as is, and
        the facade reads ``artifact_path`` from it to reload the model.

        Args:
            artifact_path: The path the ensemble artifact was saved to
                (``None`` when the ensemble was not persisted).
            k_requested: The requested ensemble size (``None`` when the
                caller did not track it; the effective size is
                ``len(self.members)``).
            weight_method: The weight conversion method used
                (``"score_weighted"`` / ``"equal"``).

        Returns:
            dict[str, Any]: The JSON-serializable report section.
        """
        section: dict[str, Any] = {
            "enabled": True,
            "strategy": "weighted_average",
            "weight_method": weight_method,
            "k_requested": k_requested,
            "k_used": len(self.members),
            "metric_user": self.metric_user,
            "metric_direction": self.metric_direction.value,
            "members": [
                {
                    "rank": member.rank,
                    "algorithm": member.name,
                    "score": member.score,
                    "weight": member.weight,
                }
                for member in self.members
            ],
        }
        if artifact_path is not None:
            section["artifact_path"] = str(artifact_path)
        return section

    def save(
        self, path: str | Path, fmt: SerializationFormat = SerializationFormat.pickle
    ) -> Path:
        """Serialize the ensemble (members + weights + live predictors).

        The payload is a single artifact written through
        ``common.serialization_utils.save_artifact``: the member records
        (weights, scores, ranks), the metric semantics and the fitted
        predictors travel together, so :meth:`load` restores a fully
        functional model without the training configuration.

        Args:
            path: The artifact path (the extension is a convention; the
                warning about a mismatching extension comes from the
                serialization utils).
            fmt: The serialization format (pickle by default; the engine
                passes ``general.serialization_format``).

        Returns:
            Path: The path the artifact was written to.
        """
        payload = {
            ENSEMBLE_ARTIFACT_KEY: True,
            "version": ENSEMBLE_ARTIFACT_VERSION,
            "metric_user": self.metric_user,
            "metric_direction": self.metric_direction.value,
            "members": [
                {
                    "name": member.name,
                    "score": member.score,
                    "weight": member.weight,
                    "rank": member.rank,
                }
                for member in self.members
            ],
            "predictors": dict(self.predictors),
        }
        resolved_path = Path(path)
        save_artifact(payload, resolved_path, fmt)
        return resolved_path

    @classmethod
    def load(
        cls, path: str | Path, fmt: SerializationFormat | None = None
    ) -> EnsembleModel:
        """Load a serialized ensemble (the :meth:`save` inverse).

        The format is auto-detected from the file extension with a
        joblib-then-pickle fallback (the same contract as the trainer
        artifacts) unless ``fmt`` is given explicitly.

        Args:
            path: The artifact path.
            fmt: The explicit serialization format; ``None`` —
                auto-detection.

        Returns:
            EnsembleModel: The restored ensemble.

        Raises:
            FileNotFoundError: The artifact file does not exist.
            ValueError: The file is not an ensemble artifact (the marker
                key is missing) or its payload version is unsupported.
        """
        resolved_path = Path(path)
        last_error: Exception | None = None
        for candidate in resolve_load_formats(resolved_path, fmt):
            try:
                payload = load_artifact(resolved_path, candidate)
            except FileNotFoundError:
                raise
            except Exception as err:  # noqa: BLE001 — try the fallback format
                last_error = err
                continue
            if not isinstance(payload, dict) or ENSEMBLE_ARTIFACT_KEY not in payload:
                raise ValueError(
                    f"{resolved_path} is not an ensemble artifact: the "
                    f"payload does not carry the {ENSEMBLE_ARTIFACT_KEY!r} "
                    "marker (was it saved by EnsembleModel.save?)"
                )
            version = payload.get("version")
            if version != ENSEMBLE_ARTIFACT_VERSION:
                raise ValueError(
                    f"Unsupported ensemble artifact version {version!r} "
                    f"in {resolved_path} (expected {ENSEMBLE_ARTIFACT_VERSION})"
                )
            members = [
                EnsembleMember(
                    name=member["name"],
                    score=member["score"],
                    weight=member["weight"],
                    rank=member["rank"],
                )
                for member in payload["members"]
            ]
            return cls(
                metric_user=payload["metric_user"],
                metric_direction=MetricDirection(payload["metric_direction"]),
                members=members,
                predictors=payload["predictors"],
            )
        raise ValueError(
            f"Could not load the ensemble artifact {resolved_path}: {last_error}"
        )

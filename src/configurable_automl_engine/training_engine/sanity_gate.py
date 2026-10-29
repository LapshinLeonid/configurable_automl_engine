"""ModelSanityGate: module for auditing model degeneracy.

Protects the final winner selection of the AutoML engine from "degenerate"
models — constant/flat predictions, step functions, unused features and
overfit learners. The module is standalone: it depends only on numpy, pandas
and scikit-learn and knows nothing about the rest of the engine.

The gate (:class:`ModelSanityGate`) holds the settings, runs the enabled
circuits (:mod:`.sanity_circuits` — the four circuits A–D, their thresholds,
NaN handling and determinism are described there), routes the circuit D
violations by ``generalization_gap_mode`` and logs the audit; the result
record is :class:`~.sanity_result.SanityCheckResult`.

``warn_only`` mode: the gate still computes every statistic and fills
``reasons`` with the would-be disqualifications, but ``is_valid`` is forced to
True and ``severity`` is capped at the "warning" level — the upper layer
decides whether to apply the verdict to winner selection. NOTE:
this is a low-level diagnostic option of the audit object itself; the pipeline
does NOT use it — select steps with ``strategy: sanity_gate`` always apply the
disqualifications (the ``warn_only`` selection mode was removed together with
``sanity_gate.mode`` / ``winner_if_active`` / the ``sanity_gate_warn_only``
report key).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from configurable_automl_engine.training_engine.sanity_circuits import (
    DeadFeatureSettings,
    check_dead_features,
    check_diversity,
    check_generalization_gap,
    check_unique,
)
from configurable_automl_engine.training_engine.sanity_result import (
    SanityCheckResult,
    _compute_severity,
    _validate_check_inputs,
)

logger = logging.getLogger(__name__)

__all__ = ["GeneralizationGapMode", "ModelSanityGate", "SanityCheckResult"]


#: Circuit D mode: ``"warn"`` — a finite gap strictly above
#: ``max_generalization_gap`` is recorded in ``soft_warnings`` (does not
#: disqualify); ``"reject"`` — a failure → ``reasons`` →
#: disqualification (the old strict behavior). Default — ``"warn"``.
#: The field name is deliberately NOT ``mode`` (a collision with the removed
#: ``sanity_gate.mode`` field, which the ``_reject_removed_mode`` validator
#: rejects with a migration error). The type is declared here, in the
#: standalone gate module, and separately in ``config/sanity_gate.py`` (without an
#: import, to keep the module independent of the rest of the engine).
GeneralizationGapMode = Literal["warn", "reject"]


@dataclass
class _CircuitsOutcome:
    """Metrics and messages collected by ``ModelSanityGate._run_circuits``."""

    diversity_ratio: float = float("nan")
    unique_ratio: float = float("nan")
    dead_features_count: int = 0
    generalization_gap: float = float("nan")
    reasons: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)


class ModelSanityGate:
    """Audit a model for degeneracy through four optional circuits.

        Standalone auditor: it depends only on numpy, pandas and scikit-learn and
        knows nothing about the rest of the engine. Each circuit is optional and
        disabled with a ``check_*`` flag. Errors and NaN are treated as a failure
        of the corresponding circuit with a clear reason (except partial NaN in
        predictions — such pairs are masked, as in ``metrics.oof_rmse``). Default
        thresholds are deliberately soft, so honest models on noisy/wide data are
        not disqualified.

        Attributes:
            min_prediction_diversity: Circuit A threshold — ``Var(y_pred_oof) /
                Var(y)`` (default 0.15, soft). Applied only in the
                non-adaptive mode.
            adaptive_diversity: Adaptive mode of circuit A relative to the
                constant model (default True): fails only for a zero
                prediction spread (the absolute threshold is not applied) and
                for models worse than the constant prediction (``R²_oof < 0``);
                ``min_prediction_diversity`` is ignored.
            min_unique_count: Absolute lower bound for nunique (circuit B).
            min_unique_ratio: Lower bound for the share of unique predictions,
                in (0, 1]. Applied as ``min_unique_ratio * N`` capped by the
                number of distinct target values ``nunique(y)`` (at the
                rounding granularity applied to predictions); the final
                threshold never exceeds that ceiling — a perfect model always
                reaches it.
            max_dead_feature_ratio: Dead-feature share threshold (circuit C);
                above it — a soft signal, and combined with a circuit A failure
                (when ``dead_features_require_low_diversity``) — a disqualification.
            dead_features_require_low_diversity: Allow the combined
                disqualification "many dead features AND low diversity".
            zero_coef_tolerance: Coefficients with ``abs(coef) <=`` this value
                count as zero (linear path of circuit C).
            max_generalization_gap: Circuit D threshold — ``RMSE_oof / RMSE_full``
                (default 1.5).
    generalization_gap_mode: Circuit D routing of a gap violation: ``"warn"``
                (default) — ``soft_warnings``, no
                disqualification (including an infinite gap);
                ``"reject"`` — a reason and disqualification (legacy strict
                default). Contract violations and non-computability stay reasons
                in both modes.
            check_permutation_sensitivity: Enable the permutation sensitivity
                audit of circuit C for nonlinear models (cost control).
            permutation_max_rows: Row subsampling cap for the permutation path.
            permutation_max_features: Cap on checked columns in the permutation
                path.
            permutation_repeats: Shuffle repeats per column.
            permutation_seed: Fixed seed for permutation determinism.
            permutation_tolerance: Relative "deadness" threshold of a feature in
                the permutation path.
            check_diversity: Enable circuit A.
            check_unique: Enable circuit B.
            check_dead_features: Enable circuit C.
            check_generalization_gap: Enable circuit D.
            warn_only: Mode without the right to disqualify (statistics are
                computed, ``is_valid`` is always True).
    """

    def __init__(
        self,
        *,
        min_prediction_diversity: float = 0.15,
        adaptive_diversity: bool = True,
        min_unique_count: int = 5,
        min_unique_ratio: float = 0.05,
        max_dead_feature_ratio: float = 0.4,
        dead_features_require_low_diversity: bool = True,
        zero_coef_tolerance: float = 1e-12,
        max_generalization_gap: float = 1.5,
        generalization_gap_mode: GeneralizationGapMode = "warn",
        check_permutation_sensitivity: bool = True,
        permutation_max_rows: int = 5000,
        permutation_max_features: int = 100,
        permutation_repeats: int = 3,
        permutation_seed: int = 42,
        permutation_tolerance: float = 1e-3,
        check_diversity: bool = True,
        check_unique: bool = True,
        check_dead_features: bool = True,
        check_generalization_gap: bool = True,
        warn_only: bool = False,
    ) -> None:
        """Initialize the gate with thresholds and circuit flags.

                Args:
                    min_prediction_diversity: circuit A threshold, in (0, 1]
                        (default 0.15 — soft). Applied only in the
                        non-adaptive mode.
                    adaptive_diversity: adaptive mode A relative to the constant
                        model (default True): fails only for a zero prediction
                        spread (the absolute threshold is not applied) and for
                        models worse than the constant prediction
                        (``R²_oof < 0``, i.e. the residual variance exceeds
                        the target variance).
                    min_unique_count: absolute lower bound for nunique (circuit B).
                    min_unique_ratio: lower bound for the share of unique predictions,
                        in (0, 1]. Applied as ``min_unique_ratio * N`` capped by
                        ``nunique(y)`` — the number of distinct target values on the
                        valid rows at the rounding granularity applied to predictions
                        (a perfect model predicting y exactly can reach
                        at most ``nunique(y)`` distinct predictions, so the threshold
                        must never exceed that ceiling).
                    max_dead_feature_ratio: dead-feature share threshold (circuit C):
                        above it — a soft signal, and combined with a circuit A failure
                        (when ``dead_features_require_low_diversity``) — disqualification.
                    dead_features_require_low_diversity: allow the combined
                        disqualification "many dead features AND low diversity".
                    zero_coef_tolerance: coefficients with ``abs(coef) <=`` this value
                        count as zero (linear path of circuit C).
                    max_generalization_gap: circuit D threshold (≥ 1, default 1.5).
        generalization_gap_mode: circuit D routing of a gap violation:
                        ``"warn"`` (default) — the violation lands
                        in ``soft_warnings`` (``is_valid`` stays True, severity ≤ 1),
                        including an infinite gap from division by zero (``RMSE_full
                        ≈ 0`` / ``RMSE_oof ≈ 0``); ``"reject"`` — legacy strict
                        behavior: a reason and disqualification. Contract violations
                        (``y_pred_full = None``) and non-computability (no finite
                        pairs) remain reasons in both modes.
                    check_permutation_sensitivity: enable the permutation sensitivity
                        audit of circuit C for nonlinear models (default True — the
                        cost-controlled permutation path runs; False — the path is
                        skipped and only the linear ``coef_`` path of circuit C works).
                    permutation_max_rows: row subsampling for permutations (>= 1,
                        default 5000).
                    permutation_max_features: cap on checked columns (>= 1,
                        default 100).
                    permutation_repeats: shuffle repeats per column.
                    permutation_seed: fixed seed for permutation determinism.
                    permutation_tolerance: relative "deadness" threshold of a feature:
                        ``|ΔMSE| / (|mse_base| + eps) < tolerance``.
                    check_diversity: enable circuit A.
                    check_unique: enable circuit B.
                    check_dead_features: enable circuit C.
                    check_generalization_gap: enable circuit D.
                    warn_only: mode without the right to disqualify (statistics are
                        computed, ``is_valid`` is always True).

                Raises:
                    ValueError: for invalid thresholds or parameter combinations.
        """
        if not 0 < min_prediction_diversity <= 1:
            raise ValueError("min_prediction_diversity must be in (0, 1]")
        if min_unique_count < 1:
            raise ValueError("min_unique_count must be >= 1")
        if not 0 < min_unique_ratio <= 1:
            raise ValueError("min_unique_ratio must be in (0, 1]")
        if not 0 <= max_dead_feature_ratio <= 1:
            raise ValueError("max_dead_feature_ratio must be in [0, 1]")
        if max_generalization_gap < 1:
            raise ValueError("max_generalization_gap must be >= 1")
        if generalization_gap_mode not in ("warn", "reject"):
            raise ValueError(
                "generalization_gap_mode must be 'warn' or 'reject'; "
                f"got {generalization_gap_mode!r}"
            )
        if zero_coef_tolerance < 0:
            raise ValueError("zero_coef_tolerance must be >= 0")
        if permutation_repeats < 1:
            raise ValueError("permutation_repeats must be >= 1")
        if permutation_max_rows < 1:
            raise ValueError("permutation_max_rows must be >= 1")
        if permutation_max_features < 1:
            raise ValueError("permutation_max_features must be >= 1")
        if permutation_tolerance < 0:
            raise ValueError("permutation_tolerance must be >= 0")

        self.min_prediction_diversity = min_prediction_diversity
        self.adaptive_diversity = adaptive_diversity
        self.min_unique_count = min_unique_count
        self.min_unique_ratio = min_unique_ratio
        self.max_dead_feature_ratio = max_dead_feature_ratio
        self.dead_features_require_low_diversity = dead_features_require_low_diversity
        self.zero_coef_tolerance = zero_coef_tolerance
        self.max_generalization_gap = max_generalization_gap
        self.generalization_gap_mode = generalization_gap_mode
        self.check_permutation_sensitivity = check_permutation_sensitivity
        self.permutation_max_rows = permutation_max_rows
        self.permutation_max_features = permutation_max_features
        self.permutation_repeats = permutation_repeats
        self.permutation_seed = permutation_seed
        self.permutation_tolerance = permutation_tolerance
        self.check_diversity = check_diversity
        self.check_unique = check_unique
        self.check_dead_features = check_dead_features
        self.check_generalization_gap = check_generalization_gap
        self.warn_only = warn_only

    # ──────────────────────────────────────────────────────────────────────── #
    #  Public API
    # ──────────────────────────────────────────────────────────────────────── #
    def check(
        self,
        *,
        y: Any,
        y_pred_oof: Any,
        y_pred_full: Any | None = None,
        X: Any | None = None,
        model: Any = None,
        candidate_name: str | None = None,
    ) -> SanityCheckResult:
        """Run the model through every enabled audit circuit.

        Args:
            y: the true target (1-D; may contain NaN — pairs are masked).
            y_pred_oof: OOF predictions aligned with ``y`` by rows (NaN for
                uncovered rows is masked).
            y_pred_full: full-fit predictions on the same rows (needed by
                circuit D); None with circuit D enabled → fail with a reason.
            X: feature matrix in the form accepted by ``model.predict`` (needed
                by the permutation path of circuit C for nonlinear models).
            model: the fitted model (needed by circuit C); pipelines are
                unwrapped to the final estimator.
            candidate_name: Name of the audited candidate. When provided, every
                audit log line is prefixed with it, so a warning (e.g. about
                dead features) is unambiguously attributed to a candidate.

        Returns:
            SanityCheckResult: the full audit result.

        Raises:
            ValueError: when the input contract is violated (None arguments,
                length mismatch between y/y_pred_oof/X).
        """
        y_arr, oof_arr = _validate_check_inputs(y, y_pred_oof, X)
        outcome = self._run_circuits(
            y_arr=y_arr, oof_arr=oof_arr, y_pred_full=y_pred_full, X=X, model=model
        )
        reasons, soft_warnings = outcome.reasons, outcome.soft_warnings

        is_valid = not reasons
        if self.warn_only:
            is_valid = True
        severity = _compute_severity(reasons, soft_warnings, self.warn_only)

        # Candidate attribution: when a candidate name is supplied, every audit
        # log line is prefixed with it so that soft warnings (e.g. about dead
        # features) are unambiguously tied to the candidate they concern.
        prefix = f"{candidate_name}: " if candidate_name else ""
        self._log_audit(prefix, reasons, soft_warnings, is_valid)

        return SanityCheckResult(
            is_valid=is_valid,
            reasons=reasons,
            soft_warnings=soft_warnings,
            diversity_ratio=outcome.diversity_ratio,
            unique_ratio=outcome.unique_ratio,
            dead_features_count=outcome.dead_features_count,
            generalization_gap=outcome.generalization_gap,
            severity=severity,
        )

    def _run_circuits(
        self,
        *,
        y_arr: np.ndarray,
        oof_arr: np.ndarray,
        y_pred_full: Any | None,
        X: Any | None,
        model: Any,
    ) -> _CircuitsOutcome:
        """Run every enabled circuit and collect the metrics and messages.

        Args:
            y_arr: the target as a float vector.
            oof_arr: OOF predictions as a float vector.
            y_pred_full: full-fit predictions (circuit D).
            X: the feature matrix (circuit C).
            model: the fitted model (circuit C).

        Returns:
            _CircuitsOutcome: the four metrics, the reasons and the soft
                warnings, in circuit order.
        """
        outcome = _CircuitsOutcome()
        reasons, soft_warnings = outcome.reasons, outcome.soft_warnings
        diversity_failed = False

        # Shared mask of finite pairs for the prediction-based circuits.
        valid_pairs = np.isfinite(y_arr) & np.isfinite(oof_arr)

        if self.check_diversity:
            outcome.diversity_ratio, d_reasons, d_warnings = check_diversity(
                y_arr,
                oof_arr,
                valid_pairs,
                adaptive=self.adaptive_diversity,
                min_prediction_diversity=self.min_prediction_diversity,
            )
            reasons.extend(d_reasons)
            soft_warnings.extend(d_warnings)
            diversity_failed = bool(d_reasons)

        if self.check_unique:
            outcome.unique_ratio, u_reasons, u_warnings = check_unique(
                y_arr,
                oof_arr,
                valid_pairs,
                min_unique_count=self.min_unique_count,
                min_unique_ratio=self.min_unique_ratio,
            )
            reasons.extend(u_reasons)
            soft_warnings.extend(u_warnings)

        if self.check_dead_features:
            outcome.dead_features_count, c_reasons, c_warnings = check_dead_features(
                model=model,
                X=X,
                y=y_arr,
                diversity_failed=diversity_failed,
                settings=self._dead_feature_settings(),
            )
            reasons.extend(c_reasons)
            soft_warnings.extend(c_warnings)

        if self.check_generalization_gap:
            outcome.generalization_gap, g_reasons, g_warnings = (
                check_generalization_gap(
                    y_arr,
                    oof_arr,
                    y_pred_full,
                    max_generalization_gap=self.max_generalization_gap,
                )
            )
            reasons.extend(g_reasons)
            # Routing of circuit D violations: warn mode
            # (default) — soft_warnings (does not disqualify,
            # including an infinite gap from division by zero); reject mode —
            # reasons (the old strict behavior). Reasons in both modes
            # remain only for contract violations (y_pred_full=None) and
            # non-computability (no finite pairs).
            if self.generalization_gap_mode == "reject":
                reasons.extend(g_warnings)
            else:
                soft_warnings.extend(g_warnings)
        return outcome

    def _dead_feature_settings(self) -> DeadFeatureSettings:
        """The circuit-C settings of this gate (read at check time)."""
        return DeadFeatureSettings(
            check_diversity=self.check_diversity,
            check_permutation_sensitivity=self.check_permutation_sensitivity,
            zero_coef_tolerance=self.zero_coef_tolerance,
            max_dead_feature_ratio=self.max_dead_feature_ratio,
            dead_features_require_low_diversity=(
                self.dead_features_require_low_diversity
            ),
            permutation_max_rows=self.permutation_max_rows,
            permutation_max_features=self.permutation_max_features,
            permutation_repeats=self.permutation_repeats,
            permutation_seed=self.permutation_seed,
            permutation_tolerance=self.permutation_tolerance,
        )

    def _log_audit(
        self,
        prefix: str,
        reasons: list[str],
        soft_warnings: list[str],
        is_valid: bool,
    ) -> None:
        """Log the audit reasons and soft warnings.

        Args:
            prefix: the candidate prefix of every line (may be empty).
            reasons: disqualification reasons.
            soft_warnings: soft signals.
            is_valid: the audit verdict.
        """
        if reasons:
            logger.info(
                "%sSanity gate%s: %d reason(s): %s",
                prefix,
                " (warn_only)" if self.warn_only else "",
                len(reasons),
                reasons,
            )
        if not soft_warnings:
            return
        # Soft warnings (dead features in linear models, "false-dead"
        # features on correlation, circuit D violations in the warn mode)
        # are logged at DEBUG/INFO level — they do not disqualify on
        # their own and must never be raised to WARNING. A deliberate
        # exception: a suspected target leakage (RMSE_oof ≈ 0) in the
        # warn mode is logged at the WARNING level — a silent
        # disqualification would swap the winner and hide the leak, so a
        # noticeable signal is needed.
        logger.info(
            "%sSanity gate: %s with %d soft warning(s) (non-disqualifying signals)",
            prefix,
            "valid" if is_valid else "disqualified",
            len(soft_warnings),
        )
        for warning in soft_warnings:
            if (
                self.generalization_gap_mode == "warn"
                and "possible target leakage" in warning
            ):
                logger.warning("%sSanity gate soft warning: %s", prefix, warning)
            else:
                logger.debug("%sSanity gate soft warning: %s", prefix, warning)

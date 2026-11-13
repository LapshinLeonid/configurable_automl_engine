"""The plans of one HPO run: the results of its pure setup steps.

``_OptimizeRunner.run()`` computes these plans once per run, in order, with
pure functions — ``plan_validation`` (``hpo.runner``) → ``plan_preprocessing``
(``hpo.trial_pipeline``) → ``plan_search_space`` (``hpo.search_space``) →
``plan_pruning`` (``hpo.pruning``) — and passes them explicitly to the trial
evaluation (``score_trial`` in ``hpo.evaluation`` returns a ``TrialScore``).
Fields holding a data frame, an estimator, a CV splitter or a callable are
excluded from the equality. The module holds definitions only and must not
import any other module of the ``hpo`` package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import optuna
    from optuna.trial import Trial

    from configurable_automl_engine.feature_selection import (
        FeatureSelectionCfg,
        FeatureSelectionMode,
    )


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    """The validated input and the resolved validation strategy.

    Attributes:
        X: The (possibly repaired) feature matrix: datetime columns expanded,
            ±inf replaced according to the input policies.
        algo: The lower-cased algorithm name.
        fs_cfg: The normalized feature selection configuration.
        is_isotonic: Whether the algorithm is ``isotonic_regression`` (feature
            selection is forcibly disabled for it).
        n_samples_eff: The number of rows a model sees in a training fold.
        method: The effective validation method (``'auto'`` resolved).
        cv: The CV splitter of ``k_fold`` / ``loo`` (``None`` for a hold-out).
        auto_decision: The ``'auto'`` decision (``None`` for an explicit
            method).
        resolved_via_auto: Whether the method was resolved from ``'auto'``.
        n_folds: The effective number of folds (for ``k_fold``).
        test_size: The requested hold-out size.
        split_test_size: The hold-out size of the ``train_test_split`` split
            (the integer row count of the ``'auto'`` decision when resolved
            from ``'auto'``, else ``test_size``).
        n_repeats: The effective number of repeats of the validation strategy
            (``1`` — the classic single-split behavior; LOO repeats are
            resolved away to ``1``).
        repeat_seeds: The deterministic seed of every repeat
            (``random_state + r``, wrapped into the sklearn seed range).
        split_aggregation: How the per-repeat trial scores are combined
            (``'median'`` by default, or ``'mean'``).
    """

    X: Any = field(compare=False)
    algo: str
    fs_cfg: FeatureSelectionCfg
    is_isotonic: bool
    n_samples_eff: int
    method: str
    cv: Any = field(compare=False)
    auto_decision: dict[str, Any] | None
    resolved_via_auto: bool
    n_folds: int
    test_size: float | int
    split_test_size: float | int
    n_repeats: int = 1
    repeat_seeds: tuple[int | None, ...] = ()
    split_aggregation: str = "median"

    @property
    def fs_mode(self) -> FeatureSelectionMode:
        """The feature selection mode."""
        return self.fs_cfg.mode


@dataclass(frozen=True, slots=True)
class PreprocessingPlan:
    """The trial preprocessor and the feature counts it implies.

    Attributes:
        X: The feature matrix after the uninformative-feature cleaning.
        dropped_features: The columns dropped by the cleaning.
        categorical_features: The categorical columns (``None`` when not
            determinable).
        numerical_features: The numerical columns (``None`` when not
            determinable).
        preprocessor: The (unfitted) preprocessor, or ``None``.
        n_features_eff: The number of features after preprocessing (the
            complexity guard ``p``).
        n_numeric_eff: The number of numerical features after preprocessing
            (``None`` — not determinable).
        n_features_size: The number of original columns (the size-mode ratio
            of the complexity guard).
        isotonic_max_feature_index: The upper bound of the isotonic
            ``feature_index`` range.
    """

    X: Any = field(compare=False)
    dropped_features: list[str]
    categorical_features: list[str] | None
    numerical_features: list[str] | None
    preprocessor: Any = field(compare=False)
    n_features_eff: int
    n_numeric_eff: int | None
    n_features_size: int
    isotonic_max_feature_index: int


@dataclass(frozen=True, slots=True)
class SearchSpacePlan:
    """The resolved search space of the run.

    Attributes:
        space_fn: The function suggesting the hyperparameters of a trial.
        space_dict: The resolved dict space (parameter → ``SearchSpaceEntry``
            or constant); empty for a custom space function.
        guard_records: The adaptive complexity guard records (the runner logs
            them).
        user_hyperparameter_keys: The parameters treated as user ranges
            (``None`` for a custom space function).
    """

    space_fn: Callable[[Trial], dict[str, Any]] = field(compare=False)
    space_dict: dict[str, Any]
    guard_records: list[dict[str, Any]]
    user_hyperparameter_keys: list[str] | None


@dataclass(frozen=True, slots=True)
class PruningPlan:
    """The early stopping setup.

    Attributes:
        pruner: The configured Optuna pruner (``None`` — pruning disabled).
        active: Whether the pruner is used (the validation has intermediate
            steps).
    """

    pruner: optuna.pruners.BasePruner | None = field(compare=False)
    active: bool


@dataclass(frozen=True, slots=True)
class TrialScore:
    """The evaluation of one trial pipeline.

    Attributes:
        value: The target metric value (``HPO_WORST_SCORE`` for a non-finite
            result).
        oof: The out-of-fold predictions by row position (NaN where the
            trial did not predict); ``None`` when not collected. With several
            repeats the shape is ``(n_repeats, n_rows)`` — one OOF vector per
            repeat; with one repeat the classic ``(n_rows,)`` vector is kept.
        repeat_scores: The per-repeat scores of a multi-split trial (the
            penalized worst-score sentinel for a failed/non-finite repeat);
            empty with one repeat (the classic single-score behavior).
    """

    value: float
    oof: np.ndarray | None = field(default=None, compare=False)
    repeat_scores: tuple[float, ...] = field(default=(), compare=False)

    @property
    def repeat_mean(self) -> float | None:
        """The mean of the per-repeat scores (``None`` without repeats)."""
        if not self.repeat_scores:
            return None
        return float(fmean(self.repeat_scores))

    @property
    def repeat_std(self) -> float | None:
        """The standard deviation of the per-repeat scores."""
        if not self.repeat_scores:
            return None
        return float(pstdev(self.repeat_scores))


@dataclass(frozen=True, slots=True)
class BestTrial:
    """The best completed trial of the phase.

    Attributes:
        params: The best parameters as reported (auxiliary keys included).
        score: The best raw score.
        model_params: The model constructor parameters (auxiliary keys
            removed).
        apply_fs: Whether the best pipeline has the feature selection step.
        n_trials_evaluated: The number of the phase's own completed trials.
    """

    params: dict[str, Any]
    score: float
    model_params: dict[str, Any]
    apply_fs: bool
    n_trials_evaluated: int

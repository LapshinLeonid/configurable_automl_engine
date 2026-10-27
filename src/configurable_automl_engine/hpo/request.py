"""The settings of one HPO run (``HpoRequest``).

``HpoRequest`` holds everything an ``optimize()`` run takes besides the data
``X`` / ``y``: the algorithm and the metric, the validation settings, the
seeds, the search space, the warm start, the preprocessing and encoding, the
feature selection, the oversampling, the pruning and the input-data
policies. ``tuner.optimize`` builds it from its keyword arguments; the
training engine builds it directly. The module holds definitions only and
must not import any other module of the ``hpo`` package. The setting groups
(``EncodingSettings``, ``OversamplingSettings``, ``InputPolicies``) live in
``common.settings``: the trainer uses the same groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from configurable_automl_engine.common.settings import (
    EncodingSettings,
    InputPolicies,
    OversamplingSettings,
)
from configurable_automl_engine.hpo.complexity_guard import ComplexityGuardConfig

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    from configurable_automl_engine.common.definitions import ValidationStrategy
    from configurable_automl_engine.feature_selection import (
        FeatureSelectionCfg,
    )
    from configurable_automl_engine.preprocessing_presets import (
        PreprocessingOverride,
    )


@dataclass(frozen=True, slots=True)
class HpoRequest:
    """Everything one HPO run of one algorithm needs besides ``X`` and ``y``.

    The defaults are the ``optimize()`` defaults. The semantics of every
    field are documented on the ``optimize()`` parameter of the same name;
    the grouped settings are ``encoding`` (the ``encoding`` /
    ``high_cardinality_*`` / ``hashing_n_components`` / ``target_encoding_*``
    arguments), ``oversampling`` (the ``data_oversampling*`` arguments) and
    ``policies`` (``inf_policy`` / ``datetime_policy`` /
    ``drop_id_like_columns``).

    Attributes:
        algo_name: The algorithm name.
        metric: The metric name to maximize.
        n_trials: The number of trials of the run.
        val_method: The validation method (``'auto'`` is resolved by the
            run; the ``validation_strategy`` alias of ``optimize()`` is
            already applied).
        n_folds: The number of folds of ``'k_fold'``.
        train_test_split_test_size: The hold-out size of
            ``'train_test_split'`` (a fraction or a number of rows).
        random_state: The seed of the data splits, the preprocessor and the
            model.
        sampler_seed: The seed of the TPE sampler (``None`` — falls back to
            ``random_state``).
        space_overrides: The search space overrides keyed by the algorithm
            name (a dict of ranges or a callable space function).
        user_hyperparameter_keys: The parameters the user set explicitly
            (``None`` — the keys of the dict override).
        limit_hyperparameters: Search only the parameters of the override.
        complexity_guard: The adaptive complexity guard settings.
        initial_params: The parameters enqueued as the first trial.
        previous_trials: The TPE history transferred into the study.
        preprocessor: A ready preprocessor (takes precedence over the column
            lists).
        categorical_features: The categorical column names.
        numerical_features: The numerical column names.
        encoding: The categorical encoding settings.
        preprocessing_override: The explicit preprocessing preset override.
        feature_selection_cfg: The feature selection settings.
        oversampling: The oversampling settings.
        pruning: The early stopping (pruning) settings.
        policies: The input-data diagnostics policies.
        fit_best_model: Whether the best model is trained on all data.
        phase_name: The pipeline phase name (log deduplication).
        collect_oof: Whether the OOF predictions of every trial are kept.
    """

    algo_name: str
    metric: str = "r2"
    n_trials: int = 50
    val_method: ValidationStrategy | str = "auto"
    n_folds: int = 5
    train_test_split_test_size: float = 0.2
    random_state: int | None = 42
    sampler_seed: int | None = None
    space_overrides: dict[str, Any] | None = None
    user_hyperparameter_keys: list[str] | None = None
    limit_hyperparameters: bool = False
    complexity_guard: ComplexityGuardConfig = field(
        default_factory=ComplexityGuardConfig
    )
    initial_params: dict[str, Any] | None = None
    previous_trials: list[FrozenTrial] | None = field(default=None, compare=False)
    preprocessor: Any | None = field(default=None, compare=False)
    categorical_features: list[str] | None = None
    numerical_features: list[str] | None = None
    encoding: EncodingSettings = field(default_factory=EncodingSettings)
    preprocessing_override: PreprocessingOverride | dict[str, Any] | None = None
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None
    oversampling: OversamplingSettings = field(default_factory=OversamplingSettings)
    pruning: dict[str, Any] | None = None
    policies: InputPolicies = field(default_factory=InputPolicies)
    fit_best_model: bool = True
    phase_name: str | None = None
    collect_oof: bool = False


def as_complexity_guard_config(value: Any) -> ComplexityGuardConfig:
    """Normalize a complexity guard setting into a ``ComplexityGuardConfig``.

    Args:
        value: ``None`` (the ``general.complexity_guard`` defaults), a
            ``ComplexityGuardConfig``, a dict or a Pydantic model with
            ``model_dump``.

    Returns:
        ComplexityGuardConfig: The runtime configuration.
    """
    if value is None:
        return ComplexityGuardConfig()
    if isinstance(value, ComplexityGuardConfig):
        return value
    return ComplexityGuardConfig.from_dict(value)

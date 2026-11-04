"""Adaptive feature preprocessing presets.

Implements automatic selection of the feature preprocessing strategy
depending on the selected regression algorithm (the user-facing description
lives in API_REFERENCE.md, "Adaptive Feature Preprocessing").

Key concepts:

    * **Preprocessing preset** (:class:`PreprocessingPreset`) — a declaratively
      described set of rules for processing numeric features: the missing value
      imputation strategy (:data:`ImputationStrategy`) and the scaling
      configuration (:class:`ScalingSpec`).
    * **Algorithm class** (:class:`AlgorithmClass`) — a group of regression
      algorithms with common preprocessing requirements.
    * **Mapping table** :data:`ALGORITHM_CLASS_MAPPING` — fixes
      the class membership of every supported algorithm. Adding a new algorithm
      does not require changing the core preprocessing logic —
      it is enough to register it in the table.

Selection rules: the algorithm name is normalized and checked against the
registry (an unknown name is rejected with :class:`ValueError`); a supported
algorithm without explicit registration falls back to the ``default`` class;
every preset field is resolved with the precedence order (later items win)
"class preset < run-level default override < per-algorithm user override" —
a field set at a higher level replaces the whole field from the lower level
(nested values are never merged).

Scaling configuration (:class:`ScalingSpec`) is an object with a ``type``
discriminator (``StandardScaling`` / ``RobustScaling`` / ``NoScaling``);
the string shorthand (``"robust"``) is accepted everywhere and normalized to
its object form by a ``BeforeValidator``, so existing configs and API calls
keep working.

The single function :func:`resolve_preprocessing_preset` is used at every stage
of the lifecycle — the HPO phase, final training and prediction — which
guarantees consistent application of the preset. :func:`merge_preprocessing_overrides`
is the public merge point that folds the run-level (``general.preprocessing``)
and per-algorithm override blocks into the single ``preprocessing_override``
argument of ``ModelTrainer`` / ``tuner.optimize`.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    model_validator,
)

from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    resolve_algorithm_name,
)

#: Missing value imputation strategies for numeric features.
ImputationStrategy = Literal["mean", "median"]

#: Deprecated alias of the string shorthand of :class:`ScalingSpec` (kept for
#: backward compatibility of imports; the canonical string form is
#: :data:`StringScaling`).
ScalingType = Literal["standard", "robust", "none"]


def _coerce_scaling_spec(value: Any) -> Any:
    """Normalize a string shorthand into the object form of a scaling spec.

    Args:
        value: ``"standard"`` / ``"robust"`` / ``"none"`` (the string
            shorthand) or an already-object form (a dict or a
            :class:`ScalingSpec` member).

    Returns:
        The object form: ``{"type": <shorthand>}`` for a string, the input
        value otherwise.
    """
    if isinstance(value, str):
        return {"type": value}
    return value


def _discriminate_scaling_type(value: Any) -> Any:
    """Extract the ``type`` discriminator tag of a scaling spec value.

    Args:
        value: A scaling spec in any pre-coercion form (a dict, the object
            form or ``None``/garbage — anything non-dict/object yields
            ``None`` and is rejected by the union).

    Returns:
        The ``type`` tag (``'standard'`` / ``'robust'`` / ``'none'``) or
        ``None`` when it cannot be determined.
    """
    if isinstance(value, dict):
        return value.get("type")
    return getattr(value, "type", None)


def _scaling_tag(t: type[BaseModel]) -> str:
    """Return the discriminator tag of a scaling spec model class."""
    return t.model_fields["type"].default  # type: ignore[return-value]


class StandardScaling(BaseModel):
    """Standard scaling: :class:`sklearn.preprocessing.StandardScaler`.

    The spec is a frozen value object: parameters cannot be mutated after
    creation, so module-level defaults stay safe to share.

    Attributes:
        type: The scaling type discriminator (always ``"standard"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["standard"] = Field(
        default="standard",
        description="The scaling type: StandardScaler (zero mean, unit variance).",
    )


class RobustScaling(BaseModel):
    """Robust scaling: :class:`sklearn.preprocessing.RobustScaler`.

    The spec is a frozen value object: parameters cannot be mutated after
    creation, so module-level defaults stay safe to share.

    Attributes:
        type: The scaling type discriminator (always ``"robust"``).
        quantile_range: The ``(low, high)`` quantiles used by the scaler:
            ``0 <= low < high <= 100``.
        with_centering: Whether to center the data with the median before
            scaling.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["robust"] = Field(
        default="robust",
        description=(
            "The scaling type: RobustScaler (median centering and quantile "
            "range; robust to outliers)."
        ),
    )
    quantile_range: tuple[float, float] = Field(
        default=(25.0, 75.0),
        description=(
            "The (low, high) quantiles of RobustScaler: 0 <= low < high <= 100."
        ),
    )
    with_centering: bool = Field(
        default=True,
        description="Center the data with the median before scaling.",
    )

    @model_validator(mode="after")
    def _validate_quantile_range(self) -> RobustScaling:
        """Validate the quantile range bounds (``0 <= lo < hi <= 100``).

        Returns:
            The validated model instance.

        Raises:
            ValueError: If the quantile range is outside the [0, 100] bounds
                or the bounds are not strictly increasing.
        """
        lo, hi = self.quantile_range
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(
                f"quantile_range must satisfy 0 <= low < high <= 100, got ({lo}, {hi})."
            )
        return self


class NoScaling(BaseModel):
    """No scaling: numeric features are passed to the model unscaled.

    The spec is a frozen value object: parameters cannot be mutated after
    creation, so module-level defaults stay safe to share.

    Attributes:
        type: The scaling type discriminator (always ``"none"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["none"] = Field(
        default="none",
        description="The scaling type: no scaling is applied.",
    )


#: The object form of a scaling configuration: a union of the per-type
#: models discriminated explicitly by the ``type`` field (``Tag`` +
#: :class:`~pydantic.Discriminator`), so validation errors are precise
#: (no cross-type union noise) and the scaler parameters are checked in the
#: branch of their own type only.
ObjectScalingSpec = Annotated[
    Annotated[StandardScaling, Tag(_scaling_tag(StandardScaling))]
    | Annotated[RobustScaling, Tag(_scaling_tag(RobustScaling))]
    | Annotated[NoScaling, Tag(_scaling_tag(NoScaling))],
    Discriminator(_discriminate_scaling_type),
]

#: String shorthand of a scaling type (e.g. ``"robust"``): accepted wherever
#: a :class:`ScalingSpec` is expected, and rendered as a plain string
#: (``enum``) alternative in the JSON schema. During validation the shorthand
#: is normalized to its object form by :func:`_coerce_scaling_spec`, so this
#: member effectively only documents/renders the schema alternative.
StringScaling = Literal["standard", "robust", "none"]

ScalingSpec = Annotated[
    ObjectScalingSpec | StringScaling,
    BeforeValidator(_coerce_scaling_spec),
    Field(
        description=(
            "Scaling of numeric features: a string shorthand "
            "('standard' | 'robust' | 'none') or an object "
            "({'type': 'standard'} | {'type': 'robust', 'quantile_range': "
            "[low, high], 'with_centering': bool} | {'type': 'none'})."
        ),
    ),
]

#: Validated accessor for a single :class:`ScalingSpec` value (used by
#: :func:`coerce_scaling` and the builder).
_SCALING_SPEC_ADAPTER: TypeAdapter[ObjectScalingSpec] = TypeAdapter(ObjectScalingSpec)


def coerce_scaling(spec: ScalingSpec) -> ObjectScalingSpec:
    """Coerce a scaling specification into its object form.

    Args:
        spec: A :class:`ScalingSpec` member, a string shorthand
            (``"standard"`` / ``"robust"`` / ``"none"``) or an object-form
            dict.

    Returns:
        The object form of the scaling spec.

    Raises:
        ValueError: If the value is not a valid scaling spec (delegated to
            Pydantic validation, a :class:`~pydantic.ValidationError`
            subclass).
    """
    coerced: Any = _coerce_scaling_spec(spec)
    if isinstance(coerced, (StandardScaling, RobustScaling, NoScaling)):
        return coerced
    obj: ObjectScalingSpec = _SCALING_SPEC_ADAPTER.validate_python(coerced)
    return obj


def scaling_type(spec: ScalingSpec) -> ScalingType:
    """The string type of a scaling spec (``'standard'`` / ``'robust'`` /
    ``'none'``).

    Args:
        spec: A scaling specification in any accepted form.

    Returns:
        The scaling type string.
    """
    obj = coerce_scaling(spec)
    result: ScalingType = obj.type
    return result


class AlgorithmClass(str, Enum):
    """Classes of regression algorithms with common preprocessing requirements.

    Attributes:
        SCALE_SENSITIVE: Scale-sensitive models (linear, kernel-based,
            distance-based) — scaling is mandatory, missing values are filled
            with the standard strategy.
        TREES: Trees and tree ensembles — scaling is not required and should
            not be applied; missing value imputation is robust to outliers.
        GLM_SKEWED: Generalized linear models with skewed distributions —
            missing value imputation that accounts for data skewness; special
            scaling requirements.
        UNIVARIATE: Specialized univariate algorithms — non-standard
            preprocessing (no scaling).
        DEFAULT: Default behavior for algorithms without explicit registration.
    """

    SCALE_SENSITIVE = "scale_sensitive"
    TREES = "trees"
    GLM_SKEWED = "glm_skewed"
    UNIVARIATE = "univariate"
    DEFAULT = "default"


class PreprocessingPreset(BaseModel):
    """A declaratively described feature preprocessing preset.

    Attributes:
        algorithm_class: The algorithm class for which the preset is defined.
        imputation_strategy: The missing value imputation strategy for numeric
            features: ``'mean'`` (standard) or ``'median'``
            (robust to outliers, accounts for skewness).
        scaling: The scaling configuration of numeric features (a
            :class:`ScalingSpec`: ``StandardScaling``, ``RobustScaling`` with
            its ``quantile_range``/``with_centering`` parameters, or
            ``NoScaling``).
        description: A human-readable description of the preset.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm_class: AlgorithmClass = Field(
        default=AlgorithmClass.DEFAULT,
        description="The algorithm class for which the preset is defined.",
    )
    imputation_strategy: ImputationStrategy = Field(
        default="mean",
        description=(
            "Missing value imputation strategy for numeric features: "
            "'mean' or 'median'."
        ),
    )
    scaling: ScalingSpec = Field(
        default_factory=StandardScaling,
        description=(
            "Scaling of numeric features: a string shorthand "
            "('standard', 'robust' or 'none') or an object "
            "({'type': 'robust', 'quantile_range': [low, high], ...}). "
            "The shorthand is normalized to its object form."
        ),
    )
    description: str = Field(
        default="",
        description="A human-readable description of the preset.",
    )

    def __str__(self) -> str:
        """Compact string for logging the selected preset."""
        return (
            f"PreprocessingPreset(class={self.algorithm_class.value!r}, "
            f"imputation_strategy={self.imputation_strategy!r}, "
            f"scaling={_format_scaling(self.scaling)})"
        )

    __repr__ = __str__


def _format_scaling(spec: ScalingSpec) -> str:
    """Format a scaling spec for the preset's log string.

    Args:
        spec: A scaling specification in any accepted form.

    Returns:
        ``'standard'`` / ``'none'`` as-is; for a robust spec the parameters
        are shown, e.g. ``robust(q=(10.0, 90.0))`` (a non-default
        ``with_centering=False`` is appended as ``, centering=False``).
    """
    obj = coerce_scaling(spec)
    if obj.type == "robust":
        text = f"robust(q={obj.quantile_range})"
        if not obj.with_centering:
            text += ", centering=False"
        return text
    return repr(obj.type)


class PreprocessingOverride(BaseModel):
    """An explicit user override of the preprocessing preset.

    It may be specified partially or fully: ``None`` in a field means "keep the
    automatically selected value". Invalid values are rejected at the config
    validation stage (Pydantic) with a clear message.
    """

    model_config = ConfigDict(extra="forbid")

    imputation_strategy: ImputationStrategy | None = Field(
        default=None,
        description=(
            "Override of the missing value imputation strategy "
            "('mean' or 'median'). None — automatic selection."
        ),
    )
    scaling: ScalingSpec | None = Field(
        default=None,
        description=(
            "Scaling override: a string shorthand ('standard', 'robust' or "
            "'none'), an object ({'type': 'robust', 'quantile_range': "
            "[low, high], ...}) or None (automatic selection). The shorthand "
            "is normalized to its object form."
        ),
    )


#: Mapping table "algorithm -> algorithm class".
#: Keys are canonical algorithm names from ``AVAILABLE_ALGORITHMS``.
ALGORITHM_CLASS_MAPPING: dict[str, AlgorithmClass] = {
    # ── Scale-sensitive models (linear, kernel-based, distance-based) ──
    "elasticnet": AlgorithmClass.SCALE_SENSITIVE,
    "sgdregressor": AlgorithmClass.SCALE_SENSITIVE,
    "ridge": AlgorithmClass.SCALE_SENSITIVE,
    "lasso": AlgorithmClass.SCALE_SENSITIVE,
    "ardregression": AlgorithmClass.SCALE_SENSITIVE,
    "svr": AlgorithmClass.SCALE_SENSITIVE,
    "nearest_neighbors_regression": AlgorithmClass.SCALE_SENSITIVE,
    "gaussian_process_regression": AlgorithmClass.SCALE_SENSITIVE,
    # ── Trees and tree ensembles ──
    "decision_tree": AlgorithmClass.TREES,
    "random_forest": AlgorithmClass.TREES,
    "extra_trees": AlgorithmClass.TREES,
    "gradient_boosting": AlgorithmClass.TREES,
    "adaboost": AlgorithmClass.TREES,
    "xgboosting": AlgorithmClass.TREES,
    # ── GLM with skewed distributions ──
    "poissonregressor": AlgorithmClass.GLM_SKEWED,
    "gammaregressor": AlgorithmClass.GLM_SKEWED,
    "tweedieregressor": AlgorithmClass.GLM_SKEWED,
    "glm": AlgorithmClass.GLM_SKEWED,
    # ── Specialized univariate algorithms ──
    "isotonic_regression": AlgorithmClass.UNIVARIATE,
}

#: Preprocessing preset for each algorithm class.
#: The ``scaling`` values are expressed in the object form of
#: :class:`ScalingSpec` (e.g. ``RobustScaling()``); they are equivalent to
#: the string shorthand and keep the table independent of the coercion.
PRESET_BY_CLASS: dict[AlgorithmClass, PreprocessingPreset] = {
    AlgorithmClass.SCALE_SENSITIVE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.SCALE_SENSITIVE,
        imputation_strategy="mean",
        scaling=StandardScaling(),
        description=(
            "Scale-sensitive models: scaling is mandatory "
            "(StandardScaler), missing values are filled with the standard strategy (mean)."
        ),
    ),
    AlgorithmClass.TREES: PreprocessingPreset(
        algorithm_class=AlgorithmClass.TREES,
        imputation_strategy="median",
        scaling=NoScaling(),
        description=(
            "Trees and tree ensembles: scaling is not required and is not "
            "applied; missing values are filled with the median (robust to outliers)."
        ),
    ),
    AlgorithmClass.GLM_SKEWED: PreprocessingPreset(
        algorithm_class=AlgorithmClass.GLM_SKEWED,
        imputation_strategy="median",
        scaling=RobustScaling(),
        description=(
            "GLM with skewed distributions: missing values are filled with the median "
            "(robust to outliers, accounts for skewness); scaling with "
            "RobustScaler (robust to outliers)."
        ),
    ),
    AlgorithmClass.UNIVARIATE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.UNIVARIATE,
        imputation_strategy="median",
        scaling=NoScaling(),
        description=(
            "Specialized univariate algorithms: scaling is not "
            "applied, missing values are filled with the median."
        ),
    ),
    AlgorithmClass.DEFAULT: PreprocessingPreset(
        algorithm_class=AlgorithmClass.DEFAULT,
        imputation_strategy="mean",
        scaling=StandardScaling(),
        description=(
            "Default behavior for algorithms without explicit registration "
            "in the mapping table: mean + StandardScaler."
        ),
    ),
}


def normalize_algorithm(algorithm: str) -> str:
    """Normalize an algorithm name: lowercased.

    Args:
        algorithm: An algorithm name (e.g. ``'Random_Forest'``).

    Returns:
        The canonical algorithm name used for lookup in the mapping table.
    """
    return resolve_algorithm_name(algorithm)


def _coerce_override(
    override: PreprocessingOverride | dict[str, Any],
) -> PreprocessingOverride:
    """Coerce a user override into a :class:`PreprocessingOverride` model."""
    if isinstance(override, PreprocessingOverride):
        return override
    if isinstance(override, dict):
        return PreprocessingOverride.model_validate(override)
    raise TypeError(
        f"preprocessing_override must be a dict or PreprocessingOverride, "
        f"got {type(override).__name__}"
    )


def merge_preprocessing_overrides(
    default: PreprocessingOverride | dict[str, Any] | None,
    specific: PreprocessingOverride | dict[str, Any] | None,
) -> PreprocessingOverride:
    """Merge two preprocessing overrides into one ("default" + "specific").

    The public merge point used by the training engine: the run-level
    ``general.preprocessing`` block (``default``) and the per-algorithm
    ``algorithms.<name>.preprocessing`` block (``specific``) are merged into
    a single override before it is handed to ``ModelTrainer`` /
    ``tuner.optimize`` (which keep their single ``preprocessing_override``
    argument). Field-wise semantics: a field set at the more specific level
    replaces the whole field of the default level; a field left as ``None``
    falls through to the default level. Nested values are never merged —
    the model only has flat scalar fields.

    Args:
        default: The run-level (general) override. ``None`` — no default.
        specific: The algorithm-level override. ``None`` — no specific
            override (the result is then just the coerced default).

    Returns:
        PreprocessingOverride: The merged override (a new model instance;
            the inputs are not mutated).

    Raises:
        TypeError: If an argument has an unsupported type.
    """
    default_model = (
        _coerce_override(default) if default is not None else PreprocessingOverride()
    )
    if specific is None:
        return default_model
    specific_model = _coerce_override(specific)
    update: dict[str, Any] = {}
    if specific_model.imputation_strategy is not None:
        update["imputation_strategy"] = specific_model.imputation_strategy
    if specific_model.scaling is not None:
        update["scaling"] = specific_model.scaling
    if not update:
        return default_model
    return default_model.model_copy(update=update)


def resolve_preprocessing_preset(
    algorithm: str,
    override: PreprocessingOverride | dict[str, Any] | None = None,
    *,
    default_override: PreprocessingOverride | dict[str, Any] | None = None,
) -> PreprocessingPreset:
    """Determine the preprocessing preset for a given algorithm.

    Selection logic:

    1. The algorithm name is normalized (lowercased) and checked
       against the registry of supported algorithms: an unknown name (typo)
       is rejected with an explicit error so that configuration mistakes are
       not hidden.
    2. The algorithm class is looked up in :data:`ALGORITHM_CLASS_MAPPING`; for a
       supported algorithm without explicit registration the ``default`` class
       is used (edge case — default behavior).
    3. Every preset field is resolved with the precedence order (later items
       win): class preset < ``default_override`` < ``override``. A field set
       at a higher level replaces the whole field from the lower level;
       a field left as ``None`` falls through to the lower level. Nested
       values are never merged.

    Args:
        algorithm: A regression algorithm name.
        override: An explicit per-algorithm preset override. ``None`` —
            no algorithm-level override.
        default_override: The run-level (general) preset override applied
            for every algorithm. Keyword-only. Fields left as ``None``
            fall through to the automatic selection; fields set here take
            priority over the class preset but yield to ``override``.

    Returns:
        The final preprocessing preset for the algorithm.

    Raises:
        ValueError: If ``algorithm`` is not a supported algorithm.
        TypeError: If an override has an unsupported type.
    """
    algo_key = normalize_algorithm(algorithm)
    if algo_key not in AVAILABLE_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            f"Available algorithms: {sorted(AVAILABLE_ALGORITHMS)}"
        )
    algo_class = ALGORITHM_CLASS_MAPPING.get(algo_key, AlgorithmClass.DEFAULT)
    preset = PRESET_BY_CLASS[algo_class]

    update: dict[str, Any] = {}
    if default_override is not None:
        default_model = _coerce_override(default_override)
        if default_model.imputation_strategy is not None:
            update["imputation_strategy"] = default_model.imputation_strategy
        if default_model.scaling is not None:
            update["scaling"] = default_model.scaling
    if override is not None:
        override_model = _coerce_override(override)
        if override_model.imputation_strategy is not None:
            update["imputation_strategy"] = override_model.imputation_strategy
        if override_model.scaling is not None:
            # The override value arrives as a string shorthand or an object
            # dict: normalize it to the object form up front, so the copy
            # below shares a validated (frozen) spec instance instead of a
            # raw dict.
            update["scaling"] = coerce_scaling(override_model.scaling)
    if update:
        preset = preset.model_copy(update=update)
    return preset


def requires_missing_indicator(algorithm: str) -> bool:
    """Whether missing value indicators are needed for the given algorithm.

    Missing value indicators are enabled for all algorithm classes except
    ``UNIVARIATE``: strictly univariate algorithms (``isotonic_regression``)
    accept exactly one feature, so an additional binary indicator column would
    violate their input contract. For the remaining algorithms the indicator
    lets the model distinguish "actually measured value" from "value recovered
    by imputation".

    Args:
        algorithm: A regression algorithm name.

    Returns:
        ``True`` — missing value indicators should be enabled; ``False`` —
        the algorithm is strictly univariate (the indicator is incompatible with
        its input).

    Raises:
        ValueError: If ``algorithm`` is not a supported
            algorithm (delegated to
            :func:`resolve_preprocessing_preset`). The caller must
            either guarantee the name is valid or handle the
            exception with a safe default value (``True``).
    """
    return (
        resolve_preprocessing_preset(algorithm).algorithm_class
        != AlgorithmClass.UNIVARIATE
    )

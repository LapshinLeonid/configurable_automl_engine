"""The single entry point that builds the feature preprocessor.

Holds :func:`build_preprocessor`: the ``ColumnTransformer`` with imputation
and the selected encoding for categorical columns and imputation plus scaling
for numerical columns. It is used both in HPO (``tuner.optimize``) and in
final training (``trainer.ModelTrainer``). It builds on the encoders and the
column selector; no other preprocessing module may import it.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    RobustScaler,
    StandardScaler,
)

from configurable_automl_engine.common.definitions import EncodingStrategy
from configurable_automl_engine.preprocessing.columns import ColumnNameSelector
from configurable_automl_engine.preprocessing.encoders import (
    _VALID_ENCODING_STRATEGIES,
    SplitCategoricalEncoder,
    _to_string_array,
)
from configurable_automl_engine.preprocessing.outliers import Winsorizer
from configurable_automl_engine.preprocessing_presets import (
    ImputationStrategy,
    NoScaling,
    ObjectScalingSpec,
    OutlierSpec,
    RobustScaling,
    ScalingSpec,
    StandardScaling,
    coerce_outliers,
    coerce_scaling,
)

logger = logging.getLogger(__name__)

#: Module-level default of the ``scaling`` argument: a singleton spec
#: instance. The spec models are frozen (``ConfigDict(frozen=True)``), so the
#: shared default cannot be mutated by a caller.
_DEFAULT_SCALING = StandardScaling()


def build_preprocessor(
    feature_names: list[str],
    categorical_features: list[str],
    numerical_features: list[str],
    encoding: EncodingStrategy = "one_hot",
    imputation_strategy: ImputationStrategy = "mean",
    scaling: ScalingSpec = _DEFAULT_SCALING,
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: EncodingStrategy | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    random_state: int | None = 42,
    force_dense_output: bool = False,
    add_indicator: bool = True,
    outliers: OutlierSpec | None = None,
    outlier_algorithm: str | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer for separate processing of data types.

    Supported categorical strategies (details in API_REFERENCE.md,
    "Categorical encoding"): ``'one_hot'`` (default, sparse one-hot),
    ``'ordinal'``, ``'target'``, ``'frequency'``, ``'hashing'``. With
    ``high_cardinality_threshold``/``high_cardinality_encoding`` set, columns
    above the cardinality threshold are encoded with the high-cardinality
    strategy and the rest with ``encoding``; the split is computed from the
    training data inside :class:`SplitCategoricalEncoder`.

    Leakage guards: target/frequency statistics are computed only from the
    training part (on each CV fold the preprocessor is refitted), and target
    encoding in ``fit_transform`` is cross-fitted — each training row is
    encoded with means that exclude the row itself.

    Numerical features are winsorized (``outliers``), imputed
    (``imputation_strategy``) and scaled (``scaling``) per the adaptive
    preprocessing preset; binary missing-value indicators are added by
    default (``add_indicator=True``) so the model can tell apart a measured
    value from an imputed one. The winsorization step comes strictly BEFORE
    the imputer: otherwise the mean/median imputation would see the extreme
    values and the 0/1 missing-value indicator columns would get clipped.
    With the default ``outliers=None`` (``method='none'``) the step is
    absent from the pipeline. Indicators are created only for features that
    had missing values on train, so the output width depends on the data.

    Columns are selected by name (:class:`ColumnNameSelector`): the column
    order in ``fit``/``transform`` does not matter; missing/duplicated
    columns raise an explicit error. ``np.ndarray`` inputs use a positional
    fallback in the training order of ``feature_names``. With
    ``encoding='ordinal'`` the integer codes are NOT scaled — for linear
    models the incomparable scale of high-cardinality codes can dominate the
    numerical features.

    Args:
        feature_names: Full list of feature names in column order.
        categorical_features: Names of the columns encoded with the selected strategy.
        numerical_features: Names of the columns subject to imputation and scaling.
        encoding: Categorical encoding strategy (see above).
        imputation_strategy: ``'mean'`` (default) or ``'median'`` (robust to
            outliers). Imputation is performed with ``add_indicator=True``.
        scaling: The scaler configuration: a string shorthand (``'standard'``
            (default), ``'robust'`` or ``'none'``) or a
            :class:`~configurable_automl_engine.preprocessing_presets.ScalingSpec`
            object (``StandardScaling`` / ``RobustScaling`` with
            ``quantile_range``/``with_centering`` / ``NoScaling``). The
            shorthand is normalized to its object form; validation happens
            through the model.
        high_cardinality_threshold: Cardinality threshold (``>= 0``);
            ``None`` — the automatic mode is disabled (the engine config
            enables it with threshold 50 + ``'frequency'``).
        high_cardinality_encoding: Encoding strategy for high-cardinality
            columns; must be set together with the threshold.
        hashing_n_components: Number of binary columns per categorical column
            for ``encoding='hashing'`` (``>= 1``).
        target_encoding_smoothing: Smoothing parameter of target encoding
            (``>= 0``); with ``m=0`` the pure per-category mean is used.
        target_encoding_fallback: Fallback value of target encoding for unknown
            categories. ``None`` — the global mean of the target variable.
        random_state: Seed for deterministic hashing encoding.
        force_dense_output: Forcibly return a dense matrix instead of a sparse
            one. Enabled for algorithms that reject ``scipy.sparse``
            (GPR/Isotonic/ARD; see
            :func:`~configurable_automl_engine.models.requires_dense_input`):
            with ``sparse_threshold=0.0`` ``ColumnTransformer`` converts sparse
            parts to dense.
        add_indicator: Whether to add binary missing-value indicators for
            numerical features. ``True`` by default; disabled for strictly
            one-dimensional algorithms (``isotonic_regression``), whose input
            contract allows exactly one feature (see
            :func:`~configurable_automl_engine.preprocessing_presets.requires_missing_indicator`).
        outliers: The outlier-handling configuration: ``None`` (the default,
            no outlier handling — the winsorization step is absent from the
            pipeline), a string shorthand (``'none'`` / ``'quantile'`` /
            ``'iqr'``) or an
            :class:`~configurable_automl_engine.preprocessing_presets.OutlierSpec`
            object (``QuantileOutliers`` with ``lower_quantile``/
            ``upper_quantile``, or ``IqrOutliers`` with ``iqr_multiplier``).
            The step is inserted before the imputer: the clipping bounds are
            learned from the raw training column, and the imputed values and
            the 0/1 indicator columns stay outside the clip.
        outlier_algorithm: The algorithm name used only to prefix the
            WARNING about collapsed (constant/all-NaN) columns with the
            algorithm context. ``None`` — the message is not prefixed.

    Returns:
        A ``ColumnTransformer`` that converts the input DataFrame into a numerical
        matrix ready to be fed to the model. If no column matched, a
        passthrough transformer is returned.

    Raises:
        ValueError: If a column name from ``categorical_features``/
            ``numerical_features`` is missing in ``feature_names``, or an
            invalid value of ``encoding``, ``imputation_strategy``,
            ``scaling`` or ``outliers`` is passed (an invalid scaling/outlier
            spec is rejected by the corresponding
            :class:`~configurable_automl_engine.preprocessing_presets.ScalingSpec`
            / ``OutlierSpec`` model), the high-cardinality parameters are
            inconsistent, or ``hashing_n_components``/``target_encoding_smoothing``
            are invalid.
    """
    if encoding not in _VALID_ENCODING_STRATEGIES:
        raise ValueError(
            f"Unknown encoding strategy: {encoding!r}. "
            f"Expected one of {_VALID_ENCODING_STRATEGIES}."
        )
    if (
        high_cardinality_encoding is not None
        and high_cardinality_encoding not in _VALID_ENCODING_STRATEGIES
    ):
        raise ValueError(
            f"Unknown high_cardinality_encoding: {high_cardinality_encoding!r}. "
            f"Expected one of {_VALID_ENCODING_STRATEGIES}."
        )
    if (high_cardinality_threshold is None) != (high_cardinality_encoding is None):
        raise ValueError(
            "high_cardinality_threshold and high_cardinality_encoding must be "
            "set together (both provided or both None)."
        )
    if high_cardinality_threshold is not None and high_cardinality_threshold < 0:
        raise ValueError(
            f"high_cardinality_threshold must be >= 0, got "
            f"{high_cardinality_threshold}."
        )
    if hashing_n_components < 1:
        raise ValueError(
            f"hashing_n_components must be >= 1, got {hashing_n_components}."
        )
    if target_encoding_smoothing < 0:
        raise ValueError(
            f"target_encoding_smoothing must be >= 0, got {target_encoding_smoothing}."
        )
    if imputation_strategy not in ("mean", "median"):
        raise ValueError(
            f"Unknown imputation strategy: {imputation_strategy!r}. "
            "Expected one of ('mean', 'median')."
        )
    # Validation of the scaling spec happens through the model (type
    # discriminator, quantile_range bounds, extra='forbid'); the string
    # shorthand is normalized to its object form here.
    try:
        scaling_spec = coerce_scaling(scaling)
    except ValidationError as err:
        raise ValueError(f"Unknown scaling spec: {scaling!r}. {err}") from err
    # Validation of the outlier spec happens through the model (method
    # discriminator, quantile bounds, extra='forbid'); the string shorthand
    # is normalized to its object form here. None/'none' — no step at all.
    winsorizer = _build_winsorizer(outliers, outlier_algorithm)

    # Validate that all requested column names are present in the training set
    # (feature_names): this is an explicit error, because the name-based
    # selector will not be able to resolve missing names.
    specified_features = list(categorical_features) + list(numerical_features)
    missing_features = [col for col in specified_features if col not in feature_names]
    if missing_features:
        raise ValueError(
            f"Unknown feature name(s): {missing_features}. "
            f"Expected columns: {feature_names}."
        )

    # Name-based selectors: for pd.DataFrame the columns are selected by name on
    # each fit/transform (column order does not matter); for inputs without
    # names (np.ndarray), a positional fallback by training order is used.
    # The selectors are picklable classes, so trainer.save()/pickle work.
    cat_selector = ColumnNameSelector(feature_names, list(categorical_features))
    num_selector = ColumnNameSelector(feature_names, list(numerical_features))

    if not categorical_features and not numerical_features:
        logger.warning(
            "No features matched for preprocessing. Defaulting to passthrough."
        )
        return ColumnTransformer(
            [("bypass", "passthrough", slice(None))],
            remainder="drop",
            sparse_threshold=0.0 if force_dense_output else 0.3,
        )

    # Numerical feature transformation pipeline: [winsorizer] -> imputation
    # -> (optionally) a scaler. add_indicator=True adds a binary
    # missing-value flag for each numerical feature that had missing values
    # in the training set: the model gets the ability to tell apart "actually
    # measured value" from "value restored by imputation". With a NoScaling
    # spec the scaling step is absent — the data is passed to the model
    # without it (for trees and ensembles). The winsorizer (when enabled via
    # ``outliers``) comes strictly BEFORE the imputer: the clipping bounds
    # must be learned from the raw training column, otherwise the mean
    # imputation sees the outliers and the 0/1 indicator columns get clipped.
    num_steps: list[tuple[str, Any]] = []
    if winsorizer is not None:
        num_steps.append(("winsorizer", winsorizer))
    num_steps.append(
        (
            "imputer",
            SimpleImputer(strategy=imputation_strategy, add_indicator=add_indicator),
        )
    )
    scaler = _build_scaler(scaling_spec)
    if scaler is not None:
        num_steps.append(("scaler", scaler))
    num_transformer = Pipeline(steps=num_steps)

    cat_transformer = Pipeline(
        steps=[
            # Convert categorical columns to a string object array. This makes the
            # pipeline robust to bool columns: SimpleImputer with
            # strategy="most_frequent" fails on numpy-bool ("SimpleImputer does
            # not support data with dtype bool"), so bool features are converted
            # to strings ("True"/"False") and encoded as regular categories.
            # Missing values (NaN/None) are preserved by _to_string_array, so the
            # imputer below actually fills them (instead of treating "nan"/"None"
            # as distinct unknown categories). feature_names_out="one-to-one"
            # registers get_feature_names_out on this step, so the preprocessor's
            # get_feature_names_out works.
            (
                "to_string",
                FunctionTransformer(_to_string_array, feature_names_out="one-to-one"),
            ),
            # keep_empty_features=True: a fully-missing categorical column would
            # otherwise be dropped by SimpleImputer (an empty feature), the
            # pipeline would then stop matching the preprocessor's get_feature_names_out
            # and the encoder dimensionality.
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "encoder",
                SplitCategoricalEncoder(
                    encoding=encoding,
                    high_cardinality_threshold=high_cardinality_threshold,
                    high_cardinality_encoding=high_cardinality_encoding,
                    hashing_n_components=hashing_n_components,
                    target_encoding_smoothing=target_encoding_smoothing,
                    target_encoding_fallback=target_encoding_fallback,
                    random_state=random_state,
                ),
            ),
        ]
    )

    transformers = []
    if categorical_features:
        transformers.append(("cat", cat_transformer, cat_selector))
    if numerical_features:
        transformers.append(("num", num_transformer, num_selector))

    return ColumnTransformer(
        transformers=(transformers if transformers else [("pass", "passthrough", [0])]),
        remainder="drop",
        # With force_dense_output we always return a dense matrix: sparse parts
        # (hashing) are converted to ndarray; otherwise the standard
        # sparse_threshold=0.3 applies (csr for any sparse part).
        sparse_threshold=0.0 if force_dense_output else 0.3,
    )


def _build_scaler(
    scaling_spec: ObjectScalingSpec,
) -> RobustScaler | StandardScaler | None:
    """Build the scaler from a validated scaling spec.

    Args:
        scaling_spec: The object form of a scaling specification
            (the output of :func:`coerce_scaling`).

    Returns:
        A ``RobustScaler``/``StandardScaler`` configured from the spec, or
        ``None`` for a ``NoScaling`` spec (the scaling step is omitted).
    """
    if isinstance(scaling_spec, NoScaling):
        return None
    if isinstance(scaling_spec, RobustScaling):
        return RobustScaler(
            quantile_range=scaling_spec.quantile_range,
            with_centering=scaling_spec.with_centering,
        )
    # StandardScaling — the only remaining member of the ScalingSpec union.
    return StandardScaler()


def _build_winsorizer(
    outliers: OutlierSpec | None, outlier_algorithm: str | None
) -> Winsorizer | None:
    """Build the winsorizer from an outlier spec, or ``None`` when disabled.

    Args:
        outliers: The outlier-handling configuration: ``None``/``'none'`` —
            no step; otherwise the clipping configuration. The spec is
            validated through the model (method discriminator, quantile
            bounds, ``extra='forbid'``); the string shorthand is normalized
            to its object form.
        outlier_algorithm: The algorithm name for the collapsed-columns
            WARNING prefix.

    Returns:
        A configured ``Winsorizer`` or ``None`` when outlier handling is off.

    Raises:
        ValueError: If the outlier spec is invalid (delegated to Pydantic
            validation, a ``ValidationError`` subclass).
    """
    if outliers is None:
        return None
    try:
        outliers_spec = coerce_outliers(outliers)
    except ValidationError as err:
        raise ValueError(f"Unknown outliers spec: {outliers!r}. {err}") from err
    if outliers_spec.method == "none":
        return None
    return Winsorizer(
        method=outliers_spec.method,
        lower_quantile=getattr(outliers_spec, "lower_quantile", 0.01),
        upper_quantile=getattr(outliers_spec, "upper_quantile", 0.99),
        iqr_multiplier=getattr(outliers_spec, "iqr_multiplier", 1.5),
        algorithm_name=outlier_algorithm,
    )

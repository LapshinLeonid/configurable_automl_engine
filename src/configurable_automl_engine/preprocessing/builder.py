"""The single entry point that builds the feature preprocessor.

Holds :func:`build_preprocessor`: the ``ColumnTransformer`` with imputation
and the selected encoding for categorical columns and imputation plus scaling
for numerical columns. It is used both in HPO (``tuner.optimize``) and in
final training (``trainer.ModelTrainer``). It builds on the encoders and the
column selector; no other preprocessing module may import it.
"""

from __future__ import annotations

import logging

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
from configurable_automl_engine.preprocessing_presets import (
    ImputationStrategy,
    ScalingType,
)

logger = logging.getLogger(__name__)


def build_preprocessor(
    feature_names: list[str],
    categorical_features: list[str],
    numerical_features: list[str],
    encoding: EncodingStrategy = "one_hot",
    imputation_strategy: ImputationStrategy = "mean",
    scaling: ScalingType = "standard",
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: EncodingStrategy | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    random_state: int | None = 42,
    force_dense_output: bool = False,
    add_indicator: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer for separate processing of data types.

        The following strategies are supported for categorical features:

        * ``'one_hot'`` (default) — ``OneHotEncoder`` with
          ``sparse_output=True``: the sparse output avoids the dense
          ``n × p`` blow-up on high-cardinality columns; the
          ``ColumnTransformer`` keeps the sparse representation when the
          total density is below ``sparse_threshold`` and returns a dense
          matrix otherwise;
        * ``'ordinal'`` — ``OrdinalEncoder`` (1 output column per column);
        * ``'target'`` — target encoding with smoothing
          (:class:`TargetEncodingTransformer`);
        * ``'frequency'`` — frequency encoding
          (:class:`FrequencyEncodingTransformer`);
        * ``'hashing'`` — deterministic hashing into a fixed number of columns
          (:class:`HashingEncodingTransformer`).

        When ``high_cardinality_threshold`` and ``high_cardinality_encoding`` are
        given, the automatic mode is enabled: columns whose number of unique values
        is strictly greater than the threshold are encoded with the high-cardinality
        strategy, and the rest — with the ``encoding`` strategy. The split is
        computed from the training data inside :class:`SplitCategoricalEncoder`, so
        the same logic applies both in HPO and in final training and prediction.

    Target/frequency-encoding statistics are computed only from the training
        part, which prevents data leakage between train and validation: on each CV
        fold the preprocessor is refitted on the training part of the fold.
        Additionally, target encoding in :meth:`TargetEncodingTransformer.fit_transform`
        performs cross-fitting: each training row is encoded with means computed
        on a fold that excludes the row itself, so the row's own ``y`` does not
        leak into its own encoded feature (removes the inflated feature-response
        correlation on training for high-cardinality columns).

        The numerical feature preprocessing strategy is set by the
        ``imputation_strategy`` and ``scaling`` parameters and is usually taken
        from the adaptive preprocessing preset (:mod:`preprocessing_presets`).

        Numerical imputation is performed with ``add_indicator=True`` by default
        (disabled via ``add_indicator=False`` for strictly one-dimensional
        algorithms): for each numerical feature that had missing values in the
        training set, a binary indicator column is added
        (1 — the value was missing, 0 — it was present). The model gets the
        ability to tell apart "actually measured value" and "value restored by
        imputation" (for example, in bioprocesses where a missing pH means that
        pH was not controlled).

        Args:
            feature_names: Full list of feature names in column order.
            categorical_features: Names of the columns encoded with the selected strategy.
            numerical_features: Names of the columns subject to imputation and scaling.
            encoding: Categorical encoding strategy: ``'one_hot'`` (default,
                sparse one-hot), ``'ordinal'``, ``'target'``, ``'frequency'``
                or ``'hashing'``.
            imputation_strategy: Missing-value imputation strategy for numerical
                features: ``'mean'`` (default) or ``'median'`` (robust to outliers,
                accounts for skewed distributions). Imputation is performed with
                ``add_indicator=True`` — binary missing-value indicators are added
                for features with missing values in the training set.
            scaling: Scaling of numerical features: ``'standard'`` (``StandardScaler``,
                default), ``'robust'`` (``RobustScaler``, robust to outliers) or
                ``'none'`` (no scaling is applied — the data is passed to the model
                as is).
            high_cardinality_threshold: Cardinality threshold (``>= 0``). Columns
                with a number of unique values strictly greater than the threshold
                are high-cardinality and encoded with the
                ``high_cardinality_encoding`` strategy. ``None`` — automatic
                mode disabled (the low-level default; the engine config enables
                the mode with threshold 50 + ``'frequency'`` by default).
            high_cardinality_encoding: Encoding strategy for high-cardinality columns.
                Must be set together with ``high_cardinality_threshold``.
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
                parts to dense. ``False`` by default — the output stays sparse for
                sparse one-hot / hashing encoding (memory savings).
            add_indicator: Whether to add binary missing-value indicators for
                numerical features (``SimpleImputer(add_indicator=...)``). ``True``
                by default; disabled for strictly one-dimensional algorithms
                (``isotonic_regression``), whose input contract allows exactly one
                feature (see
                :func:`~configurable_automl_engine.preprocessing_presets.requires_missing_indicator`).

        Returns:
            A ``ColumnTransformer`` that converts the input DataFrame into a numerical
            matrix ready to be fed to the model. If no column matched, a
            passthrough transformer is returned (behavior preserved from the
            trainer).

        Note:
            Categorical columns are converted to a string object array before
            imputation and encoding, so ``bool`` columns are handled correctly
            (without ``SimpleImputer`` failing on the bool dtype).

        Note:
            Missing-value indicators for numerical features (``add_indicator=True``,
            ``MissingIndicator(features='missing-only')``) are created only for
            features with missing values in the training set: a fully populated
            feature does not produce an indicator, so the number of output columns
            depends on the data. If at inference a missing value appears in a
            feature without missing values on train, the value is imputed but no
            indicator is created. With ``scaling != 'none'`` the indicator columns
            are standardized by the scaler together with the numerical ones (for
            trees the step is absent when ``scaling='none'``) — for linear models
            standardizing a binary flag is acceptable. The indicators are disabled
            via ``add_indicator=False`` for strictly one-dimensional algorithms
            (``isotonic_regression``), whose input contract allows exactly one
            feature. The categorical pipeline does not add indicators:
            ``SimpleImputer(strategy="most_frequent")`` works without
            ``add_indicator``.

        Note:
            Columns are selected by name via :class:`ColumnNameSelector`: for
            ``pd.DataFrame`` the column order in ``fit``/``transform`` does not
            matter — the encoder is always applied to categorical columns and the
            scaler to numerical ones; missing/duplicated columns raise an explicit
            error. For inputs without names (``np.ndarray``), a positional fallback
            by the training order of ``feature_names`` is used.

        Note:
            With ``encoding='ordinal'``, categories are encoded with integer codes
            that are NOT scaled (unlike numerical columns that go through the
            scaler). For linear models (e.g., ``elasticnet``), the incomparable
            scale of codes for high-cardinality categories can dominate the
            numerical features.

        Raises:
            ValueError: If a column name from ``categorical_features``/
                ``numerical_features`` is missing in ``feature_names``, or an
                invalid value of ``encoding``, ``imputation_strategy`` or
                ``scaling`` is passed, the high-cardinality parameters are
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
    if scaling not in ("standard", "robust", "none"):
        raise ValueError(
            f"Unknown scaling type: {scaling!r}. "
            "Expected one of ('standard', 'robust', 'none')."
        )

    # Validate that all requested column names are present in the training set
    # (feature_names). Previously, missing names were silently skipped; now
    # this is an explicit error — the name-based selector will not be able to
    # resolve them.
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

    # Numerical feature transformation pipeline: imputation + (optionally) a scaler.
    # add_indicator=True adds a binary missing-value flag for each numerical
    # feature that had missing values in the training set: the model gets the
    # ability to tell apart "actually measured value" from "value restored by
    # imputation". With scaling='none' the scaling step is absent — the data
    # is passed to the model without it (for trees and ensembles).
    num_steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(strategy=imputation_strategy, add_indicator=add_indicator),
        )
    ]
    if scaling != "none":
        scaler = RobustScaler() if scaling == "robust" else StandardScaler()
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

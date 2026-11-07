"""Preprocessing: a single entry point for building the feature preprocessor.

The user-facing description of the encoders and the high-cardinality mode
lives in API_REFERENCE.md ("Categorical encoding"); this module holds the
building blocks:

* :func:`_clean_uninformative_features` — removal of uninformative columns
  (fully empty and strictly constant) before feature type detection;
  applied identically in the HPO phase and in final training so both work
  with a consistent feature set;
* :func:`detect_feature_types` — automatic detection of categorical and
  numerical columns;
* :func:`build_preprocessor` — assembly of the ``sklearn.ColumnTransformer``
  (categorical encoding + numerical imputation/scaling; missing-value
  indicators are added by default);
* :class:`Winsorizer` — the standalone winsorization transformer
  (:mod:`.outliers`; used by :func:`build_preprocessor` for the
  ``outliers`` setting — the first step of the numeric pipeline);
* :class:`_FoldSafeQuantileTransformer` — the fold-safe quantile scaler
  (:mod:`.quantile`; used by :func:`build_preprocessor` for
  ``scaling: quantile``).

Key invariants:

* numerical imputation uses ``add_indicator=True`` — binary indicator
  columns are added for features with training-set missing values;
* the sparse-vs-dense output follows ``ColumnTransformer``'s
  ``sparse_threshold=0.3``, with ``force_dense_output`` for algorithms that
  reject sparse input;
* the high-cardinality split is computed at fit time from training data
  only, so it never depends on validation data;
* the single entry point is used in BOTH training locations (HPO and final
  fit), which prevents preprocessing logic from drifting between the stages.

Package layout: :mod:`.encoders` (the encoders), :mod:`.columns`
(:class:`ColumnNameSelector`), :mod:`.features` (type detection,
uninformative-column removal, feature-count estimate), :mod:`.outliers`
(:class:`Winsorizer`, standalone), :mod:`.quantile`
(:class:`_FoldSafeQuantileTransformer`), :mod:`.builder`
(:func:`build_preprocessor`). This ``__init__`` only re-exports the public
names (the import path stays the same).
"""

from __future__ import annotations

from configurable_automl_engine.common.definitions import EncodingStrategy
from configurable_automl_engine.preprocessing.builder import build_preprocessor
from configurable_automl_engine.preprocessing.columns import ColumnNameSelector
from configurable_automl_engine.preprocessing.encoders import (
    FrequencyEncodingTransformer,
    HashingEncodingTransformer,
    SplitCategoricalEncoder,
    TargetEncodingTransformer,
    _to_string_array,
)
from configurable_automl_engine.preprocessing.features import (
    _clean_uninformative_features,
    detect_feature_types,
    estimate_n_features_after_preprocessing,
)
from configurable_automl_engine.preprocessing.outliers import Winsorizer
from configurable_automl_engine.preprocessing.quantile import (
    _FoldSafeQuantileTransformer,
)
from configurable_automl_engine.preprocessing.target_transform import (
    TargetTransformer,
    fit_transform_target,
    normalize_target_transform,
    validate_target_for_transform,
    wrap_with_target_transform,
)

__all__ = [
    "ColumnNameSelector",
    "EncodingStrategy",
    "FrequencyEncodingTransformer",
    "HashingEncodingTransformer",
    "SplitCategoricalEncoder",
    "TargetEncodingTransformer",
    "TargetTransformer",
    "Winsorizer",
    "_FoldSafeQuantileTransformer",
    "_clean_uninformative_features",
    "_to_string_array",
    "build_preprocessor",
    "detect_feature_types",
    "estimate_n_features_after_preprocessing",
    "fit_transform_target",
    "normalize_target_transform",
    "validate_target_for_transform",
    "wrap_with_target_transform",
]

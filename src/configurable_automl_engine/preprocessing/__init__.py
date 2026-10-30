"""Preprocessing: a single entry point for building the feature preprocessor.

The module encapsulates the low-level data preparation building blocks:

    1. :func:`_clean_uninformative_features` — removal of uninformative
       columns (fully empty and strictly constant) before feature type
       detection and preprocessor assembly. It is applied identically in the
       HPO phase (``tuner.optimize``) and in final training
       (``trainer.ModelTrainer``), so that both phases work with a consistent
       set of features.
    2. :func:`detect_feature_types` — automatic detection of categorical and
       numerical columns from a ``pd.DataFrame``.
    3. :func:`build_preprocessor` — assembly of an ``sklearn.ColumnTransformer``
       with categorical preprocessing (``most_frequent`` imputation + the
       selected encoding strategy) and scaling for numerical features.
       The numerical feature handling strategy (imputation strategy and
       scaling type) is set via the ``imputation_strategy``/``scaling``
       parameters and is usually taken from the adaptive preprocessing preset
       (:mod:`preprocessing_presets`), which is selected automatically based
       on the regression algorithm. Numerical imputation is performed with
       ``add_indicator=True`` (the default): for features with missing values
       in the training set, binary missing-value indicator columns are added.

Supported categorical encoding strategies:

* **one_hot** (default) — ``OneHotEncoder`` with ``sparse_output=True``:
      each category becomes a binary column; unknown categories are ignored.
      The sparse output avoids the dense ``n × p`` blow-up (and the
      ``n × n`` dense allocations of the default ``OneHotEncoder`` on
      high-cardinality ID columns) — the ``ColumnTransformer`` keeps the
      sparse representation whenever the total density is below
      ``sparse_threshold=0.3`` and returns a dense matrix otherwise.
    * **ordinal** — ``OrdinalEncoder``: one numerical column per category;
      unknown categories are mapped to ``-1``.
    * **target** — :class:`TargetEncodingTransformer`: mean of the target
      variable per category with smoothing; statistics are computed only on
      the training part (in ``fit``), the fallback for unknown categories is
      the global mean (or a configurable value).
      :meth:`~TargetEncodingTransformer.fit_transform` performs
      cross-fitting: each training row is encoded with means that exclude its
      own ``y``, which removes the "row into its own feature" leakage on
      high-cardinality columns.
    * **frequency** — :class:`FrequencyEncodingTransformer`: relative
      frequency of a category in the training set; unknown categories -> ``0.0``.
    * **hashing** — :class:`HashingEncodingTransformer`: deterministic
      hashing of categories into a fixed number of binary columns
      (``n_components``); unknown categories are hashed the same way.
      The output is sparse (``scipy.sparse.csr_matrix``), like the standard
      sklearn ``FeatureHasher``, which avoids OOM on high-cardinality columns
      of large datasets. For algorithms that reject sparse input
      (GPR/Isotonic/ARD), ``build_preprocessor`` forcibly returns a dense
      matrix (``force_dense_output``).

An automatic mode is supported for high-cardinality columns: columns whose
number of unique values exceeds ``high_cardinality_threshold`` are encoded
with the ``high_cardinality_encoding`` strategy, and the rest with the
``encoding`` strategy. The split is computed in :meth:`SplitCategoricalEncoder.fit`
from the actual (training) data, so the decision does not depend on the
validation data. The engine config (``GeneralCfg``) enables the mode by
default (threshold 50, ``high_cardinality_encoding='frequency'``); passing
``None`` for both parameters disables it (the default of the low-level
``build_preprocessor``/``SplitCategoricalEncoder`` API).

The single preprocessor entry point is used in BOTH training locations — the
hyperparameter tuning phase (``tuner.optimize``) and final training
(``trainer.ModelTrainer``) — which prevents preprocessing logic from drifting
between the stages.

Layout of the package:

* :mod:`.encoders` — the categorical encoders and their helpers;
* :mod:`.columns` — :class:`ColumnNameSelector` (column selection by name);
* :mod:`.features` — feature type detection, removal of uninformative
  columns, the estimate of the feature count after preprocessing;
* :mod:`.builder` — :func:`build_preprocessor`.

This ``__init__`` only re-exports the public names (the import path
``configurable_automl_engine.preprocessing`` stays the same).
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

__all__ = [
    "ColumnNameSelector",
    "EncodingStrategy",
    "FrequencyEncodingTransformer",
    "HashingEncodingTransformer",
    "SplitCategoricalEncoder",
    "TargetEncodingTransformer",
    "_clean_uninformative_features",
    "_to_string_array",
    "build_preprocessor",
    "detect_feature_types",
    "estimate_n_features_after_preprocessing",
]

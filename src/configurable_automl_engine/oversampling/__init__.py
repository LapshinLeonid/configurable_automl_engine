"""
Data Oversampling Engine: An intelligent sample expander.
The class provides class balancing and synthetic data expansion
while preserving data types and supporting multithreading.

Key features:
    1. Multi-Algorithm: Supports Random, SMOTE and ADASYN.
    2. Noise Injection: Optional Gaussian noise for regularization.
    3. Type Integrity: The `_restore_dtypes` mechanism prevents type
       degradation (e.g. Categorical -> Object cast) after Numpy transforms.
    4. Thread-Safety: Uses an RLock for
    safe access to parameters inside a Pipeline.

The engine is regression-only, so by default any numeric target (float or
integer) is treated as a regression target and oversampled through quantile
bins: the sampling strategy is computed on binned ``y`` regardless of
integrality. The classification path is kept only for standalone
``DataOversampler`` calls with an explicit ``task="classification"``
argument.

Layout of the package:

* :mod:`.sampler` — :class:`DataOversampler` and :func:`oversample`;
* :mod:`.targets` — binning of a regression target and decoding of the
  resampled target;
* :mod:`.smoter` — the SMOTER-style generator for a regression target;
* :mod:`.columns` — categorical and binary-indicator column detection;
* :mod:`.experimental` — the experimental status flag and its WARNING.

This ``__init__`` only re-exports the public names (the import path
``configurable_automl_engine.oversampling`` stays the same).
"""

from configurable_automl_engine.oversampling.columns import (
    _binary_indicator_columns,
    _is_categorical_col,
)
from configurable_automl_engine.oversampling.experimental import (
    EXPERIMENTAL_WARNING_MESSAGE,
    OVERSAMPLING_EXPERIMENTAL,
    reset_experimental_warnings,
    warn_experimental_once,
)
from configurable_automl_engine.oversampling.sampler import (
    DataOversampler,
    oversample,
)
from configurable_automl_engine.oversampling.smoter import _SMOTERGenerator
from configurable_automl_engine.oversampling.targets import (
    CONTINUOUS_TARGET_MIN_BINS,
    CONTINUOUS_TARGET_QUANTILES,
    DEFAULT_TARGET_BINS,
    MIN_TARGET_BINS,
    _bin_continuous_target,
    _bin_regression_target,
    _BinnedTarget,
    _decode_continuous_y,
    _is_continuous_target,
    _prepare_continuous_bins,
)

__all__ = [
    "CONTINUOUS_TARGET_MIN_BINS",
    "CONTINUOUS_TARGET_QUANTILES",
    "DEFAULT_TARGET_BINS",
    "EXPERIMENTAL_WARNING_MESSAGE",
    "MIN_TARGET_BINS",
    "OVERSAMPLING_EXPERIMENTAL",
    "DataOversampler",
    "_BinnedTarget",
    "_SMOTERGenerator",
    "_bin_continuous_target",
    "_bin_regression_target",
    "_binary_indicator_columns",
    "_decode_continuous_y",
    "_is_categorical_col",
    "_is_continuous_target",
    "_prepare_continuous_bins",
    "oversample",
    "reset_experimental_warnings",
    "warn_experimental_once",
]

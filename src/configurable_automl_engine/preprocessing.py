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

* **one_hot** (default) — ``OneHotEncoder``: each category becomes a
      binary column; unknown categories are ignored.
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
validation data.

The single preprocessor entry point is used in BOTH training locations — the
hyperparameter tuning phase (``tuner.optimize``) and final training
(``trainer.ModelTrainer``) — which prevents preprocessing logic from drifting
between the stages.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from configurable_automl_engine.preprocessing_presets import (
    ImputationStrategy,
    ScalingType,
)

logger = logging.getLogger(__name__)

EncodingStrategy = Literal["one_hot", "ordinal", "target", "frequency", "hashing"]

_VALID_ENCODING_STRATEGIES = (
    "one_hot",
    "ordinal",
    "target",
    "frequency",
    "hashing",
)


def _to_string_array(X: Any) -> np.ndarray:
    """Convert a categorical matrix to a string object array, keeping missing values.

    ``np.ndarray.astype(str)`` produces fixed-width unicode (``<U``), while
    ``SimpleImputer`` only accepts ``object`` arrays, so after converting to
    strings the result is cast back to the ``object`` dtype.

    Missing values must stay missing: otherwise ``NaN`` becomes the literal
    ``"nan"`` and ``None`` (e.g. from a JSON payload) becomes ``"None"``, so
    the ``most_frequent`` imputer never sees a gap — the two spellings of "no
    value" turn into two different categories and produce different
    predictions at inference. The missing mask (``pd.isna``) is therefore
    restored on the string array for every dtype, including float columns
    whose missing ``NaN`` values would otherwise stringify to ``"nan"``.
    """
    arr = np.asarray(X)
    string_arr = arr.astype(str).astype(object)
    missing = pd.isna(arr)
    if missing.any():
        string_arr[missing] = np.nan
    return string_arr


def _as_2d(X: Any) -> np.ndarray:
    """Reduce the transformer input to a two-dimensional array (n_samples, n_cols)."""
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _validate_target(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Prepare (X, y) for target encoding.

    Returns a two-dimensional categorical array ``X_arr`` and a numeric
    float ``y_float`` (an ``ndarray`` of shape ``(n_samples,)``). The checks
    are shared by :meth:`TargetEncodingTransformer.fit` and
    :meth:`TargetEncodingTransformer.fit_transform`, so that both training
    paths reject an invalid target the same way.

    Args:
        X: Matrix of categorical features.
        y: Target variable. Required — without it, encoding is impossible.

    Returns:
        The ``(X_arr, y_float)`` tuple.

    Raises:
        ValueError: If ``y`` is not passed, is non-numeric or contains
            non-finite values.
    """
    if y is None:
        raise ValueError("TargetEncodingTransformer requires y during fit().")
    X_arr = _as_2d(X)
    y_arr = np.asarray(y).ravel()
    try:
        y_float = y_arr.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "TargetEncodingTransformer requires a numeric target y."
        ) from e
    if not np.isfinite(y_float).all():
        raise ValueError(
            "TargetEncodingTransformer does not support non-finite values in y."
        )
    return X_arr, y_float


# --------------------------------------------------------------------------- #
#                       Custom categorical encoders                           #
# --------------------------------------------------------------------------- #


class TargetEncodingTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Target encoding (mean encoding) with smoothing and a fallback value.

    For each categorical column, the mean of the target variable per category
    is blended with the global mean through the smoothing parameter ``m``:

        encoded(cat) = (n_cat * mean_cat + m * global_mean) / (n_cat + m)

    All statistics are computed **only** from the provided ``y``,
    so information from the validation/test part does not affect the encoding
    (no train/test leakage). Categories absent from the training set are
    mapped to the fallback value (the global mean or a configurable value
    via ``fallback``).

    :meth:`fit_transform` additionally performs cross-fitting: each training
    row is encoded with means computed on a fold that excludes the row
    itself, so the row's own ``y`` value does not participate in its own
    encoding (no "row into its own feature" leakage that inflates the
    feature-response correlation on training). The full-sample statistics are
    stored on the encoder itself, and the subsequent :meth:`transform`
    (including on new data) uses exactly those.

    Args:
        smoothing: Smoothing parameter ``m`` (``>= 0``). With ``m=0`` the
            pure per-category mean is used; with larger ``m``, rare categories
            approach the global mean.
        fallback: Value for categories absent from the training set.
            ``None`` — use the global mean of the target variable.
        cv: Number of cross-fitting folds in :meth:`fit_transform` (``>= 2``).
            When the number of rows is smaller than ``cv``, the number of
            folds is reduced to the number of rows (leave-one-out), so that
            small samples do not fail.
        random_state: Seed for shuffling the folds in :meth:`fit_transform`
            (determinism of the cross-fitting).

    Note:
        The statistic keys keep the original category types (as in
        :class:`FrequencyEncodingTransformer`): standalone calls with numeric
        categories (``int`` codes, ``float``, ``bool``) correctly find their
        matches in :meth:`transform`. The matching is performed by pandas by
        dtype compatibility: ``int`` and ``float`` are interchangeable, while
        ``bool`` is a separate dtype. Due to the standard Python equality
        semantics (``True == 1``), mixing ``bool`` and ``int`` ``0/1`` can
        produce surprising results, so the categories of a single feature must
        keep one dtype in ``fit`` and ``transform``. When used through
        :func:`build_preprocessor`, categories are converted to strings in
        advance (the ``to_string`` step), which rules out dtype mismatches.
    """

    def __init__(
        self,
        smoothing: float = 20.0,
        fallback: float | None = None,
        cv: int = 5,
        random_state: int | None = 0,
    ):
        self.smoothing = smoothing
        self.fallback = fallback
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> TargetEncodingTransformer:
        """Compute target statistics from the training data.

        The statistics are computed over the whole provided sample; the
        resulting encoder is applied to new data through :meth:`transform`.
        For training without the "row into its own feature" leakage, use
        :meth:`fit_transform` (cross-fitting).

        Args:
            X: Matrix of categorical features (rows after imputation).
            y: Target variable. Required — without it, encoding is impossible.

        Returns:
            Fitted transformer. Statistics for each column are stored in
            ``statistics_`` as ``dict[Any, float]``: keys mirror the original
            category values (types are not normalized to strings).

        Raises:
            ValueError: If ``y`` is not passed or contains NaN.
        """
        X_arr, y_float = _validate_target(X, y)

        self.global_mean_ = float(np.mean(y_float))
        self.fallback_ = (
            self.global_mean_ if self.fallback is None else float(self.fallback)
        )
        self.statistics_: list[dict[Any, float]] = []
        for col in range(X_arr.shape[1]):
            group = pd.DataFrame(
                {"cat": pd.Series(X_arr[:, col]), "target": y_float}
            ).groupby("cat")["target"]
            agg = group.agg(["mean", "count"])
            smoothed = (
                agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing
            ) / (agg["count"] + self.smoothing)
            self.statistics_.append(
                {cat: float(value) for cat, value in smoothed.items()}
            )
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        """Encode training rows with cross-fitting.

        Each row is encoded with statistics computed on the fold it is **not**
        in (``KFold(cv, shuffle=True)``): the row's own ``y`` value does not
        participate in its own encoding, so the encoded feature does not
        correlate with the response on the training set because of leakage
        (critical for high-cardinality columns with rare categories). For an
        unbiased :meth:`transform` on new data, the full-sample statistics are
        stored on the encoder itself.

        When the number of rows is smaller than ``cv``, the number of folds is
        reduced to the number of rows (leave-one-out); a single row is encoded
        with the full statistics (cross-fitting is impossible; the value
        coincides with the own-row contribution in any case).

        Args:
            X: Matrix of categorical features.
            y: Target variable. Required.
            **fit_params: Not used (accepted for compatibility with sklearn
                pipelines).

        Returns:
            Numeric matrix of the same shape as ``X``: encoded values without
            the contribution of the row's own ``y``.

        Raises:
            ValueError: If ``y`` is not passed, is non-numeric or contains
                non-finite values, or ``cv < 2``.
        """
        X_arr, y_float = _validate_target(X, y)
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2, got {self.cv}.")
        # The full statistics — for the subsequent transform() on new data.
        self.fit(X_arr, y_float)

        n_samples = X_arr.shape[0]
        if n_samples < 2:
            # Cross-fitting with row exclusion is impossible: encode the single
            # row with the full statistics. They are already fixed by the
            # self.fit(X_arr, y_float) call above; defensively confirm their
            # presence with an explicit check_is_fitted so that transform()
            # does not rely on the implicit call order (the value in any case
            # coincides with the row's own y at m=0 and is close to it at m>0).
            check_is_fitted(self, attributes=["statistics_"])
            return self.transform(X_arr)

        # The number of folds cannot exceed the number of rows (KFold fails at
        # n_splits > n_samples): for small samples reduce it to LOO.
        n_splits = min(self.cv, n_samples)
        out = np.empty(X_arr.shape, dtype=np.float64)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(X_arr):
            fold = clone(self).fit(X_arr[train_idx], y_float[train_idx])
            out[val_idx] = fold.transform(X_arr[val_idx])
        return out

    def transform(self, X: Any) -> np.ndarray:
        """Replace categories with their target values.

        Args:
            X: Matrix of categorical features.

        Returns:
            Numeric matrix of the same shape (1 column per input column).
            Unknown categories are mapped to the fallback value. Matches are
            looked up by dtype-compatible keys (see the class docstring): numeric
            categories in ``X`` must have the same dtype as in the training set.
        """
        check_is_fitted(self, attributes=["statistics_"])
        X_arr = _as_2d(X)
        out = np.empty((X_arr.shape[0], X_arr.shape[1]), dtype=np.float64)
        for col, stats in enumerate(self.statistics_):
            mapped = pd.Series(X_arr[:, col]).map(stats)
            out[:, col] = mapped.fillna(self.fallback_).to_numpy(dtype=np.float64)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names (1 column per input column)."""
        check_is_fitted(self, attributes=["statistics_"])
        names = _feature_names(input_features, len(self.statistics_))
        return np.asarray(names, dtype=object)


class FrequencyEncodingTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Frequency encoding: relative frequency of a category in the training set.

    Deterministic encoding that does not require ``y``. Unknown categories at
    prediction time are mapped to ``0.0`` (documented fallback). Frequencies
    are computed only from the training part in :meth:`fit`.
    """

    def fit(self, X: Any, y: Any = None) -> FrequencyEncodingTransformer:
        """Compute category frequencies from the training data.

        Args:
            X: Matrix of categorical features.
            y: Not used (accepted for sklearn API compatibility).

        Returns:
            Fitted transformer.
        """
        X_arr = _as_2d(X)
        n_rows = X_arr.shape[0]
        self.frequencies_: list[dict[str, float]] = []
        self.n_columns_ = X_arr.shape[1]
        for col in range(X_arr.shape[1]):
            counts = pd.Series(X_arr[:, col]).value_counts(dropna=False)
            self.frequencies_.append((counts / n_rows).to_dict())
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Replace categories with their frequencies.

        Args:
            X: Matrix of categorical features.

        Returns:
            Numeric matrix of the same shape. Unknown categories -> ``0.0``.
        """
        check_is_fitted(self, attributes=["frequencies_"])
        X_arr = _as_2d(X)
        out = np.empty((X_arr.shape[0], X_arr.shape[1]), dtype=np.float64)
        for col, freqs in enumerate(self.frequencies_):
            mapped = pd.Series(X_arr[:, col]).map(freqs)
            out[:, col] = mapped.fillna(0.0).to_numpy(dtype=np.float64)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names (1 column per input column)."""
        check_is_fitted(self, attributes=["frequencies_"])
        names = _feature_names(input_features, self.n_columns_)
        return np.asarray(names, dtype=object)


class HashingEncodingTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Feature hashing: deterministic encoding of categories by hash.

    Each category is hashed to an index in ``[0, n_components)`` using
    ``hashlib.blake2b`` (the salt depends on ``random_state``), after which a
    binary column is set. The number of output features is fixed
    (``n_components`` per input column) and controlled by configuration — the
    dimensionality does not depend on the column cardinality.

    The output of :meth:`transform` is a sparse matrix ``scipy.sparse.csr_matrix``
    (nnz = n_rows · n_cols), like the standard ``FeatureHasher``: a dense
    ``float64`` matrix of shape ``(n_rows, n_cols · n_components)``
    would cause OOM on real tables with high-cardinality columns.

    Hashing is fully deterministic given fixed settings and ``random_state``:
    unknown categories at prediction time are hashed the same way, so no
    failures occur. Hash collisions are possible (different categories -> one
    column) — a documented limitation.

    Args:
        n_components: Number of binary columns per input categorical column
            (``>= 1``).
        random_state: Seed for the hashing salt (reproducibility).
    """

    def __init__(self, n_components: int = 16, random_state: int | None = 42):
        self.n_components = n_components
        self.random_state = random_state

    def _hash_token(self, token: Any, salt: str) -> int:
        """Deterministically hash a category into ``[0, n_components)``.

        The salt is prepended to the message (rather than passed as the blake2b
        ``salt`` parameter, which is limited to 16 bytes), so the salt length is
        unrestricted and different ``random_state`` values are guaranteed to
        produce different projections.
        """
        digest = hashlib.blake2b(digest_size=8)
        digest.update(salt.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(token).encode("utf-8", errors="replace"))
        return int.from_bytes(digest.digest(), byteorder="little") % self.n_components

    def fit(self, X: Any, y: Any = None) -> HashingEncodingTransformer:
        """Prepare salts for each column.

        Args:
            X: Matrix of categorical features.
            y: Not used (accepted for sklearn API compatibility).

        Returns:
            Fitted transformer.
        """
        X_arr = _as_2d(X)
        self.n_columns_ = X_arr.shape[1]
        self.salts_ = [
            f"hashing_col_{i}_seed_{self.random_state}" for i in range(self.n_columns_)
        ]
        return self

    def transform(self, X: Any) -> sp.csr_matrix:
        """Encode categories as binary hash columns.

        Returns a sparse matrix ``scipy.sparse.csr_matrix``
        (COO → CSR, ``nnz = n_samples * n_columns``, values ``1.0``),
        not a dense array: hashing is intended for very high-cardinality
        columns on large datasets, and a dense ``float64`` matrix of shape
        ``(n_rows, n_cols * n_components)`` would cause OOM. The format matches
        the standard ``sklearn.feature_extraction.FeatureHasher``.

        Args:
            X: Matrix of categorical features.

        Returns:
            Sparse binary matrix of shape
            ``(n_samples, n_columns * n_components)``.
        """
        check_is_fitted(self, attributes=["salts_"])
        X_arr = _as_2d(X)
        n_rows, n_cols = X_arr.shape
        # Each row of each column contributes exactly one non-zero element,
        # so nnz = n_rows * n_cols. Hash collisions within a single column are
        # impossible for one row (one category per row), so no duplicates arise
        # in COO, and the sum over rows is always n_cols (binarity).
        col_blocks: list[np.ndarray] = []
        for col in range(n_cols):
            uniq, inverse = np.unique(X_arr[:, col], return_inverse=True)
            hashed = np.array(
                [self._hash_token(token, self.salts_[col]) for token in uniq],
                dtype=np.int64,
            )
            col_blocks.append(col * self.n_components + hashed[inverse])
        rows = np.tile(np.arange(n_rows), n_cols)
        cols = np.concatenate(col_blocks) if col_blocks else np.empty(0, dtype=np.int64)
        data = np.ones(n_rows * n_cols, dtype=np.float64)
        out = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(n_rows, n_cols * self.n_components),
            dtype=np.float64,
        )
        return out.tocsr()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names (``n_components`` per input column)."""
        check_is_fitted(self, attributes=["salts_"])
        base = _feature_names(input_features, self.n_columns_)
        names = [
            f"{base[i]}__h{j}"
            for i in range(self.n_columns_)
            for j in range(self.n_components)
        ]
        return np.asarray(names, dtype=object)


def _feature_names(input_features: Any, n_columns: int) -> list[str]:
    """Build a list of names for ``n_columns`` input features."""
    if input_features is None:
        return [f"cat_{i}" for i in range(n_columns)]
    features = list(input_features)
    if len(features) != n_columns:
        raise ValueError(
            f"input_features has {len(features)} elements, expected {n_columns}."
        )
    return features


class SplitCategoricalEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Per-column categorical encoder with an optional high-cardinality mode.

    At :meth:`fit`/:meth:`fit_transform` time the cardinality of each column is
        computed from the actual data. Columns whose cardinality exceeds
        ``high_cardinality_threshold`` are encoded with the
        ``high_cardinality_encoding`` strategy; the rest — with the ``encoding``
        strategy. If the threshold is not set (``None``), all columns are encoded
        with the ``encoding`` strategy (default behavior).

        Encoder outputs are concatenated horizontally in the order "default columns
        first, then high-cardinality columns". The split depends only on the
        training data, so the strategy decision does not "peek" into validation.

        :meth:`fit_transform` delegates to the internal encoders' own
        ``fit_transform``: for target encoding this includes cross-fitting, so the
        training rows are encoded without their own ``y``, and the full statistics
        are stored on the internal encoder for the subsequent :meth:`transform`.

        Args:
            encoding: Encoding strategy for regular columns.
            high_cardinality_threshold: Cardinality threshold (``>= 0``); columns
                whose number of unique values is strictly greater than the threshold
                are considered high-cardinality. ``None`` — mode disabled.
            high_cardinality_encoding: Strategy for high-cardinality columns.
                Must be set together with ``high_cardinality_threshold``.
            hashing_n_components: Number of binary columns for hashing encoding.
            target_encoding_smoothing: Smoothing parameter of target encoding.
            target_encoding_fallback: Fallback value of target encoding
                (``None`` — global mean).
            random_state: Seed for deterministic hashing encoding.
    """

    def __init__(
        self,
        encoding: EncodingStrategy = "one_hot",
        high_cardinality_threshold: int | None = None,
        high_cardinality_encoding: EncodingStrategy | None = None,
        hashing_n_components: int = 16,
        target_encoding_smoothing: float = 20.0,
        target_encoding_fallback: float | None = None,
        random_state: int | None = 42,
    ):
        self.encoding = encoding
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.hashing_n_components = hashing_n_components
        self.target_encoding_smoothing = target_encoding_smoothing
        self.target_encoding_fallback = target_encoding_fallback
        self.random_state = random_state

    def _make_encoder(self, strategy: EncodingStrategy) -> BaseEstimator:
        """Create an encoder by strategy name.

        The branches cover all valid values of ``EncodingStrategy``;
        strategy validation is performed in :meth:`fit`, so this branch is
        unreachable for correct calls.
        """
        if strategy == "one_hot":
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if strategy == "ordinal":
            return OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        if strategy == "target":
            return TargetEncodingTransformer(
                smoothing=self.target_encoding_smoothing,
                fallback=self.target_encoding_fallback,
            )
        if strategy == "frequency":
            return FrequencyEncodingTransformer()
        if strategy == "hashing":
            return HashingEncodingTransformer(
                n_components=self.hashing_n_components,
                random_state=self.random_state,
            )
        raise AssertionError(
            f"Unreachable: strategy {strategy!r} must be validated in fit(), "
            f"expected one of {_VALID_ENCODING_STRATEGIES}."
        )

    def _prepare(self, X: Any) -> np.ndarray:
        """Validate parameters, split columns, and create the encoders.

        The shared part of :meth:`fit` and :meth:`fit_transform`: strategy
        validation, cardinality computation, the default/HC column split, and
        the creation of the internal encoders (without training — the calling
        method trains them through ``fit``/``fit_transform``).

        Args:
            X: Matrix of categorical features.

        Returns:
            The two-dimensional numeric ``X`` array (after ``_as_2d``).

        Raises:
            ValueError: If an unknown encoding strategy is set or the
                high-cardinality mode is misconfigured.
        """
        if self.encoding not in _VALID_ENCODING_STRATEGIES:
            raise ValueError(
                f"Unknown encoding strategy: {self.encoding!r}. "
                f"Expected one of {_VALID_ENCODING_STRATEGIES}."
            )
        if (
            self.high_cardinality_encoding is not None
            and self.high_cardinality_encoding not in _VALID_ENCODING_STRATEGIES
        ):
            raise ValueError(
                f"Unknown high_cardinality_encoding: "
                f"{self.high_cardinality_encoding!r}. "
                f"Expected one of {_VALID_ENCODING_STRATEGIES}."
            )
        if (
            self.high_cardinality_threshold is not None
            and self.high_cardinality_encoding is None
        ):
            raise ValueError(
                "high_cardinality_encoding must be set when "
                "high_cardinality_threshold is provided."
            )

        X_arr = _as_2d(X)
        n_columns = X_arr.shape[1]
        cardinalities = [len(np.unique(X_arr[:, col])) for col in range(n_columns)]
        self.cardinalities_ = cardinalities

        threshold = self.high_cardinality_threshold
        default_columns: list[int] = []
        hc_columns: list[int] = []
        if threshold is not None:
            for col, card in enumerate(cardinalities):
                if card > threshold:
                    hc_columns.append(col)
                else:
                    default_columns.append(col)
        else:
            default_columns = list(range(n_columns))
        self.default_columns_ = default_columns
        self.hc_columns_ = hc_columns

        # The internal encoders are created here, in fit()/fit_transform(),
        # unconditionally — including for an empty column set (n_columns=0):
        # the default_encoder_/hc_encoder_ attributes are guaranteed to exist
        # after training. They are trained/applied only for the non-empty
        # default_columns_/hc_columns_ (see fit/fit_transform).
        self.default_encoder_ = self._make_encoder(self.encoding)
        self.hc_encoder_: BaseEstimator | None = (
            self._make_encoder(self.high_cardinality_encoding)
            if self.high_cardinality_encoding is not None
            else None
        )

        logger.debug(
            "SplitCategoricalEncoder: %d default columns, %d high-cardinality "
            "columns (threshold=%r), cardinalities=%s",
            len(default_columns),
            len(hc_columns),
            threshold,
            cardinalities,
        )
        return X_arr

    @staticmethod
    def _concat_parts(
        parts: list[np.ndarray | sp.spmatrix], n_rows: int
    ) -> np.ndarray | sp.spmatrix:
        """Concatenate the default- and HC-encoder outputs into one matrix.

        If at least one part is sparse (hashing encoding), the parts are
        concatenated via ``sp.hstack`` (CSR); otherwise — via ``np.hstack``.

        Args:
            parts: Encoder outputs in concatenation order.
            n_rows: Number of rows of the input matrix (for an empty result).

        Returns:
            A numeric matrix (dense or sparse).
        """
        if not parts:
            return np.empty((n_rows, 0), dtype=np.float64)
        if any(sp.issparse(part) for part in parts):
            # scipy-stubs do not describe heterogeneous sequences
            # (ndarray | spmatrix) for hstack: cast the blocks to a common type.
            sparse_blocks = cast(list[sp.spmatrix], parts)
            return sp.hstack(sparse_blocks).tocsr()
        # All parts are dense (the sparse branch above) — a plain np.hstack.
        dense_blocks = cast(list[np.ndarray], parts)
        return np.hstack(dense_blocks)

    def fit(self, X: Any, y: Any = None) -> SplitCategoricalEncoder:
        """Split columns by cardinality and fit the encoders.

        The internal encoders (``default_encoder_``/``hc_encoder_``) are created
        only in :meth:`fit`/:meth:`fit_transform`: attributes with a trailing
        underscore store the fitted state, so the constructor only saves the
        parameters, while ``clone()``/``set_params()``/``get_params()`` work
        through the base ``BaseEstimator`` implementation without overrides.

        Args:
            X: Matrix of categorical features (rows after imputation).
            y: Target variable (required only for target encoding).

        Returns:
            Fitted encoder.

        Raises:
            ValueError: If an unknown encoding strategy is set or the
                high-cardinality mode is misconfigured.
        """
        X_arr = self._prepare(X)
        if self.default_columns_:
            self.default_encoder_.fit(X_arr[:, self.default_columns_], y)

        if self.hc_columns_:
            # The invariant is guaranteed by the validation in _prepare: when
            # the threshold is set the HC strategy must be given, so
            # hc_encoder_ was created.
            assert self.hc_encoder_ is not None
            self.hc_encoder_.fit(X_arr[:, self.hc_columns_], y)
        return self

    def fit_transform(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> np.ndarray | sp.spmatrix:
        """Encode columns through the ``fit_transform`` of the internal encoders.

        Delegates to :meth:`TargetEncodingTransformer.fit_transform`
        (cross-fitting) for target encoding; for the other strategies, the
        standard ``fit_transform`` (equivalent to ``fit`` + ``transform``).
        The internal encoders stay fitted on the full data, so the subsequent
        :meth:`transform` (new data) works in the usual way.

        Args:
            X: Matrix of categorical features.
            y: Target variable (required for target encoding).
            **fit_params: Not used (accepted for compatibility with sklearn
                pipelines).

        Returns:
            A numeric matrix (dense or sparse) combining the outputs of the
            default and high-cardinality encoders.

        Raises:
            ValueError: If an unknown encoding strategy is set or the
                high-cardinality mode is misconfigured.
        """
        X_arr = self._prepare(X)
        # An empty column set (n_columns=0): the encoders are already created
        # in _prepare (the invariant above), but there is nothing to
        # train/apply — return an empty (n_rows, 0) matrix without touching
        # the encoders.
        if not self.default_columns_ and not self.hc_columns_:
            return np.empty((X_arr.shape[0], 0), dtype=np.float64)
        parts: list[np.ndarray | sp.spmatrix] = []
        if self.default_columns_:
            parts.append(
                self.default_encoder_.fit_transform(X_arr[:, self.default_columns_], y)
            )
        if self.hc_columns_:
            # The invariant is guaranteed by the validation in _prepare: when
            # the threshold is set the HC strategy must be given, so
            # hc_encoder_ was created.
            assert self.hc_encoder_ is not None
            parts.append(self.hc_encoder_.fit_transform(X_arr[:, self.hc_columns_], y))
        return self._concat_parts(parts, X_arr.shape[0])

    def transform(self, X: Any) -> np.ndarray | sp.spmatrix:
        """Encode columns with the selected strategies and concatenate the result.

        Uses the statistics stored on the internal encoders during
        ``fit``/``fit_transform`` (for target encoding — the full-sample
        statistics, without cross-fitting).

        Args:
            X: Matrix of categorical features.

        Returns:
            Numeric matrix (dense or sparse) combining the outputs of the
            default and high-cardinality encoders.
        """
        check_is_fitted(self, attributes=["default_columns_"])
        X_arr = _as_2d(X)
        parts: list[np.ndarray | sp.spmatrix] = []
        if self.default_columns_:
            parts.append(
                self.default_encoder_.transform(X_arr[:, self.default_columns_])
            )
        if self.hc_columns_ and self.hc_encoder_ is not None:
            parts.append(self.hc_encoder_.transform(X_arr[:, self.hc_columns_]))
        return self._concat_parts(parts, X_arr.shape[0])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names in concatenation order."""
        check_is_fitted(self, attributes=["default_columns_"])
        n_columns = len(self.default_columns_) + len(self.hc_columns_)
        base = _feature_names(input_features, n_columns)
        default_names = list(self.default_columns_)
        hc_names = list(self.hc_columns_)
        names: list[str] = []
        if default_names:
            names.extend(
                self.default_encoder_.get_feature_names_out(
                    [base[i] for i in default_names]
                )
            )
        if hc_names and self.hc_encoder_ is not None:
            names.extend(
                self.hc_encoder_.get_feature_names_out([base[i] for i in hc_names])
            )
        return np.asarray(names, dtype=object)


class ColumnNameSelector:
    """Serializable column selector by names for ``ColumnTransformer``.

    Replaces positional index lists when assembling the preprocessor:
    ``ColumnTransformer`` treats a list of ``int`` as a positional slice and
    ignores the ``pd.DataFrame`` column names, so prediction on a reordered
    DataFrame silently applies the encoder to numerical columns and the scaler
    to categorical ones.

    The selector is a picklable object (an ordinary class with plain attributes,
    not a closure), so it survives ``trainer.save()``/pickle correctly.
    On each call (sklearn resolves callable selectors at the
    ``fit_transform``/``transform`` stage) the names are mapped to positions
    against the actual input ``X``:

    * ``pd.DataFrame`` — mapping by column name; column order does not
      matter, missing or duplicated columns raise an explicit error;
    * any other inputs (``np.ndarray``, ``pd.Series``, etc.) — positional
      fallback by training order (``feature_names``).

    Args:
        feature_names: Full list of feature names in training order.
        column_names: Column names that this selector must select.

    Raises:
        ValueError: If for a ``pd.DataFrame`` some of the ``column_names`` are
            missing or occur more than once, or a name cannot be resolved
            against the training order for non-DataFrame inputs.
    """

    def __init__(self, feature_names: list[str], column_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.column_names = list(column_names)

    def __call__(self, X: Any) -> list[int]:
        if isinstance(X, pd.DataFrame):
            return _resolve_column_positions_by_name(X, self.column_names)
        return _resolve_column_positions_by_order(self.feature_names, self.column_names)


def _resolve_column_positions_by_name(
    X: pd.DataFrame, column_names: list[str]
) -> list[int]:
    """Map column names to positions in the actual DataFrame.

    Args:
        X: Input DataFrame against which the mapping is performed.
        column_names: Column names to select.

    Returns:
        List of positions of the selected columns in ``column_names`` order.

    Raises:
        ValueError: If some of the names are missing from ``X`` or occur more
            than once in ``X`` (ambiguous selection by name).
    """
    missing = [name for name in column_names if name not in X.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in the input DataFrame. "
            f"Available columns: {list(X.columns)}."
        )
    duplicated = [name for name in column_names if list(X.columns).count(name) > 1]
    if duplicated:
        raise ValueError(
            f"Column(s) {duplicated} appear more than once in the input "
            "DataFrame; ambiguous selection by name is not supported."
        )
    # After the uniqueness check, index() gives the exact column position
    # (pandas-stubs: get_loc may return a slice/mask for duplicates).
    return [list(X.columns).index(name) for name in column_names]


def _resolve_column_positions_by_order(
    feature_names: list[str], column_names: list[str]
) -> list[int]:
    """Map column names to positions by training order.

    Used as a fallback for inputs without column names (``np.ndarray``):
    the position of a name is determined by its place in ``feature_names``.

    Args:
        feature_names: Full list of feature names in training order.
        column_names: Column names to select.

    Returns:
        List of positions of the selected columns in ``column_names`` order.

    Raises:
        ValueError: If a name does not occur in ``feature_names`` exactly once.
    """
    positions: list[int] = []
    for name in column_names:
        occurrences = feature_names.count(name)
        if occurrences != 1:
            raise ValueError(
                f"Cannot resolve column {name!r} by training order: it occurs "
                f"{occurrences} time(s) in feature_names {feature_names}."
            )
        positions.append(feature_names.index(name))
    return positions


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Automatically classify columns into numerical and categorical.

    Columns with dtype ``object``, ``category`` and ``bool`` are considered
    categorical; numerical columns have a dtype from the ``number`` family.

    Args:
        X: Feature DataFrame (without the target variable).

    Returns:
        A tuple ``(categorical_features, numerical_features)`` — lists of
        column names of each type.

    Note:
        The function expects exactly ``pd.DataFrame``; for ``np.ndarray``
        without column names auto-detection is impossible (documented limitation).

    Note:
        Classification is based on the column ``dtype``. This intentionally
        differs from the value-based detection in ``oversampling._is_categorical_col``:
        here the preprocessor handles data before oversampling, so ``bool`` and
        'numeric strings' (``'10','20'``) are treated as categories (one-hot).
        The oversampler receives already encoded numerical features and applies
        its own value-based logic for standalone calls.
    """
    categorical = X.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    return categorical, numerical


def _clean_uninformative_features(
    X: pd.DataFrame,
    max_nan_ratio: float = 0.95,
    min_unique: int = 2,
) -> pd.DataFrame:
    """Drop uninformative columns: empty and strictly constant ones.

    A column is considered uninformative if at least one of the following
    conditions holds:

    1. Missing ratio ``>= max_nan_ratio`` (0.95 by default): imputation with
       mean/median replaces almost all values with a single constant; after
       scaling the column either carries no information (zero contribution to
       distances) or biases the distance matrix of metric models
       (SVR/KNN/Ridge).
    2. Number of unique values excluding NaN ``< min_unique`` (2 by default):
       a strictly constant column has zero variance after imputation and
       scaling; for ``RobustScaler`` division by a zero IQR (``q75 == q25``)
       yields NaN/±inf, which breaks training. A constant categorical column
       is useless even after one-hot encoding (a constant binary column), so
       it is also dropped.

    The decision is made **only from the training data** (no leakage) and is
    fixed at the fitting stage: a column that is empty on train but
    informative at inference will be dropped in production too (documented
    behavior). This is consistent with ``detect_feature_types``, which is also
    computed on the full data. The function is idempotent: a repeated call on
    already cleaned data does not change the result.

    Guard rules:

    * A DataFrame without columns (``X.shape[1] == 0``) or without rows
      (``X.shape[0] == 0``) is returned unchanged — cleaning does not fail.
    * The single column is never dropped, even if it is constant or empty:
      otherwise an empty feature matrix would remain.
    * If **all** columns fall under removal (with ``P >= 2``), a
      :class:`ValueError` is raised: an empty feature set is not allowed. This
      is analogous to ``FeatureSelector.min_features_guard``
      (``feature_selection.py``), but at the data preparation stage the error
      is explicit rather than a silent passthrough.

    Args:
        X: Feature DataFrame (without the target variable).
        max_nan_ratio: Missing ratio threshold in ``(0, 1]``: columns with a
            NaN ratio ``>=`` the threshold are dropped. ``1.0`` — only fully
            empty columns are dropped.
        min_unique: Minimum number of unique values excluding NaN (``>= 1``)
            at which a column is considered informative. The value ``2``
            (default) drops strictly constant columns; the value ``1`` keeps
            them (useful only together with the missing-ratio rule).

    Returns:
        A copy of ``X`` without the uninformative columns (the original object
        is not mutated). If there is nothing to drop, the original ``X`` is
        returned.

    Raises:
        TypeError: If ``X`` is not a ``pd.DataFrame``. For ``np.ndarray``
            without column names, cleaning is impossible (documented
            limitation, as with :func:`detect_feature_types`).
        ValueError: If the thresholds are invalid or all columns fall under
            removal (an empty feature set).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "_clean_uninformative_features expects a pandas.DataFrame, got "
            f"{type(X).__name__}. Column names are required to identify "
            "uninformative features (same limitation as detect_feature_types)."
        )
    if not 0.0 < max_nan_ratio <= 1.0:
        raise ValueError(f"max_nan_ratio must be in (0, 1], got {max_nan_ratio!r}.")
    if min_unique < 1:
        raise ValueError(f"min_unique must be >= 1, got {min_unique!r}.")

    n_cols = X.shape[1]
    # An empty DataFrame (0 rows / 0 columns) is handled gracefully without
    # failures: there is nothing to clean.
    if n_cols == 0 or X.shape[0] == 0:
        return X
    # The single column is always kept, even if it is constant or empty:
    # otherwise an empty feature matrix would remain.
    if n_cols == 1:
        return X

    # 1. Columns with a missing ratio >= max_nan_ratio (including fully empty ones).
    nan_ratios = X.isna().mean()
    drop_cols = list(nan_ratios[nan_ratios >= max_nan_ratio].index)

    # 2. Strictly constant columns: exactly one unique value excluding NaN (for a
    # fully empty column nunique() == 0, but such columns are already covered
    # by the missing-ratio rule at max_nan_ratio <= 1.0).
    for col in X.columns:
        if col not in drop_cols and X[col].dropna().nunique() < min_unique:
            drop_cols.append(col)

    if not drop_cols:
        logger.debug(
            "No uninformative columns to drop (max_nan_ratio=%s, min_unique=%d, "
            "n_columns=%d)",
            max_nan_ratio,
            min_unique,
            n_cols,
        )
        return X
    if len(drop_cols) >= n_cols:
        raise ValueError(
            "All columns are uninformative and would be dropped: "
            f"{drop_cols}. Refusing to produce an empty feature matrix."
        )

    logger.warning("Dropping uninformative/empty columns: %s", drop_cols)
    logger.debug(
        "Cleanup thresholds: max_nan_ratio=%s, min_unique=%d; dropped %d of %d columns",
        max_nan_ratio,
        min_unique,
        len(drop_cols),
        n_cols,
    )
    return X.drop(columns=drop_cols)


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

        * ``'one_hot'`` (default) — ``OneHotEncoder``;
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
            encoding: Categorical encoding strategy: ``'one_hot'`` (default),
                ``'ordinal'``, ``'target'``, ``'frequency'`` or ``'hashing'``.
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
                ``high_cardinality_encoding`` strategy. ``None`` (default) —
                automatic mode disabled.
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
                hashing encoding (memory savings).
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

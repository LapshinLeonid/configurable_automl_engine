"""Categorical encoders of the preprocessing.

Holds the encoders that the preprocessor applies to categorical columns —
:class:`TargetEncodingTransformer`, :class:`FrequencyEncodingTransformer`,
:class:`HashingEncodingTransformer` and :class:`SplitCategoricalEncoder` (the
per-column split between the main and the high-cardinality strategy) — and
their helpers (``_to_string_array`` and the input checks). The fitted
encoders are pickled inside saved models, so their module path is part of the
artifact format (``trainer_state.ARTIFACT_FORMAT_VERSION``). The module must
not import the other preprocessing modules.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.utils.validation import check_is_fitted

from configurable_automl_engine.common.definitions import EncodingStrategy

logger = logging.getLogger(__name__)


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
            # sparse_output=True (the default of OneHotEncoder): the
            # pre-processed matrix stays sparse when its density is below
            # sparse_threshold, which prevents the dense n×n float blow-up on
            # high-cardinality columns (an ID-like column was the worst case:
            # it expanded into n binary columns and the dense float64 output
            # allocated n×n floats per fold). Algorithms that reject sparse
            # input (GPR/Isotonic/ARD) get a dense matrix via
            # build_preprocessor(force_dense_output=True).
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
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

        If at least one part is sparse (sparse one-hot or hashing encoding),
        the parts are concatenated via ``sp.hstack`` (CSR); otherwise — via
        ``np.hstack``.

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

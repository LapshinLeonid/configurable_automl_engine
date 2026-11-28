"""Shared type aliases and structural protocols.

The aliases name the data shapes and callables that cross module boundaries:
``Frame`` and ``Target`` for the normalized feature matrix and target vector,
``FrameInput``/``TargetInput`` for the looser shapes accepted at the public
entry points, ``NamesArray`` for the object arrays of feature names,
``Scorer``/``MetricFn`` for the two callable shapes of the metric layer, and
the ``Params*``/``JSON*`` families for parameter mappings and serialized
payloads. The protocols ``Estimator`` and ``Predictor`` describe the minimal
fitted-model interface the trainer, the tuner and the ensemble rely on.

Keeping the aliases in the shared layer lets the higher-level modules stay
free of explicit ``Any`` annotations without repeating the same unions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd
from automl_contracts.protocols import ScorerCallable
from scipy import sparse as sp

from configurable_automl_engine.common.shared_frame import SharedDataFrame

__all__ = [
    "Array",
    "BoolArray",
    "DataFrame",
    "Estimator",
    "EstimatorFactory",
    "FloatArray",
    "Frame",
    "FrameInput",
    "IntArray",
    "JSONDict",
    "MetricFn",
    "NamesArray",
    "Params",
    "ParamsInput",
    "Predictor",
    "Scorer",
    "Series",
    "SparseMatrix",
    "Target",
    "TargetInput",
]

#: A NumPy array with any element type.
Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.number]]

#: A NumPy array of float64 values (predictions, metrics, targets).
FloatArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

#: A NumPy array of integer values (bins, codes, indices).
IntArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.intp]]

#: A boolean NumPy array (row masks and support masks).
BoolArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]

#: A NumPy array of feature names (object dtype, not numeric).
NamesArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.object_]]

#: A scipy sparse matrix of float values. The preprocessing pipeline emits
#: ``float64`` sparse matrices, so that element type is the supported one.
SparseMatrix: TypeAlias = sp.spmatrix[np.float64]

#: A pandas frame (the aliases module is the single place that names the
#: library's default generic arguments).
DataFrame: TypeAlias = pd.DataFrame

if TYPE_CHECKING:
    #: A pandas series with any element type, as returned by ``DataFrame``
    #: columns. The element type is open, so the annotation spells ``Any``;
    #: like the open argument list of ``MetricFn`` it is confined to this
    #: module (see the mypy override in ``pyproject.toml``).
    Series: TypeAlias = pd.Series[Any]
else:
    # pandas 3 removed runtime generic subscripting of Series.
    Series = pd.Series

#: A feature matrix in its normalized in-memory form.
Frame: TypeAlias = pd.DataFrame | Array

#: A target vector in its normalized in-memory form.
Target: TypeAlias = Series | Array

#: A feature matrix as accepted at the public entry points: a frame, a
#: shared-memory frame wrapped for multiprocessing, a NumPy array or a
#: nested sequence of numbers.
FrameInput: TypeAlias = Frame | SharedDataFrame | Sequence[Sequence[float]]

#: A target as accepted at the public entry points: a vector, a single-column
#: frame, a shared-memory frame or a sequence of numbers.
TargetInput: TypeAlias = Target | DataFrame | SharedDataFrame | Sequence[float]

#: A raw metric function: ``fn(y_true, y_pred, ...) -> float``. Individual
#: metrics may require extra keyword or positional arguments (e.g. the global
#: range of ``global_nrmse``), so the exact signature is metric-specific.
MetricFn: TypeAlias = Callable[..., float]

#: An sklearn-style scorer object called as ``scorer(estimator, X, y)``.
Scorer: TypeAlias = ScorerCallable

#: A JSON-serializable payload built inside the engine and handed to a
#: serializer. It is an output-only shape, never a public input contract, so
#: the heterogeneous value type is intentional.
JSONDict: TypeAlias = dict[str, object]

#: A mutable mapping of model or pipeline parameters assembled and updated
#: inside the engine.
Params: TypeAlias = dict[str, object]

#: A parameter mapping accepted at the public entry points. A ``Mapping`` is
#: covariant in the value type, so ``dict[str, int]`` and ``dict[str, float]``
#: are accepted; the engine copies it into a mutable :data:`Params`.
ParamsInput: TypeAlias = Mapping[str, object]


@runtime_checkable
class Predictor(Protocol):
    """A fitted model that produces predictions for a feature matrix."""

    def predict(self, X: Frame) -> np.ndarray:
        """Return the prediction vector for ``X``."""
        ...  # pragma: no cover - protocol definition


class Estimator(Predictor, Protocol):
    """A model that can be fitted and then used for predictions."""

    def fit(self, X: Frame, y: Target, **kwargs: object) -> Self:
        """Fit the model on ``X`` and ``y`` and return the fitted model."""
        ...  # pragma: no cover - protocol definition


#: A callable that builds an unfitted estimator from parameters.
EstimatorFactory: TypeAlias = Callable[..., Estimator]

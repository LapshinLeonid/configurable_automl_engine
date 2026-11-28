"""Tests for the shared type aliases and structural protocols."""

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.common.types import (
    DataFrame,
    FrameInput,
    ParamsInput,
    Predictor,
    Series,
    SparseMatrix,
    TargetInput,
)
from configurable_automl_engine.experiment.ensemble import (
    EnsembleMember,
    EnsembleModel,
)
from configurable_automl_engine.trainer import ModelTrainer


class _DummyPredictor:
    """Minimal object exposing the ``Predictor`` interface."""

    def predict(self, X: FrameInput) -> np.ndarray:
        """Return one zero prediction per input row."""
        return np.zeros(len(X))


def test_runtime_aliases_point_to_pandas_objects() -> None:
    """The runtime fallbacks resolve to the pandas classes."""
    assert Series is pd.Series
    assert DataFrame is pd.DataFrame


def test_sparse_matrix_alias_resolves_to_the_scipy_class() -> None:
    """The sparse alias stays a plain class at runtime for every scipy release."""
    assert SparseMatrix is sp.spmatrix


def test_public_input_aliases_accept_documented_shapes() -> None:
    """The public input aliases are importable and usable in annotations."""
    frame: FrameInput = pd.DataFrame({"a": [1.0, 2.0]})
    target: TargetInput = [1.0, 2.0]
    assert isinstance(frame, pd.DataFrame)
    assert list(target) == [1.0, 2.0]


def test_model_trainer_satisfies_predictor_protocol() -> None:
    """A ``ModelTrainer`` structurally matches the ``Predictor`` protocol."""
    assert isinstance(ModelTrainer(), Predictor)


def test_ensemble_model_satisfies_predictor_protocol() -> None:
    """An ``EnsembleModel`` structurally matches the ``Predictor`` protocol."""
    ensemble = EnsembleModel(
        metric_user="rmse",
        metric_direction="minimize",
        members=[EnsembleMember(name="dummy", score=1.0, weight=1.0, rank=1)],
        predictors={"dummy": _DummyPredictor()},
    )
    assert isinstance(ensemble, Predictor)


def test_hyperparams_accept_read_only_mapping() -> None:
    """A read-only mapping is copied into a mutable parameter dict."""
    params: ParamsInput = MappingProxyType({"alpha": 0.5})
    trainer = ModelTrainer(hyperparams=params)
    assert trainer.hyperparams == {"alpha": 0.5}
    assert not isinstance(trainer.hyperparams, MappingProxyType)


def test_hyperparams_rejects_non_mapping() -> None:
    """A non-mapping hyperparameters value is rejected."""
    with pytest.raises(TrainingError, match="hyperparams must be a dictionary"):
        ModelTrainer(hyperparams=[("alpha", 0.5)])  # type: ignore[arg-type]

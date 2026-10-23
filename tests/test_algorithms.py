import pytest
import pandas as pd
import numpy as np
from configurable_automl_engine.trainer import ModelTrainer

# List of all supported abbreviations from your models.py
ALGORITHMS = [
    "elasticnet",
    "dt",
    "sgd",
    "knn",
    "gpr",
    "svr",
    "isotonic",
    "ard",
    "rf",
    "glm",
    "xgboost",
    "ridge_regression",
]

# List of all supported algorithms
ALGORITHMS = [
    "elasticnet",
    "dt",
    "sgd",
    "knn",
    "gpr",
    "svr",
    "isotonic",
    "ard",
    "rf",
    "glm",
    "xgboost",
    "ridge_regression",
]


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_every_algorithm_can_fit_and_predict(algo, small_dataset):
    """
    Test that every algorithm can fit and predict.
    Uses the centralized small_dataset fixture.
    """
    # 1. Prepare data from the fixture
    target_column = "target"
    X = small_dataset.drop(columns=[target_column])
    y = small_dataset[target_column]
    # 2. Initialize the trainer
    # IMPORTANT: Make sure ModelTrainer accepts the 'algorithm' argument
    trainer = ModelTrainer(algorithm=algo)
    # 3. Train
    trainer.fit(X, y)
    # 4. Predict
    preds = trainer.predict(X)
    # 5. Checks (Assertions)
    # 5. Checks (Assertions)
    assert preds.shape[0] == y.shape[0], (
        f"Algorithm {algo}: prediction size mismatch"
    )
    assert not np.isnan(preds).any(), f"Algorithm {algo} returned NaN in predictions"

    # Check that the computed metric is available in the universal val_score attribute
    assert hasattr(trainer, "val_score"), (
        f"Algorithm {algo} did not set the val_score attribute"
    )
    assert trainer.val_score is not None, (
        f"Algorithm {algo}: val_score is empty after training"
    )

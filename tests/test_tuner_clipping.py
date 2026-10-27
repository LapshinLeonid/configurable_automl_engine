import pytest
import pandas as pd
import numpy as np
import math
from unittest.mock import MagicMock, patch
from sklearn.linear_model import Ridge
from configurable_automl_engine.hpo.default_spaces import FloatSpace

from configurable_automl_engine.hpo.default_spaces import (
    SearchSpaceEntry,
    clip_search_space,
    IntSpace,
)
from configurable_automl_engine.tuner import optimize

from configurable_automl_engine.common.validation_utils import get_effective_train_size
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.trainer import ModelTrainer, train_model, TrainingError


def test_clip_search_space_logic():
    """
    Checks the basic clipping logic:
    1. n_neighbors is limited to n_samples - 1
    2. min_samples_leaf is limited to n_samples
    3. Low is pulled up to High if High became smaller than the original Low
    """
    space = {
        "n_neighbors": SearchSpaceEntry.model_validate([5, 100, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([10, 20, "int"]),
    }

    # n_samples = 3
    # n_neighbors: limit = 2. New high = 2. New low = min(5, 2) = 2.
    # min_samples_leaf: limit = 3. New high = 3. New low = min(10, 3) = 3.
    clipped = clip_search_space(space, n_samples=3)

    assert clipped["n_neighbors"].high == 2
    assert clipped["n_neighbors"].low == 2
    assert isinstance(clipped["n_neighbors"].config, IntSpace)

    assert clipped["min_samples_leaf"].high == 3
    assert clipped["min_samples_leaf"].low == 3  # Fixed: low is now 3, not 1



def test_optimize_integration_clipping(monkeypatch):
    """
    Checks that optimize() applies clipping to the passed parameters.
    """
    # Create 20 rows to avoid division issues during validation
    X = pd.DataFrame(np.random.rand(20, 2), columns=["a", "b"])
    y = pd.Series(np.random.rand(20))

    # Intentionally inflated bounds
    overrides = {
        "nearest_neighbors_regression": {
            "n_neighbors": SearchSpaceEntry.model_validate([15, 100, "int"]),
        }
    }

    # 1. Mock the metric so it always returns 0.5 (avoid NaN)
    monkeypatch.setattr(
        "configurable_automl_engine.hpo.evaluation.get_scorer_object",
        lambda name, global_y=None: lambda model, X, y: 0.5,
    )

    # 2. Mock model creation to check the incoming n_neighbors
    def mock_create_model(algo, **params):
        if algo == "xgboosting":  # Skip initialization at the start of optimize
            return Ridge()
        # For KNN the limit is n_samples-1 = 19
        if algo == "nearest_neighbors_regression" and params.get("n_neighbors", 0) > 19:
            pytest.fail(f"n_neighbors {params['n_neighbors']} exceeds limit 19")
        return Ridge()

    monkeypatch.setattr(
        "configurable_automl_engine.hpo.runner.create_model", mock_create_model
    )

    # 3. Mock cross-validation (in case the strategy changes)
    monkeypatch.setattr(
        "configurable_automl_engine.hpo.evaluation.model_selection.cross_val_score",
        lambda *a, **kw: [0.5, 0.5],
    )

    # Run the optimization
    opt_res = optimize(
        "nearest_neighbors_regression",
        X,
        y,
        n_trials=2,
        space_overrides=overrides,
        validation_strategy="train_test_split",
        random_state=42,
    )
    assert opt_res is not None
    _, best_params, _ = opt_res.model, opt_res.params, opt_res.score


    # With 20 samples the n_neighbors limit is 19.
    # The original range [15, 100] must become [15, 19].
    assert 15 <= best_params["n_neighbors"] <= 19


def test_clipping_preserves_original_space():
    """Ensure clipping does not mutate the original objects (Deep Copy)."""
    original_entry = SearchSpaceEntry.model_validate([10, 100, "int"])
    space = {"n_neighbors": original_entry}

    _ = clip_search_space(space, n_samples=5)

    # The original object in the space dict must not change
    assert original_entry.high == 100
    assert original_entry.low == 10


def test_clip_extreme_small_data():
    """Checks clipping with n_samples = 1."""
    space = {"n_neighbors": SearchSpaceEntry.model_validate([5, 10, "int"])}
    # n=1, limit = max(1, 1-1) = 1.
    clipped = clip_search_space(space, n_samples=1)
    assert clipped["n_neighbors"].low == 1
    assert clipped["n_neighbors"].high == 1


def test_clip_ignores_unrelated_params():
    """Checks that clipping leaves parameters outside the restriction list untouched."""
    space = {
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.1, "float"]),
    }
    clipped = clip_search_space(space, n_samples=5)
    assert clipped["max_depth"].high == 32
    assert clipped["learning_rate"].low == 0.01


def test_clip_preserves_types():
    """Checks that FloatSpace remains FloatSpace after clipping."""
    space = {"min_samples_leaf": SearchSpaceEntry.model_validate([0.1, 0.5, "float"])}
    # With n=10 the limit is 10.0. The bounds 0.1 and 0.5 must not change,
    # but it is important that the type stays FloatSpace.
    clipped = clip_search_space(space, n_samples=10)
    assert isinstance(clipped["min_samples_leaf"].config, FloatSpace)


def test_clip_handles_categorical():
    """Checks that categorical parameters pass through the filter unchanged."""
    space = {
        "weights": SearchSpaceEntry.model_validate(
            [["uniform", "distance"], "categorical"]
        )
    }
    clipped = clip_search_space(space, n_samples=10)
    assert clipped["weights"].dist_type == "categorical"
    assert "uniform" in clipped["weights"].config.options


@pytest.mark.parametrize(
    "n_total, strategy, n_folds, test_size, expected",
    [
        # K-Fold: floor(100 * (1 - 1/5)) = 80
        (100, ValidationStrategy.k_fold, 5, 0.2, 80),
        # K-Fold: floor(10 * (1 - 1/3)) = floor(6.66) = 6
        (10, "k_fold", 3, 0.2, 6),
        # LOO: 100 - 1 = 99
        (100, ValidationStrategy.loo, 5, 0.2, 99),
        # Train-Test Split: floor(100 * (1 - 0.25)) = 75
        (100, ValidationStrategy.train_test_split, 5, 0.25, 75),
        # Safety test (clamping) test_size: 1.5 -> 0.99. floor(100 * 0.01) = 1
        (100, ValidationStrategy.train_test_split, 5, 1.5, 1),
        # Edge case test: n=0
        (0, ValidationStrategy.k_fold, 5, 0.2, 0),
        # Minimum for Neff when n_total >= 2
        (2, ValidationStrategy.train_test_split, 5, 0.99, 1),
    ],
)
def test_get_effective_train_size_logic(
    n_total, strategy, n_folds, test_size, expected
):
    result = get_effective_train_size(n_total, strategy, n_folds, test_size)
    assert result == expected


# ──────────────────────────────────────────────────────────────────────
# 2. Tests for trainer.py (Refactoring ModelTrainer)
# ──────────────────────────────────────────────────────────────────────


def test_model_trainer_no_longer_splits_internally():
    """Checks that ModelTrainer now trains on all passed data."""
    df = pd.DataFrame({"feature1": np.random.rand(10), "target": np.random.rand(10)})

    # Initialization without test_size (it is no longer in the signature)
    trainer = ModelTrainer(algorithm="ridge")

    with patch.object(
        ModelTrainer, "_fit_internal", wraps=trainer._fit_internal
    ) as mock_fit:
        trainer.fit(df, y="target")

        # Check that _fit_internal received 10 rows (the whole df), not a part
        args, _ = mock_fit.call_args
        # args[0] is X_train
        assert len(args[0]) == 10
        assert trainer.val_score is not None


def test_train_model_facade_compatibility():
    """Checks that the train_model facade function does not fail without test_size."""
    X = pd.DataFrame(np.random.rand(10, 2))
    y = pd.Series(np.random.rand(10))

    # Call of the old API (params_or_metric is a dict of parameters)
    # test_size is now simply ignored inside or missing
    score = train_model(
        cfg_or_algo="ridge",
        metric_or_testsize="r2",
        params_or_metric={"alpha": 1.0},
        X=X,
        y=y,
    )
    assert isinstance(score, float)


# ──────────────────────────────────────────────────────────────────────
# 3. Tests for tuner.py (KNN & Space Clipping)
# ──────────────────────────────────────────────────────────────────────



@patch("configurable_automl_engine.hpo.search_space.clip_search_space")
@patch("configurable_automl_engine.hpo.runner.create_model")
@patch("optuna.create_study")
def test_optimize_calls_clipping_with_effective_size(
    mock_study, mock_create, mock_clip
):
    """Checks that during optimization clipping is called with the effective size."""
    X = pd.DataFrame(np.random.rand(20, 2))
    y = pd.Series(np.random.rand(20))

    # Simulate an external search-space config
    space_overrides = {"ridge": {"alpha": [0.1, 1.0]}}

    # Run optimization with the k_fold strategy (n_folds=2)
    # Neff = floor(20 * (1 - 1/2)) = 10
    try:
        optimize(
            algo_name="ridge",
            X=X,
            y=y,
            validation_strategy="k_fold",
            n_folds=2,
            n_trials=1,
            space_overrides=space_overrides,
        )
    except Exception:
        pass  # Only the clipping call matters here

    # Check that clip_search_space received n_samples=10 (Neff)
    mock_clip.assert_called_once()
    assert mock_clip.call_args[0][1] == 10



def test_model_trainer_raises_error_on_empty_data():
    """Checks that TrainingError is raised when empty data is passed."""
    trainer = ModelTrainer(algorithm="ridge")

    # Scenario 1: empty DataFrame
    empty_df = pd.DataFrame()
    empty_y = pd.Series(dtype=float)

    with pytest.raises(TrainingError, match="Data is empty"):
        trainer.fit(empty_df, empty_y)

    # Scenario 2: empty numpy array
    empty_X_np = np.array([]).reshape(0, 2)
    empty_y_np = np.array([])

    with pytest.raises(TrainingError, match="Data is empty"):
        trainer.fit(empty_X_np, empty_y_np)

    # Scenario 3: data with columns but no rows
    df_zero_rows = pd.DataFrame(columns=["a", "b"])
    y_zero_rows = pd.Series(dtype=float)

    with pytest.raises(TrainingError, match="Data is empty"):
        trainer.fit(df_zero_rows, y_zero_rows)


def test_get_effective_train_size_raises_on_invalid_string():
    """Checks that the function raises on an unknown string."""
    with pytest.raises(ValueError, match="Unknown validation strategy string"):
        get_effective_train_size(100, strategy="magic_split")


def test_get_effective_train_size_raises_on_invalid_type():
    """Checks that the function raises on an invalid type (e.g. None)."""
    with pytest.raises(ValueError, match="Unsupported validation strategy type"):
        get_effective_train_size(100, strategy=None)

    with pytest.raises(ValueError, match="Unsupported validation strategy type"):
        get_effective_train_size(100, strategy=123.45)


def test_clip_search_space_raises_on_negative_samples():
    """
    Checks that clip_search_space raises ValueError for a negative n_samples.
    """
    # Create a minimal valid search space for the test
    space = {"n_neighbors": SearchSpaceEntry.model_validate([1, 50, "int"])}

    # Scenario 1: negative value
    with pytest.raises(ValueError, match="must be positive"):
        clip_search_space(space, n_samples=-1)

    with pytest.raises(ValueError, match="Got -100"):
        clip_search_space(space, n_samples=-100)


def test_clip_search_space_raises_on_zero_samples():
    """Checks that 0 samples now raises an error instead of returning the space."""
    space = {"n_neighbors": SearchSpaceEntry.model_validate([1, 50, "int"])}

    with pytest.raises(ValueError, match="must be positive"):
        clip_search_space(space, n_samples=0)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Step-grid alignment after clipping
# ──────────────────────────────────────────────────────────────────────────────


def test_clip_aligns_high_to_step_grid():
    """A clipped upper bound is re-aligned to the step grid.

    Regression: ``n_estimators`` 50–1000 with a step of 100 clipped to
    50–300 produced ``(300 - 50) % 100 != 0``; Optuna silently shifted the
    upper bound to 250. The stored range must describe the grid Optuna
    actually searches — ``high`` is aligned to
    ``low + floor((high - low) / step) * step``.
    """
    space = {"min_samples_leaf": SearchSpaceEntry.model_validate([50, 1000, "int", 100])}
    clipped = clip_search_space(space, n_samples=300)
    entry = clipped["min_samples_leaf"]
    assert isinstance(entry.config, IntSpace)
    assert (entry.config.low, entry.config.high) == (50, 250)


def test_clip_preserves_aligned_high_when_divisible():
    """A clipped high that already lies on the step grid stays unchanged."""
    space = {"min_samples_leaf": SearchSpaceEntry.model_validate([50, 1000, "int", 50])}
    clipped = clip_search_space(space, n_samples=300)
    entry = clipped["min_samples_leaf"]
    # (300 - 50) % 50 == 0 -> high stays 300.
    assert (entry.config.low, entry.config.high) == (50, 300)


def test_clip_empty_step_grid_fixes_value():
    """A clipped range whose grid is empty collapses to a single value (fixed).

    The grid anchored at 50 with a step of 100 contains no value in (50, 130];
    Optuna could only ever sample 50, so the parameter must be fixed at 50
    instead of storing a misleading range.
    """
    space = {"min_samples_leaf": SearchSpaceEntry.model_validate([50, 130, "int", 100])}
    clipped = clip_search_space(space, n_samples=300)
    entry = clipped["min_samples_leaf"]
    assert isinstance(entry.config, IntSpace)
    assert (entry.config.low, entry.config.high) == (50, 50)


def test_clip_aligns_float_step_grid():
    """Float ranges with a step are aligned too (exact decimal arithmetic)."""
    space = {
        "min_samples_leaf": SearchSpaceEntry.model_validate([0.1, 1.0, "float", 0.3])
    }
    # n=300: the physical limit is far above the range, so only the alignment
    # applies: (1.0 - 0.1) / 0.3 = 3 exactly -> high stays 1.0.
    clipped = clip_search_space(space, n_samples=300)
    entry = clipped["min_samples_leaf"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.low == pytest.approx(0.1)
    assert entry.config.high == pytest.approx(1.0)


def test_clip_align_does_not_mutate_original_entry():
    """Step alignment must not mutate the source space (deep copy)."""
    original = SearchSpaceEntry.model_validate([50, 1000, "int", 100])
    space = {"min_samples_leaf": original}
    _ = clip_search_space(space, n_samples=300)
    assert (original.config.low, original.config.high) == (50, 1000)
    assert original.config.step == 100


def test_align_high_to_step_matches_optuna():
    """``align_high_to_step`` matches Optuna's internal high adjustment.

    Optuna narrows a stepped distribution whose range is not divisible by the
    step: ``high = low + floor((high - low) / step) * step``. The helper must
    reproduce the same values so the stored bounds describe the real search
    grid.
    """
    from configurable_automl_engine.hpo.default_spaces import (
        align_high_to_step,
    )

    # The exact scenario covered: 50-300 with a step of 100 -> 250.
    assert align_high_to_step(50, 300, 100) == 250
    # Divisible ranges stay unchanged.
    assert align_high_to_step(50, 300, 50) == 300
    # An empty grid collapses to the anchor value.
    assert align_high_to_step(50, 130, 100) == 50
    # Float steps use exact decimal arithmetic.
    assert align_high_to_step(0.1, 1.0, 0.3) == 1.0

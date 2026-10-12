import numpy as np
import pytest
import logging
from unittest.mock import MagicMock
from configurable_automl_engine.training_engine.metrics import (
    _rmse,
    _nrmse,
    direction_label,
    get_metric,
    is_error_metric,
    is_greater_better,
    get_scorer_object,
    to_sklearn_name,
    to_user_value,
    user_direction,
    NRMSEZeroRangeError,
    _global_nrmse,
    get_global_nrmse_scorer,
)


def test_nrmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    nrmse = get_metric("nrmse")
    val = nrmse(y_true, y_pred)

    # manual check
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    expected = rmse / (y_true.max() - y_true.min())

    assert np.isclose(val, expected, atol=1e-8)
    assert not is_greater_better("nrmse")


# Test RMSE
def test_rmse_calculation():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    # MSE = (0.5^2 + 0.5^2 + 0 + 1^2) / 4 = 1.5 / 4 = 0.375
    # RMSE = sqrt(0.375) approx 0.61237
    result = _rmse(y_true, y_pred)
    assert isinstance(result, float)
    assert result == pytest.approx(np.sqrt(0.375))


def test_nrmse_normal_case():
    y_true = np.array([0, 10])
    y_pred = np.array([0, 5])
    # RMSE = sqrt((0^2 + 5^2)/2) = sqrt(12.5) approx 3.535
    # Denom = 10 - 0 = 10
    # Result = 3.535 / 10 = 0.3535
    assert _nrmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5) / 10)


def test_nrmse_zero_range_coverage():
    """Covers lines 35 (Error class) and 53-57 (denom < 1e-6)"""
    y_true = np.array([1, 1, 1])  # Constant target, range = 0
    y_pred = np.array([1, 2, 3])

    # 1. Check the zero-range handling logic (lines 53-57)
    # The code returns float('inf') instead of raising an exception
    result = _nrmse(y_true, y_pred)
    assert result == float("inf")

    # 2. Cover the NRMSEZeroRangeError class declaration (line 35)
    # Just instantiate it so coverage tools count the line as executed
    err = NRMSEZeroRangeError("Target range is too small")
    assert isinstance(err, ValueError)
    assert str(err) == "Target range is too small"


# Test registries and helpers


def test_get_metric():
    # Check correct retrieval
    assert get_metric("rmse") == _rmse
    assert get_metric("NRMSE") == _nrmse

    # Check the KeyError exception
    with pytest.raises(KeyError, match="Metric 'unknown' not implemented"):
        get_metric("unknown")


@pytest.mark.parametrize(
    "metric_name, expected",
    [
        # 1. Custom registry: the direction is stored explicitly next to the scorer
        ("r2", True),
        ("R2", True),
        ("rmse", False),
        ("mae", False),
        ("mse", False),
        ("nrmse", False),
        # neg_ registry metric: inverted error -> maximization
        ("neg_root_mean_squared_error", True),
        # 2. sklearn scorers: a larger value is always better — error metrics
        # are already inverted into neg_ scorers, score metrics are returned as is
        ("accuracy", True),
        ("explained_variance", True),
        ("neg_log_loss", True),
        ("neg_mean_absolute_percentage_error", True),
        ("neg_mean_squared_error", True),
        ("neg_mean_absolute_error", True),
    ],
)
def test_is_greater_better(metric_name, expected):
    assert is_greater_better(metric_name) == expected


def test_is_greater_better_contract_from_issue():
    """Contract: the direction is determined by the scorer object."""
    assert is_greater_better("neg_root_mean_squared_error") is True
    assert is_greater_better("r2") is True
    assert is_greater_better("mae") is False


def test_is_greater_better_unknown_metric_raises():
    """An unknown metric has no direction — a clear ValueError.

    The error message explains how to fix it (registry or sklearn),
    similarly to get_metric.
    """
    with pytest.raises(ValueError, match="not implemented"):
        is_greater_better("unknown_custom_metric")


def test_is_greater_better_global_nrmse():
    """global_nrmse is a dynamic error: "lower is better".

    Regression: before the fix,
    is_greater_better('global_nrmse') raised ValueError (sklearn does not
    know such a scorer), which crashed ModelTrainer.fit() with
    metric='global_nrmse' (called from the debug log).
    """
    assert is_greater_better("global_nrmse") is False


def test_is_error_metric_user_presentation():
    """is_error_metric reflects the user-facing representation of a metric.

    Inverted error scorers (incl. neg_ metrics) are marked as errors,
    score metrics are not (the label in the train_model log).
    """
    assert is_error_metric("global_nrmse") is True
    assert is_error_metric("rmse") is True
    assert is_error_metric("mae") is True
    assert is_error_metric("mse") is True
    assert is_error_metric("nrmse") is True
    assert is_error_metric("neg_root_mean_squared_error") is True
    assert is_error_metric("neg_log_loss") is True
    assert is_error_metric("r2") is False
    assert is_error_metric("accuracy") is False
    with pytest.raises(ValueError, match="not implemented"):
        is_error_metric("unknown_custom_metric")


def test_to_user_value_natural_semantics():
    """"Raw" scorer values are converted to user-facing semantics.

    RMSE/MAE/MSE/NRMSE and all neg_ metrics are returned positive,
    score metrics (R², accuracy) — as is.
    """
    assert to_user_value("rmse", -0.123) == pytest.approx(0.123)
    assert to_user_value("neg_root_mean_squared_error", -0.123) == pytest.approx(0.123)
    assert to_user_value("mae", -0.05) == pytest.approx(0.05)
    assert to_user_value("mse", -1.5) == pytest.approx(1.5)
    assert to_user_value("nrmse", -0.3) == pytest.approx(0.3)
    assert to_user_value("neg_log_loss", -0.6) == pytest.approx(0.6)
    assert to_user_value("neg_mean_squared_error", -2.0) == pytest.approx(2.0)
    assert to_user_value("r2", 0.85) == pytest.approx(0.85)
    assert to_user_value("accuracy", 0.9) == pytest.approx(0.9)


def test_to_user_value_global_nrmse():
    """global_nrmse is a dynamic error: the value is inverted."""
    assert to_user_value("global_nrmse", -0.25) == pytest.approx(0.25)


# ──────────────────────────────────────────────────────────────────────────────
# user_direction: metric direction in the user-facing semantics
# ──────────────────────────────────────────────────────────────────────────────
def test_user_direction_minimize_for_errors():
    """Error metrics in user-facing semantics — "lower is better"."""
    assert user_direction("rmse") == "minimize"
    assert user_direction("mae") == "minimize"
    assert user_direction("mse") == "minimize"
    assert user_direction("nrmse") == "minimize"
    assert user_direction("global_nrmse") == "minimize"
    assert user_direction("neg_root_mean_squared_error") == "minimize"
    assert user_direction("neg_log_loss") == "minimize"
    assert user_direction("neg_mean_squared_error") == "minimize"
    assert user_direction("neg_mean_absolute_error") == "minimize"


def test_user_direction_maximize_for_score_metrics():
    """Score metrics (R², accuracy, etc.) — "higher is better"."""
    assert user_direction("r2") == "maximize"
    assert user_direction("R2") == "maximize"
    assert user_direction("accuracy") == "maximize"
    assert user_direction("explained_variance") == "maximize"


def test_user_direction_contrast_with_greater_is_better():
    """Contrast with the optimizer semantics.

    The optimizer maximizes neg_ metrics (is_greater_better=True), but in
    user-facing semantics these are errors — "lower is better". The finalist
    pool corridor formula must use user_direction, not greater_is_better.
    """
    assert is_greater_better("neg_root_mean_squared_error") is True
    assert user_direction("neg_root_mean_squared_error") == "minimize"


def test_direction_label_for_logs():
    """Metric direction for logs: "min/max better".

    User-facing values in logs are accompanied by the direction from
    user_direction; for neg_ metrics this is "min better", although the
    optimizer maximizes them (greater_is_better is not used).
    """
    assert direction_label("rmse") == "min better"
    assert direction_label("mae") == "min better"
    assert direction_label("neg_root_mean_squared_error") == "min better"
    assert direction_label("nrmse") == "min better"
    assert direction_label("r2") == "max better"
    assert direction_label("R2") == "max better"
    assert direction_label("accuracy") == "max better"


def test_user_direction_unknown_metric_raises():
    """An unknown metric has no direction — a clear ValueError."""
    with pytest.raises(ValueError, match="not implemented"):
        user_direction("unknown_custom_metric")


def test_is_greater_better_scorer_without_sign_fallback(monkeypatch):
    """A scorer without a _sign attribute: conservative fallback True.

    Covers the defensive branch for hypothetical custom scorers whose
    direction is not exposed.
    """
    import configurable_automl_engine.training_engine.metrics as metrics_mod

    class NoSignScorer:
        pass

    monkeypatch.setattr(
        metrics_mod, "sklearn_get_scorer", lambda name: NoSignScorer()
    )
    assert is_greater_better("some_custom_metric") is True


def test_is_greater_better_reflects_registry_mutation():
    """Mutation of the _SCORER_OBJECTS registry is visible immediately, no stale cache.

    Regression: the direction from the custom registry is
    read without caching, so runtime registry changes (registering a custom
    metric, changing its direction) are picked up immediately.
    """
    from sklearn.metrics import make_scorer, mean_squared_error

    import configurable_automl_engine.training_engine.metrics as metrics_mod

    custom_name = "dyn_registry_metric"
    assert custom_name not in metrics_mod._SCORER_OBJECTS

    # Before registration the metric is unknown -> ValueError.
    with pytest.raises(ValueError):
        is_greater_better(custom_name)

    # Register an "error" (lower is better) and check the direction.
    metrics_mod._SCORER_OBJECTS[custom_name] = (
        make_scorer(mean_squared_error, greater_is_better=False),
        False,
    )
    try:
        assert is_greater_better(custom_name) is False
        # Change the direction in the registry — the result must update,
        # despite preceding calls (there is no registry cache).
        scorer, _ = metrics_mod._SCORER_OBJECTS[custom_name]
        metrics_mod._SCORER_OBJECTS[custom_name] = (scorer, True)
        assert is_greater_better(custom_name) is True
    finally:
        del metrics_mod._SCORER_OBJECTS[custom_name]


def test_get_scorer_object():
    # Check custom objects (including lambdas in _SCORER_OBJECTS)
    scorer = get_scorer_object("rmse")
    assert callable(scorer)

    # Check nrmse
    nrmse_scorer = get_scorer_object("nrmse")
    assert callable(nrmse_scorer)
    # Check standard sklearn metrics (sklearn_get_scorer call)
    std_scorer = get_scorer_object("explained_variance")
    assert hasattr(std_scorer, "_score_func")


# Cover line 107 (to_sklearn_name)
def test_to_sklearn_name():
    # Dictionary case
    assert to_sklearn_name("rmse") == "neg_root_mean_squared_error"
    # Default case (line 107: returned as is in lower case)
    assert to_sklearn_name("R2") == "r2"
    assert to_sklearn_name("Unknown_Metric") == "unknown_metric"


# Test the lambda functions in the registries (additional coverage)
def test_neg_rmse_lambda():
    # Cover the lambda in _METRICS["neg_root_mean_squared_error"]
    neg_rmse_func = get_metric("neg_root_mean_squared_error")
    y_true = np.array([0, 2])
    y_pred = np.array([0, 0])
    # Expected value: -sqrt((0^2 + 2^2)/2) = -sqrt(2) ≈ -1.414
    assert neg_rmse_func(y_true, y_pred) == pytest.approx(-np.sqrt(2.0))


def test_global_nrmse_coverage(caplog):
    """
    The test covers:
    1. ValueError raised in get_scorer_object when global_y is not passed.
    2. Successful scorer creation via get_global_nrmse_scorer.
    3. The target_range < 1e-6 branch in _global_nrmse (protection against division by zero).
    4. The standard _global_nrmse computation.
    """

    # 1. Check the exception: global_nrmse called without global_y
    with pytest.raises(
        ValueError, match="For 'global_nrmse', 'global_y' must be passed"
    ):
        get_scorer_object("global_nrmse", global_y=None)
    # 2. Check successful scorer creation
    y_full = np.array([10.0, 20.0, 30.0])  # range = 20.0
    scorer = get_scorer_object("global_nrmse", global_y=y_full)

    # Verify it is a scorer object (Callable)
    assert callable(scorer)
    # 3. Check the target_range < 1e-6 branch in _global_nrmse
    # Call the function directly with a critically small range
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])

    with caplog.at_level(logging.WARNING):
        res_inf = _global_nrmse(y_true, y_pred, target_range=1e-7)
        assert res_inf == float("inf")
        assert "Global target_range is too small" in caplog.text
    # 4. Check the standard computation (main return branch)
    # RMSE for (1,2) and (1,2) = 0. Range = 10. Result = 0/10 = 0.0
    res_zero = _global_nrmse(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), target_range=10.0
    )
    assert res_zero == 0.0
    # Computation with concrete values:
    # y_true=[0, 2], y_pred=[0, 0] -> MSE = (0^2 + 2^2)/2 = 2 -> RMSE = sqrt(2) ≈ 1.4142
    # target_range = 2
    # Result = 1.4142 / 2 = 0.7071...
    y_t = np.array([0.0, 2.0])
    y_p = np.array([0.0, 0.0])
    expected_rmse = np.sqrt(2.0)
    expected_nrmse = expected_rmse / 2.0

    assert np.isclose(_global_nrmse(y_t, y_p, target_range=2.0), expected_nrmse)

import logging
from unittest.mock import patch, MagicMock

import pytest
from sklearn.linear_model import ARDRegression, ElasticNet
from sklearn.svm import SVR

from configurable_automl_engine.models import (
    LEGACY_PARAM_MAPPINGS,
    _get_constructor_param_info,
    _get_factory,
    clean_hyperparameters,
    create_model,
    get_allowed_hyperparameters,
    remap_legacy_param_names,
)


def test_get_constructor_param_info_ard_regression():
    """_get_constructor_param_info returns correct accepted params for ARDRegression."""
    accepted, accepts_kwargs = _get_constructor_param_info(ARDRegression)
    # max_iter is a known parameter
    assert "max_iter" in accepted
    # self should never be included
    assert "self" not in accepted
    # ~kwargs should not be present (ARDRegression does not accept **kwargs)
    assert not accepts_kwargs


def test_get_constructor_param_info_cache():
    """Calling with the same class returns the cached result."""
    result1 = _get_constructor_param_info(ElasticNet)
    result2 = _get_constructor_param_info(ElasticNet)
    assert result1 is result2  # same tuple object from lru_cache


def test_get_constructor_param_info_different_classes():
    """Different classes return different cached results."""
    r1 = _get_constructor_param_info(ElasticNet)
    r2 = _get_constructor_param_info(SVR)
    assert r1 != r2


def test_get_constructor_param_info_skips_self():
    """
    Cover the 'if name == "self": continue' branch (line 152).
    Use a class whose __init__ has only 'self' and **kwargs to also
    cover the VAR_KEYWORD branch (line 154).
    """

    class _ModelWithKwargs:
        def __init__(self, **kwargs):
            pass

    accepted, accepts_kwargs = _get_constructor_param_info(_ModelWithKwargs)
    # 'self' must never appear in accepted params
    assert "self" not in accepted
    # The class accepts **kwargs
    assert accepts_kwargs is True
    # No positional params beyond self
    assert accepted == frozenset()


def test_get_constructor_param_info_self_only():
    """
    Cover the 'if name == "self": continue' branch with a class
    that has ONLY 'self' (no params, no **kwargs).
    """

    class _ModelSelfOnly:
        def __init__(self):
            pass

    accepted, accepts_kwargs = _get_constructor_param_info(_ModelSelfOnly)
    assert "self" not in accepted
    assert accepts_kwargs is False
    assert accepted == frozenset()


def test_legacy_param_mappings_contains_ard():
    """LEGACY_PARAM_MAPPINGS has the expected entry for ardregression."""
    assert "ardregression" in LEGACY_PARAM_MAPPINGS
    assert LEGACY_PARAM_MAPPINGS["ardregression"]["n_iter"] == "max_iter"


def test_clean_hyperparameters_remapping():
    """clean_hyperparameters remaps n_iter -> max_iter for ARDRegression."""
    raw = {"n_iter": 500, "tol": 1e-4}
    cleaned = clean_hyperparameters("ardregression", ARDRegression, raw)
    assert "max_iter" in cleaned
    assert cleaned["max_iter"] == 500
    assert "n_iter" not in cleaned
    assert cleaned["tol"] == 1e-4


def test_clean_hyperparameters_drops_unknown_params():
    """clean_hyperparameters drops params not in the constructor signature."""
    raw = {"alpha": 1.0, "totally_fake_param": 42}
    cleaned = clean_hyperparameters("elasticnet", ElasticNet, raw)
    assert "alpha" in cleaned
    assert "totally_fake_param" not in cleaned


def test_clean_hyperparameters_remapped_key_kept_if_valid():
    """If the remapped key IS valid for the target class, it is kept."""
    raw = {"n_iter": 500}
    # ElasticNet DOES accept max_iter
    cleaned = clean_hyperparameters("ardregression", ElasticNet, raw)
    assert "max_iter" in cleaned
    assert cleaned["max_iter"] == 500


def test_clean_hyperparameters_empty_dict():
    """An empty hyperparams dict returns an empty dict."""
    cleaned = clean_hyperparameters("elasticnet", ElasticNet, {})
    assert cleaned == {}


def test_clean_hyperparameters_no_remapping_for_other_algo():
    """non-ard algo does not trigger remapping."""
    raw = {"n_iter": 500, "alpha": 1.0}
    cleaned = clean_hyperparameters("elasticnet", ElasticNet, raw)
    # n_iter is not a valid ElasticNet param -> dropped, not remapped
    assert "n_iter" not in cleaned
    assert "alpha" in cleaned


def test_create_model_ard_with_legacy_n_iter():
    """ARDRegression can be created with legacy 'n_iter' param."""
    model = create_model("ardregression", n_iter=300, tol=1e-4)
    assert isinstance(model, ARDRegression)
    # The clean function should have remapped n_iter -> max_iter
    assert model.max_iter == 300


def test_create_model_svr_default_max_iter():
    """SVR gets max_iter=10000 when not provided."""
    model = create_model("svr", C=1.0)
    assert model.max_iter == 10000


def test_create_model_svr_user_max_iter():
    """SVR respects user-provided max_iter."""
    model = create_model("svr", C=1.0, max_iter=5000)
    assert model.max_iter == 5000


def test_create_model_svr_explicit_sigmoid_kernel_allowed():
    """Explicitly requested kernel='sigmoid' keeps working.

    A regression test: the DEFAULT_SPACES['svr'] change excludes sigmoid/poly
    only from the automatic HPO search. Validation is built on
    get_allowed_hyperparameters (SVR constructor signature), so an explicit
    'sigmoid' choice via hyperparameters / space_overrides is not forbidden.
    """
    allowed = get_allowed_hyperparameters("svr")
    assert "kernel" in allowed

    model = create_model("svr", C=1.0, kernel="sigmoid")
    assert isinstance(model, SVR)
    assert model.kernel == "sigmoid"


def test_clean_hyperparameters_logs_remapped(caplog):
    """clean_hyperparameters logs at DEBUG level when remapping."""
    with caplog.at_level(logging.DEBUG, logger="configurable_automl_engine.models"):
        raw = {"n_iter": 500, "tol": 1e-4}
        clean_hyperparameters("ardregression", ARDRegression, raw)
    assert any("Remapped" in record.message for record in caplog.records)


def test_clean_hyperparameters_logs_dropped(caplog):
    """clean_hyperparameters logs at DEBUG level when dropping unknown param."""
    with caplog.at_level(logging.DEBUG, logger="configurable_automl_engine.models"):
        raw = {"alpha": 1.0, "fake_param": 99}
        clean_hyperparameters("elasticnet", ElasticNet, raw)
    assert any("Dropped" in record.message for record in caplog.records)


def test_remap_legacy_param_names_renames_ard_n_iter():
    """remap_legacy_param_names turns n_iter into max_iter for ardregression."""
    raw = {"n_iter": 500, "tol": 1e-4}
    remapped = remap_legacy_param_names("ardregression", raw)
    assert remapped == {"max_iter": 500, "tol": 1e-4}
    # The input dict is not mutated.
    assert raw == {"n_iter": 500, "tol": 1e-4}


def test_remap_legacy_param_names_alias_resolution():
    """The 'ard' alias resolves to ardregression for the remap."""
    raw = {"n_iter": 300}
    assert remap_legacy_param_names("ard", raw) == {"max_iter": 300}


def test_remap_legacy_param_names_modern_key_wins():
    """If both n_iter and max_iter are present, the modern name wins."""
    raw = {"n_iter": 300, "max_iter": 900}
    assert remap_legacy_param_names("ardregression", raw) == {"max_iter": 900}


def test_remap_legacy_param_names_other_algo_unchanged():
    """Algorithms without legacy mappings are copied unchanged."""
    raw = {"alpha": 1.0, "max_iter": 500}
    assert remap_legacy_param_names("elasticnet", raw) == raw


def test_remap_legacy_param_names_empty():
    """An empty dict stays empty."""
    assert remap_legacy_param_names("ardregression", {}) == {}


def test_create_model_invalid_type():
    """
    Verify TypeError is raised when the algorithm argument is not a string.
    """
    # Pass a list instead of a string
    with pytest.raises(TypeError, match="Algorithm must be a string"):
        create_model(algorithm=["elasticnet"])

    # Pass a number instead of a string
    with pytest.raises(TypeError, match="Algorithm must be a string"):
        create_model(algorithm=123)


def test_create_model_import_error_for_missing_package():
    """
    Coverage of line 107: verify ImportError is raised when the model class
    in the factory is None (simulating a missing XGBoost).
    """
    # Patch _FACTORY to temporarily replace the value with None
    # This simulates the situation where XGBRegressor was not imported
    with patch("configurable_automl_engine.models._FACTORY") as mocked_factory:
        # Configure the mock so that None is returned for 'xgboosting'
        mocked_factory.__contains__.return_value = True
        mocked_factory.__getitem__.return_value = None

        # Try to create a model through the 'xgboost' alias
        with pytest.raises(ImportError, match="requires an additional package"):
            create_model("xgboost")


def test_create_model_unknown_algorithm():
    """
    Additional test for the unknown-algorithm branch (lines 102-103).
    """
    with pytest.raises(ValueError, match="Unknown algorithm: 'unknown_model'"):
        create_model("unknown_model")


def test_create_model_gpr_default_kernel():
    """
    Verify the specific logic for GaussianProcessRegressor.
    """
    model = create_model("gpr")
    # Verify the RBF kernel was set by default
    assert hasattr(model, "kernel")
    # In sklearn GPR, the kernel is kept in the parameters after initialization
    assert "RBF" in str(model.get_params()["kernel"])


def test_create_model_gpr_default_kernel_has_amplitude_and_noise():
    """GPR default kernel = amplitude × RBF + white noise (not a bare RBF).

    Regression: a bare RBF(1.0) without an output-scale
    amplitude and without an explicit noise term behaves like the mean
    predictor, while GPR is one of the most suitable models on small
    samples and provides prediction intervals.
    """
    model = create_model("gpr")
    kernel = model.get_params()["kernel"]
    kernel_str = str(kernel)
    # amplitude * base + white noise: ConstantKernel in the Product,
    # WhiteKernel in the Sum.
    assert "WhiteKernel" in kernel_str


def test_create_model_gpr_defaults_normalize_and_restarts():
    """GPR defaults: normalize_y=True, n_restarts_optimizer>0, random_state set.

    Target normalization and several optimizer restarts with a fixed seed
    are part of the default configuration, so the model is not a "mean
    predictor" and reproducible across runs.
    """
    model = create_model("gpr")
    assert model.normalize_y is True
    assert model.n_restarts_optimizer >= 1
    assert model.random_state is not None


@pytest.mark.parametrize("kernel_type", ["rbf", "matern", "rational_quadratic"])
def test_create_model_gpr_kernel_type(kernel_type):
    """kernel_type selects the GPR kernel family (rbf/matern/rational_quadratic)."""
    model = create_model("gpr", kernel_type=kernel_type)
    kernel = model.get_params()["kernel"]
    kernel_str = str(kernel)
    if kernel_type == "rbf":
        assert "RBF" in kernel_str
    elif kernel_type == "matern":
        assert "Matern" in kernel_str
    else:
        assert "RationalQuadratic" in kernel_str
    # Every family keeps the amplitude (Product) and the noise component
    # (WhiteKernel in the outer Sum).
    assert "WhiteKernel" in kernel_str
    assert hasattr(kernel, "k1") and hasattr(kernel, "k2")
    assert type(kernel.k1).__name__ == "Product"
    assert type(kernel.k1.k1).__name__ == "ConstantKernel"


def test_create_model_gpr_kernel_type_not_in_params():
    """kernel_type is a tuner-only hint and never a model parameter."""
    model = create_model("gpr", kernel_type="matern")
    assert "kernel_type" not in model.get_params()


# --- get_allowed_hyperparameters (single source of truth for validation) ---
def test_get_allowed_hyperparameters_constructor_params():
    """The allowed set is built from the estimator constructor signature."""
    allowed = get_allowed_hyperparameters("elasticnet")
    assert "alpha" in allowed
    assert "l1_ratio" in allowed
    assert "fit_intercept" in allowed
    assert "bad_param" not in allowed


def test_get_allowed_hyperparameters_nondefault_params_included():
    """Params outside DEFAULT_SPACES are included in the allowed set."""
    rf_allowed = get_allowed_hyperparameters("random_forest")
    assert "min_samples_split" in rf_allowed
    assert "max_features" in rf_allowed
    assert "warm_start" in rf_allowed

    gpr_allowed = get_allowed_hyperparameters("gaussian_process_regression")
    assert "alpha" in gpr_allowed


def test_get_allowed_hyperparameters_legacy_names_included():
    """Legacy names from LEGACY_PARAM_MAPPINGS count as allowed."""
    allowed = get_allowed_hyperparameters("ardregression")
    assert "max_iter" in allowed
    assert "n_iter" in allowed  # remapped to max_iter


def test_get_allowed_hyperparameters_isotonic_feature_index_included():
    """``feature_index`` (an IsotonicDataTransformer hint, not a constructor
    parameter) counts as an allowed isotonic hyperparameter (issue #150)."""
    allowed = get_allowed_hyperparameters("isotonic_regression")
    assert "feature_index" in allowed
    assert "increasing" in allowed
    assert "out_of_bounds" in allowed

    alias_allowed = get_allowed_hyperparameters("isotonic")
    assert "feature_index" in alias_allowed


def test_get_allowed_hyperparameters_alias_resolution():
    """Algorithm aliases give the same set as the canonical name."""
    assert get_allowed_hyperparameters("rf") == get_allowed_hyperparameters(
        "random_forest"
    )
    assert get_allowed_hyperparameters("xgboost") == get_allowed_hyperparameters(
        "xgboosting"
    )


def test_get_allowed_hyperparameters_unknown_algo():
    """Unknown algorithm → None (the check is skipped softly)."""
    assert get_allowed_hyperparameters("no_such_algo") is None


def test_get_allowed_hyperparameters_missing_dependency(monkeypatch):
    """Missing dependency (estimator=None) → None."""
    monkeypatch.setattr(
        "configurable_automl_engine.models._FACTORY", {"xgboosting": None}
    )
    assert get_allowed_hyperparameters("xgboosting") is None


def test_get_allowed_hyperparameters_var_kwargs_returns_none(monkeypatch):
    """A constructor with **kwargs does not restrict the param set → None."""

    class _FlexibleEstimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "configurable_automl_engine.models._FACTORY",
        {"flex": _FlexibleEstimator},
    )
    assert get_allowed_hyperparameters("flex") is None


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic seeding of stochastic estimators (issue #144)
# ──────────────────────────────────────────────────────────────────────────────


def test_create_model_seeds_stochastic_estimators():
    """Stochastic estimators get a fixed random_state by default.

    Trees/forests/boosting and randomized linear models draw randomness at
    fit time; without a fixed seed identical ``train_best_model`` runs
    produced different scores/best parameters (issue #144). ``create_model``
    must inject the engine seed for every estimator accepting
    ``random_state``.
    """
    from sklearn.ensemble import RandomForestRegressor

    model = create_model("random_forest", n_estimators=10)
    assert isinstance(model, RandomForestRegressor)
    assert model.random_state == 42


def test_create_model_respects_explicit_random_state():
    """An explicit random_state always wins over the injected default."""
    from sklearn.ensemble import RandomForestRegressor

    model = create_model("random_forest", n_estimators=10, random_state=7)
    assert isinstance(model, RandomForestRegressor)
    assert model.random_state == 7


def test_create_model_none_random_state_normalized_to_default():
    """random_state=None is normalized to the fixed engine seed.

    ``None`` requests a fresh RNG per fit (sklearn semantics), which would
    break reproducibility exactly like an omitted seed — the explicit-none
    case is treated as "no seed set" and normalized.
    """
    model = create_model("random_forest", n_estimators=10, random_state=None)
    assert model.random_state == 42


def test_create_model_estimators_without_random_state_unaffected():
    """Estimators that do not accept random_state are left untouched.

    Non-stochastic algorithms (kNN, SVR, isotonic, ARD, ...) have no
    ``random_state`` parameter — the seeding must not add one (and must not
    break the constructor).
    """
    model = create_model("knn", n_neighbors=5)
    assert model.get_params(deep=False).get("random_state") is None
    model = create_model("svr", C=1.0)
    assert model.get_params(deep=False).get("random_state") is None


def test_create_model_xgboost_gets_seeded_when_installed():
    """XGBoost accepts **kwargs; the default seed is injected for it too."""
    from configurable_automl_engine.models import _FACTORY

    if _FACTORY.get("xgboosting") is None:
        pytest.skip("xgboost is not installed")
    model = create_model("xgboosting", n_estimators=10)
    assert model.get_params().get("random_state") == 42

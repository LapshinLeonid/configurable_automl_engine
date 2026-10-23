"""Tests for manual range handling (issue #127).

Coverage:
- A range whose type is inferred from the boundary numbers is re-resolved
  against the parameter type: ``[1, 50]`` for a real-valued parameter
  (e.g. ``C`` of an SVR, ``alpha`` of a Tweedie) stays a real range instead
  of silently becoming an integer range.
- Scale parameters (``alpha``, ``C``, ``epsilon``, ``gamma``,
  ``learning_rate``) without an explicit type default to the logarithmic
  scale.
- A parameter whose DEFAULT_SPACES entry is ``float_log`` (e.g. ``alpha_1``
  of an ARD regression) also defaults to the logarithmic scale (issue #140).
- An explicitly typed range (``[1, 50, "int"]``) is never rewritten.
- The tuner warns when the best trial lies at a range boundary.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from configurable_automl_engine.common.hyperopt_defaults import (
    FloatSpace,
    IntSpace,
    SearchSpaceEntry,
    normalize_hyperparameter_entry,
    resolve_implicit_range_type,
)
from configurable_automl_engine.training_engine.config_parser import Config
from configurable_automl_engine.tuner import _OptimizeRunner, optimize

# A minimal valid base config: elasticnet search + terminal select.
BASE = {
    "general": {
        "comparison_metric": "r2",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "phases": [
            {"name": "search", "n_trials": 2, "action": "all_algorithms"},
            {
                "name": "final_select",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
    },
    "algorithms": {"ridge": {"enable": True}},
}


def _config(algo_name: str, hyperparameters: dict) -> Config:
    cfg_data = BASE | {
        "algorithms": {algo_name: {"enable": True, "hyperparameters": hyperparameters}}
    }
    return Config.model_validate(cfg_data)


# ──────────────────────────────────────────────────────────────────────────────
# 1. The range type is taken from the parameter type, not the boundary numbers
# ──────────────────────────────────────────────────────────────────────────────


def test_svr_C_integer_bounds_stay_real():
    """[1, 50] for C of an SVR is a real range, not an integer one."""
    cfg = _config("svr", {"C": [1, 50]})
    entry = cfg.algorithms.svr.hyperparameters["C"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.type == "float_log"
    assert entry.low == 1.0
    assert entry.high == 50.0


def test_tweedie_alpha_integer_bounds_stay_real():
    """[1, 50] for alpha of a Tweedie is a real range, not an integer one."""
    cfg = _config("tweedieregressor", {"alpha": [1, 50]})
    entry = cfg.algorithms.tweedieregressor.hyperparameters["alpha"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.type == "float_log"


def test_ridge_alpha_mixed_bounds_stay_real():
    """[1, 50.0] (mixed int/float bounds) stays a real range."""
    cfg = _config("ridge", {"alpha": [1, 50.0]})
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert isinstance(entry.config, FloatSpace)


def test_int_parameter_integer_bounds_stay_int():
    """An integer parameter keeps an integer range: n_estimators [10, 500]."""
    cfg = _config("random_forest", {"n_estimators": [10, 500], "max_depth": [3, 20]})
    rf = cfg.algorithms.random_forest.hyperparameters
    assert isinstance(rf["n_estimators"].config, IntSpace)
    assert isinstance(rf["max_depth"].config, IntSpace)


def test_int_parameter_not_in_default_space_from_signature():
    """min_samples_split (absent from DEFAULT_SPACES) is int from the signature."""
    cfg = _config("random_forest", {"min_samples_split": [2, 10]})
    entry = cfg.algorithms.random_forest.hyperparameters["min_samples_split"]
    assert isinstance(entry.config, IntSpace)
    assert entry.config.low == 2
    assert entry.config.high == 10


# ──────────────────────────────────────────────────────────────────────────────
# 2. Scale parameters default to the logarithmic scale
# ──────────────────────────────────────────────────────────────────────────────


def test_scale_param_positive_bounds_default_to_log():
    """alpha with positive integer bounds → float_log."""
    cfg = _config("ridge", {"alpha": [1, 50]})
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "float_log"


def test_scale_param_non_positive_low_falls_back_to_float():
    """alpha [0, 1.0] → plain float (log scale requires low > 0)."""
    cfg = _config("ridge", {"alpha": [0.0, 1.0]})
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "float"
    assert isinstance(entry.config, FloatSpace)


def test_learning_rate_integer_bounds_log():
    """learning_rate [1, 100] for gradient_boosting → float_log."""
    cfg = _config("gradient_boosting", {"learning_rate": [1, 100]})
    entry = cfg.algorithms.gradient_boosting.hyperparameters["learning_rate"]
    assert entry.dist_type == "float_log"


# ──────────────────────────────────────────────────────────────────────────────
# 2a. Parameters with a float_log DEFAULT_SPACES entry default to log scale
#     (issue #140: ARD alpha_1/alpha_2/lambda_1/lambda_2 are not in
#     SCALE_PARAMS, so their inferred [1e-6, 0.1] became a linear range)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "param",
    ["alpha_1", "alpha_2", "lambda_1", "lambda_2"],
)
def test_ard_float_log_default_params_inferred_as_log(param):
    """[1e-6, 0.1] for ARD alpha_1/alpha_2/lambda_1/lambda_2 → float_log."""
    cfg = _config("ardregression", {param: [1e-6, 0.1]})
    entry = cfg.algorithms.ardregression.hyperparameters[param]
    assert entry.dist_type == "float_log"
    assert isinstance(entry.config, FloatSpace)


def test_ard_integer_bounds_stay_float_log():
    """ARD alpha_1 [1, 50] is float_log, not an integer range."""
    cfg = _config("ardregression", {"alpha_1": [1, 50]})
    entry = cfg.algorithms.ardregression.hyperparameters["alpha_1"]
    assert entry.dist_type == "float_log"
    assert isinstance(entry.config, FloatSpace)


def test_ard_float_log_default_non_positive_low_falls_back_to_float():
    """ARD alpha_1 [0, 0.1] → plain float (log scale requires low > 0)."""
    cfg = _config("ardregression", {"alpha_1": [0.0, 0.1]})
    entry = cfg.algorithms.ardregression.hyperparameters["alpha_1"]
    assert entry.dist_type == "float"
    assert isinstance(entry.config, FloatSpace)


def test_ard_explicit_float_preserved_despite_float_log_default():
    """[1e-6, 0.1, 'float'] for ARD alpha_1 stays a plain float range."""
    cfg = _config("ardregression", {"alpha_1": [1e-6, 0.1, "float"]})
    entry = cfg.algorithms.ardregression.hyperparameters["alpha_1"]
    assert entry.dist_type == "float"
    assert isinstance(entry.config, FloatSpace)


def test_sgd_eta0_float_log_default_inferred_as_log():
    """eta0 of an SGD regressor (float_log in DEFAULT_SPACES) → float_log."""
    cfg = _config("sgdregressor", {"eta0": [1e-4, 1e-1]})
    entry = cfg.algorithms.sgdregressor.hyperparameters["eta0"]
    assert entry.dist_type == "float_log"
    assert isinstance(entry.config, FloatSpace)


def test_plain_float_default_stays_linear():
    """l1_ratio/subsample (float in DEFAULT_SPACES) stay plain float."""
    cfg = _config("elasticnet", {"l1_ratio": [0.2, 0.8]})
    entry = cfg.algorithms.elasticnet.hyperparameters["l1_ratio"]
    assert entry.dist_type == "float"
    cfg = _config("gradient_boosting", {"subsample": [0.5, 1.0]})
    entry = cfg.algorithms.gradient_boosting.hyperparameters["subsample"]
    assert entry.dist_type == "float"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Explicitly typed ranges are preserved
# ──────────────────────────────────────────────────────────────────────────────


def test_explicit_float_type_preserved():
    """[1, 50, 'float'] for alpha stays a plain float range (no log rewrite)."""
    cfg = _config("ridge", {"alpha": [1, 50, "float"]})
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "float"
    assert isinstance(entry.config, FloatSpace)


def test_explicit_int_type_preserved():
    """[1, 50, 'int'] for alpha is honored even though alpha is a float param."""
    cfg = _config("ridge", {"alpha": [1, 50, "int"]})
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "int"
    assert isinstance(entry.config, IntSpace)


def test_explicit_float_log_type_preserved():
    """[1e-6, 1.0, 'float_log'] stays float_log."""
    cfg = _config("ridge", {"alpha": [1e-6, 1.0, "float_log"]})
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "float_log"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Unit-level: normalize_hyperparameter_entry / resolve_implicit_range_type
# ──────────────────────────────────────────────────────────────────────────────


def test_normalize_skips_explicit_type():
    entry = SearchSpaceEntry.model_validate([1, 50, "int"])
    assert entry.explicit_type is True
    assert normalize_hyperparameter_entry("ridge", "alpha", entry) is entry


def test_normalize_retypes_inferred_entry():
    entry = SearchSpaceEntry.model_validate([1, 50])
    assert entry.explicit_type is False
    assert isinstance(entry.config, IntSpace)
    normalized = normalize_hyperparameter_entry("ridge", "alpha", entry)
    assert isinstance(normalized.config, FloatSpace)
    assert normalized.config.type == "float_log"
    # The original entry is not mutated.
    assert isinstance(entry.config, IntSpace)


def test_normalize_preserves_inferred_int_for_int_param():
    entry = SearchSpaceEntry.model_validate([1, 10])
    normalized = normalize_hyperparameter_entry("random_forest", "max_depth", entry)
    assert isinstance(normalized.config, IntSpace)


def test_normalize_keeps_fractional_bounds_for_int_param():
    """Fractional bounds for an int parameter are not silently truncated.

    [1.5, 2.5] stays a float range: int(1.5)=1 would change the user's
    range, so the original inference (float) is kept.
    """
    entry = SearchSpaceEntry.model_validate([1.5, 2.5])
    assert isinstance(entry.config, FloatSpace)
    normalized = normalize_hyperparameter_entry("random_forest", "max_depth", entry)
    assert isinstance(normalized.config, FloatSpace)
    assert normalized.low == 1.5
    assert normalized.high == 2.5


def test_resolve_implicit_scale_param_log():
    assert resolve_implicit_range_type("ridge", "alpha", 1.0, 50.0) == "float_log"
    assert resolve_implicit_range_type("svr", "epsilon", 1e-3, 1.0) == "float_log"


def test_resolve_implicit_float_log_default_param_log():
    """A float_log default entry (ARD alpha_1, SGD eta0) resolves to log."""
    assert (
        resolve_implicit_range_type("ardregression", "alpha_1", 1e-6, 0.1)
        == "float_log"
    )
    assert (
        resolve_implicit_range_type("ardregression", "lambda_2", 1e-6, 0.1)
        == "float_log"
    )
    assert (
        resolve_implicit_range_type("sgdregressor", "eta0", 1e-4, 1e-1)
        == "float_log"
    )


def test_resolve_implicit_float_log_default_non_positive_bounds_float():
    """A float_log default with a non-positive bound falls back to float."""
    assert (
        resolve_implicit_range_type("ardregression", "alpha_1", 0.0, 0.1) == "float"
    )


def test_resolve_implicit_plain_float_default_stays_float():
    """A float default entry (l1_ratio, subsample) stays linear."""
    assert resolve_implicit_range_type("elasticnet", "l1_ratio", 0.2, 0.8) == "float"
    assert (
        resolve_implicit_range_type("gradient_boosting", "subsample", 0.5, 1.0)
        == "float"
    )


def test_resolve_implicit_non_scale_unknown_param_none():
    """An unknown parameter type → None (the boundary inference is kept)."""
    assert resolve_implicit_range_type("no_such_algo", "unknown", 1.0, 2.0) is None


# ──────────────────────────────────────────────────────────────────────────────
# 5. Warning when the optimum lies at a range boundary
# ──────────────────────────────────────────────────────────────────────────────


def _runner_with_space(space: dict) -> _OptimizeRunner:
    runner = _OptimizeRunner.__new__(_OptimizeRunner)
    runner.algo = "ridge"
    runner.search_space_dict = space
    return runner


def test_warn_at_high_boundary(caplog):
    runner = _runner_with_space(
        {"alpha": SearchSpaceEntry.model_validate([0.1, 1.0, "float"])}
    )
    with caplog.at_level(logging.WARNING):
        runner._warn_if_optimum_at_boundary({"alpha": 1.0})
    assert "may lie beyond the range" in caplog.text
    assert "alpha" in caplog.text


def test_warn_at_low_boundary(caplog):
    runner = _runner_with_space(
        {"alpha": SearchSpaceEntry.model_validate([0.1, 1.0, "float"])}
    )
    with caplog.at_level(logging.WARNING):
        runner._warn_if_optimum_at_boundary({"alpha": 0.1})
    assert "may lie beyond the range" in caplog.text


def test_no_warn_at_middle_value(caplog):
    runner = _runner_with_space(
        {"alpha": SearchSpaceEntry.model_validate([0.1, 1.0, "float"])}
    )
    with caplog.at_level(logging.WARNING):
        runner._warn_if_optimum_at_boundary({"alpha": 0.55})
    assert "may lie beyond the range" not in caplog.text


def test_warn_skips_categorical_and_constants(caplog):
    runner = _runner_with_space(
        {
            "kernel": SearchSpaceEntry.model_validate(
                [["rbf", "linear"], "categorical"]
            ),
            "max_iter": 1000,
        }
    )
    with caplog.at_level(logging.WARNING):
        runner._warn_if_optimum_at_boundary({"kernel": "rbf", "max_iter": 1000})
    assert "may lie beyond the range" not in caplog.text


def test_warn_skips_empty_space(caplog):
    runner = _runner_with_space({})
    with caplog.at_level(logging.WARNING):
        runner._warn_if_optimum_at_boundary({"alpha": 0.1})
    assert "may lie beyond the range" not in caplog.text


def test_warn_at_boundary_with_tolerance(caplog):
    """A value near a boundary within the range-width tolerance triggers."""
    runner = _runner_with_space(
        {"alpha": SearchSpaceEntry.model_validate([0.1, 1.0, "float"])}
    )
    with caplog.at_level(logging.WARNING):
        runner._warn_if_optimum_at_boundary({"alpha": 0.1 + 9e-7})
    assert "may lie beyond the range" in caplog.text


def test_warn_only_abs_tolerance_no_false_positive_on_wide_range(caplog):
    """The tolerance is the range-width fraction, not a relative tolerance.

    For [0, 1e9] the abs_tol is 1e-6 · 1e9 = 1000: a value within 1000 of a
    boundary IS at the boundary. A value farther than the tolerance (e.g.
    2000) must NOT warn — with rel_tol=1e-9 alone the tolerance would be
    ~1.0 and 2000 would also be accepted, which is the false positive this
    test guards.
    """
    runner = _runner_with_space(
        {"alpha": SearchSpaceEntry.model_validate([0.0, 1e9, "float"])}
    )
    with caplog.at_level(logging.WARNING):
        # Within the abs tolerance but far beyond any rel_tol=1e-9.
        runner._warn_if_optimum_at_boundary({"alpha": 500.0})
    assert "may lie beyond the range" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # Beyond the abs tolerance of 1000.
        runner._warn_if_optimum_at_boundary({"alpha": 2000.0})
    assert "may lie beyond the range" not in caplog.text


def test_inferred_float_log_entry_used_in_search_space():
    """End-to-end: the Config re-types the range; the entry reaches the space."""
    cfg = _config("svr", {"C": [1, 50], "epsilon": [0.5, 1.0]})
    C = cfg.algorithms.svr.hyperparameters["C"]
    eps = cfg.algorithms.svr.hyperparameters["epsilon"]
    assert C.dist_type == "float_log"
    assert eps.dist_type == "float_log"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dict-form entries (canonical model_dump output) keep their explicit type
# ──────────────────────────────────────────────────────────────────────────────


def test_dict_form_explicit_int_not_rewritten():
    """The dict form {'config': {'type': 'int', ...}} stays int for a float param.

    The dict form is the canonical output of ``model_dump()`` and a valid
    Config input: its explicit type must be honored, not silently rewritten to
    ``float_log`` (regression found in review).
    """
    cfg = _config("ridge", {"alpha": {"config": {"type": "int", "low": 1, "high": 50}}})
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "int"
    assert isinstance(entry.config, IntSpace)
    assert entry.explicit_type is True


def test_dict_form_explicit_float_not_rewritten_to_log():
    """A dict-form explicit 'float' for alpha stays a plain float."""
    cfg = _config(
        "ridge",
        {"alpha": {"config": {"type": "float", "low": 1, "high": 50}}},
    )
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "float"
    assert isinstance(entry.config, FloatSpace)
    assert entry.explicit_type is True


def test_dict_form_explicit_float_log_preserved():
    """A dict-form explicit 'float_log' stays float_log."""
    cfg = _config(
        "ridge",
        {"alpha": {"config": {"type": "float_log", "low": 0.1, "high": 50}}},
    )
    entry = cfg.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "float_log"
    assert entry.explicit_type is True


def test_dict_form_categorical_not_marked_explicit():
    """A dict-form categorical entry is not marked explicit (no numeric type)."""
    entry = SearchSpaceEntry.model_validate(
        {"config": {"type": "categorical", "options": ["rbf", "linear"]}}
    )
    assert entry.explicit_type is False


# ──────────────────────────────────────────────────────────────────────────────
# 7. dump → reload round trip preserves the explicit type
# ──────────────────────────────────────────────────────────────────────────────


def test_dump_reload_round_trip_explicit_int_preserved():
    """An explicit [1, 50, 'int'] survives model_dump(mode='json') → reload.

    The internal explicit_type flag travels with the dump so that a saved and
    reloaded config does not convert an explicit int range into float_log
    (regression found in review).
    """
    cfg = _config("ridge", {"alpha": [1, 50, "int"]})
    dumped = cfg.model_dump(mode="json")
    cfg_reloaded = Config.model_validate(dumped)
    entry = cfg_reloaded.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "int"
    assert isinstance(entry.config, IntSpace)
    assert entry.explicit_type is True


def test_dump_reload_round_trip_explicit_float_log_preserved():
    """An explicit float_log range survives the dump → reload round trip."""
    cfg = _config("ridge", {"alpha": [1e-2, 50.0, "float_log"]})
    dumped = cfg.model_dump(mode="json")
    cfg_reloaded = Config.model_validate(dumped)
    entry = cfg_reloaded.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "float_log"
    assert entry.explicit_type is True


def test_dump_reload_round_trip_inferred_still_retyped():
    """An inferred [1, 50] for C of an SVR is still re-typed after a round trip.

    The inferred flag (explicit_type=False) is preserved by the dump, so a
    reloaded config keeps the "infer against the parameter type" behavior.
    """
    cfg = _config("svr", {"C": [1, 50]})
    assert cfg.algorithms.svr.hyperparameters["C"].dist_type == "float_log"
    dumped = cfg.model_dump(mode="json")
    cfg_reloaded = Config.model_validate(dumped)
    entry = cfg_reloaded.algorithms.svr.hyperparameters["C"]
    assert entry.dist_type == "float_log"
    assert entry.explicit_type is False


def test_dump_reload_round_trip_inferred_ard_float_log_preserved():
    """An inferred ARD alpha_1 range stays float_log after a round trip."""
    cfg = _config("ardregression", {"alpha_1": [1e-6, 0.1]})
    assert cfg.algorithms.ardregression.hyperparameters["alpha_1"].dist_type == "float_log"
    dumped = cfg.model_dump(mode="json")
    cfg_reloaded = Config.model_validate(dumped)
    entry = cfg_reloaded.algorithms.ardregression.hyperparameters["alpha_1"]
    assert entry.dist_type == "float_log"
    assert entry.explicit_type is False


def test_dump_reload_round_trip_dict_form_int_preserved():
    """A dict-form entry survives the dump → reload round trip as int."""
    cfg = _config("ridge", {"alpha": {"config": {"type": "int", "low": 1, "high": 50}}})
    dumped = cfg.model_dump(mode="json")
    cfg_reloaded = Config.model_validate(dumped)
    entry = cfg_reloaded.algorithms.ridge.hyperparameters["alpha"]
    assert entry.dist_type == "int"
    assert entry.explicit_type is True


# ──────────────────────────────────────────────────────────────────────────────
# 8. Direct optimize() path is normalized too (same semantics as the Config path)
# ──────────────────────────────────────────────────────────────────────────────


def test_normalize_space_dict_retypes_inferred_entry():
    """_normalize_space_dict re-resolves [1, 50] for C of an SVR to float_log."""
    runner = _OptimizeRunner.__new__(_OptimizeRunner)
    runner.algo = "svr"
    space = {"C": [1, 50], "kernel": [["rbf", "linear"], "categorical"]}
    normalized = runner._normalize_space_dict(space)
    assert isinstance(normalized["C"], SearchSpaceEntry)
    assert normalized["C"].dist_type == "float_log"
    # Categorical lists are parsed to SearchSpaceEntry too (no longer constants).
    assert isinstance(normalized["kernel"], SearchSpaceEntry)
    assert normalized["kernel"].dist_type == "categorical"


def test_normalize_space_dict_retypes_float_log_default_param():
    """_normalize_space_dict re-resolves ARD alpha_1 [1e-6, 0.1] to float_log.

    The direct optimize() path (issue #140): alpha_1 is absent from
    SCALE_PARAMS, but its DEFAULT_SPACES entry is float_log, so an inferred
    range must be searched on a logarithmic scale.
    """
    runner = _OptimizeRunner.__new__(_OptimizeRunner)
    runner.algo = "ardregression"
    space = {"alpha_1": [1e-6, 0.1], "max_iter": 500}
    normalized = runner._normalize_space_dict(space)
    assert normalized["alpha_1"].dist_type == "float_log"
    assert isinstance(normalized["alpha_1"].config, FloatSpace)
    assert normalized["max_iter"] == 500


def test_normalize_space_dict_keeps_explicit_types():
    """Explicitly typed entries in space_overrides are not rewritten."""
    runner = _OptimizeRunner.__new__(_OptimizeRunner)
    runner.algo = "svr"
    space = {
        "C": [1, 50, "int"],  # Explicitly int, even for a float param.
        "epsilon": SearchSpaceEntry.model_validate([0.5, 1.0, "float_log"]),
    }
    normalized = runner._normalize_space_dict(space)
    assert normalized["C"].dist_type == "int"
    assert normalized["epsilon"].dist_type == "float_log"


def test_normalize_space_dict_keeps_constants():
    """Plain constants pass through unchanged."""
    runner = _OptimizeRunner.__new__(_OptimizeRunner)
    runner.algo = "svr"
    normalized = runner._normalize_space_dict({"max_iter": 1000})
    assert normalized == {"max_iter": 1000}


def test_normalize_space_dict_int_param_stays_int():
    """n_estimators [10, 500] for a random forest stays int."""
    runner = _OptimizeRunner.__new__(_OptimizeRunner)
    runner.algo = "random_forest"
    normalized = runner._normalize_space_dict({"n_estimators": [10, 500]})
    assert normalized["n_estimators"].dist_type == "int"
    assert isinstance(normalized["n_estimators"].config, IntSpace)


def test_optimize_direct_dict_override_retypes_C(monkeypatch):
    """A raw list [1, 50] through space_overrides searches C as float_log.

    The exact issue #127 bug: before the fix, ``space_overrides`` with a raw
    list treated C as an integer-only range (and a two-element list was even
    handled as a constant by the dynamic space).
    """
    import numpy as np
    import pandas as pd

    X = pd.DataFrame(np.random.default_rng(0).random((60, 4)))
    y = pd.Series(np.random.default_rng(1).random(60))

    seen_types: list[str] = []

    def mock_create_model(algo: str, **params):
        if algo == "svr" and "C" in params:
            seen_types.append(type(params["C"]).__name__)
        from sklearn.linear_model import Ridge

        return Ridge()

    monkeypatch.setattr(
        "configurable_automl_engine.tuner.create_model", mock_create_model
    )
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.get_scorer_object",
        lambda name, global_y=None: lambda model, X, y: 0.5,
    )
    monkeypatch.setattr(
        "configurable_automl_engine.tuner._evaluate_pooled_cv",
        lambda *a, **kw: 0.5,
    )
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.model_selection.cross_val_score",
        lambda *a, **kw: [0.5, 0.5],
    )

    opt_res = optimize(
        "svr",
        X,
        y,
        n_trials=5,
        random_state=42,
        validation_strategy="train_test_split",
        space_overrides={"svr": {"C": [1, 50]}},
    )
    assert opt_res is not None
    assert seen_types, "C was never passed to the model — the test is useless"
    # Before the fix the range was integer-only, so C arrived as int.
    assert all(t == "float" for t in seen_types), f"C arrived as {seen_types}"
    assert isinstance(opt_res.params["C"], float)

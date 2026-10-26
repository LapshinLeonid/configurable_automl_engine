import pytest
import logging
import re
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import (
    GeneralCfg,
    OversamplingCfg,
    SearchSpaceEntry,
    AlgoCfg,
    Config,
    read_config,
    HPOPhaseCfg,
    FeatureSelectionCfg,
    FeatureSelectionMode,
    FeatureSelectionMethod,
    SanityGateCfg,
    ERR_REMOVED_REFINE_WINNER,
    ERR_REMOVED_SANITY_GATE_MODE,
)
from configurable_automl_engine.common.hyperopt_defaults import (
    NumericSpace,
    FloatSpace,
    IntSpace,
)
from configurable_automl_engine.common.definitions import (
    ValidationStrategy,
    SerializationFormat,
)

from unittest.mock import patch

# Terminal select step: the last phase of any valid config.
TERMINAL_SELECT = {
    "name": "final_select",
    "n_trials": 1,
    "action": "select",
    "strategy": "cv_best",
    "output": "winner",
}

BASE = {
    "general": {
        "comparison_metric": "nrmse",
        "validation_strategy": "k_fold",
        "n_folds": 5,
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            TERMINAL_SELECT,
        ],
    },
    "algorithms": {"elasticnet": {"enable": True}},
}


def test_n_folds_ok():
    cfg = Config.model_validate(BASE)
    assert cfg.general.n_folds == 5


def test_general_random_state_default_42():
    """By default, general.random_state=42 (the documented engine seed)."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.random_state == 42


def test_general_random_state_explicit_int():
    """An explicit general.random_state int is preserved."""
    cfg_data = BASE | {"general": {**BASE["general"], "random_state": 7}}
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.random_state == 7


def test_general_random_state_explicit_none():
    """An explicit general.random_state=None is preserved (non-reproducible run)."""
    cfg_data = BASE | {"general": {**BASE["general"], "random_state": None}}
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.random_state is None


def test_general_random_state_invalid_type_rejected():
    """A non-integer seed is rejected (int | None only)."""
    cfg_data = BASE | {"general": {**BASE["general"], "random_state": 1.5}}
    with pytest.raises(ValidationError):
        Config.model_validate(cfg_data)


def test_categorical_encoding_ordinal_valid():
    """A config with categorical_encoding='ordinal' passes Config validation."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "categorical_encoding": "ordinal"}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.categorical_encoding == "ordinal"


def test_categorical_encoding_default_one_hot():
    """By default, categorical_encoding='one_hot'."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.categorical_encoding == "one_hot"


def test_categorical_encoding_invalid_rejected():
    """A config with categorical_encoding='binary' is rejected (Literal)."""
    cfg_data = BASE | {"general": {**BASE["general"], "categorical_encoding": "binary"}}
    with pytest.raises(ValidationError):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize(
    "enc", ["one_hot", "ordinal", "target", "frequency", "hashing"]
)
def test_categorical_encoding_new_strategies_valid(enc):
    """All supported strategies (including the new ones) pass validation."""
    cfg_data = BASE | {"general": {**BASE["general"], "categorical_encoding": enc}}
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.categorical_encoding == enc


def test_high_cardinality_encoding_pair_valid():
    """A consistent threshold + high_cardinality_encoding pair is accepted."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "categorical_encoding": "one_hot",
            "high_cardinality_threshold": 20,
            "high_cardinality_encoding": "target",
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.high_cardinality_threshold == 20
    assert cfg.general.high_cardinality_encoding == "target"


def test_high_cardinality_threshold_zero_allowed():
    """A threshold of zero is allowed (all non-empty columns become high-cardinality)."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": 0,
            "high_cardinality_encoding": "hashing",
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.high_cardinality_threshold == 0


def test_high_cardinality_threshold_negative_rejected():
    """A negative threshold is rejected during configuration validation."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": -1,
            "high_cardinality_encoding": "target",
        }
    }
    with pytest.raises(ValidationError, match="high_cardinality_threshold"):
        Config.model_validate(cfg_data)


def test_high_cardinality_encoding_unsupported_rejected():
    """An unsupported high-cardinality strategy is rejected."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": 10,
            "high_cardinality_encoding": "binary",
        }
    }
    with pytest.raises(ValidationError):
        Config.model_validate(cfg_data)


def test_high_cardinality_pair_must_be_set_together():
    """Only one parameter of the pair is set -> a clear error is raised.

    The defaults (threshold 50 + encoding 'frequency') mean that both
    half-settings are now self-consistent: setting only the threshold uses the
    default 'frequency', and setting only the encoding uses the default
    threshold 50. The pair must be explicitly disabled together (both None).
    """
    only_threshold = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": 10,
        }
    }
    # The threshold alone is valid now: the encoding defaults to 'frequency'.
    cfg = Config.model_validate(only_threshold)
    assert cfg.general.high_cardinality_threshold == 10
    assert cfg.general.high_cardinality_encoding == "frequency"

    only_encoding = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_encoding": "target",
        }
    }
    # The encoding alone is valid now: the threshold defaults to 50.
    cfg2 = Config.model_validate(only_encoding)
    assert cfg2.general.high_cardinality_threshold == 50
    assert cfg2.general.high_cardinality_encoding == "target"

    # Explicitly disabling the pair must be done together.
    partial_disable = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": None,
        }
    }
    with pytest.raises(ValidationError, match="must be set together"):
        Config.model_validate(partial_disable)


def test_high_cardinality_pair_explicitly_disabled():
    """Both parameters set to null disable the automatic mode."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": None,
            "high_cardinality_encoding": None,
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.high_cardinality_threshold is None
    assert cfg.general.high_cardinality_encoding is None


def test_hashing_n_components_zero_rejected():
    """hashing_n_components=0 is rejected."""
    cfg_data = BASE | {"general": {**BASE["general"], "hashing_n_components": 0}}
    with pytest.raises(ValidationError, match="hashing_n_components"):
        Config.model_validate(cfg_data)


def test_target_encoding_smoothing_negative_rejected():
    """Negative target-encoding smoothing is rejected."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "target_encoding_smoothing": -0.5}
    }
    with pytest.raises(ValidationError, match="target_encoding_smoothing"):
        Config.model_validate(cfg_data)


def test_target_encoding_fallback_any_float():
    """target_encoding_fallback accepts a float or None."""
    for fallback in (0.0, -5.5, 42.0):
        cfg_data = BASE | {
            "general": {**BASE["general"], "target_encoding_fallback": fallback}
        }
        cfg = Config.model_validate(cfg_data)
        assert cfg.general.target_encoding_fallback == fallback


def test_new_params_absent_defaults():
    """A config without the new parameters uses the documented defaults.

    The automatic high-cardinality mode is enabled by default:
    threshold 50 with the 'frequency' strategy, so high-cardinality columns do
    not silently expand into a dense one-hot matrix. To disable the mode, set
    both parameters to null explicitly.
    """
    cfg = Config.model_validate(BASE)
    assert cfg.general.categorical_encoding == "one_hot"
    assert cfg.general.high_cardinality_threshold == 50
    assert cfg.general.high_cardinality_encoding == "frequency"
    assert cfg.general.hashing_n_components == 16
    assert cfg.general.target_encoding_smoothing == 20.0
    assert cfg.general.target_encoding_fallback is None


def test_n_folds_bad():
    bad = BASE | {"general": {**BASE["general"], "n_folds": 1}}
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_n_folds_ignored_for_loo():
    loo = BASE | {
        "general": {**BASE["general"], "validation_strategy": "loo", "n_folds": 1}
    }
    cfg = Config.model_validate(loo)  # should not raise
    assert cfg.general.validation_strategy == ValidationStrategy.loo


# --- Tests for OversamplingCfg ---
def test_oversampling_warn_useless_multiplier(caplog):
    # Warning when multiplier=1 and enable=True
    with caplog.at_level(logging.WARNING):
        OversamplingCfg(enable=True, multiplier=1.0)

    assert "Oversampling multiplier = 1 ➜ class balance will not change" in caplog.text


def test_oversampling_algorithm_random_with_noise_rejected():
    """random_with_noise was removed from the interface: a config using it
    is rejected during validation with a clear error message."""
    with pytest.raises(ValidationError, match="random_with_noise"):
        OversamplingCfg(enable=True, multiplier=2.0, algorithm="random_with_noise")


def test_oversampling_unknown_algorithm_rejected():
    """An unknown algorithm is rejected by config validation."""
    with pytest.raises(ValidationError):
        OversamplingCfg(enable=True, multiplier=2.0, algorithm="magic_boost")


def test_oversampling_target_bins_default_and_alias():
    """target_bins defaults to 8 and is configurable via its flat alias."""
    cfg = OversamplingCfg()
    assert cfg.target_bins == 8
    cfg_alias = OversamplingCfg(data_oversampling_target_bins=16)
    assert cfg_alias.target_bins == 16


def test_oversampling_target_bins_less_than_two_rejected():
    """A target_bins value below 2 is rejected during validation."""
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        OversamplingCfg(data_oversampling_target_bins=1)


def test_oversampling_noise_defaults_and_aliases():
    """add_noise / noise_level default to off / 0.01 and accept flat aliases."""
    cfg = OversamplingCfg()
    assert cfg.add_noise is False
    assert cfg.noise_level == pytest.approx(0.01)
    cfg_alias = OversamplingCfg(
        data_oversampling=True,
        data_oversampling_add_noise=True,
        data_oversampling_noise_level=0.05,
    )
    assert cfg_alias.add_noise is True
    assert cfg_alias.noise_level == pytest.approx(0.05)


@pytest.mark.parametrize("bad_level", [0.0, -0.1])
def test_oversampling_noise_level_non_positive_rejected(bad_level):
    """A non-positive noise_level is rejected during validation."""
    with pytest.raises(ValidationError, match="greater than 0"):
        OversamplingCfg(data_oversampling_noise_level=bad_level)


def test_oversampling_noise_without_enable_warns(caplog):
    """add_noise without enabled oversampling logs a 'no effect' warning."""
    with caplog.at_level(logging.WARNING):
        OversamplingCfg(data_oversampling_add_noise=True)
    assert "has no effect" in caplog.text


# --- Tests for AlgoCfg ---

# --- Tests for the root Config and the API ---
def test_config_no_enabled_algorithms():
    # Test for the _must_have_enabled validator
    algo_disabled = AlgoCfg(enable=False)
    with pytest.raises(
        ValidationError, match=".*At least one algorithm must be enabled.*"
    ):
        Config(general=GeneralCfg(phases=[]), algorithms={"elasticnet": algo_disabled})


def test_read_config_integration(tmp_path):
    # Test for read_config and correct YAML loading
    yaml_content = """
    general:
      comparison_metric: "r2"
      phases:
        - name: "search"
          n_trials: 10
          action: "all_algorithms"
        - name: "final_select"
          n_trials: 1
          action: "select"
          strategy: "cv_best"
          output: "winner"
      validation_strategy: "k_fold"
      n_folds: 3
    algorithms:
      random_forest:
        enable: true
        hyperparameters:
          n_estimators: [[10, 50, 100], "categorical"]
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = read_config(config_file)
    assert config.general.n_folds == 3
    assert hasattr(config.algorithms, "random_forest")
    assert getattr(config.algorithms, "random_forest").enable is True


# ─────────────────── Timeouts in threads mode ───────────────────
def test_threads_mode_rejects_phase_timeout():
    """phase_timeout + parallel_mode='threads' is a config error."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "parallel_mode": "threads",
            "phase_timeout": 60,
        }
    }
    with pytest.raises(ValidationError, match="not supported in parallel_mode='threads'"):
        Config.model_validate(cfg_data)


def test_threads_mode_rejects_task_timeout():
    """task_timeout + parallel_mode='threads' is a config error."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "parallel_mode": "threads",
            "task_timeout": 30,
        }
    }
    with pytest.raises(ValidationError, match="not supported in parallel_mode='threads'"):
        Config.model_validate(cfg_data)


def test_threads_mode_rejects_both_timeouts():
    """Both timeouts + parallel_mode='threads' — a config error naming both."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "parallel_mode": "threads",
            "phase_timeout": 60,
            "task_timeout": 30,
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(cfg_data)
    msg = str(exc_info.value)
    assert "general.phase_timeout=60" in msg
    assert "general.task_timeout=30" in msg


def test_threads_mode_without_timeouts_valid():
    """Threads mode without timeouts is valid (the default config)."""
    cfg_data = BASE | {"general": {**BASE["general"], "parallel_mode": "threads"}}
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.parallel_mode == "threads"
    assert cfg.general.phase_timeout is None
    assert cfg.general.task_timeout is None


def test_processes_mode_allows_timeouts():
    """Timeouts are allowed in processes mode (hard timeout)."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "parallel_mode": "processes",
            "phase_timeout": 60,
            "task_timeout": 30,
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.parallel_mode == "processes"
    assert cfg.general.phase_timeout == 60
    assert cfg.general.task_timeout == 30


# Tests for (successful GeneralCfg validation)
def test_general_cfg_valid_n_folds():
    """The _check_n_folds validator completes successfully."""
    cfg = GeneralCfg(
        phases=[
            HPOPhaseCfg(name="test", n_trials=1),
            HPOPhaseCfg(
                name="final_select",
                n_trials=1,
                action="select",
                strategy="cv_best",
                output="winner",
            ),
        ],
        validation_strategy=ValidationStrategy.k_fold,
        n_folds=3,
    )
    assert cfg.n_folds == 3


def test_general_cfg_input_diagnostics_policies_defaults():
    """The input-data diagnostics policies have the documented defaults."""
    cfg = GeneralCfg(phases=[])
    assert cfg.inf_policy == "as_missing"
    assert cfg.datetime_policy == "expand"
    assert cfg.drop_id_like_columns is False


def test_general_cfg_input_diagnostics_policies_overridable():
    """The input-data diagnostics policies accept all documented values."""
    cfg = GeneralCfg(
        phases=[],
        inf_policy="error",
        datetime_policy="drop",
        drop_id_like_columns=True,
    )
    assert cfg.inf_policy == "error"
    assert cfg.datetime_policy == "drop"
    assert cfg.drop_id_like_columns is True


def test_general_cfg_unknown_inf_policy_rejected():
    """An unknown inf_policy value is rejected by config validation."""
    with pytest.raises(ValidationError, match="inf_policy"):
        GeneralCfg(phases=[], inf_policy="bogus")


def test_general_cfg_unknown_datetime_policy_rejected():
    """An unknown datetime_policy value is rejected by config validation."""
    with pytest.raises(ValidationError, match="datetime_policy"):
        GeneralCfg(phases=[], datetime_policy="bogus")


# 3. Tests for AlgoCfg._must_not_be_empty
def test_algo_cfg_empty_paths():
    """Checks that an empty string is rejected in module paths."""
    with pytest.raises(ValidationError) as exc_info:
        AlgoCfg(tuner="", hyperparameters={})

    error_msg = str(exc_info.value)

    # Assert specific parts of the message
    assert "tuner" in error_msg
    assert "not a valid dotted path" in error_msg
    assert "Value error" in error_msg

    with pytest.raises(ValidationError) as exc_info:
        AlgoCfg(trainer_module="")

    error_msg = str(exc_info.value)

    assert "trainer_module" in error_msg
    assert "not a valid dotted path" in error_msg
    assert "Value error" in error_msg


# Additional test for the n_folds logic (boundary conditions)
def test_general_cfg_invalid_n_folds_kfold():
    """Covers the validation error when n_folds < 2 for k_fold."""
    with pytest.raises(ValidationError, match="(?s).*n_folds must be ≥ 2.*"):
        GeneralCfg(
            phases=[
                HPOPhaseCfg(name="test", n_trials=1),
                HPOPhaseCfg(
                    name="final_select",
                    n_trials=1,
                    action="select",
                    strategy="cv_best",
                    output="winner",
                ),
            ],
            validation_strategy=ValidationStrategy.k_fold,
            n_folds=1,
        )


# Triggering a ValueError
def test_general_cfg_coverage_line_83():
    """
    We pass n_folds=0, which should trigger the first exception in
    _check_n_folds regardless of the chosen validation strategy.
    """
    # Use the first available Enum element to avoid an AttributeError
    strategy = list(ValidationStrategy)[0]

    with pytest.raises(ValidationError, match="`n_folds` must be at least 1"):
        GeneralCfg(
            phases=[
                HPOPhaseCfg(name="test", n_trials=1),
                HPOPhaseCfg(
                    name="final_select",
                    n_trials=1,
                    action="select",
                    strategy="cv_best",
                    output="winner",
                ),
            ],
            validation_strategy=strategy,
            n_folds=0,  # This triggers the raise on line 83
        )


# Successful return of v in AlgoCfg
def test_algo_cfg_coverage_line_205():
    """The module path value is returned successfully."""
    # When creating a valid AlgoCfg, the _must_not_be_empty validator
    # should return the value v (line 205)
    algo = AlgoCfg(tuner="path.to.tuner", trainer_module="path.to.trainer")
    assert algo.tuner == "path.to.tuner"
    assert algo.trainer_module == "path.to.trainer"


@patch("configurable_automl_engine.training_engine.config_parser.is_installed")
def test_joblib_not_installed_raises_error(mock_is_installed):
    """Test for the exception raised when joblib is missing."""
    # Simulate the package being absent
    mock_is_installed.return_value = False

    data = {
        "general": {
            "phases": [
                {"name": "search", "n_trials": 10},
                TERMINAL_SELECT,
            ],
            "serialization_format": "joblib",
            "validation_strategy": "k_fold",
            "n_folds": 5,
        },
        "algorithms": {"any_algo": {"enable": True}},
    }

    with pytest.raises(ValidationError, match=".*serialization_format='joblib'.*"):
        Config.model_validate(data)


@patch("configurable_automl_engine.training_engine.config_parser.is_installed")
def test_missing_algorithm_dependency_raises_error(mock_is_installed):
    """Test for the exception raised when an algorithm's library is missing."""
    mock_is_installed.return_value = False

    data = {
        "general": {
            "phases": [
                {"name": "search", "n_trials": 10},
                TERMINAL_SELECT,
            ],
            "validation_strategy": "train_test_split",
        },
        "algorithms": {"xgboosting": {"enable": True}},
    }

    expected_msg = (
        "Algorithm 'xgboosting' is enabled, but the package 'xgboost' is not installed"
    )
    with pytest.raises(ValueError, match=expected_msg):
        Config.model_validate(data)


def test_config_skips_disabled_algorithms_dependency_check():
    """
    Verifies that the dependency validator ignores disabled algorithms.
    Covers the line: if not algo_cfg.enable: continue
    """

    # Configuration data:
    # 1. 'some_custom_algo' is enabled (to pass the _must_have_enabled validator)
    # 2. 'xgboost' is disabled. Even if the xgboost library is not installed,
    #    no error should be raised thanks to 'continue'.
    config_data = {
        "general": {"phases": [{"name": "test", "n_trials": 1}, TERMINAL_SELECT]},
        "algorithms": {"elasticnet": {"enable": True}, "xgboosting": {"enable": False}},
    }
    # Simulate the xgboost library being absent from the system
    with patch(
        "configurable_automl_engine.common.dependency_utils.is_installed",
        return_value=False,
    ):
        # If continue works, the config object is created successfully.
        # Otherwise, a ValueError: "Algorithm 'xgboosting' is enabled..." is raised.
        config = Config.model_validate(config_data)

    assert getattr(config.algorithms, "xgboosting").enable is False
    assert hasattr(config.algorithms, "xgboosting")


# 1. Test for: if self.low > self.high: raise ValueError(...)
def test_numeric_space_range_validation():
    # Error case: low > high
    # Use re.escape so that brackets (10.0) are not treated as a regex group
    expected_msg = re.escape("low (10.0) must be <= high (5.0)")

    with pytest.raises(ValidationError, match=expected_msg):
        NumericSpace(type="base", low=10.0, high=5.0)

    # Success case: low == high (allowed)
    n = NumericSpace(type="base", low=5.0, high=5.0)
    assert n.low == 5.0


# 2. Test for: if self.type == "float_log" and self.step is not None:
def test_float_log_step_forbidden():
    with pytest.raises(
        ValidationError, match="The 'step' parameter is not supported for 'float_log'"
    ):
        FloatSpace(type="float_log", low=1.0, high=10.0, step=0.1)


# 2a. Test for: float_log requires low > 0
@pytest.mark.parametrize("bad_low", [0, -1, -0.0001])
def test_float_log_non_positive_low_rejected(bad_low):
    """float_log with low <= 0 is rejected during validation.

    Parameterized over the lower-bound value: covers both the compact list
    form ``[low, high, "float_log"]`` via SearchSpaceEntry (including the
    integer ``0``) and the dict form via FloatSpace.
    """
    with pytest.raises(
        ValidationError, match="low must be > 0 for log-scale distributions"
    ):
        SearchSpaceEntry.model_validate([bad_low, 1, "float_log"])

    with pytest.raises(
        ValidationError, match="low must be > 0 for log-scale distributions"
    ):
        FloatSpace(type="float_log", low=bad_low, high=1.0)


def test_float_log_positive_low_valid():
    """Positive scenario: float_log with a strictly positive low is accepted."""
    entry = SearchSpaceEntry.model_validate([1e-6, 1.0, "float_log"])
    assert entry.dist_type == "float_log"
    assert entry.low == 1e-6
    assert entry.high == 1.0
    assert entry.step is None


def test_float_log_non_positive_low_rejected_in_config():
    """A config with float_log and low <= 0 is rejected at Config.model_validate."""
    cfg_data = BASE | {
        "algorithms": {
            "ridge": {
                "enable": True,
                "hyperparameters": {"alpha": [0.0, 1.0, "float_log"]},
            }
        }
    }
    with pytest.raises(
        ValidationError, match="low must be > 0 for log-scale distributions"
    ):
        Config.model_validate(cfg_data)


# 3. Test for: if self.step is not None and self.step <= 0: (in FloatSpace and IntSpace)
def test_step_positive_validation():
    # For FloatSpace
    with pytest.raises(ValidationError, match="Step must be positive. Got -1.0"):
        FloatSpace(type="float", low=0.0, high=1.0, step=-1.0)

    # For IntSpace
    with pytest.raises(ValidationError, match="Step must be positive. Got 0"):
        IntSpace(type="int", low=1, high=10, step=0)


# 4. Test for: _parse_list_to_dict (validation of various list formats)
def test_parse_list_to_dict_logic():
    # Check len(data) >= 3 and payload["step"] = data[3]
    raw_data = [1, 10, "int", 2]
    entry = SearchSpaceEntry.model_validate(raw_data)
    assert entry.dist_type == "int"
    assert entry.step == 2

    # Test the case of a non-list input (should be returned as is).
    # Pydantic raises a validation error later if it is not a dict,
    # but _parse_list_to_dict itself should pass the data through.
    not_a_list = {"config": {"type": "int", "low": 1, "high": 5}}
    entry_from_dict = SearchSpaceEntry.model_validate(not_a_list)
    assert entry_from_dict.low == 1


# 5. Test for: the bounds property (building a list from the object)
def test_search_space_bounds_property():
    # For numeric with a step
    entry_int = SearchSpaceEntry.model_validate([1, 10, "int", 2])
    assert entry_int.bounds == [1, 10, "int", 2]

    # For categorical
    cat_data = [["a", "b"], "categorical"]
    entry_cat = SearchSpaceEntry.model_validate(cat_data)
    assert entry_cat.bounds == [["a", "b"], "categorical"]


# 5a. Test of the DEFAULT_SPACES['svr'] composition: only reliable kernels
def test_default_svr_kernel_space_only_reliable_kernels():
    """Only the rbf and linear kernels remain in the SVR search space.

    Positive scenario: options == ["rbf", "linear"].
    Negative scenario: 'sigmoid' and 'poly' are excluded from the search.
    """
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES

    kernel_entry = DEFAULT_SPACES["svr"]["kernel"]
    assert kernel_entry.dist_type == "categorical"
    assert kernel_entry.config.options == ["rbf", "linear"]
    assert "sigmoid" not in kernel_entry.config.options
    assert "poly" not in kernel_entry.config.options


def test_default_svr_gamma_stays_in_space():
    """The gamma parameter remains in the SVR search space.

    It is used for rbf and ignored for linear by sklearn, so it does not
    need to be removed.
    """
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES

    svr_space = DEFAULT_SPACES["svr"]
    assert "gamma" in svr_space
    gamma_entry = svr_space["gamma"]
    # A numeric RBF width (not just the "scale"/"auto" heuristics) that
    # covers ~1 / p of standardized features on both sides.
    assert gamma_entry.dist_type == "float_log"
    assert gamma_entry.low <= 1e-3 and gamma_entry.high >= 1.0


def test_default_spaces_cover_common_optima():
    """Default spaces do not clamp typical optima at a boundary.

    Ridge alpha on standardized data is often 10–1000 (~22 on diabetes);
    forests search max_features as a fraction of the features.
    """
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES

    ridge_alpha = DEFAULT_SPACES["ridge"]["alpha"]
    assert ridge_alpha.dist_type == "float_log"
    assert ridge_alpha.high >= 1e3
    for algo in ("random_forest", "extra_trees"):
        max_features = DEFAULT_SPACES[algo]["max_features"]
        assert max_features.dist_type == "float"
        assert 0.0 < max_features.low < max_features.high <= 1.0


# 6. Test for: _check_algorithm_dependencies (checking installed packages)
def test_algorithm_dependency_check():
    # Mock the mapping and the installation-check function
    with (
        patch(
            "configurable_automl_engine.training_engine.config_parser.ALGO_PACKAGE_MAPPING",
            {"xgboosting": "xgboost_pkg"},
        ),
        patch(
            "configurable_automl_engine.training_engine.config_parser.is_installed"
        ) as mock_installed,
    ):
        # Scenario: the package is NOT installed
        mock_installed.return_value = False

        config_data = {
            "general": {
                "phases": [{"name": "p1", "n_trials": 1}, TERMINAL_SELECT],
                "validation_strategy": "k_fold",
                "n_folds": 2,
            },
            "algorithms": {"xgboosting": {"enable": True}},
        }

        expected_msg = "Algorithm 'xgboosting' is enabled, but the package 'xgboost_pkg' is not installed"
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            Config.model_validate(config_data)
        # Scenario: the package is installed
        mock_installed.return_value = True
        cfg = Config.model_validate(config_data)
        assert getattr(cfg.algorithms, "xgboosting").enable is True


# 7. Additional test for n_folds (the general GeneralCfg logic)
def test_general_cfg_n_folds():
    base_phases = [{"name": "test", "n_trials": 1}, TERMINAL_SELECT]

    # Error case: n_folds < 1
    with pytest.raises(ValidationError, match="`n_folds` must be at least 1"):
        GeneralCfg(phases=base_phases, n_folds=0)

    # Error case: k_fold requires n_folds >= 2
    with pytest.raises(ValidationError, match=".*n_folds must be ≥ 2 for k-fold.*"):
        GeneralCfg(phases=base_phases, validation_strategy="k_fold", n_folds=1)


def test_get_unknown_hyperparameters_none():
    cfg = AlgoCfg(hyperparameters=None)
    assert cfg.get_unknown_hyperparameters("xgboosting") == []


def test_get_unknown_hyperparameters_skipped_when_unresolvable():
    """If the estimator cannot be resolved (unknown algorithm / missing dependency),
    the check is skipped gracefully — the list of unknowns is empty."""
    cfg = AlgoCfg(hyperparameters={"a": [1, 10]})
    assert cfg.get_unknown_hyperparameters("no_such_algo") == []


def test_get_unknown_hyperparameters_valid():
    cfg = AlgoCfg(
        hyperparameters={"alpha": [1e-4, 10.0]}  # valid ElasticNet parameter
    )

    assert cfg.get_unknown_hyperparameters("elasticnet") == []


def test_get_unknown_hyperparameters_legacy_name_valid():
    """The legacy name n_iter is considered valid for ardregression."""
    cfg = AlgoCfg(hyperparameters={"n_iter": [100, 500]})
    assert cfg.get_unknown_hyperparameters("ardregression") == []


def test_get_unknown_hyperparameters_unknown():
    cfg = AlgoCfg(
        hyperparameters={"bad_param": [1, 10]}  # not in the ElasticNet signature
    )

    assert cfg.get_unknown_hyperparameters("elasticnet") == ["bad_param"]


def test_validator_allows_none():
    cfg = AlgoCfg(tuner=None, trainer_module=None)
    assert cfg.tuner is None
    assert cfg.trainer_module is None


def test_validator_valid_path():
    cfg = AlgoCfg(tuner="a.b", trainer_module="x.y.z")
    assert cfg.tuner == "a.b"


def test_validator_invalid_path():
    with pytest.raises(ValueError):
        AlgoCfg(tuner="invalid-path")


# --- Tests for the preprocessing override ---
def test_algo_cfg_preprocessing_valid_override():
    """A valid preprocessing override block is accepted by the config."""
    cfg = AlgoCfg(preprocessing={"imputation_strategy": "median", "scaling": "none"})
    assert cfg.preprocessing is not None
    assert cfg.preprocessing.imputation_strategy == "median"
    assert cfg.preprocessing.scaling == "none"


def test_algo_cfg_preprocessing_partial_override():
    """Partial override: unset fields remain None."""
    cfg = AlgoCfg(preprocessing={"scaling": "robust"})
    assert cfg.preprocessing.scaling == "robust"
    assert cfg.preprocessing.imputation_strategy is None


def test_algo_cfg_preprocessing_default_none():
    """Without a preprocessing block, the auto-selected default is None."""
    assert AlgoCfg().preprocessing is None


def test_algo_cfg_preprocessing_invalid_scaling_rejected():
    """An invalid scaling value is rejected during validation
    with a clear error message (negative scenario)."""
    with pytest.raises(ValidationError, match="scaling"):
        AlgoCfg(preprocessing={"scaling": "quantile"})


def test_algo_cfg_preprocessing_invalid_imputation_rejected():
    with pytest.raises(ValidationError, match="imputation_strategy"):
        AlgoCfg(preprocessing={"imputation_strategy": "mode"})


def test_algo_cfg_preprocessing_unknown_field_rejected():
    """Unknown fields in the override block are rejected (extra='forbid')."""
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        AlgoCfg(preprocessing={"unknown_field": 1})


def test_config_preprocessing_override_end_to_end():
    """The preprocessing block in an algorithm config reaches Config without loss."""
    cfg_data = BASE | {
        "algorithms": {
            "elasticnet": {
                "enable": True,
                "preprocessing": {"imputation_strategy": "median"},
            }
        }
    }
    cfg = Config.model_validate(cfg_data)
    algo_cfg = getattr(cfg.algorithms, "elasticnet")
    assert algo_cfg.preprocessing.imputation_strategy == "median"
    assert algo_cfg.preprocessing.scaling is None


def test_hyperparameter_compatibility_error():
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]  # elasticnet

    cfg_data = {
        "general": {"phases": [{"name": "p1", "n_trials": 1}, TERMINAL_SELECT]},
        "algorithms": {
            algo_name: {"enable": True, "hyperparameters": {"bad": [1, 10]}}
        },
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)


def test_hyperparameter_compatibility_error_lists_allowed():
    """The error message contains the actual list of allowed hyperparameters.

    Regression test: previously the literal ``{allowed}`` was substituted
    into the text (the result of ``sorted(...)`` was discarded), making the
    message useless. The allowed set is now built from the estimator's
    constructor signature.
    """
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]  # elasticnet

    cfg_data = BASE | {
        "algorithms": {algo_name: {"enable": True, "hyperparameters": {"bad": [1, 10]}}}
    }

    with pytest.raises(ValueError) as exc_info:
        Config.model_validate(cfg_data)

    msg = str(exc_info.value)
    assert "unknown hyperparameters ['bad']" in msg
    assert "Allowed parameters:" in msg
    assert "'alpha'" in msg  # the actual ElasticNet parameter appears in the list
    assert "'l1_ratio'" in msg
    assert "{allowed}" not in msg


def test_hyperparameter_compatibility_valid_hyperparameters_pass():
    """Compatible hyperparameters do not cause a validation error."""
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]  # elasticnet

    cfg_data = BASE | {
        "algorithms": {
            algo_name: {"enable": True, "hyperparameters": {"alpha": [1e-4, 10.0]}}
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, algo_name).enable is True


def test_isotonic_feature_index_accepted_by_config_validation():
    """``feature_index`` is a valid isotonic hyperparameter in Config.

    The value is not an ``IsotonicRegression`` constructor parameter but a
    data-adapter hint consumed by ``IsotonicDataTransformer``, so the
    compatibility check must accept it.
    """
    cfg_data = BASE | {
        "algorithms": {
            "isotonic_regression": {
                "enable": True,
                "hyperparameters": {
                    "feature_index": [0, 3],
                    "increasing": [[True, False], "categorical"],
                },
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, "isotonic_regression").enable is True


def test_hyperparameter_compatibility_nondefault_valid_params():
    """Non-default but estimator-valid parameters pass validation.

    Regression test: min_samples_split and max_features are absent from
    DEFAULT_SPACES['random_forest'] but are accepted by the
    RandomForestRegressor constructor.
    """
    cfg_data = BASE | {
        "algorithms": {
            "random_forest": {
                "enable": True,
                "hyperparameters": {
                    "min_samples_split": [2, 10],
                    "max_features": [0.1, 1.0],
                    "warm_start": [[True, False], "categorical"],
                },
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    rf_cfg = getattr(cfg.algorithms, "random_forest")
    assert rf_cfg.enable is True
    assert set(rf_cfg.hyperparameters) == {
        "min_samples_split",
        "max_features",
        "warm_start",
    }


def test_hyperparameter_compatibility_gpr_alpha():
    """alpha for gaussian_process_regression passes validation.

    alpha is a standard GaussianProcessRegressor constructor parameter, and
    the non-empty DEFAULT_SPACES entry does not restrict user overrides.
    """
    cfg_data = BASE | {
        "algorithms": {
            "gaussian_process_regression": {
                "enable": True,
                "hyperparameters": {"alpha": [1e-10, 1.0]},
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, "gaussian_process_regression").enable is True


def test_hyperparameter_compatibility_gpr_kernel_type():
    """kernel_type (the GPR kernel selector) passes validation."""
    cfg_data = BASE | {
        "algorithms": {
            "gaussian_process_regression": {
                "enable": True,
                "hyperparameters": {
                    "kernel_type": [
                        ["rbf", "matern", "rational_quadratic"],
                        "categorical",
                    ]
                },
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, "gaussian_process_regression").enable is True


def test_hyperparameter_compatibility_constant_value():
    """A single (constant) value of a valid parameter passes.

    SearchSpaceEntry treats [value] as a categorical with a single option —
    fixing any valid parameter should not be rejected.
    """
    cfg_data = BASE | {
        "algorithms": {
            "random_forest": {
                "enable": True,
                "hyperparameters": {"min_samples_split": [2]},
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert (
        getattr(cfg.algorithms, "random_forest")
        .hyperparameters["min_samples_split"]
        .dist_type
        == "categorical"
    )


def test_hyperparameter_compatibility_legacy_name():
    """The legacy name n_iter for ardregression passes validation."""
    cfg_data = BASE | {
        "algorithms": {
            "ardregression": {
                "enable": True,
                "hyperparameters": {"n_iter": [100, 500]},
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, "ardregression").enable is True


def test_hyperparameter_compatibility_typo_rejected():
    """A typo in a parameter name is still rejected."""
    cfg_data = BASE | {
        "algorithms": {
            "random_forest": {
                "enable": True,
                "hyperparameters": {"min_samples_spilt": [2, 10]},
            }
        }
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)


def test_hyperparameter_compatibility_gpr_typo_rejected():
    """A typo for gaussian_process_regression is now rejected too.

    Previously, the empty DEFAULT_SPACES['gaussian_process_regression']
    registry silently accepted any keys, including typos.
    """
    cfg_data = BASE | {
        "algorithms": {
            "gaussian_process_regression": {
                "enable": True,
                "hyperparameters": {"aplha": [1e-10, 1.0]},
            }
        }
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)


def test_hyperparameter_compatibility_skips_disabled():
    """Disabled algorithms are skipped by the compatibility check.

    algo_b is disabled and holds an invalid hyperparameter: the test passes
    only thanks to the ``enable=False`` guard.
    """
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_a, algo_b = AVAILABLE_ALGORITHMS[0], AVAILABLE_ALGORITHMS[1]

    cfg_data = BASE | {
        "algorithms": {
            algo_a: {"enable": True},
            algo_b: {"enable": False, "hyperparameters": {"bad": [1, 10]}},
        }
    }

    cfg = Config.model_validate(cfg_data)  # should not raise
    assert getattr(cfg.algorithms, algo_a).enable is True
    assert getattr(cfg.algorithms, algo_b).enable is False


def test_hyperparameter_compatibility_aggregates_errors():
    """Errors for multiple algorithms are aggregated into a single exception."""
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_a, algo_b = AVAILABLE_ALGORITHMS[0], AVAILABLE_ALGORITHMS[1]

    cfg_data = BASE | {
        "algorithms": {
            algo_a: {"enable": True, "hyperparameters": {"bad": [1, 10]}},
            algo_b: {"enable": True, "hyperparameters": {"wrong": [0, 1]}},
        }
    }

    with pytest.raises(ValueError) as exc_info:
        Config.model_validate(cfg_data)

    msg = str(exc_info.value)
    assert f"Algorithm '{algo_a}': unknown hyperparameters ['bad']" in msg
    assert f"Algorithm '{algo_b}': unknown hyperparameters ['wrong']" in msg


# ─────────────────── Tests for PruningCfg (early stopping) ───────────────────
def test_pruning_disabled_by_default():
    """Early stopping is disabled by default: a config without the block behaves as before."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.pruning.enable is False
    assert cfg.general.pruning.strategy.value == "median"
    assert cfg.general.pruning.min_steps == 1
    assert cfg.general.pruning.n_startup_trials == 5
    assert cfg.general.pruning.reduction_factor == 3


def test_pruning_valid_median_config():
    """A valid median-pruner configuration passes validation."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 5,
            "pruning": {
                "enable": True,
                "strategy": "median",
                "min_steps": 2,
                "n_startup_trials": 3,
            },
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.pruning.enable is True
    assert cfg.general.pruning.strategy.value == "median"
    assert cfg.general.pruning.min_steps == 2
    assert cfg.general.pruning.n_startup_trials == 3


def test_pruning_valid_hyperband_config():
    """A valid hyperband-pruner configuration passes validation."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 5,
            "pruning": {
                "enable": True,
                "strategy": "hyperband",
                "min_steps": 1,
                "reduction_factor": 2,
            },
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.pruning.strategy.value == "hyperband"
    assert cfg.general.pruning.reduction_factor == 2


def test_pruning_unknown_strategy_rejected():
    """An unknown strategy is rejected during configuration validation."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "strategy": "unknown_strategy"},
        }
    }
    with pytest.raises(ValidationError, match="unknown_strategy"):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("bad_min_steps", [0, -1, -100])
def test_pruning_invalid_min_steps_rejected(bad_min_steps):
    """Invalid min_steps values (< 1) are rejected."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "min_steps": bad_min_steps},
        }
    }
    with pytest.raises(ValidationError, match="min_steps"):
        Config.model_validate(cfg_data)


def test_pruning_invalid_n_startup_trials_rejected():
    """An invalid n_startup_trials value (< 1) is rejected."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "n_startup_trials": 0},
        }
    }
    with pytest.raises(ValidationError, match="n_startup_trials"):
        Config.model_validate(cfg_data)


def test_pruning_invalid_reduction_factor_rejected():
    """An invalid reduction_factor value (< 2) is rejected."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "strategy": "hyperband", "reduction_factor": 1},
        }
    }
    with pytest.raises(ValidationError, match="reduction_factor"):
        Config.model_validate(cfg_data)


def test_pruning_conflict_kfold_min_steps_exceeds_folds():
    """Validation conflict: min_steps > n_folds with k_fold is rejected."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "pruning": {"enable": True, "min_steps": 3},
        }
    }
    with pytest.raises(
        ValidationError,
        match="pruning.min_steps .* cannot be greater than general.n_folds",
    ):
        Config.model_validate(cfg_data)


def test_pruning_disabled_skips_kfold_conflict_check():
    """With enable=False the min_steps/n_folds conflict is not checked (backward compatibility)."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "pruning": {"enable": False, "min_steps": 10},
        }
    }
    cfg = Config.model_validate(cfg_data)  # should not raise
    assert cfg.general.pruning.enable is False


def test_pruning_train_test_split_warns_and_allows(caplog):
    """Pruning + train_test_split: the config is accepted with a warning.

    For validation strategies without natural steps, the pruner is not
    applied — the behavior is documented and safe.
    """
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "train_test_split",
            "pruning": {"enable": True, "min_steps": 1},
        }
    }
    with caplog.at_level(logging.WARNING):
        cfg = Config.model_validate(cfg_data)
    assert cfg.general.pruning.enable is True
    assert "early stopping will NOT be applied" in caplog.text


def test_pruning_allows_loo_and_auto():
    """Pruning is allowed for loo and auto (validates without errors)."""
    for strategy in ("loo", "auto"):
        cfg_data = BASE | {
            "general": {
                **BASE["general"],
                "validation_strategy": strategy,
                "pruning": {"enable": True, "min_steps": 1},
            }
        }
        cfg = Config.model_validate(cfg_data)
        assert cfg.general.pruning.enable is True


# ─────────────── Tests for feature_selection ───────────────
def test_feature_selection_defaults_backward_compat():
    """A config without the feature_selection block parses with the default
    disabled mode.

    Backward compatibility: the absence of the section does not change the
    library's behavior, and all parameters keep their defaults.
    """
    cfg = Config.model_validate(BASE)
    fs = cfg.general.feature_selection
    assert fs.mode == FeatureSelectionMode.disabled
    assert fs.method == FeatureSelectionMethod.importance
    assert fs.percentile == 50.0
    assert fs.min_features == 2
    assert fs.variance_threshold == 0.0
    assert fs.n_estimators == 50


@pytest.mark.parametrize("mode", ["disabled", "always", "auto"])
def test_feature_selection_explicit_modes_valid(mode):
    """All supported feature-selection modes pass validation."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"mode": mode}}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.feature_selection.mode == FeatureSelectionMode(mode)


@pytest.mark.parametrize(
    "method", ["importance", "percentile", "mutual_info", "variance"]
)
def test_feature_selection_methods_valid(method):
    """Each of the four feature-selection methods parses correctly."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"method": method}}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.feature_selection.method == FeatureSelectionMethod(method)


def test_feature_selection_full_block_end_to_end():
    """The full feature_selection block reaches Config without loss."""
    fs_block = {
        "mode": "always",
        "method": "mutual_info",
        "percentile": 20.5,
        "min_features": 3,
        "variance_threshold": 0.1,
        "n_estimators": 100,
    }
    cfg_data = BASE | {"general": {**BASE["general"], "feature_selection": fs_block}}
    cfg = Config.model_validate(cfg_data)
    fs = cfg.general.feature_selection
    assert fs.mode == FeatureSelectionMode.always
    assert fs.method == FeatureSelectionMethod.mutual_info
    assert fs.percentile == 20.5
    assert fs.min_features == 3
    assert fs.variance_threshold == 0.1
    assert fs.n_estimators == 100


def test_feature_selection_percentile_int_coerced_to_float():
    """An integer percentile literal from YAML is coerced to float without errors."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"percentile": 50}}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.feature_selection.percentile == 50.0
    assert isinstance(cfg.general.feature_selection.percentile, float)


def test_feature_selection_model_standalone():
    """FeatureSelectionCfg can be used directly (without Config)."""
    fs = FeatureSelectionCfg(mode="auto", method="variance", n_estimators=25)
    assert fs.mode == FeatureSelectionMode.auto
    assert fs.method == FeatureSelectionMethod.variance
    assert fs.n_estimators == 25


@pytest.mark.parametrize(
    "field,value",
    [
        ("percentile", 100.0),
        ("percentile", 0.01),
        ("min_features", 1),
        ("variance_threshold", 0.0),
        ("n_estimators", 10),
    ],
)
def test_feature_selection_boundary_values_valid(field, value):
    """Boundary-valid parameter values are accepted by the schema."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {field: value}}
    }
    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.general.feature_selection, field) == value


@pytest.mark.parametrize("mode", ["magic", 123, "ALWAYS"])
def test_feature_selection_invalid_mode_rejected(mode):
    """An unknown or case-mismatched mode is rejected.

    Enum value comparison is strict: 'ALWAYS' is not equal to 'always'.
    """
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"mode": mode}}
    }
    with pytest.raises(ValidationError, match="feature_selection"):
        Config.model_validate(cfg_data)
    with pytest.raises(ValidationError, match=str(mode)):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("method", ["pca", "rfe"])
def test_feature_selection_invalid_method_rejected(method):
    """An unknown feature-selection method is rejected during parsing."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"method": method}}
    }
    with pytest.raises(ValidationError, match=method):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("percentile", [0.0, -10.0, 105.0])
def test_feature_selection_invalid_percentile_rejected(percentile):
    """A percentile outside the (0, 100] range is rejected."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"percentile": percentile}}
    }
    with pytest.raises(ValidationError, match="percentile"):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("min_features", [0, -1])
def test_feature_selection_invalid_min_features_rejected(min_features):
    """min_features < 1 is rejected (protection against an empty matrix)."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "feature_selection": {"min_features": min_features},
        }
    }
    with pytest.raises(ValidationError, match="min_features"):
        Config.model_validate(cfg_data)


def test_feature_selection_negative_variance_threshold_rejected():
    """A negative variance_threshold is rejected."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "feature_selection": {"variance_threshold": -0.01},
        }
    }
    with pytest.raises(ValidationError, match="variance_threshold"):
        Config.model_validate(cfg_data)


def test_feature_selection_small_n_estimators_rejected():
    """n_estimators < 10 is rejected (the ge=10 threshold)."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"n_estimators": 5}}
    }
    with pytest.raises(ValidationError, match="n_estimators"):
        Config.model_validate(cfg_data)


def test_feature_selection_extra_field_rejected():
    """Extra keys inside the block are rejected (extra='forbid')."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "feature_selection": {"unknown_param": 42},
        }
    }
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        Config.model_validate(cfg_data)


def test_feature_selection_invalid_rejected_even_when_disabled():
    """Strict schema: invalid parameters are rejected even at mode='disabled'."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "feature_selection": {"mode": "disabled", "n_estimators": 3},
        }
    }
    with pytest.raises(ValidationError, match="n_estimators"):
        Config.model_validate(cfg_data)


def test_read_config_feature_selection_yaml(tmp_path):
    """A real YAML with the feature_selection section parses into a correct Config."""
    yaml_content = """
    general:
      comparison_metric: "r2"
      phases:
        - name: "search"
          n_trials: 5
          action: "all_algorithms"
        - name: "final_select"
          n_trials: 1
          action: "select"
          strategy: "cv_best"
          output: "winner"
      validation_strategy: "k_fold"
      n_folds: 3
      feature_selection:
        mode: "always"
        method: "percentile"
        percentile: 25
        min_features: 4
        variance_threshold: 0.0
        n_estimators: 30
    algorithms:
      elasticnet:
        enable: true
    """
    config_file = tmp_path / "test_config_fs.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = read_config(config_file)
    fs = config.general.feature_selection
    assert fs.mode == FeatureSelectionMode.always
    assert fs.method == FeatureSelectionMethod.percentile
    assert fs.percentile == 25.0
    assert fs.min_features == 4
    assert fs.variance_threshold == 0.0
    assert fs.n_estimators == 30


def test_read_config_feature_selection_absent_defaults(tmp_path):
    """A YAML without the feature_selection section defaults to the disabled mode."""
    yaml_content = """
    general:
      comparison_metric: "r2"
      phases:
        - name: "search"
          n_trials: 5
          action: "all_algorithms"
        - name: "final_select"
          n_trials: 1
          action: "select"
          strategy: "cv_best"
          output: "winner"
      validation_strategy: "k_fold"
      n_folds: 3
    algorithms:
      elasticnet:
        enable: true
    """
    config_file = tmp_path / "test_config_no_fs.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = read_config(config_file)
    fs = config.general.feature_selection
    assert fs.mode == FeatureSelectionMode.disabled
    assert fs.method == FeatureSelectionMethod.importance
    assert fs.percentile == 50.0
    assert fs.min_features == 2
    assert fs.variance_threshold == 0.0
    assert fs.n_estimators == 50


# ─────────────────────── Sanity Gate v2 ───────────────────────


def _sg_cfg(sanity_gate_block: dict) -> dict:
    """A config with the ``general.sanity_gate`` block on top of the base one."""
    return {
        "general": {**BASE["general"], "sanity_gate": sanity_gate_block},
        "algorithms": BASE["algorithms"],
    }


def test_sanity_gate_v2_defaults():
    """Defaults of the sanity_gate v2 block: soft thresholds from the spec.

    The mode field was removed — the section is only a source of thresholds.
    """
    cfg = Config.model_validate(BASE)
    sg = cfg.general.sanity_gate
    assert "mode" not in SanityGateCfg.model_fields
    assert sg.top_k_candidates == 3
    assert sg.corridor_delta == 0.15
    assert sg.corridor_mode == "auto"
    assert sg.min_prediction_diversity == 0.15
    assert sg.adaptive_diversity is True  # constant-model baseline by default
    assert sg.min_unique_count == 5
    assert sg.min_unique_ratio == 0.05
    assert sg.max_dead_feature_ratio == 0.4
    assert sg.dead_features_require_low_diversity is True
    assert sg.max_generalization_gap == 1.5
    assert sg.generalization_gap_mode == "warn"
    assert sg.check_permutation_sensitivity is True
    assert sg.permutation_max_rows == 5000
    assert sg.permutation_max_features == 100
    assert sg.permutation_repeats == 3
    assert sg.permutation_seed == 42
    assert sg.audit_time_budget_seconds == 0
    assert sg.check_diversity is True
    assert sg.check_unique is True
    assert sg.check_dead_features is True
    assert sg.check_generalization_gap is True


def test_sanity_gate_v2_full_block_parses():
    """The full sanity_gate v2 block parses."""
    block = {
        "top_k_candidates": 5,
        "corridor_delta": 0.2,
        "corridor_mode": "multiplicative",
        "enforce_family_diversity": True,
        "algorithm_families": {"ridge": "linear", "svr": "kernel"},
        "allowed_families": ["linear", "kernel"],
        "family_diversity_multiplier": 1.3,
        "min_prediction_diversity": 0.25,
        "adaptive_diversity": True,
        "min_unique_count": 10,
        "min_unique_ratio": 0.1,
        "max_dead_feature_ratio": 0.5,
        "dead_features_require_low_diversity": False,
        "zero_coef_tolerance": 1e-10,
        "max_generalization_gap": 2.0,
        "generalization_gap_mode": "reject",
        "check_permutation_sensitivity": False,
        "permutation_max_rows": 2000,
        "permutation_max_features": 50,
        "permutation_repeats": 5,
        "permutation_seed": 7,
        "permutation_tolerance": 1e-2,
        "audit_time_budget_seconds": 120.5,
        "check_diversity": True,
        "check_unique": True,
        "check_dead_features": True,
        "check_generalization_gap": True,
    }
    cfg = Config.model_validate(_sg_cfg(block))
    sg = cfg.general.sanity_gate
    assert sg.top_k_candidates == 5
    assert sg.corridor_delta == 0.2
    assert sg.corridor_mode == "multiplicative"
    assert sg.enforce_family_diversity is True
    assert sg.algorithm_families == {"ridge": "linear", "svr": "kernel"}
    assert sg.allowed_families == ["linear", "kernel"]
    assert sg.family_diversity_multiplier == 1.3
    assert sg.min_prediction_diversity == 0.25
    assert sg.adaptive_diversity is True
    assert sg.min_unique_count == 10
    assert sg.min_unique_ratio == 0.1
    assert sg.max_dead_feature_ratio == 0.5
    assert sg.dead_features_require_low_diversity is False
    assert sg.zero_coef_tolerance == 1e-10
    assert sg.max_generalization_gap == 2.0
    assert sg.generalization_gap_mode == "reject"
    assert sg.check_permutation_sensitivity is False
    assert sg.permutation_max_rows == 2000
    assert sg.permutation_max_features == 50
    assert sg.permutation_repeats == 5
    assert sg.permutation_seed == 7
    assert sg.permutation_tolerance == 1e-2
    assert sg.audit_time_budget_seconds == 120.5
    assert sg.check_diversity is True
    assert sg.check_unique is True
    assert sg.check_dead_features is True
    assert sg.check_generalization_gap is True


@pytest.mark.parametrize("legacy_mode", ["off", "warn_only", "active", "weird"])
def test_sanity_gate_v2_mode_removed_with_migration_hint(legacy_mode):
    """The sanity_gate.mode field was removed: a config using it is rejected
    with a clear error and a migration hint to an explicit select phase.

    The error is identified by a stable code
    (``ERR_REMOVED_SANITY_GATE_MODE`` in error['type']); the meaningful
    parts of the hint (mentioning the field and the select phase) are
    checked separately.
    """
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(_sg_cfg({"mode": legacy_mode}))
    error_types = {e["type"] for e in exc_info.value.errors()}
    assert ERR_REMOVED_SANITY_GATE_MODE in error_types
    msg = str(exc_info.value)
    assert "sanity_gate.mode" in msg
    assert "select" in msg


@pytest.mark.parametrize("corridor_mode", ["multiplicative", "additive", "auto"])
def test_sanity_gate_v2_corridor_mode_valid(corridor_mode):
    """All three corridor forms are accepted by the schema."""
    cfg = Config.model_validate(_sg_cfg({"corridor_mode": corridor_mode}))
    assert cfg.general.sanity_gate.corridor_mode == corridor_mode


def test_sanity_gate_v2_corridor_mode_invalid_rejected():
    """An unknown corridor form is rejected during validation."""
    with pytest.raises(ValidationError, match="corridor_mode"):
        Config.model_validate(_sg_cfg({"corridor_mode": "geometric"}))


@pytest.mark.parametrize("gap_mode", ["warn", "reject"])
def test_sanity_gate_v2_generalization_gap_mode_valid(gap_mode):
    """Both generalization_gap_mode values are accepted by the schema."""
    cfg = Config.model_validate(_sg_cfg({"generalization_gap_mode": gap_mode}))
    assert cfg.general.sanity_gate.generalization_gap_mode == gap_mode


@pytest.mark.parametrize(
    "bad_value", ["invalid", "", "warn_only", "active", "off", 1, None]
)
def test_sanity_gate_v2_generalization_gap_mode_invalid_rejected(bad_value):
    """An unknown value of generalization_gap_mode is rejected."""
    with pytest.raises(ValidationError, match="generalization_gap_mode"):
        Config.model_validate(_sg_cfg({"generalization_gap_mode": bad_value}))


@pytest.mark.parametrize(
    "field,value",
    [
        # Boundary values (equality with the threshold is valid).
        ("min_prediction_diversity", 1.0),
        ("min_unique_ratio", 1.0),
        ("max_dead_feature_ratio", 0.0),
        ("max_dead_feature_ratio", 1.0),
        ("max_generalization_gap", 1.0),
        ("top_k_candidates", 1),
        ("min_unique_count", 1),
        # corridor_delta → 0 (strictly positive, but arbitrarily small).
        ("corridor_delta", 1e-9),
        # Multipliers and tolerances: non-negative/positive boundaries.
        ("family_diversity_multiplier", 1e-9),
        ("zero_coef_tolerance", 0.0),
        ("permutation_tolerance", 0.0),
        ("audit_time_budget_seconds", 0.0),
    ],
)
def test_sanity_gate_v2_threshold_boundaries_valid(field, value):
    """Boundary threshold values of the v2 block pass validation."""
    cfg = Config.model_validate(_sg_cfg({field: value}))
    assert getattr(cfg.general.sanity_gate, field) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_prediction_diversity", 0),
        ("min_prediction_diversity", -0.1),
        ("min_prediction_diversity", 1.5),
        ("min_unique_ratio", 0),
        ("min_unique_ratio", 1.5),
        ("max_dead_feature_ratio", -0.1),
        ("max_dead_feature_ratio", 1.1),
        ("max_generalization_gap", 0.5),
        ("max_generalization_gap", 0.99),
        ("top_k_candidates", 0),
        ("top_k_candidates", -1),
        ("min_unique_count", 0),
        ("corridor_delta", 0),
        ("corridor_delta", 1.0),
        ("corridor_delta", -0.1),
        ("family_diversity_multiplier", 0),
        ("family_diversity_multiplier", -1),
        ("zero_coef_tolerance", -1e-6),
        ("permutation_tolerance", -1e-6),
        ("permutation_repeats", 0),
        ("permutation_max_rows", 0),
        ("permutation_max_features", 0),
        ("audit_time_budget_seconds", -0.1),
    ],
)
def test_sanity_gate_v2_threshold_out_of_bounds_rejected(field, value):
    """Threshold values out of range are rejected with the field name mentioned."""
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: value}))


def test_sanity_gate_v2_extra_forbid():
    """Extra keys inside the sanity_gate block are rejected (extra='forbid')."""
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        Config.model_validate(_sg_cfg({"unknown_sanity_key": True}))


@pytest.mark.parametrize(
    "field",
    [
        "check_diversity",
        "check_unique",
        "check_dead_features",
        "check_generalization_gap",
        "adaptive_diversity",
        "dead_features_require_low_diversity",
        "enforce_family_diversity",
        "check_permutation_sensitivity",
    ],
)
def test_sanity_gate_v2_check_flags_accept_bools_only(field):
    """Circuit flag fields accept bool; incompatible types are rejected.

    In lax mode Pydantic coerces ``1``/``"yes"`` to True, but lists,
    dicts and None — incompatible types — are rejected.
    """
    for flag_value in (True, False):
        cfg = Config.model_validate(_sg_cfg({field: flag_value}))
        assert getattr(cfg.general.sanity_gate, field) == flag_value
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: [1]}))
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: {"value": True}}))
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: None}))


def test_sanity_gate_v2_empty_block_is_valid_thresholds_only():
    """An empty sanity_gate block (``{}``) is valid — the section is a
    source of thresholds, not a behavioral flag."""
    cfg = Config.model_validate(_sg_cfg({}))
    sg = cfg.general.sanity_gate
    assert "mode" not in SanityGateCfg.model_fields
    assert sg.corridor_delta == 0.15
    assert sg.min_prediction_diversity == 0.15
    assert sg.adaptive_diversity is True
    assert sg.max_generalization_gap == 1.5
    assert sg.generalization_gap_mode == "warn"


# ─────────────── Pipeline phases: refine / select ───────────────


def _phase(**overrides: object) -> dict:
    """A dict for a single pipeline phase on top of defaults."""
    base: dict = {"name": "phase", "n_trials": 1, "action": "all_algorithms"}
    base.update(overrides)
    return base


def _cfg_with_phases(*phases: dict) -> dict:
    """A config with the given phase sequence on top of BASE."""
    return {
        "general": {**BASE["general"], "phases": list(phases)},
        "algorithms": BASE["algorithms"],
    }


# ── Positive scenarios ──


@pytest.mark.parametrize(
    "phase",
    [
        {"name": "refine_all", "n_trials": 5, "action": "refine"},
        {
            "name": "sel_winner",
            "n_trials": 1,
            "action": "select",
            "strategy": "cv_best",
            "output": "winner",
        },
        {
            "name": "sel_topk",
            "n_trials": 1,
            "action": "select",
            "strategy": "cv_best",
            "output": "top_k",
            "k": 1,
        },
        {
            "name": "sel_pool",
            "n_trials": 1,
            "action": "select",
            "strategy": "cv_best",
            "output": "pool",
        },
        {
            "name": "gate_winner",
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "winner",
        },
        {
            "name": "gate_topk",
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "top_k",
            "k": 3,
        },
        {
            "name": "gate_pool",
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "pool",
        },
    ],
)
def test_phase_actions_valid_combinations(phase):
    """All valid action/strategy/output/k combinations are parsed by the schema.

    Phases with output='winner' come last (the terminal step); the rest go
    in the middle of the pipeline before the terminal select.
    """
    is_terminal = phase.get("output") == "winner"
    phases = [_phase(), phase] if is_terminal else [_phase(), phase, TERMINAL_SELECT]
    cfg = Config.model_validate(_cfg_with_phases(*phases))
    parsed = cfg.general.phases[1]
    assert parsed.action == phase["action"]
    assert parsed.strategy == phase.get("strategy")
    assert parsed.output == phase.get("output")
    assert parsed.k == phase.get("k")


def test_refine_default_fields_are_none():
    """For all_algorithms/refine phases the strategy/output/k fields are None."""
    cfg = Config.model_validate(
        _cfg_with_phases(
            _phase(action="all_algorithms"), _phase(action="refine"), TERMINAL_SELECT
        )
    )
    for phase in cfg.general.phases[:2]:
        assert phase.strategy is None
        assert phase.output is None
        assert phase.k is None


def test_select_immediately_after_select_is_valid():
    """select immediately after select is valid (edge case): top_k → winner."""
    cfg = Config.model_validate(
        _cfg_with_phases(
            _phase(),
            _phase(action="select", strategy="sanity_gate", output="top_k", k=2),
            TERMINAL_SELECT,
        )
    )
    assert [p.action for p in cfg.general.phases] == [
        "all_algorithms",
        "select",
        "select",
    ]


def test_refine_right_after_all_algorithms_is_valid():
    """refine right after all_algorithms is valid (edge case)."""
    cfg = Config.model_validate(
        _cfg_with_phases(_phase(), _phase(action="refine"), TERMINAL_SELECT)
    )
    assert [p.action for p in cfg.general.phases] == [
        "all_algorithms",
        "refine",
        "select",
    ]


def test_sanity_gate_section_without_sanity_select_is_valid():
    """A sanity_gate section without select phases using strategy:
    sanity_gate is valid: the section is a source of thresholds, not a
    behavioral flag."""
    data = _cfg_with_phases(_phase(), TERMINAL_SELECT)
    data["general"]["sanity_gate"] = {
        "top_k_candidates": 2,
        "corridor_delta": 0.3,
        "audit_time_budget_seconds": 60,
    }
    cfg = Config.model_validate(data)
    assert cfg.general.sanity_gate.top_k_candidates == 2
    assert cfg.general.sanity_gate.corridor_delta == 0.3


def test_empty_phases_list_is_still_accepted():
    """An empty phase list is not checked by cross-validation (the protection
    against an empty pipeline stays a runtime check in train_best_model)."""
    cfg = Config(
        general=GeneralCfg(phases=[]), algorithms={"elasticnet": {"enable": True}}
    )
    assert cfg.general.phases == []


# ── Negative scenarios ──


def test_refine_winner_rejected_with_migration_hint():
    """The 'refine_winner' action was removed: the config is rejected with
    a migration hint (not silently ignored).

    The test depends on the stable error code
    (``ERR_REMOVED_REFINE_WINNER`` in error['type']) and separately on the
    meaningful parts of the hint text; the exact wording of the hint may
    change without breaking the test.
    """
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(
            _cfg_with_phases(_phase(action="refine_winner"), TERMINAL_SELECT)
        )
    error_types = {e["type"] for e in exc_info.value.errors()}
    assert ERR_REMOVED_REFINE_WINNER in error_types
    msg = str(exc_info.value)
    assert "refine_winner" in msg
    assert "has been removed" in msg
    assert "select" in msg and "refine" in msg
    assert "API_REFERENCE.md" in msg


def test_select_first_phase_rejected():
    """select cannot be the first phase: there is nothing to rank or audit."""
    with pytest.raises(ValidationError, match="cannot be the first phase"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(action="select", strategy="cv_best", output="winner")
            )
        )


def test_phases_after_terminal_select_rejected():
    """Phases after the terminal select (output: winner) are forbidden."""
    with pytest.raises(ValidationError, match="terminal"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(),
                TERMINAL_SELECT,
                _phase(action="refine"),
            )
        )


def test_select_winner_in_middle_rejected():
    """A select(output: winner) in the middle of the pipeline is rejected."""
    with pytest.raises(ValidationError, match="terminal"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(),
                _phase(action="select", strategy="cv_best", output="winner"),
                TERMINAL_SELECT,
            )
        )


@pytest.mark.parametrize(
    "last_phase",
    [
        _phase(),  # all_algorithms last
        _phase(action="refine"),  # refine last
        _phase(action="select", strategy="cv_best", output="top_k", k=1),
        _phase(action="select", strategy="cv_best", output="pool"),
    ],
)
def test_last_phase_must_be_terminal_select(last_phase):
    """The last phase must be a select with output: winner."""
    with pytest.raises(ValidationError, match="must be a 'select' phase"):
        Config.model_validate(_cfg_with_phases(_phase(), last_phase))


def test_last_phase_select_without_winner_output_rejected():
    """The last phase is a select but with output != winner — an error."""
    with pytest.raises(ValidationError, match="must be a 'select' phase"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(),
                _phase(action="select", strategy="cv_best", output="pool"),
            )
        )


def test_top_k_without_k_rejected():
    """output: top_k without k >= 1 is rejected."""
    with pytest.raises(ValidationError, match="top_k.*requires.*k"):
        Config.model_validate(
            HPOPhaseCfg(n_trials=1, action="select", strategy="cv_best", output="top_k")
        )


def test_k_zero_or_negative_rejected():
    """k < 1 is rejected by the field constraint (ge=1)."""
    with pytest.raises(ValidationError, match="k"):
        Config.model_validate(
            HPOPhaseCfg(
                n_trials=1, action="select", strategy="cv_best", output="top_k", k=0
            )
        )


@pytest.mark.parametrize("output", ["winner", "pool"])
def test_k_with_winner_or_pool_rejected(output):
    """k is forbidden with output: winner|pool."""
    with pytest.raises(ValidationError, match="k is not allowed"):
        Config.model_validate(
            HPOPhaseCfg(
                n_trials=1, action="select", strategy="cv_best", output=output, k=2
            )
        )


@pytest.mark.parametrize("field", ["strategy", "output", "k"])
def test_select_fields_forbidden_on_non_select_phases(field):
    """The strategy/output/k fields are forbidden on all_algorithms/refine phases."""
    value: object = (
        "cv_best" if field == "strategy" else "winner" if field == "output" else 2
    )
    with pytest.raises(ValidationError, match="only allowed for action 'select'"):
        HPOPhaseCfg(action="refine", n_trials=3, **{field: value})  # type: ignore[arg-type]


def test_select_without_strategy_rejected():
    """A select without strategy is rejected with a hint of allowed values."""
    with pytest.raises(ValidationError, match="requires fields \\['strategy'\\]"):
        HPOPhaseCfg(n_trials=1, action="select", output="winner")


def test_select_without_output_rejected():
    """A select without output is rejected with a hint of allowed values."""
    with pytest.raises(ValidationError, match="requires fields \\['output'\\]"):
        HPOPhaseCfg(n_trials=1, action="select", strategy="cv_best")


def test_select_without_strategy_and_output_rejected():
    """A select without both fields lists both in the error message."""
    with pytest.raises(ValidationError, match="\\['strategy', 'output'\\]"):
        HPOPhaseCfg(n_trials=1, action="select")


# ─────────── Runtime: select/refine execution ───────────


def test_refine_phase_executes_with_warm_start():
    """refine is executable: warm-start HPO for the whole set.

    The config [all_algorithms → refine → terminal select] runs both HPO
    phases (one HPO run per algorithm per phase; initial_params are passed
    to refine) and finishes with the artifact of the terminal select phase.
    """
    import numpy as np
    import pandas as pd
    from configurable_automl_engine.training_engine.component import (
        PhaseResult,
        train_best_model,
    )

    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
    df["target"] = 2.0 * df["a"] + rng.randn(30) * 0.1

    cfg_refine = _cfg_with_phases(
        _phase(), _phase(action="refine", n_trials=1), TERMINAL_SELECT
    )
    with patch(
        "configurable_automl_engine.training_engine.component._run_hpo",
        side_effect=lambda **kw: PhaseResult(score=0.9, params={"p": 1}),
    ) as mock_hpo:
        res = train_best_model(cfg_refine, df, target="target")

    assert res["algorithm"] in BASE["algorithms"]
    assert "sanity_gate" not in res
    # elasticnet is the only algorithm in BASE: 1 call in the
    # all_algorithms phase (without initial_params) + 1 call in the refine
    # phase (with a warm start).
    assert mock_hpo.call_count == 2
    first_call = mock_hpo.call_args_list[0].kwargs
    refine_call = mock_hpo.call_args_list[1].kwargs
    assert first_call.get("initial_params") is None
    assert refine_call.get("initial_params") == {"p": 1}


def test_select_phase_executes_in_pipeline():
    """Select-phase execution works: the terminal select(cv_best, winner)
    runs and returns the artifact."""
    import numpy as np
    import pandas as pd
    from configurable_automl_engine.training_engine.component import (
        PhaseResult,
        train_best_model,
    )

    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
    df["target"] = 2.0 * df["a"] + rng.randn(30) * 0.1

    res = train_best_model(
        _cfg_with_phases(_phase(), TERMINAL_SELECT), df, target="target"
    )
    assert res["algorithm"] in BASE["algorithms"]
    assert "sanity_gate" not in res


# ─────────── M1/M2: boundary phase combinations ───────────


def test_refine_first_phase_is_valid():
    """refine as the first phase is valid (M1): the input working set is the
    full set of enabled algorithms (no warm start from a previous phase);
    the execution semantics are finalized by the refine task."""
    cfg = Config.model_validate(
        _cfg_with_phases(_phase(action="refine"), TERMINAL_SELECT)
    )
    assert [p.action for p in cfg.general.phases] == ["refine", "select"]


def test_select_cv_best_pool_is_valid():
    """select(strategy: cv_best, output: pool) is valid (M2): a corridor
    finalists pool (corridor δ ∩ top_k_candidates, thresholds from the
    sanity_gate section), ranked by CV score WITHOUT a ModelSanityGate audit."""
    phase = _phase(action="select", strategy="cv_best", output="pool")
    cfg = Config.model_validate(_cfg_with_phases(_phase(), phase, TERMINAL_SELECT))
    parsed = cfg.general.phases[1]
    assert parsed.strategy == "cv_best"
    assert parsed.output == "pool"
    assert parsed.k is None


def test_k_exceeding_top_k_candidates_is_accepted():
    """k > sanity_gate.top_k_candidates is not checked at the config level
    (M3): the finalists pool is first capped by top_k_candidates, and the
    exact interaction of k with the pool is decided at execution time."""
    data = _cfg_with_phases(_phase(), TERMINAL_SELECT)
    data["general"]["sanity_gate"] = {"top_k_candidates": 3}
    cfg = Config.model_validate(
        _cfg_with_phases(
            _phase(),
            _phase(action="select", strategy="sanity_gate", output="top_k", k=10),
            TERMINAL_SELECT,
        )
    )
    assert cfg.general.phases[1].k == 10
    assert cfg.general.sanity_gate.top_k_candidates == 3
    _ = data  # the sanity_gate section is valid on its own


# ─────────────────── adaptive complexity guard ──────────────────── #
def test_complexity_guard_defaults():
    """The complexity_guard section defaults to apply/lossless/regularize."""
    cfg = Config.model_validate(BASE)
    guard = cfg.general.complexity_guard
    assert guard.mode.value == "apply"
    assert guard.lossless is True
    assert guard.regularize is True
    assert guard.user_range_policy.value == "user_wins"
    assert guard.small_n_threshold == 50
    assert guard.medium_n_threshold == 300
    assert guard.size_basis.value == "rows"


def test_complexity_guard_custom_values():
    """A custom complexity_guard section is parsed."""
    cfg = Config.model_validate(
        BASE
        | {
            "general": {
                **BASE["general"],
                "complexity_guard": {
                    "mode": "warn",
                    "lossless": False,
                    "regularize": True,
                    "user_range_policy": "engine_wins",
                    "small_n_threshold": 100,
                    "small_np_ratio_threshold": 8.0,
                    "medium_n_threshold": 500,
                    "medium_np_ratio_threshold": 30.0,
                    "size_basis": "rows",
                },
            }
        }
    )
    guard = cfg.general.complexity_guard
    assert guard.mode.value == "warn"
    assert guard.lossless is False
    assert guard.user_range_policy.value == "engine_wins"
    assert guard.small_n_threshold == 100
    assert guard.medium_n_threshold == 500
    assert guard.size_basis.value == "rows"


def test_complexity_guard_mode_off_valid():
    """mode: off is a valid value."""
    cfg = Config.model_validate(
        BASE
        | {"general": {**BASE["general"], "complexity_guard": {"mode": "off"}}}
    )
    assert cfg.general.complexity_guard.mode.value == "off"


def test_complexity_guard_unknown_mode_rejected():
    """An unknown mode value is rejected."""
    with pytest.raises(ValidationError):
        Config.model_validate(
            BASE
            | {"general": {**BASE["general"], "complexity_guard": {"mode": "nope"}}}
        )


def test_complexity_guard_unknown_field_rejected():
    """extra='forbid': an unknown field in complexity_guard is rejected."""
    with pytest.raises(ValidationError):
        Config.model_validate(
            BASE
            | {
                "general": {
                    **BASE["general"],
                    "complexity_guard": {"mode": "apply", "bogus": 1},
                }
            }
        )


def test_complexity_guard_thresholds_ordering_rejected():
    """small_n_threshold must be less than medium_n_threshold."""
    with pytest.raises(ValidationError, match="small_n_threshold"):
        Config.model_validate(
            BASE
            | {
                "general": {
                    **BASE["general"],
                    "complexity_guard": {
                        "small_n_threshold": 300,
                        "medium_n_threshold": 100,
                    },
                }
            }
        )


def test_complexity_guard_ratio_thresholds_ordering_rejected():
    """The small ratio threshold must be below the medium one."""
    with pytest.raises(ValidationError, match="small_np_ratio_threshold"):
        Config.model_validate(
            BASE
            | {
                "general": {
                    **BASE["general"],
                    "complexity_guard": {
                        "small_np_ratio_threshold": 20.0,
                        "medium_np_ratio_threshold": 5.0,
                    },
                }
            }
        )


def test_complexity_guard_non_positive_threshold_rejected():
    """Thresholds must be positive."""
    with pytest.raises(ValidationError):
        Config.model_validate(
            BASE
            | {
                "general": {
                    **BASE["general"],
                    "complexity_guard": {"small_n_threshold": 0},
                }
            }
        )
    with pytest.raises(ValidationError):
        Config.model_validate(
            BASE
            | {
                "general": {
                    **BASE["general"],
                    "complexity_guard": {"small_np_ratio_threshold": 0.0},
                }
            }
        )


# ─────────────── default pipeline (phases / general omitted) ─────────────── #


def _assert_default_pipeline(phases):
    assert [p.action for p in phases] == ["all_algorithms", "select"]
    search, final = phases
    assert search.n_trials == 50
    assert final.strategy == "cv_best"
    assert final.output == "winner"
    assert final.k is None


def test_minimal_config_only_algorithms():
    cfg = Config.model_validate({"algorithms": {"ridge": {}}})
    assert cfg.general.comparison_metric == "r2"
    _assert_default_pipeline(cfg.general.phases)


def test_general_without_phases_uses_default_pipeline():
    cfg = Config.model_validate(
        {"general": {"comparison_metric": "rmse"}, "algorithms": {"ridge": {}}}
    )
    _assert_default_pipeline(cfg.general.phases)


def test_default_phases_are_not_shared_between_configs():
    first = GeneralCfg()
    second = GeneralCfg()
    assert first.phases is not second.phases
    first.phases[0].n_trials = 3
    assert second.phases[0].n_trials == 50


def test_phase_n_trials_defaults_to_50():
    assert HPOPhaseCfg().n_trials == 50
    select = HPOPhaseCfg(action="select", strategy="cv_best", output="winner")
    assert select.n_trials == 50


def test_explicit_empty_phases_list_is_kept():
    # An explicit empty list is kept as is (not replaced by the default).
    cfg = Config.model_validate(
        {"general": {"phases": []}, "algorithms": {"ridge": {}}}
    )
    assert cfg.general.phases == []


def test_n_trials_zero_still_rejected():
    with pytest.raises(ValidationError):
        HPOPhaseCfg(n_trials=0)


def test_minimal_config_without_enabled_algorithm_rejected():
    with pytest.raises(ValidationError, match="At least one algorithm"):
        Config.model_validate({"algorithms": {"ridge": {"enable": False}}})


def test_missing_algorithms_section_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({})

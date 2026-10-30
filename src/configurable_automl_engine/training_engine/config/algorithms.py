"""Config schema of the ``algorithms`` and ``oversampling`` sections.

Holds ``AlgoCfg`` (one algorithm: enable flag, search space, preprocessing
override), the dynamic ``AlgorithmsConfig`` model with one optional field per
known algorithm, the rejection of the removed plug-in keys, and
``OversamplingCfg``. The module must not import the other config modules.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)
from pydantic_core import PydanticCustomError

from configurable_automl_engine.common.definitions import (
    ALGO_PACKAGE_MAPPING,
)
from configurable_automl_engine.hpo.default_spaces import (
    SearchSpaceEntry,
)
from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    get_allowed_hyperparameters,
)
from configurable_automl_engine.preprocessing_presets import PreprocessingOverride

ERR_REMOVED_PLUGIN_KEY = "removed_plugin_module_key"

#: The per-algorithm keys of the custom tuner/trainer plug-in, removed in 0.5.0.
REMOVED_ALGORITHM_KEYS = ("tuner", "trainer_module")


def _removed_plugin_key_error(path: str) -> PydanticCustomError:
    """The targeted error for a removed custom tuner/trainer key.

    Args:
        path: The config path of the key (e.g. ``algorithms.ridge.tuner``).

    Returns:
        PydanticCustomError: The error with the ``ERR_REMOVED_PLUGIN_KEY``
            type and a migration hint.
    """
    return PydanticCustomError(
        ERR_REMOVED_PLUGIN_KEY,
        f"`{path}` was removed in 0.5.0: custom tuner/trainer modules are no "
        "longer supported; delete the key. See 'Migrating older configs' in "
        "API_REFERENCE.md.",
    )


# ──────────────── oversampling ──────────────── #
class OversamplingAlgorithm(str, Enum):
    random = "random"
    smote = "smote"
    adasyn = "adasyn"


class OversamplingCfg(BaseModel):
    """Class balancing and data synthesis parameters.

    Provides configuration of methods for resolving the target-variable imbalance.
    Uses an aliases mechanism to map flat YAML keys into a structured object.
    Attributes:
        enable (bool): Global flag enabling oversampling.
        multiplier (float): Target sample-increase factor.
        algorithm (OversamplingAlgorithm): Selected generation method
            (SMOTE, ADASYN, etc.).
        target_bins (int): Number of quantile bins for the regression path.
        add_noise (bool): Whether to add Gaussian noise to the numeric
            features of the oversampled training set.
        noise_level (float): Noise standard deviation as a fraction of each
            column's standard deviation.
    """

    # accept alias names
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    enable: bool = Field(
        default=False,
        alias="data_oversampling",
        description=("Flag enabling data balancing. Applies only to the training set"),
    )
    multiplier: float = Field(
        default=1.0,
        alias="data_oversampling_multiplier",
        ge=1.0,
        description="How many times to increase the number of minority-class examples",
    )
    algorithm: OversamplingAlgorithm = Field(
        default=OversamplingAlgorithm.random,
        alias="data_oversampling_algorithm",
        description="Algorithm for synthesizing new data (Random, SMOTE, ADASYN)",
    )
    target_bins: int = Field(
        default=8,
        alias="data_oversampling_target_bins",
        ge=2,
        description=(
            "Number of quantile bins for the regression oversampling strategy"
        ),
    )
    add_noise: bool = Field(
        default=False,
        alias="data_oversampling_add_noise",
        description=(
            "Add Gaussian noise to the numeric features of the oversampled "
            "training set (categorical columns are not noised)"
        ),
    )
    noise_level: float = Field(
        default=0.01,
        alias="data_oversampling_noise_level",
        gt=0.0,
        description=(
            "Gaussian noise standard deviation as a fraction of each numeric "
            "column's standard deviation"
        ),
    )

    @model_validator(mode="after")
    def _validate_oversampling_logic(self) -> OversamplingCfg:
        if self.enable and self.multiplier == 1.0:
            logging.getLogger(__name__).warning(
                "Oversampling multiplier = 1 ➜ class balance will not change."
            )
        if self.add_noise and not self.enable:
            logging.getLogger(__name__).warning(
                "data_oversampling_add_noise = true has no effect while "
                "data_oversampling is disabled."
            )
        return self


# ───────────────── algorithms ───────────────── #


class AlgoCfg(BaseModel):
    """Technical configuration of a specific ML algorithm in the pipeline.

    Stores the algorithm enablement settings, the overridden search space
    and the preprocessing preset override.
    Attributes:
        enable (bool): Flag for the algorithm's participation in the current experiment.
        limit_hyperparameters (bool): Restrict the search to the user-listed
            hyperparameters only (no DEFAULT_SPACES entries).
        hyperparameters (Dict | None): Custom parameter bounds
            overriding the default values.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_plugin_keys(cls, data: Any) -> Any:
        """Reject the removed ``tuner`` / ``trainer_module`` keys with a hint.

        Args:
            data: Raw input data of the algorithm section.

        Returns:
            Any: The data unchanged, if no removed key is present.

        Raises:
            PydanticCustomError: If a removed key is present.
        """
        if isinstance(data, dict):
            for key in REMOVED_ALGORITHM_KEYS:
                if key in data:
                    raise _removed_plugin_key_error(key)
        return data

    enable: bool = Field(
        default=True, description="Whether to use this algorithm in the AutoML pipeline"
    )
    limit_hyperparameters: bool = Field(
        default=False,
        description=(
            "Restrict the search to ONLY the parameters listed in "
            "'hyperparameters'; every other parameter stays at the estimator "
            "default (no DEFAULT_SPACES entries are added). false (default) — "
            "user ranges override and are merged into DEFAULT_SPACES."
        ),
    )
    hyperparameters: dict[str, SearchSpaceEntry] | None = Field(
        default=None,
        description=(
            "Keys must correspond to the algorithm's allowed hyperparameters: "
            "the constructor parameters of the sklearn estimator (including legacy "
            "names from LEGACY_PARAM_MAPPINGS). The source of truth is "
            "models.get_allowed_hyperparameters."
        ),
    )
    preprocessing: PreprocessingOverride | None = Field(
        default=None,
        description=(
            "Explicit override of the feature preprocessing preset for this "
            "algorithm. Set partially or fully: 'imputation_strategy' "
            "('mean'|'median') and/or 'scaling' ('standard'|'robust'|'none'). "
            "Takes precedence over the automatic choice based on the algorithm "
            "class. None — automatic selection."
        ),
    )

    def get_required_package(self, algo_name: str) -> str | None:
        """Determine the name of the external Python package required by the algorithm.

        Lookup logic:
        1. Consults the global `ALGO_PACKAGE_MAPPING`.
        2. Maps the internal algorithm name (for example, 'xgboosting')
            to its PyPI name.

        Args:
            algo_name (str): Unique identifier of the algorithm.
        Returns:
            Optional[str]: Name of the package to install via pip, or None
                if no dependency is defined.
        """
        return ALGO_PACKAGE_MAPPING.get(algo_name)

    def get_unknown_hyperparameters(self, algo_name: str) -> list[str]:
        """Return the hyperparameters incompatible with the given algorithm.

        Checks the keys of `self.hyperparameters` against the allowed set of
        estimator constructor parameters (see
        `models.get_allowed_hyperparameters`), including legacy names from
        `LEGACY_PARAM_MAPPINGS`. If the allowed set cannot be determined
        (the algorithm is unknown, the dependency is not installed, or the
        constructor accepts **kwargs), the check is skipped (soft fallback).

        Args:
            algo_name (str): Unique identifier of the algorithm.
        Returns:
            List[str]: List of invalid keys. An empty list means everything is correct.
        """
        if self.hyperparameters is None:
            return []
        allowed = get_allowed_hyperparameters(algo_name)

        if allowed is None:
            return []

        return [k for k in self.hyperparameters if k not in allowed]

    @model_validator(mode="after")
    def _limit_requires_hyperparameters(self) -> AlgoCfg:
        """Reject the meaningless ``limit_hyperparameters: true`` without ``hyperparameters``.

        The limited mode searches ONLY the user-listed parameters; with an
        empty (or missing) ``hyperparameters`` block there would be nothing
        to search and every parameter would silently stay at the estimator
        default. The combination is a configuration error, not a silent no-op.
        """
        if self.limit_hyperparameters and not self.hyperparameters:
            raise ValueError(
                "limit_hyperparameters=true requires a non-empty 'hyperparameters' "
                "block: the limited search space is built ONLY from the listed "
                "parameters, so an empty block would leave nothing to optimize."
            )
        return self


class _AlgorithmsConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_plugin_keys(cls, data: Any) -> Any:
        """Reject ``algorithms.<name>.tuner`` / ``trainer_module`` with a hint.

        Checked here, before the per-algorithm validation, so that the message
        names the full config path of the key.

        Args:
            data: Raw input data of the ``algorithms`` section.

        Returns:
            Any: The data unchanged, if no removed key is present.

        Raises:
            PydanticCustomError: If an algorithm section has a removed key.
        """
        if isinstance(data, dict):
            for name, section in data.items():
                if not isinstance(section, dict):
                    continue
                for key in REMOVED_ALGORITHM_KEYS:
                    if key in section:
                        raise _removed_plugin_key_error(f"algorithms.{name}.{key}")
        return data


AlgorithmsConfig = create_model(
    "AlgorithmsConfig",
    __base__=_AlgorithmsConfigBase,
    **{name: (AlgoCfg | None, Field(default=None)) for name in AVAILABLE_ALGORITHMS},
)  # type: ignore[call-overload]

if TYPE_CHECKING:
    # use a static alias for mypy
    AlgorithmsConfigType = _AlgorithmsConfigBase
else:
    # runtime — dynamic create_model
    AlgorithmsConfigType = AlgorithmsConfig

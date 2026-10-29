"""The root config model and the YAML reader.

Holds ``Config`` (the ``general`` / ``oversampling`` / ``algorithms``
sections and the checks that need more than one section: enabled
algorithms, installed dependencies, hyperparameter compatibility) and
``read_config``. It builds on the other config modules; none of them may
import it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.hpo.default_spaces import (
    normalize_hyperparameter_entry,
)
from configurable_automl_engine.models import (
    get_allowed_hyperparameters,
)
from configurable_automl_engine.training_engine.config.algorithms import (
    AlgorithmsConfigType,
    OversamplingCfg,
)
from configurable_automl_engine.training_engine.config.general import GeneralCfg


# ─────────────────── root ──────────────────── #
class Config(BaseModel):
    """Root object of the whole AutoML configuration system.

    Aggregates all configuration sections and performs the final validation
    of the parameter graph integrity, including system dependency checks.
    Attributes:
        general (GeneralCfg): General experiment parameters.
        oversampling (OversamplingCfg): Data preprocessing settings.
        algorithms (AlgorithmsConfig): Registry of available and active algorithms.
    """

    model_config = ConfigDict(extra="forbid")

    general: GeneralCfg = Field(
        default_factory=GeneralCfg,
        description="General experiment and validation settings",
    )
    oversampling: OversamplingCfg = Field(
        default=OversamplingCfg(), description="Data balancing settings"
    )
    algorithms: AlgorithmsConfigType = Field(
        ...,
        description=(
            "Dictionary of algorithms, where the key is the algorithm name "
            "(for example, 'xgboosting', 'random_forest')"
        ),
    )

    @field_validator("algorithms")
    @classmethod
    def _must_have_enabled(cls, v: Any) -> Any:
        # Since 'v' is now an AlgorithmsConfig (object),
        # we get all its fields via .model_dump()
        enabled_algorithms = [
            algo
            for algo in v.model_dump().values()
            if algo is not None and algo.get("enable") is True
        ]

        if not enabled_algorithms:
            raise ValueError("At least one algorithm must be enabled (enable: true)")
        return v

    @model_validator(mode="after")
    def _check_algorithm_dependencies(self) -> Config:
        # model_fields works through an instance too, but it is deprecated
        # (PydanticDeprecatedSince211, removal in V3.0) — use the class instead
        for name in type(self.algorithms).model_fields:
            # Get the attribute value (could be None if not provided in data)
            algo_cfg = getattr(self.algorithms, name)

            # Skip if the algorithm entry is missing (None)
            if algo_cfg is None:
                continue

            # Ensure it has an 'enable' attribute before checking it
            # If it's a Pydantic model, check attribute, otherwise get() as dict
            enabled = getattr(algo_cfg, "enable", False)

            if enabled:
                # Safely get the required package
                required_pkg = getattr(
                    algo_cfg, "get_required_package", lambda n: None
                )(name)

                if required_pkg and not is_installed(required_pkg):
                    raise ValueError(
                        f"Algorithm '{name}' is enabled, but the package "
                        f"'{required_pkg}' is not installed. "
                        f"Please run: pip install {required_pkg}"
                    )
        return self

    @model_validator(mode="after")
    def _check_hyperparameter_compatibility(self) -> Config:
        errors = []
        for name in type(self.algorithms).model_fields:
            algo_cfg = getattr(self.algorithms, name)
            if algo_cfg is None or not algo_cfg.enable:
                continue
            unknown = algo_cfg.get_unknown_hyperparameters(name)
            if unknown:
                allowed = sorted(get_allowed_hyperparameters(name) or set())
                errors.append(
                    f"Algorithm '{name}': unknown hyperparameters {unknown}. "
                    f"Allowed parameters: {allowed}"
                )
        if errors:
            raise ValueError("\n".join(errors))
        return self

    @model_validator(mode="after")
    def _normalize_hyperparameter_types(self) -> Config:
        """Re-resolve the type of inferred (not explicitly typed) ranges.

        The type of a range is taken from the parameter's type — the default
        search space or the estimator signature — not from how the boundary
        numbers are written. So ``[1, 50]`` for a real-valued parameter
        (e.g. ``C`` of an SVR, ``alpha`` of a Tweedie) stays a real range
        instead of silently becoming an integer range. A parameter whose
        default space entry is ``float_log`` (e.g. ``alpha_1``/``lambda_1``
        of an ARD regression), as well as scale parameters (``alpha``,
        ``C``, ``epsilon``, ``gamma``, ``learning_rate``), default to the
        logarithmic scale without an explicit type. Explicitly typed ranges
        (``[1, 50, "int"]``) are left untouched.
        """
        for name in type(self.algorithms).model_fields:
            algo_cfg = getattr(self.algorithms, name)
            if algo_cfg is None or not algo_cfg.enable:
                continue
            hyperparams = algo_cfg.hyperparameters
            if not hyperparams:
                continue
            # Update the existing dict in place: the Config was built from the
            # input data, and other code may hold a reference to the same
            # hyperparameters mapping (e.g. through model_dump) — replacing
            # the dict would silently detach those references.
            for param, entry in list(hyperparams.items()):
                normalized = normalize_hyperparameter_entry(name, param, entry)
                if normalized is not entry:
                    hyperparams[param] = normalized
        return self


# ────────────────── API ────────────────────── #
def read_config(path: str | Path) -> Config:
    """Load, parse, and validate an experiment configuration file.

    How it works:
    1. Opens the file at the given path in UTF-8 encoding.
    2. Safely loads the YAML structure into a Python dictionary.
    3. Triggers cascading Pydantic validation
        to build a typed `Config` object.
    Args:
        path (str | Path): Path to a .yaml or .yml configuration file.
    Returns:
        Config: The root configuration object, ready for use in the AutoML engine.

    Raises:
        FileNotFoundError: If the file at the given path is not found.
        ValidationError: If the file structure does not match the schema or
            the parameter logic is violated.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)

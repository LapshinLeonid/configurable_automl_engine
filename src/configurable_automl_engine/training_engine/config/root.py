"""The engine configuration entry point.

``Config`` is re-exported from the contract. ``read_config`` keeps the YAML
reader in the engine and applies runtime validation after the structural
validation.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from automl_contracts.config import Config

from ..config_runtime import validate_runtime_config


def read_config(path: str | Path) -> Config:
    """Load, parse and validate an experiment configuration file.

    Args:
        path: Path to a ``.yaml`` or ``.yml`` configuration file.

    Returns:
        Config: The configuration with structural and runtime checks applied.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the configuration structure is invalid.
        ValueError: If a runtime check fails.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return validate_runtime_config(Config.model_validate(data))


__all__ = ["Config", "read_config"]

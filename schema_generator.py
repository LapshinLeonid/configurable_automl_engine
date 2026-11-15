"""Compatibility wrapper for the contract JSON Schema generator."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from automl_contracts import Config

ROOT_SCHEMA = Path("config.schema.json")


def clean_description(value: Any) -> None:
    """Normalize multiline descriptions recursively in place."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "description" and isinstance(item, str):
                value[key] = inspect.cleandoc(item).strip()
            else:
                clean_description(item)
    elif isinstance(value, list):
        for item in value:
            clean_description(item)


def build_schema() -> dict[str, Any]:
    """Return the deterministic JSON Schema of the contract ``Config``."""
    schema = Config.model_json_schema()
    clean_description(schema)
    return schema


def save_schema(filename: str | Path = ROOT_SCHEMA) -> Path:
    """Write the root compatibility mirror to the current working directory.

    The script keeps the historical CLI contract: the default target is
    ``config.schema.json`` relative to the caller's working directory, and the
    file has no trailing newline. ``scripts/compare_with_base.py`` runs the
    generator from a temporary workdir and compares its bytes with the base
    generator output.

    Args:
        filename: Output path. The default is relative to the current working
            directory.

    Returns:
        The written path.
    """
    path = Path(filename)
    with path.open("w", encoding="utf-8") as file:
        json.dump(build_schema(), file, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    """Generate the root compatibility mirror."""
    print(f"Schema saved: {save_schema()}")


if __name__ == "__main__":
    main()

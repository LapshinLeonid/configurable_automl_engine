"""Generate the canonical JSON Schema of the contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from automl_contracts import Config

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "automl_contracts"
    / "schemas"
    / "config.schema.json"
)


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


def save_schema(output: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write the canonical schema to ``output`` and return the written path.

    The file intentionally has no trailing newline: the root compatibility
    mirror is kept byte-for-byte identical, and the historical generator also
    wrote the JSON without a final newline.

    Args:
        output: Output path. The default is the package-data schema location.

    Returns:
        The written path.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(build_schema(), file, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    """Generate the canonical schema."""
    print(f"Schema saved: {save_schema()}")


if __name__ == "__main__":
    main()

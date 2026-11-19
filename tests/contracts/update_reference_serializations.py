"""Regenerate the serialized reference artifacts of the contract tests.

Usage::

    python -m tests.contracts.update_reference_serializations
"""

from __future__ import annotations

import json
from pathlib import Path

from automl_contracts import Config, ExperimentResult

DATA_DIR = Path(__file__).resolve().parent / "data"


def write_reference_serializations() -> list[Path]:
    """Write the serialized reference config and result."""
    config = Config.model_validate(
        json.loads((DATA_DIR / "reference_config.json").read_text(encoding="utf-8"))
    )
    result = ExperimentResult.model_validate(
        json.loads((DATA_DIR / "reference_result.json").read_text(encoding="utf-8"))
    )
    config_path = DATA_DIR / "reference_config_serialized.json"
    result_path = DATA_DIR / "reference_result_serialized.json"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(json.loads(result.to_json()), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return [config_path, result_path]


def main() -> None:
    """Write the serialized reference artifacts and print their locations."""
    for path in write_reference_serializations():
        print(f"Reference artifact written: {path}")


if __name__ == "__main__":
    main()

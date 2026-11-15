"""Regenerate the golden artifacts of the contract tests.

Usage::

    python -m tests.contracts.update_golden

A public-API or serialization change is a contract change: review the diff,
add a ``CHANGELOG.md`` entry and bump ``automl-contracts/pyproject.toml``
according to SemVer before committing the regenerated files.
"""

from __future__ import annotations

import json
from pathlib import Path

from automl_contracts import Config, ExperimentResult

from tests.contracts._contract_snapshot import GOLDEN_DIR, write_golden

DATA_DIR = Path(__file__).resolve().parent / "data"


def write_reference_serializations() -> list[Path]:
    """Write the serialized reference config and result."""
    config = Config.model_validate(
        json.loads((DATA_DIR / "reference_config.json").read_text(encoding="utf-8"))
    )
    result = ExperimentResult.model_validate(
        json.loads((DATA_DIR / "reference_result.json").read_text(encoding="utf-8"))
    )
    config_path = GOLDEN_DIR / "reference_config_serialized.json"
    result_path = GOLDEN_DIR / "reference_result_serialized.json"
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
    """Write all golden artifacts and print their locations."""
    paths = [
        write_golden(),
        *write_reference_serializations(),
    ]
    for path in paths:
        print(f"Golden artifact written: {path}")


if __name__ == "__main__":
    main()

"""Regenerate the golden artifacts of the contract tests.

Usage::

    python -m tests.contracts.update_golden

A public-API or serialization change is a contract change: review the diff,
add a ``CHANGELOG.md`` entry and bump ``automl-contracts/pyproject.toml``
according to SemVer before committing the regenerated files.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from automl_contracts import Config, ExperimentResult

from tests.contracts._contract_snapshot import (
    GOLDEN_DIR,
    expand_import_matrix,
    parse_engine_imports_from_source,
    write_golden,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
ENGINE_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "engine" / "golden"


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


def write_engine_import_matrix() -> Path:
    """Write the old-import compatibility matrix parsed from CONTRACTS.md."""
    paths: dict[str, list[str]] = OrderedDict()
    for path, symbol in expand_import_matrix(parse_engine_imports_from_source()):
        paths.setdefault(path, [])
        if symbol not in paths[path]:
            paths[path].append(symbol)
    ENGINE_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    target = ENGINE_GOLDEN_DIR / "import_matrix.json"
    target.write_text(
        json.dumps({"paths": paths}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    """Write all golden artifacts and print their locations."""
    paths = [
        write_golden(),
        *write_reference_serializations(),
        write_engine_import_matrix(),
    ]
    for path in paths:
        print(f"Golden artifact written: {path}")


if __name__ == "__main__":
    main()

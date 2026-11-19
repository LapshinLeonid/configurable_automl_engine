"""Serialization and schema stability on the declared Pydantic versions.

The ADR supports ``pydantic>=2.12.5,<3``.  The test installs the lower bound
and the latest allowed version into isolated target directories and runs the
same serialization/schema checks under each of them.

The installation is network-dependent, so the module is skipped unless
``AUTOML_RUN_NETWORK_TESTS=1`` (set in CI).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SRC = REPO_ROOT / "automl-contracts" / "src"
DATA_DIR = Path(__file__).resolve().parent / "data"
SCHEMA = CONTRACT_SRC / "automl_contracts" / "schemas" / "config.schema.json"
GENERATOR = REPO_ROOT / "automl-contracts" / "scripts" / "generate_schema.py"

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOML_RUN_NETWORK_TESTS") != "1",
    reason="set AUTOML_RUN_NETWORK_TESTS=1 to install Pydantic versions",
)


def install_pydantic(target: Path, requirement: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(target),
            requirement,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def check_script() -> str:
    return """
import importlib.util
import json
import sys

import pydantic
import automl_contracts
from automl_contracts import Config, ExperimentResult

assert "configurable_automl_engine" not in sys.modules

reference_config = json.load(open(sys.argv[1], encoding="utf-8"))
reference_result = json.load(open(sys.argv[2], encoding="utf-8"))
expected_config = json.load(open(sys.argv[3], encoding="utf-8"))
expected_result = json.load(open(sys.argv[4], encoding="utf-8"))

config = Config.model_validate(reference_config)
assert config.model_dump(mode="json") == expected_config
result = ExperimentResult.model_validate(reference_result)
assert json.loads(result.to_json()) == expected_result
assert ExperimentResult.from_json(result.to_json()) == result

spec = importlib.util.spec_from_file_location("generator", sys.argv[5])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import json as _json
payload = _json.dumps(module.build_schema(), indent=2, ensure_ascii=False).encode()
assert payload == open(sys.argv[6], "rb").read()
print(pydantic.VERSION)
"""


@pytest.mark.parametrize(
    "requirement",
    ["pydantic==2.12.5", "pydantic>=2.12.5,<3"],
    ids=["minimum", "latest"],
)
def test_serialization_and_schema_on_pydantic_versions(
    requirement: str, tmp_path: Path
) -> None:
    target = tmp_path / "site"
    target.mkdir()
    install_pydantic(target, requirement)

    script = tmp_path / "check.py"
    script.write_text(check_script(), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(target), str(CONTRACT_SRC)])
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(DATA_DIR / "reference_config.json"),
            str(DATA_DIR / "reference_result.json"),
            str(DATA_DIR / "reference_config_serialized.json"),
            str(DATA_DIR / "reference_result_serialized.json"),
            str(GENERATOR),
            str(SCHEMA),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    version = Version(result.stdout.strip().splitlines()[-1])
    if requirement == "pydantic==2.12.5":
        assert version == Version("2.12.5")
    else:
        assert Version("2.12.5") <= version < Version("3")

"""Isolated installation and use of the contract without the engine."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = REPO_ROOT / "automl-contracts"
CANONICAL_SCHEMA = (
    CONTRACT_ROOT / "src" / "automl_contracts" / "schemas" / "config.schema.json"
)
DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def contract_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("contract-wheel")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(output),
            str(CONTRACT_ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = sorted(output.glob("automl_contracts-*.whl"))
    assert len(wheels) == 1, [wheel.name for wheel in wheels]
    return wheels[0]


def test_wheel_package_data_and_metadata(contract_wheel: Path) -> None:
    with zipfile.ZipFile(contract_wheel) as archive:
        names = set(archive.namelist())
        assert "automl_contracts/py.typed" in names
        schema_name = "automl_contracts/schemas/config.schema.json"
        assert schema_name in names
        assert archive.read(schema_name) == CANONICAL_SCHEMA.read_bytes()
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")
    assert "Name: automl-contracts" in metadata
    assert "Requires-Dist: pydantic<3,>=2.12.5" in metadata
    for forbidden in (
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "optuna",
        "PyYAML",
        "joblib",
    ):
        assert forbidden not in metadata


def test_contract_is_usable_in_a_clean_venv(
    contract_wheel: Path, tmp_path: Path
) -> None:
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    python = venv / "bin" / "python"
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            str(contract_wheel),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    reference = DATA_DIR / "reference_config.json"
    example = DATA_DIR / "example_config.json"
    script = tmp_path / "consumer.py"
    script.write_text(
        """
import json
import sys
from importlib.resources import files

import automl_contracts as contracts
from automl_contracts import Config
from automl_contracts.protocols import MetricFunction, ScorerCallable

for forbidden in ("configurable_automl_engine", "numpy", "pandas", "sklearn", "scipy", "optuna", "yaml"):
    assert forbidden not in sys.modules, forbidden
try:
    import configurable_automl_engine  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("the engine must not be installed in the clean venv")

with open(sys.argv[1], encoding="utf-8") as file:
    reference = json.load(file)
with open(sys.argv[2], encoding="utf-8") as file:
    example = json.load(file)

config = Config.model_validate(reference)
assert config.general.comparison_metric == "r2"
assert config.general.n_repeats == 2
assert Config.model_validate(example).general.path_to_model.name == "diabetes_model.pkl"

# Runtime checks stay in the engine: the contract accepts platform/runtime
# combinations and unknown algorithm packages structurally.
Config.model_validate(
    {
        "general": {"parallel_mode": "threads", "phase_timeout": 30},
        "algorithms": {"xgboosting": {}},
    }
)

def error_code(payload):
    try:
        Config.model_validate(payload)
    except Exception as error:
        return error.errors()[0]["type"]
    raise AssertionError("the contract must reject the payload")

assert error_code({**example, "unknown_section": {}}) == "extra_forbidden"
assert error_code(
    {
        "general": {
            "phases": [{"name": "bad", "n_trials": 1, "action": "refine_winner"}]
        },
        "algorithms": {"ridge": {}},
    }
) == contracts.ERR_REMOVED_REFINE_WINNER
assert error_code(
    {
        "general": {"sanity_gate": {"mode": "active"}},
        "algorithms": {"ridge": {}},
    }
) == contracts.ERR_REMOVED_SANITY_GATE_MODE
assert error_code(
    {
        "general": {
            "phases": [{"name": "bad", "n_trials": 1, "action": "not_a_phase"}]
        },
        "algorithms": {"ridge": {}},
    }
) == "literal_error"

schema = files("automl_contracts").joinpath("schemas/config.schema.json")
assert schema.is_file()
assert json.loads(schema.read_text(encoding="utf-8"))["$defs"]

assert callable(MetricFunction)
assert callable(ScorerCallable)
assert contracts.__version__ == "0.1.0"
print("CONTRACT-OK")
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(python), str(script), str(reference), str(example)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONTRACT-OK" in result.stdout


def test_contract_import_does_not_require_the_engine_source() -> None:
    code = (
        "import sys, automl_contracts\n"
        "assert 'configurable_automl_engine' not in sys.modules\n"
        "print(automl_contracts.__version__)\n"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=CONTRACT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr

"""Architecture guarantees for the contract and the engine."""

from __future__ import annotations

import io
import tarfile
import textwrap
import zipfile
from pathlib import Path

from tools.contract_boundaries import (
    CONTRACT_ROOT,
    CONTRACT_SRC,
    ENGINE_PYPROJECT,
    ENGINE_SRC,
    contract_public_names,
    inspect_contract_sdist,
    inspect_contract_wheel,
    inspect_engine_artifacts,
    run_static_checks,
    scan_contract_dynamic_imports,
    scan_contract_imports,
    scan_contract_metadata,
    scan_contract_package_data,
    scan_contract_runtime_access,
    scan_contract_type_strings,
    scan_engine_contract_usage,
    scan_engine_duplicates,
    scan_ruff_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _wheel(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _sdist(path: Path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _contract_wheel_members(version: str = "0.1.0") -> dict[str, str]:
    return {
        "automl_contracts/__init__.py": '"""Contract."""\n',
        "automl_contracts/py.typed": "",
        "automl_contracts/schemas/config.schema.json": "{}",
        f"automl_contracts-{version}.dist-info/METADATA": textwrap.dedent(
            f"""\
            Metadata-Version: 2.1
            Name: automl-contracts
            Version: {version}
            Requires-Dist: pydantic>=2.12.5,<3
            Requires-Dist: pydantic-core>=2,<3
            """
        ),
    }


def _contract_sdist_members(version: str = "0.1.0") -> dict[str, str]:
    prefix = f"automl-contracts-{version}"
    return {
        f"{prefix}/pyproject.toml": textwrap.dedent(
            f"""\
            [project]
            name = "automl-contracts"
            version = "{version}"
            """
        ),
        f"{prefix}/src/automl_contracts/__init__.py": '"""Contract."""\n',
        f"{prefix}/src/automl_contracts/py.typed": "",
        f"{prefix}/src/automl_contracts/schemas/config.schema.json": "{}",
    }


def test_repository_static_checks_pass() -> None:
    assert run_static_checks(REPO_ROOT) == []


def test_contract_imports_are_clean() -> None:
    assert scan_contract_imports(CONTRACT_SRC) == []


def test_contract_has_no_hidden_type_references() -> None:
    assert scan_contract_type_strings(CONTRACT_SRC) == []


def test_contract_has_no_dynamic_imports() -> None:
    assert scan_contract_dynamic_imports(CONTRACT_SRC) == []


def test_contract_has_no_runtime_access() -> None:
    assert scan_contract_runtime_access(CONTRACT_SRC) == []


def test_contract_metadata_is_clean() -> None:
    assert scan_contract_metadata() == []


def test_contract_package_data_is_declared() -> None:
    assert scan_contract_package_data(CONTRACT_ROOT) == []


def test_ruff_policy_is_shared_with_engine() -> None:
    assert scan_ruff_policy(ENGINE_PYPROJECT, CONTRACT_ROOT / "pyproject.toml") == []


def test_engine_has_no_contract_duplicates() -> None:
    assert scan_engine_duplicates(ENGINE_SRC, CONTRACT_SRC) == []


def test_engine_depends_on_contract() -> None:
    assert scan_engine_contract_usage(ENGINE_SRC, ENGINE_PYPROJECT) == []


def test_engine_import_is_reported_as_direction_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import configurable_automl_engine
        """,
    )
    violations = scan_contract_imports(tmp_path)
    assert len(violations) == 1
    assert violations[0].rule == "dependency-direction"
    assert "never back" in violations[0].message


def test_type_checking_engine_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from configurable_automl_engine import Config
        """,
    )
    violations = scan_contract_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dependency-direction"]


def test_heavy_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import numpy as np
        """,
    )
    violations = scan_contract_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["contract-dependency"]
    assert "numpy" in violations[0].message


def test_try_except_import_error_is_still_a_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        try:
            import sklearn
        except ImportError:
            sklearn = None
        """,
    )
    violations = scan_contract_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["contract-dependency"]
    assert "sklearn" in violations[0].message


def test_string_annotation_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        def use(value: "configurable_automl_engine.Config") -> None:
            return None
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]
    assert "forward references" in violations[0].message


def test_string_type_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        ConfigAlias = "configurable_automl_engine.Config"
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_dynamic_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        importlib.import_module("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_runtime_environment_access_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        os.getenv("CONTRACT_MODE")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_allowed_contract_imports_pass(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import json
        from pydantic import BaseModel
        from pydantic_core import PydanticCustomError
        """,
    )
    assert scan_contract_imports(tmp_path) == []


def test_forbidden_declarative_dependency_is_reported(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "automl-contracts"
        version = "0.1.0"
        dependencies = ["pydantic>=2", "numpy>=2"]

        [tool.mypy]
        strict = true

        [tool.importlinter]
        exclude_type_checking_imports = false

        [[tool.importlinter.contracts]]
        type = "forbidden"
        source_modules = ["automl_contracts"]
        forbidden_modules = ["configurable_automl_engine"]
        """,
    )
    violations = scan_contract_metadata(pyproject)
    assert [violation.rule for violation in violations] == ["contract-metadata"]
    assert "numpy" in violations[0].message


def test_engine_duplicate_class_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "engine.py",
        """
        class Config:
            pass
        """,
    )
    violations = scan_engine_duplicates(tmp_path, CONTRACT_SRC)
    assert [violation.rule for violation in violations] == ["engine-duplicate"]
    assert "Config" in violations[0].message


def test_engine_duplicate_constant_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "engine.py",
        """
        NO_OUTLIERS = object()
        """,
    )
    violations = scan_engine_duplicates(tmp_path, CONTRACT_SRC)
    assert [violation.rule for violation in violations] == ["engine-duplicate"]


def test_contract_names_include_public_submodule_symbols() -> None:
    names = contract_public_names(CONTRACT_SRC)
    assert "Config" in names
    assert "ScalingType" in names
    assert "to_sklearn_name" in names


def test_contract_wheel_with_engine_file_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "automl_contracts-0.1.0-py3-none-any.whl"
    members = _contract_wheel_members()
    members["configurable_automl_engine/__init__.py"] = ""
    _wheel(wheel, members)
    violations = inspect_contract_wheel(wheel, "0.1.0")
    assert any(violation.rule == "artifact-content" for violation in violations)


def test_valid_contract_wheel_passes(tmp_path: Path) -> None:
    wheel = tmp_path / "automl_contracts-0.1.0-py3-none-any.whl"
    _wheel(wheel, _contract_wheel_members())
    assert inspect_contract_wheel(wheel, "0.1.0") == []


def test_contract_wheel_version_mismatch_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "automl_contracts-0.1.0-py3-none-any.whl"
    _wheel(wheel, _contract_wheel_members())
    violations = inspect_contract_wheel(wheel, "0.2.0")
    assert any(violation.rule == "artifact-metadata" for violation in violations)


def test_contract_wheel_with_forbidden_dependency_is_rejected(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "automl_contracts-0.1.0-py3-none-any.whl"
    members = _contract_wheel_members()
    members["automl_contracts-0.1.0.dist-info/METADATA"] += (
        "Requires-Dist: scikit-learn>=1.8\n"
    )
    _wheel(wheel, members)
    violations = inspect_contract_wheel(wheel, "0.1.0")
    assert any(violation.rule == "artifact-metadata" for violation in violations)


def test_contract_sdist_with_engine_file_is_rejected(tmp_path: Path) -> None:
    sdist = tmp_path / "automl-contracts-0.1.0.tar.gz"
    members = _contract_sdist_members()
    members["automl-contracts-0.1.0/src/configurable_automl_engine/__init__.py"] = ""
    _sdist(sdist, members)
    violations = inspect_contract_sdist(sdist, "0.1.0")
    assert any(violation.rule == "artifact-content" for violation in violations)


def test_valid_contract_sdist_passes(tmp_path: Path) -> None:
    sdist = tmp_path / "automl-contracts-0.1.0.tar.gz"
    _sdist(sdist, _contract_sdist_members())
    assert inspect_contract_sdist(sdist, "0.1.0") == []


def test_engine_wheel_with_contract_sources_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "engine-0.0.0-py3-none-any.whl"
    _wheel(wheel, {"automl_contracts/__init__.py": ""})
    violations = inspect_engine_artifacts(wheel=wheel)
    assert [violation.rule for violation in violations] == ["artifact-separation"]


def test_engine_sdist_with_contract_sources_is_rejected(tmp_path: Path) -> None:
    sdist = tmp_path / "engine-0.0.0.tar.gz"
    _sdist(sdist, {"engine-0.0.0/src/automl_contracts/__init__.py": ""})
    violations = inspect_engine_artifacts(sdist=sdist)
    assert [violation.rule for violation in violations] == ["artifact-separation"]

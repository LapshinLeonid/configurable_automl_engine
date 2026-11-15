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
    scan_ci_configuration,
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


def test_builtins_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import builtins

        builtins.__import__("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_builtins_import_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from builtins import __import__ as imp

        imp("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_assignment_alias_of_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        f = __import__
        f("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_assignment_aliases_of_runtime_calls_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        run = os.system
        run("id")
        read = open
        read("/etc/passwd")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_callable_alias_is_scoped_per_function(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def harmless():
            f = len
            f([])

        def dangerous():
            f = os.system
            f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_callable_alias_reassignment_is_ordered(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def run():
            f = len
            f([])
            f = os.system
            f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_os_file_io_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        os.open("x", os.O_RDONLY)
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]
    assert "os.open" in violations[0].message


def test_imported_os_open_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from os import open as osopen

        osopen("x", 0)
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_exec_and_eval_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        exec("import configurable_automl_engine")
        eval("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_literal_getattr_dynamic_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        getattr(importlib, "import_module")("configurable_automl_engine")
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


def test_os_lstat_and_scandir_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        os.lstat("x")
        os.scandir("x")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 2


def test_process_and_network_access_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import ftplib
        import http.client
        import os
        import smtplib
        import urllib.request

        os.system("true")
        os.popen("true")
        os.fork()
        os.execv("x", [])
        os.kill(1, 0)
        urllib.request.urlopen("http://example.com")
        http.client.HTTPSConnection("example.com")
        smtplib.SMTP("example.com")
        ftplib.FTP("example.com")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 9
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_tempfile_and_environment_surface_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import sys
        import tempfile

        tempfile.NamedTemporaryFile()
        os.get_exec_path()
        sys.argv
        sys.path
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 4
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_io_and_process_identity_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import io
        import os
        import sys

        io.FileIO("x")
        os.getpid()
        sys.stdin
        sys.stdout
        sys.stderr
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 5
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_path_working_directory_helpers_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        Path.cwd()
        Path.home()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 2


def test_pure_path_methods_are_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        Path("x").joinpath("y")
        Path("x").with_suffix(".json")
        Path("x").as_posix()
        Path("x").match("*.py")
        Path("x").is_absolute()
        Path("x").relative_to("x")
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_path_instance_io_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        Path("x").read_text()
        Path("x").write_text("...")
        Path("x").exists()
        Path("x").open()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 4


def test_imported_runtime_attributes_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from os import environ
        from sys import platform

        environ.get("HOME")
        platform
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    messages = {violation.message for violation in violations}
    assert len(violations) == 2
    assert any("os.environ" in message for message in messages)
    assert any("sys.platform" in message for message in messages)


def test_imported_runtime_function_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from os import getenv

        getenv("HOME")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_metadata_checks_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        Path("x").is_symlink()
        Path("x").is_socket()
        Path("x").is_fifo()
        Path("x").is_block_device()
        Path("x").is_char_device()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 5


def test_chained_path_io_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        Path("x").joinpath("y").read_text()
        Path("x").resolve().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_path_variable_io_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        p = Path("x")
        p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_attribute_io_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Model:
            path: Path

            def load(self) -> str:
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_unbound_path_io_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        Path.read_text(Path("x"))
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_field_with_default_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        from pydantic import BaseModel, Field

        class Model(BaseModel):
            path: Path = Field(default=Path("model.bin"))

            def load(self) -> str:
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_optional_path_annotation_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def load(path: Path | None) -> str | None:
            return None if path is None else path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_slash_composition_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        base = Path("x")
        (base / "model.json").read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_pure_path_operations_are_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        from pathlib import Path

        Path("a").joinpath("b")
        os.path.join("a", "b")
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_ordinary_string_values_are_not_type_annotations(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from typing import Annotated, Literal

        from pydantic import Field

        SUPPORTED_FORMATS = ("pickle", "joblib")
        FORMATS = Literal["numpy", "pandas"]
        DEFAULT_FORMAT = "joblib"
        value: str = "joblib"
        score: Annotated[float, Field(description="numpy bound")]
        """,
    )
    assert scan_contract_type_strings(tmp_path) == []


def test_annotated_type_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import TypeAlias

        ConfigAlias: TypeAlias = "configurable_automl_engine.Config"
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_subscript_type_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import TypeAlias

        ConfigAlias: TypeAlias = list["configurable_automl_engine.Config"]
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_plain_subscript_type_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        ConfigAlias = list["configurable_automl_engine.Config"]
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_imported_type_alias_name_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import TypeAlias as TA

        ConfigAlias: TA = "configurable_automl_engine.Config"
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_aliased_forward_ref_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import ForwardRef as FR

        Alias = FR("configurable_automl_engine.Config")
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_aliased_type_var_bound_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import TypeVar as TV

        T = TV("T", bound="numpy.float32")
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_aliased_cast_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import cast as c

        value = c("configurable_automl_engine.X", object())
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_lowercase_subscript_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        Alias = list["numpy.float32"]
        Alias2 = dict[str, "numpy.ndarray"]
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"string-annotation"}


def test_chained_string_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        Alias = Copy = "configurable_automl_engine.X"
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_data_subscript_is_not_a_type_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        MAPPING = {"key": "numpy.float32"}
        DEFAULT = MAPPING["numpy.float32"]
        """,
    )
    assert scan_contract_type_strings(tmp_path) == []


def test_nested_type_aliases_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from typing import TYPE_CHECKING, TypeAlias

        if TYPE_CHECKING:
            Alias1: TypeAlias = list["configurable_automl_engine.Config"]

        try:
            Alias2 = list["configurable_automl_engine.Config"]
        except ImportError:
            Alias2 = None

        class Holder:
            Alias3 = "configurable_automl_engine.Config"

        def build():
            Alias4: TypeAlias = "configurable_automl_engine.Config"
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert len(violations) == 4
    assert {violation.rule for violation in violations} == {"string-annotation"}


def test_dotted_data_constant_is_not_a_type_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        DEFAULT_DTYPE = "numpy.float32"
        DEFAULT_FORMAT = "joblib"
        """,
    )
    assert scan_contract_type_strings(tmp_path) == []


def test_ci_commented_commands_do_not_satisfy_contracts_task(
    tmp_path: Path,
) -> None:
    ci = tmp_path / "ci.yaml"
    _write(
        ci,
        """
        workflows:
          quality-checks:
            tasks:
              - name: contracts-guarantees
                cubes:
                  - name: static
                    script:
                      - |
                        # scripts/check_contract_boundaries.py
                        scripts/verify_contract_distribution.py
                        scripts/smoke_contract_install.py
                        python3.11 -m build automl-contracts
                        twine check automl-contracts/dist/*
                        python3.11 -m venv /tmp/venv
                        pytest tests/contracts
                        mypy src
                        ruff check src/ automl-contracts/src/
                        ruff check tools/contract_boundaries.py
                        mypy tools/contract_boundaries.py
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any("static boundary check" in violation.message for violation in violations)


def test_ci_requires_a_dedicated_contracts_task(tmp_path: Path) -> None:
    ci = tmp_path / "ci.yaml"
    _write(
        ci,
        """
        workflows:
          quality-checks:
            tasks:
              - name: run-quality-checks
                cubes:
                  - name: legacy
                    script:
                      - |
                        ruff check src/ automl-contracts/src/
                        mypy src
                        pytest tests/contracts
                        python3.11 -m venv /tmp/venv
                        python3.11 -m build automl-contracts
                        twine check automl-contracts/dist/*
                        python3.11 scripts/check_contract_boundaries.py
                        python3.11 scripts/verify_contract_distribution.py
                        python3.11 scripts/smoke_contract_install.py
                        ruff check tools/contract_boundaries.py
                        mypy tools/contract_boundaries.py
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any(
        "no contracts-guarantees task" in violation.message for violation in violations
    )


def test_ci_checks_commands_inside_contracts_task(tmp_path: Path) -> None:
    ci = tmp_path / "ci.yaml"
    _write(
        ci,
        """
        workflows:
          quality-checks:
            tasks:
              - name: contracts-guarantees
                cubes:
                  - name: static
                    script:
                      - |
                        ruff check src/ automl-contracts/src/
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert violations
    assert all(
        "contracts-guarantees task" in violation.message for violation in violations
    )


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


def test_engine_contract_reexports_are_not_duplicates(tmp_path: Path) -> None:
    _write(
        tmp_path / "engine.py",
        """
        import automl_contracts
        from automl_contracts import Config as _Config

        Config = automl_contracts.Config
        ConfigAgain = _Config
        """,
    )
    assert scan_engine_duplicates(tmp_path, CONTRACT_SRC) == []


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

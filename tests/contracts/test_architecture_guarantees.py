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


def test_if_branch_alias_is_merged(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        if True:
            f = os.system
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_try_finally_alias_is_merged(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        try:
            f = os.system
        finally:
            pass
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_for_loop_alias_is_merged(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        for f in [os.system]:
            pass
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_module_alias_after_function_is_visible(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def run():
            f("id")

        f = os.system
        run()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_tuple_assignment_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        f, _ = os.system, None
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_tuple_assignment_import_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        f, _ = __import__, None
        f("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_tuple_alias_through_name_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        pair = (os.system, None)
        f, _ = pair
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_tuple_alias_through_name_for_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        pair = (__import__, None)
        f, _ = pair
        f("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_tuple_for_target_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        for f, _ in [(os.system, None)]:
            f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert "os.system" in violations[0].message


def test_conditional_callable_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        run = os.system if flag else os.popen
        run("id")

        run2 = os.system or os.popen
        run2("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2


def test_literal_subscript_callable_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        run = (os.system, os.popen)[0]
        run("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1


def test_conditional_import_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        f = __import__ if flag else eval
        f("configurable_automl_engine")

        g = (__import__, eval)[0]
        g("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 2


def test_named_for_iterable_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        runs = [os.system]
        for f in runs:
            f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert any("os.system" in violation.message for violation in violations)


def test_dict_literal_subscript_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        run = {"run": os.system}["run"]
        run("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_default_parameter_callable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def f(run=os.system):
            run("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_attribute_assignment_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Holder:
            pass

        holder = Holder()
        holder.run = os.system
        holder.run("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert any("os.system" in violation.message for violation in violations)


def test_returned_callable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def get_run():
            return os.system

        run = get_run()
        run("id")
        get_run()("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert any("os.system" in violation.message for violation in violations)


def test_returned_import_callable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        def get_import():
            return __import__

        f = get_import()
        f("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_walrus_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        if (f := os.system):
            f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_walrus_import_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        if (f := __import__):
            f("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_named_expression_in_same_expression_is_ordered(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        (f := os.system) and f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_named_expression_statement_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        (f := __import__)
        f("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_named_expression_assign_value_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        f = (g := os.system)
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_named_expression_as_callee_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        (f := os.system)("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_named_expression_as_attribute_base_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        (m := os).system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_named_expression_as_import_callee_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        (f := __import__)("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_named_expression_as_path_base_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        (p := Path("x")).read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_named_expression_bound_path_method_alias_is_tracked(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        read = (p := Path("x")).read_text
        read()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_named_expression_runtime_attribute_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import sys

        (s := sys).platform
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_named_expression_in_assert_is_tracked(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        assert (f := os.system) and f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1


def test_try_star_alias_is_merged(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        try:
            pass
        except* Exception:
            f = os.system
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 1
    assert "os.system" in violations[0].message


def test_bound_path_method_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        read = Path("x").read_text
        read()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}
    assert all("Path.read_text" in violation.message for violation in violations)


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


def test_literal_dict_dynamic_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        importlib.__dict__["import_module"]("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_literal_dict_builtins_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import builtins

        builtins.__dict__["__import__"]("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_literal_dict_dynamic_import_assignment_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        f = importlib.__dict__["import_module"]
        f("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_calls_in_annotations_and_handler_types_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        def f(x: __import__("configurable_automl_engine")):
            pass

        Alias: __import__("configurable_automl_engine") = 1

        try:
            pass
        except __import__("configurable_automl_engine"):
            pass
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_call_in_return_annotation_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def g() -> os.system("id"):
            pass
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_getattr_path_variable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        p = Path("x")
        getattr(p, "read_text")()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_getattr_vars_module_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        getattr(vars(os), "system")("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_getattr_vars_import_module_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        getattr(vars(importlib), "import_module")("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_vars_and_builtins_dynamic_import_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        vars(importlib)["import_module"]("configurable_automl_engine")
        __builtins__["__import__"]("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"dynamic-import"}


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


def test_literal_getattr_runtime_access_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        getattr(os, "system")("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_literal_getattr_assignment_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        f = getattr(os, "system")
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert any("os.system" in violation.message for violation in violations)


def test_open_ended_literal_indirection_forms_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        from pathlib import Path

        vars(os)["system"]("id")
        os.__dict__.get("system")("id")
        os.__getattribute__("system")("id")
        __builtins__["open"]("/etc/passwd")
        getattr(Path("x"), "read_text")()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 5
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_literal_dict_assignment_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        f = os.__dict__["system"]
        f("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert any("os.system" in violation.message for violation in violations)


def test_literal_getattr_keeps_alternative_aliases(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import json
        import os

        if flag:
            m = os
        else:
            m = json

        getattr(m, "system")("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert any("os.system" in violation.message for violation in violations)


def test_literal_getattr_runtime_attributes_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import sys

        getattr(sys, "platform")
        getattr(os, "environ")
        sys.__dict__["path"]
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_literal_dict_runtime_access_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        os.__dict__["system"]("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_stdlib_filesystem_modules_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import dbm
        import fileinput
        import glob
        import mmap
        import shelve
        import shutil
        import tarfile
        import webbrowser
        import zipfile

        glob.glob("/etc/*")
        zipfile.ZipFile("/etc/passwd")
        tarfile.open("/etc/passwd")
        dbm.open("/tmp/x", "c")
        shelve.open("/tmp/x")
        fileinput.input()
        mmap.mmap(0, 1)
        shutil.disk_usage("/")
        webbrowser.open("https://example.com")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 9
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_remaining_stdlib_runtime_apis_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import ntpath
        import posixpath
        import shutil
        import sys

        posixpath.exists("x")
        ntpath.exists("x")
        shutil.make_archive("x", "zip")
        shutil.unpack_archive("x")
        shutil.chown("x", user="root")
        sys.modules["configurable_automl_engine"]
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 6
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


def test_path_walrus_taint_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        (p := Path("x"))
        p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_for_target_taint_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        for p in [Path("x")]:
            p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_conditional_and_bool_taint_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        a = Path("x") if flag else Path("y")
        a.read_text()

        b = Path("x") or Path("y")
        b.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2


def test_path_named_for_iterable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        paths = [Path("x")]
        for p in paths:
            p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations


def test_path_with_target_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        with Path("x") as p:
            p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations


def test_path_comprehension_target_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        [p.read_text() for p in [Path("x")]]
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations


def test_path_taint_is_cleared_by_rebinding(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        p = Path("x")
        p = "abc"
        p.strip()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_path_taint_survives_a_single_branch(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        p = "abc"
        if flag:
            p = Path("x")
        getattr(p, "read_text")()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_taint_is_cleared_when_every_branch_rebinds(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        p = Path("x")
        if flag:
            p = "a"
        else:
            p = "b"
        getattr(p, "read_text")()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_path_tuple_target_taint_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        first, second = Path("x"), 1
        first.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_taint_survives_a_try_branch(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        p = "abc"
        try:
            p = Path("x")
        except OSError:
            pass
        getattr(p, "read_text")()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_except_handler_name_is_not_path_tainted(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        p = Path("x")
        try:
            pass
        except Exception as p:
            getattr(p, "read_text")()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


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


def test_path_taint_does_not_leak_between_classes(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        class A:
            path: Path

        class B:
            path: str

            def f(self) -> str:
                return self.path.strip()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_path_taint_does_not_leak_between_functions(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        def a(path: Path) -> None:
            pass

        def b(path: str) -> str:
            return path.strip()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_path_return_annotation_is_tainted(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def model_path() -> Path:
            return Path("x")

        model_path().read_text()
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


def test_shadowed_runtime_module_is_not_the_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import sys

        def f(sys):
            return sys.platform
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_module_rebinding_hides_later_access(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import sys

        sys = object()
        x = sys.argv
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_function_rebinding_hides_later_access(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import sys

        def f():
            sys = object()
            return sys.argv
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_class_rebinding_hides_later_access(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class C:
            os = object()
            x = os.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_function_rebinding_does_not_hide_earlier_access(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import sys

        def f():
            x = sys.argv
            sys = object()
            return x
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_rebinding_does_not_hide_earlier_access(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class C:
            x = os.environ
            os = None
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_except_handler_name_clears_callable_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        e = os.system
        try:
            pass
        except Exception as e:
            e("id")
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_module_rebinding_does_not_hide_earlier_access(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import sys

        x = sys.argv
        sys = object()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_shadowed_runtime_attribute_is_not_the_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from sys import platform

        def f(platform):
            return platform
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_shadowed_aliased_import_is_not_the_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import sys as s

        def f(s):
            return s.argv
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_decorator_runs_before_definition_shadowing(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        @__import__("configurable_automl_engine")
        def __import__(*args):
            pass
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_open_decorator_runs_before_definition_shadowing(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        @open("x")
        def open(*args):
            pass
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_shadowed_open_is_not_the_builtin(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from collections.abc import Callable

        def open(path: str) -> str:
            return path

        def f(open: Callable[[str], str]) -> str:
            return open("x")

        open("x")
        f(lambda value: value)
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_path_property_result_is_not_a_callable_sentinel(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        suffix = Path("model.txt").suffix
        suffix.upper()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


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


def test_union_string_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        Other = "str | numpy.ndarray"
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert [violation.rule for violation in violations] == ["string-annotation"]


def test_lowercase_qualified_string_alias_is_a_documented_limitation(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        Alias = "numpy.ndarray"
        DEFAULT_DTYPE = "numpy.float32"
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


def test_ci_trailing_comment_does_not_satisfy_contracts_task(
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
                        true  # scripts/check_contract_boundaries.py
                        true  # scripts/verify_contract_distribution.py
                        true  # scripts/smoke_contract_install.py
                        true  # python3.11 -m build automl-contracts
                        true  # twine check
                        true  # python3.11 -m venv
                        true  # pytest tests/contracts
                        true  # mypy src
                        true  # ruff check src/ automl-contracts/src/
                        true  # ruff check tools/contract_boundaries.py
                        true  # mypy tools/contract_boundaries.py
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any("static boundary check" in violation.message for violation in violations)


def test_missing_github_workflow_is_reported(tmp_path: Path) -> None:
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
                        scripts/check_contract_boundaries.py
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any(
        "GitHub CI configuration is missing" in violation.message
        for violation in violations
    )


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


def test_ci_requires_pip_check_and_artifact_verification(
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
                        scripts/check_contract_boundaries.py
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
                        ruff format --check src/
                        lint_imports
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    messages = {violation.message for violation in violations}
    assert any("pip check" in message for message in messages)
    assert any("engine wheel verification" in message for message in messages)
    assert any("engine sdist verification" in message for message in messages)


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
    assert any(
        "contracts-guarantees task" in violation.message for violation in violations
    )


def test_runtime_value_aliases_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import sys

        sys_alias = sys
        x = sys_alias.platform

        os_alias = os
        y = os_alias.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_runtime_value_alias_rebinding_is_ordered(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import sys

        alias = sys
        alias = object()
        x = alias.argv
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_runtime_value_alias_in_annotation_and_walrus_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import sys

        annotated: object = sys
        x = annotated.argv

        (assigned := sys)
        y = assigned.platform
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_runtime_store_target_is_not_an_access(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from sys import platform

        platform = "custom"
        x = platform
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_pipe_data_is_not_a_string_annotation(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        FORMATS = "numpy|pandas"
        """,
    )
    assert scan_contract_type_strings(tmp_path) == []


def test_in_memory_io_streams_are_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import io

        io.StringIO()
        io.BytesIO()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_file_backed_io_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import io

        io.FileIO("x")
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


def test_engine_duplicate_function_without_all_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "engine.py",
        """
        def check_n_folds(n_folds, validation_strategy):
            return None
        """,
    )
    violations = scan_engine_duplicates(tmp_path, CONTRACT_SRC)
    assert [violation.rule for violation in violations] == ["engine-duplicate"]
    assert "check_n_folds" in violations[0].message


def test_engine_module_level_logger_is_not_a_contract_symbol(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "engine.py",
        """
        logger = object()
        """,
    )
    assert scan_engine_duplicates(tmp_path, CONTRACT_SRC) == []


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
    assert "check_n_folds" in names


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


def test_path_read_before_rebinding_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        p = Path("x")
        p.read_text()
        p = "abc"
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_read_before_rebinding_in_function_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def load():
            p = Path("x")
            p.read_text()
            p = "abc"
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_conditional_module_rebinding_keeps_the_access_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        if flag:
            os = None
        x = os.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_conditional_class_rebinding_keeps_the_access_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class C:
            if flag:
                os = None
            x = os.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_loop_class_rebinding_keeps_the_access_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class D:
            for os in []:
                pass
            x = os.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_filesystem_and_logging_file_handlers_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import logging
        import os

        os.unlink("x")
        os.link("a", "b")
        os.utime("x")
        os.mknod("x")
        os.mkfifo("x")
        logging.FileHandler("x.log")
        logging.basicConfig(filename="x.log")
        logging.handlers.RotatingFileHandler("x.log")
        logging.config.fileConfig("logging.ini")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    reported = {violation.message.split("'")[1] for violation in violations}
    assert reported == {
        "logging.FileHandler",
        "logging.basicConfig",
        "logging.config.fileConfig",
        "logging.handlers.RotatingFileHandler",
        "os.link",
        "os.mkfifo",
        "os.mknod",
        "os.unlink",
        "os.utime",
    }


def test_concrete_pathlib_classes_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import pathlib
        from pathlib import PosixPath, WindowsPath

        PosixPath("x").read_text()
        WindowsPath("x").read_text()
        pathlib.PosixPath("x").read_text()
        pathlib.WindowsPath("x").read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 4
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_globals_locals_and_operator_indirection_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import operator
        import os

        globals()["os"].system("id")
        locals()["os"].system("id")
        operator.attrgetter("system")(os)("id")
        operator.methodcaller("system", "id")(os)
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    reported = {violation.message.split("'")[1] for violation in violations}
    assert {
        "globals",
        "locals",
        "operator.attrgetter",
        "operator.methodcaller",
    } <= reported


def test_computed_getattr_on_module_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        name = "sys" + "tem"
        getattr(os, name)("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_getattr_path_class_spellings_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import pathlib
        from pathlib import Path

        getattr(Path, "read_text")(Path("x"))
        getattr(pathlib.Path, "write_text")(pathlib.Path("x"), "d")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    reported = {violation.message.split("'")[1] for violation in violations}
    assert reported == {"pathlib.Path.read_text", "pathlib.Path.write_text"}


def test_forbidden_callables_in_value_position_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import functools
        import os

        functools.partial(os.system)("id")
        sorted(["id"], key=os.system)
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.message.split("'")[1] for violation in violations} == {
        "os.system"
    }


def test_dynamic_import_callable_in_value_position_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        list(map(__import__, ["configurable_automl_engine"]))
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_functional_type_constructors_are_scanned(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from dataclasses import make_dataclass
        from pydantic import create_model
        from typing import NamedTuple, TypedDict

        create_model("M", x=("numpy.ndarray", ...))
        create_model("M2", x=("configurable_automl_engine.Config", ...))
        TypedDict("T", {"a": "numpy.ndarray"})
        NamedTuple("N", [("a", "numpy.ndarray")])
        make_dataclass("D", [("a", "numpy.ndarray")])
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert len(violations) == 5
    assert {violation.rule for violation in violations} == {"string-annotation"}


def test_interpreter_version_metadata_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import sys

        if sys.version_info >= (3, 12):
            Alias = int
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_process_metadata_helpers_are_still_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        os.cpu_count()
        os.getcwd()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    reported = {violation.message.split("'")[1] for violation in violations}
    assert reported == {"os.cpu_count", "os.getcwd"}


def test_shadowed_module_dict_subscript_is_not_the_import(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        os = object()
        os.__dict__["system"]
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_bound_module_dictionary_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        d = vars(os)
        run = d["system"]
        run("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}
    assert all("os.system" in violation.message for violation in violations)


def test_bound_module_dunder_dict_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        d = os.__dict__
        d["system"]("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]
    assert "os.system" in violations[0].message


def test_bound_builtins_dictionary_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        d = __builtins__
        d["__import__"]("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]
    assert "builtins.__import__" in violations[0].message


def test_aliased_path_class_getattr_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import pathlib as pl
        from pathlib import Path as P

        getattr(P, "read_text")(P("x"))
        getattr(pl.Path, "write_text")(pl.Path("x"), "d")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    reported = {violation.message.split("'")[1] for violation in violations}
    assert reported == {"pathlib.Path.read_text", "pathlib.Path.write_text"}


def test_path_branch_merge_checks_both_orders(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        if flag:
            p = Path("x")
        else:
            p = "abc"
        getattr(p, "read_text")()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_imported_attribute_rebinding_is_not_an_access(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from os import environ

        environ = {}
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_builtins_module_indirection_is_a_dynamic_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        __builtins__.__import__("configurable_automl_engine")
        __builtins__.__dict__["__import__"]("configurable_automl_engine")
        vars(__builtins__)["__import__"]("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_import_alias_access_before_rebinding_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import sys as s

        x = s.argv
        s = object()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_value_alias_access_before_rebinding_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import sys

        alias = sys
        x = alias.argv
        alias = object()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_access_after_import_alias_rebinding_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import sys as s

        s = object()
        x = s.argv
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_named_literal_container_callable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        runs = [os.system]
        runs[0]("id")

        run = runs[0]
        run("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.message.split("'")[1] for violation in violations} == {
        "os.system"
    }


def test_named_literal_container_path_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        paths = [Path("x")]
        paths[0].read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_unannotated_helper_return_is_path_tainted(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def model_path():
            return Path("x")

        model_path().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_self_path_assigned_in_init_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Model:
            def __init__(self):
                self.path = Path("x")

            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_remaining_filesystem_apis_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import posix

        os.fstat(1)
        os.getxattr("x", "y")
        os.setxattr("x", "y", b"z")
        os.removexattr("x", "y")
        os.sync()
        posix.open("x", 0)
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    reported = {violation.message.split("'")[1] for violation in violations}
    assert reported == {
        "os.fstat",
        "os.getxattr",
        "os.removexattr",
        "os.setxattr",
        "os.sync",
        "posix.open",
    }


def test_wildcard_runtime_imports_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from os import *
        from posix import *
        from pathlib import *
        """,
    )
    violations = scan_contract_imports(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"wildcard-import"}


def test_unspaced_union_and_subscript_type_strings_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        Alias = "numpy.ndarray|pandas.DataFrame"
        Alias2 = "list[numpy.ndarray]"
        """,
    )
    violations = scan_contract_type_strings(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"string-annotation"}


def test_function_local_module_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def load():
            module = os
            return module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_local_module_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class C:
            module = os
            value = module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_conditional_import_alias_rebinding_keeps_the_access(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import sys as s

        if flag:
            s = None
        x = s.argv
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_bound_method_value_positions_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import functools
        from pathlib import Path

        functools.partial(Path("x").read_text)()
        handlers = [Path("x").read_text]

        def get_reader():
            return Path("x").read_text
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) >= 3
    assert {violation.rule for violation in violations} == {"runtime-access"}
    assert all("Path.read_text" in violation.message for violation in violations)


def test_path_variable_bound_method_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def get_reader():
            path = Path("x")
            return path.read_text
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_comprehension_target_does_not_leak_path_taint(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        p = "abc"
        items = [p for p in [Path("x")]]
        p.strip()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_module_dict_get_indirection_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import sys

        vars(os).get("system")("id")
        x = vars(sys).get("argv")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.message.split("'")[1] for violation in violations} == {
        "os.system",
        "sys.argv",
    }


def test_module_dict_get_dynamic_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        vars(importlib).get("import_module")("os")
        __builtins__.get("__import__")("os")
        vars(__builtins__).get("__import__")("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_getattr_module_dict_indirection_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        getattr(importlib, "__dict__")["import_module"]("os")
        getattr(importlib, "__dict__").get("import_module")("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_other_stdlib_wildcard_imports_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from ctypes import *
        from operator import *
        from threading import *
        from urllib import *
        """,
    )
    violations = scan_contract_imports(tmp_path)
    assert len(violations) == 4
    assert {violation.rule for violation in violations} == {"wildcard-import"}


def test_typing_wildcard_import_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from typing import *

        Alias = Optional[int]
        """,
    )
    assert scan_contract_imports(tmp_path) == []


def test_dict_iteration_callable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib
        import os

        for f in {importlib.import_module: 1}:
            f("os")

        for g in {os.system: 1}:
            g("id")
        """,
    )
    runtime = scan_contract_runtime_access(tmp_path)
    dynamic = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in runtime] == ["runtime-access"]
    assert [violation.rule for violation in dynamic] == ["dynamic-import"]


def test_class_method_path_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class C:
            def get(self):
                return Path("x")

        C().get().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_lambda_path_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        g = lambda: Path("x")
        g().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_ci_echoed_commands_do_not_satisfy_contracts_task(
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
                        echo "python scripts/check_contract_boundaries.py scripts/verify_contract_distribution.py scripts/smoke_contract_install.py python3.11 -m build automl-contracts twine check python3.11 -m venv pytest tests/contracts mypy src ruff check src/ automl-contracts/src/ ruff check tools/contract_boundaries.py mypy tools/contract_boundaries.py ruff format --check lint_imports pip check --engine-wheel --engine-sdist"
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any("static boundary check" in violation.message for violation in violations)


def test_path_parents_property_is_pure(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        for parent in Path("a/b").parents:
            parent.name
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_builtins_module_chain_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import builtins

        builtins.__dict__["sys"].modules["os"].system("id")
        getattr(__builtins__, "sys").modules["os"].system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_additional_os_and_nt_apis_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import nt
        import os

        os.pathconf("/tmp", "PC_NAME_MAX")
        os.lchmod("x", 0o644)
        os.lchown("x", 0, 0)
        os.unsetenv("X")
        value = os.environb[b"PATH"]
        nt.system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    reported = {violation.message.split("'")[1] for violation in violations}
    assert reported == {
        "nt.system",
        "os.environb",
        "os.lchmod",
        "os.lchown",
        "os.pathconf",
        "os.unsetenv",
    }


def test_ci_python_c_print_does_not_satisfy_contracts_task(
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
                        python3.11 -c "print('scripts/check_contract_boundaries.py scripts/verify_contract_distribution.py scripts/smoke_contract_install.py python3.11 -m build automl-contracts twine check python3.11 -m venv pytest tests/contracts mypy src ruff check src/ automl-contracts/src/ ruff check tools/contract_boundaries.py mypy tools/contract_boundaries.py ruff format --check lint_imports pip check --engine-wheel --engine-sdist')"
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any("static boundary check" in violation.message for violation in violations)


def test_class_self_path_and_method_chain_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class C:
            def __init__(self):
                self.path = Path("x")

            def get(self):
                return self.path

        C().get().read_text()

        class D:
            def base(self):
                return Path("x")

            def get(self):
                return self.base()

        D().get().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_lambda_forbidden_callables_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib
        import os

        loader = lambda name=importlib.import_module: name(
            "configurable_automl_engine"
        )
        loader()

        run = lambda: os.system
        run()("id")
        """,
    )
    runtime = scan_contract_runtime_access(tmp_path)
    dynamic = scan_contract_dynamic_imports(tmp_path)
    assert runtime
    assert dynamic
    assert {violation.rule for violation in runtime} == {"runtime-access"}
    assert {violation.rule for violation in dynamic} == {"dynamic-import"}


def test_dynamic_builtins_values_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        list(map(__builtins__.__import__, ["configurable_automl_engine"]))
        list(
            map(
                __builtins__.__dict__["__import__"],
                ["configurable_automl_engine"],
            )
        )
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_value_position_path_and_comprehension_callables_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        from pathlib import Path

        def get():
            return Path("x")

        reader = get().read_text

        class Model:
            def __init__(self):
                self.path = Path("x")

            def reader(self):
                return self.path.read_text

        runs = [run for run in [os.system]]
        runs[0]("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    reported = {violation.message.split("'")[1] for violation in violations}
    assert "Path.read_text" in reported
    assert "os.system" in reported


def test_logging_basic_config_without_filename_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import logging

        logging.basicConfig(level=logging.INFO)
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_logging_basic_config_with_filename_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import logging

        logging.basicConfig(filename="x.log")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_namespace_callables_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib
        import os

        class X:
            run = os.system
            load = importlib.import_module

        X.run("id")
        X.load("configurable_automl_engine")
        """,
    )
    runtime = scan_contract_runtime_access(tmp_path)
    dynamic = scan_contract_dynamic_imports(tmp_path)
    assert runtime
    assert dynamic
    assert {violation.rule for violation in runtime} == {"runtime-access"}
    assert {violation.rule for violation in dynamic} == {"dynamic-import"}


def test_self_attribute_callable_and_method_return_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Y:
            def __init__(self):
                self.run = os.system

            def go(self):
                self.run("id")

        class Z:
            def get(self):
                return os.system

        Z().get()("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_forbidden_decorator_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        @os.system
        def payload():
            pass
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_comprehension_target_callable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        [f("id") for f in [os.system]]
        {f("id") for f in [os.system]}
        {k: f("id") for k in [1] for f in [os.system]}
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_ci_python_c_printed_lint_import_does_not_satisfy_task(
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
                        python3.11 scripts/check_contract_boundaries.py
                        python3.11 scripts/verify_contract_distribution.py
                        python3.11 scripts/smoke_contract_install.py
                        python3.11 -m build automl-contracts
                        twine check
                        python3.11 -m venv
                        pytest tests/contracts
                        mypy src
                        ruff check src/ automl-contracts/src/
                        ruff check tools/contract_boundaries.py
                        mypy tools/contract_boundaries.py
                        ruff format --check
                        python3.11 -c "print('from importlinter.cli import lint_imports')"
                        pip check
                        --engine-wheel
                        --engine-sdist
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any("import-linter" in violation.message for violation in violations)


def test_self_path_facts_do_not_leak_between_classes(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        class A:
            def __init__(self):
                self.path = Path("x")

        class B:
            def __init__(self):
                self.path = object()

            def go(self):
                return self.path.read_text()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_self_path_fact_is_cleared_by_reassignment(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        class A:
            def __init__(self):
                self.path = Path("x")
                self.path = object()

            def go(self):
                return self.path.read_text()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_self_path_branch_may_analysis_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class A:
            def __init__(self, flag):
                if flag:
                    self.path = Path("x")
                else:
                    self.path = None

            def go(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_self_path_union_over_methods_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class A:
            def __init__(self):
                self.path = Path("x")

            def reset(self):
                self.path = None

            def go(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_alias_chain_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            _os = os
            run = _os.system

        X.run("id")

        class Y:
            mod = os

        Y.mod.system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_class_importlib_alias_chain_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class X:
            _il = importlib
            load = _il.import_module

        X.load("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_class_body_unpacking_and_branches_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import sys

        class X:
            first, second = os, None
            first.system("id")

        class Y:
            if flag:
                mod = os
            else:
                mod = sys

        Y.mod.system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_ci_python_c_import_only_does_not_satisfy_task(
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
                        python3.11 scripts/check_contract_boundaries.py
                        python3.11 scripts/verify_contract_distribution.py
                        python3.11 scripts/smoke_contract_install.py
                        python3.11 -m build automl-contracts
                        twine check
                        python3.11 -m venv
                        pytest tests/contracts
                        mypy src
                        ruff check src/ automl-contracts/src/
                        ruff check tools/contract_boundaries.py
                        mypy tools/contract_boundaries.py
                        ruff format --check
                        python3.11 -c "from importlinter.cli import lint_imports"
                        pip check
                        --engine-wheel
                        --engine-sdist
        """,
    )
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any("import-linter" in violation.message for violation in violations)


def test_self_path_from_annotated_parameter_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Model:
            def __init__(self, artifact_dir: Path):
                self.path = artifact_dir / "model.json"

            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_self_path_from_local_variable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Model:
            def __init__(self):
                base = Path("artifacts")
                self.path = base / "model.json"

            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_self_path_from_path_returning_method_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Model:
            def __init__(self):
                self.path = self.make()

            def make(self) -> Path:
                return Path("x")

            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_inherited_self_path_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Base:
            def __init__(self):
                self.path = Path("x")

        class Child(Base):
            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_body_for_with_match_and_named_expression_aliases_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib
        import os

        class Loop:
            for mod in [os]:
                run = mod.system

        Loop.run("id")

        class Context:
            with importlib as mod:
                load = mod.import_module

        Context.load("configurable_automl_engine")

        class Match:
            match 1:
                case 1:
                    mod = os
                case _:
                    mod = None

        Match.mod.system("id")

        class Named:
            (mod := os)

        Named.mod.system("id")
        """,
    )
    runtime = scan_contract_runtime_access(tmp_path)
    dynamic = scan_contract_dynamic_imports(tmp_path)
    assert runtime
    assert dynamic
    assert {violation.rule for violation in runtime} == {"runtime-access"}
    assert {violation.rule for violation in dynamic} == {"dynamic-import"}


def test_class_body_comprehension_callable_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            runs = [f for f in [os.system]]

        X.runs[0]("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_tuple_and_list_bindings_keep_element_aliases(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib
        import os

        class Tuple:
            a, b = (os, importlib)

        Tuple.a.system("id")
        Tuple.b.import_module("configurable_automl_engine")

        class List:
            mods = [os, importlib]

        List.mods[0].system("id")
        """,
    )
    runtime = scan_contract_runtime_access(tmp_path)
    dynamic = scan_contract_dynamic_imports(tmp_path)
    assert runtime
    assert dynamic
    assert {violation.rule for violation in runtime} == {"runtime-access"}
    assert {violation.rule for violation in dynamic} == {"dynamic-import"}


def test_self_path_in_match_and_except_star_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class MatchModel:
            def __init__(self, flag):
                match flag:
                    case 1:
                        self.path = Path("x")
                    case _:
                        self.path = None

            def load(self):
                return self.path.read_text()

        class TryModel:
            def __init__(self):
                try:
                    self.path = Path("x")
                except* ValueError:
                    self.path = None

            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_module_path_facts_reach_self_attributes(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        DEFAULT_PATH = Path("model.pkl")

        def model_path() -> Path:
            return Path("x")

        class First:
            def __init__(self):
                self.path = DEFAULT_PATH

            def load(self):
                return self.path.read_text()

        class Second:
            def __init__(self):
                self.path = model_path()

            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_local_name_does_not_clear_self_path_fact(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Model:
            def __init__(self):
                self.path = Path("x")
                path = None

            def load(self):
                return self.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_nested_walrus_in_class_assignment_is_bound(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            run = (mod := os).system

        X.mod.system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_class_level_path_annotation_and_property_are_order_independent(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Annotated:
            def load(self):
                return self.path.read_text()

            path: Path

        class Property:
            def load(self):
                return self.path.read_text()

            @property
            def path(self) -> Path:
                return Path("x")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_class_container_subscripts_keep_each_element(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib
        import os

        class X:
            mods = [importlib, os]
            tools = {"load": os.system}

        X.mods[1].system("id")
        X.tools["load"]("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_self_container_subscript_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            def __init__(self):
                self.mods = [os, os.system]

            def go(self):
                self.mods[1]("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_augmented_division_preserves_path_taint(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        p = Path("x")
        p /= "nested"
        p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_inline_literal_path_subscript_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        [Path("x")][0].read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_additional_runtime_deny_list_entries_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import bz2
        import codecs
        import gzip
        import lzma
        import os
        import tarfile
        import zipfile

        codecs.open("x")
        gzip.open("x")
        bz2.open("x")
        lzma.open("x")
        tarfile.open("x")
        zipfile.ZipFile("x")
        os.abort()
        os._exit(1)
        os.waitpid(1, 0)
        os.umask(0)
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    reported = {violation.message.split("'")[1] for violation in violations}
    assert reported == {
        "bz2.open",
        "codecs.open",
        "gzip.open",
        "lzma.open",
        "os._exit",
        "os.abort",
        "os.umask",
        "os.waitpid",
        "tarfile.open",
        "zipfile.ZipFile",
    }


def test_nested_function_inherits_enclosing_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def outer():
            alias = os

            def inner():
                return alias.system("id")

            return inner()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_container_is_visible_through_self_and_get(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            tools = {"run": os.system}
            runs = [os.system]

            def go(self):
                self.tools["run"]("id")
                self.runs[0]("id")

        X.tools.get("run")("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_escaped_container_subscript_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class X:
            mods = [None, importlib]

        f = X.mods[1]
        f.import_module("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_container_subscript_from_method_return_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class X:
            mods = [None, importlib]

            def get(self):
                return self.mods[1]

        X().get().import_module("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_negative_and_computed_container_indices_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [None, os]

        index = 1
        X.mods[-1].system("id")
        X.mods[index].system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_global_assignment_propagates_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        alias = None

        def setup():
            global alias
            alias = os

        setup()
        alias.system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_shadowed_module_is_not_aliased(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        os = object()
        alias = os
        alias.system("id")
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_computed_getattr_on_importlib_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        name = "import_" + "module"
        getattr(importlib, name)("os")
        importlib.__dict__[name]("os")
        vars(importlib)[name]("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_computed_container_index_checks_every_candidate(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import json
        import math
        import os

        class ListOps:
            ops = [math.sqrt, os.system]

        index = 1
        ListOps.ops[index].system("id")

        class DictOps:
            ops = {"safe": json.loads, "run": os.system}

        key = "run"
        DictOps.ops[key].system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_class_level_call_return_container_element_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class X:
            mods = [None, importlib]

            def get(self):
                return X.mods[1]

        X.get().import_module("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_element_alias_does_not_leak_into_function_scope(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mods = [os]

        def f(X):
            value = X.mods[0]
            return value.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_computed_container_index_unions_all_candidates(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import math
        import os

        class X:
            mods = [os, math]

        index = 1
        X.mods[index].environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_function_local_element_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class X:
            mods = [None, importlib]

        def load():
            alias = X.mods[1]
            alias.import_module("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_element_alias_does_not_leak_into_parameters(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mods = [os]

        def f(X):
            value = X.mods[0]
            return value.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_method_attribute_chain_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mod = os

            def get(self):
                return self.mod.system

        X().get()("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_alias_call_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class X:
            mods = [None, importlib]

            def get(self):
                return X.mods[1]

        Y = X
        Y.get().import_module("configurable_automl_engine")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_class_level_path_container_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class X:
            paths = [Path("x")]

        X.paths[0].read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_module_element_alias_is_visible_in_nested_scope(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [os]

        alias = X.mods[0]

        def f():
            return alias.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_element_alias_merges_out_of_compound_statements(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [None, os]

        if True:
            alias = X.mods[1]

        alias.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_escaped_computed_container_index_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import math
        import os

        class X:
            mods = [os, math]

        index = 1
        alias = X.mods[index]
        alias.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_method_computed_container_index_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import math
        import os

        class X:
            mods = [os, math]

            def get(self, index):
                return self.mods[index]

        X().get(1).environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_alias_applies_to_container_attributes(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [os]

        Y = X
        Y.mods[0].environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_rebound_class_facts_are_not_reused(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mods = [os]

            def get(self):
                return self.mods[0]

        X = object()
        X.get().environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_container_alias_index_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [None, os]

        runs = X.mods
        runs[1].environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_container_alias_safe_index_is_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mods = [None, os]

        runs = X.mods
        runs[0].environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_for_target_from_class_container_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [None, os]

        for f in X.mods:
            f.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_for_target_from_container_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [os]

        runs = X.mods
        for f in runs:
            f.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_tuple_unpacking_from_class_container_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [None, os]

        first, second = X.mods
        second.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_comprehension_target_from_class_container_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [os]

        values = [mod.environ for mod in X.mods]
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_literal_container_element_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        alias = [os][0]
        alias.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_literal_container_for_target_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        for alias in [os]:
            alias.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_method_returning_bare_class_attribute_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mod = os
            mods = {"run": os}

            def get_self(self):
                return self.mod

            @classmethod
            def get_cls(cls):
                return cls.mod

            @staticmethod
            def get_class():
                return X.mod

        X().get_self().environ
        X.get_cls().environ
        X.get_class().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_transitive_class_alias_container_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [None, os]

        Y = X
        Z = Y
        Z.mods[1].environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_function_local_class_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mods = [None, os]

        def load():
            Y = X
            return Y.mods[1].environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_local_class_shadow_does_not_reuse_module_facts(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mod = os

            def get(self):
                return self.mod

        class Loader:
            mod = object()

            def get(self):
                return object()

        def load():
            X = Loader()
            return X.get().environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_local_class_shadow_ignores_class_attributes(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mod = os

        class Loader:
            mod = object()

        def load():
            X = Loader()
            return X.mod.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_inherited_class_facts_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Base:
            mod = os
            mods = [os]

            def get(self):
                return self.mod

        class X(Base):
            pass

        X.mod.environ
        X.mods[0].environ
        X().get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 3
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_inherited_path_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Base:
            def get(self) -> Path:
                return Path("x")

        class C(Base):
            pass

        C().get().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_facts_do_not_depend_on_method_order(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            def get(self):
                return self.x

            def __init__(self):
                self.x = os

        X().get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_mapping_values_iteration_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        handlers = {"run": os.system}
        for handler in handlers.values():
            handler("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_mapping_items_iteration_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        handlers = {"run": os.system}
        for name, handler in handlers.items():
            handler("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_mapping_items_binds_only_value_position(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        handlers = {"run": os.system}
        for name, handler in handlers.items():
            name("id")
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_mapping_values_dynamic_iteration_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        handlers = {"load": importlib.import_module}
        for name, handler in handlers.items():
            handler("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_tuple_for_target_over_tuple_elements_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        for name, mod in [(None, os)]:
            mod.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_tuple_for_target_dynamic_elements_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        for name, loader in [(None, importlib)]:
            loader.import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_method_parameter_shadow_is_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mod = os

            def get(self, mod):
                return mod

        class Safe:
            environ = {"A": "1"}

        X().get(Safe()).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_method_local_shadow_is_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mod = os

            def get(self):
                mod = Safe()
                return mod

        class Safe:
            environ = {"A": "1"}

        X().get().environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_local_class_alias_construction_chain_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            mod = os.system

            def get(self):
                return self.mod

        def load():
            Y = X
            return Y().get()("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_local_class_alias_path_chain_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class X:
            def get(self) -> Path:
                return Path("x")

        def load():
            Y = X
            return Y().get().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_incremental_mapping_values_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        handlers = {}
        handlers["run"] = os.system
        for handler in handlers.values():
            handler("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_incremental_mapping_values_dynamic_import_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        handlers = {}
        handlers["load"] = importlib.import_module
        for handler in handlers.values():
            handler("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_nested_mapping_values_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        handlers = {"a": {"run": os.system}}
        for handler in handlers["a"].values():
            handler("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_wrapped_mapping_values_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        handlers = {"run": os.system}
        for handler in list(handlers.values()):
            handler("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_for_target_unpacks_variable_of_tuples(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        pairs = [(None, os)]
        for name, mod in pairs:
            mod.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_for_target_unpacks_variable_tuple_of_tuples(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        pairs = ((None, importlib),)
        for name, loader in pairs:
            loader.import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_class_alias_as_base_keeps_attributes(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Base:
            mod = os

        Alias = Base

        class X(Alias):
            pass

        X.mod.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_alias_as_base_keeps_method_returns(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Base:
            def get(self):
                return os

        Alias = Base

        class X(Alias):
            pass

        X().get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_alias_as_base_keeps_path_returns(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Base:
            def get(self) -> Path:
                return Path("x")

        Alias = Base

        class X(Alias):
            pass

        X().get().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_method_return_unions_over_branches(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import json
        import os

        class X:
            mod = os
            safe = json

            def get(self, flag):
                if flag:
                    return self.mod
                return self.safe

        X().get(True).environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_method_return_unions_path_over_branches(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class X:
            def get(self, flag):
                if flag:
                    return Path("x")
                return object()

        X().get(True).read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_unresolved_attribute_override_shadows_inherited_fact(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Base:
            mod = os

        class X(Base):
            mod = Safe()

        class Safe:
            environ = {}

        X.mod.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_unresolved_method_override_shadows_inherited_return(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Base:
            def get(self):
                return os

        class Y(Base):
            def get(self):
                return Safe()

        class Safe:
            environ = {}

        Y().get().environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_import_shadow_in_method_is_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mod = os

            def get(self):
                import json as mod
                return mod

        X().get().environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_exception_shadow_in_method_is_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Holder(Exception):
            environ = {}

        class X:
            mod = os

            def get(self):
                try:
                    raise Holder()
                except Holder as mod:
                    return mod

        X().get().environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_match_capture_shadow_in_method_is_not_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class X:
            mod = os

            def get(self, value):
                match value:
                    case mod:
                        return mod

        class Safe:
            environ = {}

        X().get(Safe()).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_runpy_execution_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import runpy

        runpy.run_path("/tmp/payload.py")
        runpy.run_module("payload")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_importlib_machinery_loader_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from importlib.machinery import SourceFileLoader

        SourceFileLoader("m", "/tmp/payload.py").load_module()
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_zipimport_execution_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import zipimport

        zipimport.zipimporter("/tmp/payload.zip").load_module("m")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_pkgutil_loader_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import pkgutil

        pkgutil.get_loader("payload")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_getattr_runpy_execution_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import runpy

        getattr(runpy, "run_path")("/tmp/payload.py")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert violations
    assert {violation.rule for violation in violations} == {"dynamic-import"}


def test_nested_subscript_mutation_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        handlers = {}
        handlers["a"] = {}
        handlers["a"]["run"] = os.system
        for handler in handlers["a"].values():
            handler("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_nested_subscript_mutation_dynamic_import_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        handlers = {}
        handlers["a"] = {}
        handlers["a"]["load"] = importlib.import_module
        for name, handler in handlers["a"].items():
            handler("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_attribute_subscript_mutation_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            def __init__(self):
                self.handlers = {}
                self.handlers["run"] = os.system

            def use(self):
                for handler in self.handlers.values():
                    handler("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_nested_mapping_overwrite_clears_stale_alias(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        handlers = {"a": {"run": os.system}}
        handlers["a"] = {}
        for handler in handlers["a"].values():
            handler("id")
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_runtime_access_through_call_argument_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def get(mod):
            return mod.environ

        get(os)
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_dynamic_import_through_call_argument_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        def load(loader):
            loader.import_module("os")

        load(importlib)
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_path_access_through_call_argument_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def read(p):
            return p.read_text()

        read(Path("x"))
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_super_delegated_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Base:
            def get(self):
                return os

        class X(Base):
            def get(self):
                return super().get()

        X().get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_super_delegated_path_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Base:
            def get(self):
                return Path("x")

        class X(Base):
            def get(self):
                return super().get()

        X().get().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_instance_class_method_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def get(self):
                return os

        loader = Loader()
        loader.get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_instance_class_method_return_dynamic_import_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class Loader:
            def get(self):
                return importlib

        loader = Loader()
        loader.get().import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_self_and_cls_method_returns_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class X:
            def get(self):
                return os

            def use(self):
                return self.get().environ

        class Y:
            @classmethod
            def get(cls):
                return os

            @classmethod
            def use(cls):
                return cls.get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert len(violations) == 2
    assert {violation.rule for violation in violations} == {"runtime-access"}


def test_annotated_instance_path_field_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Config:
            path: Path

        cfg = Config()
        cfg.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_assigned_instance_path_field_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Config:
            def __init__(self, path: Path):
                self.path = path

        cfg = Config(Path("x"))
        cfg.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_identity_helper_return_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def identity(value):
            return value

        identity(os).environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_identity_helper_dynamic_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        def identity(value):
            return value

        identity(importlib).import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_path_parents_subscript_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        Path("x").parents[0].read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_ci_requires_source_scope_format_check(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / ".sourcecraft" / "ci.yaml"
    lines = source.read_text(encoding="utf-8").splitlines()
    kept = [
        line
        for line in lines
        if "ruff format --check src/ automl-contracts/src/" not in line
    ]
    assert len(kept) < len(lines)
    ci = tmp_path / "ci.yaml"
    ci.write_text("\n".join(kept) + "\n", encoding="utf-8")
    violations = scan_ci_configuration(ci, tmp_path / "missing-github.yaml")
    assert any(
        "ruff format check for both packages" in violation.message
        for violation in violations
    )


def test_identity_result_path_argument_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def identity(value):
            return value

        def use(value):
            value.read_text()

        use(identity(Path("x")))
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_identity_direct_path_result_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def identity(value):
            return value

        identity(Path("x")).read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_identity_result_non_path_argument_is_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        def identity(value):
            return value

        def use(value):
            value.strip()

        use(identity("safe"))
        identity("safe").strip()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_annotated_instance_path_parameters_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class Config:
            path: Path
            general: General

        def read_direct(cfg: Config) -> str:
            return cfg.path.read_text()

        def read_nested(cfg: Config) -> str:
            return cfg.general.path.read_text()

        def read_via_local(cfg: Config) -> str:
            p = cfg.path
            return p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 3


def test_optional_annotated_instance_path_parameter_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path
        from typing import Optional

        class Config:
            path: Path

        def read(cfg: Optional[Config]) -> str:
            assert cfg is not None
            return cfg.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_instance_path_parameter_from_call_argument_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Config:
            path: Path

        def read(cfg) -> str:
            return cfg.path.read_text()

        read(Config())
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_non_path_instance_fields_are_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path
        from typing import Optional

        class General:
            title: str

        class Config:
            path: Path
            title: str
            general: Optional[General]

        def read(cfg: Config) -> str:
            return cfg.title.strip() + str(cfg.general)
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_class_method_parameter_return_runtime_access_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def get(self, module):
                return module

        Loader().get(os).system("id")
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_method_self_alias_return_runtime_access_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            module = os

            def get(self):
                alias = self
                return alias.module

        Loader().get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_class_method_parameter_and_self_alias_path_returns_are_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Loader:
            def get(self, path):
                return path

        class Aliased:
            path = Path("x")

            def get(self):
                alias = self
                return alias.path

        Loader().get(Path("x")).read_text()
        Aliased().get().read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 2


def test_class_method_parameter_return_dynamic_import_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class Loader:
            def get(self, module):
                return module

        Loader().get(importlib).import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_inherited_class_method_parameter_return_runtime_access_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Base:
            def get(self, module):
                return module

        class Child(Base):
            pass

        Child().get(os).environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_inherited_class_method_parameter_return_dynamic_import_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        class Base:
            def get(self, module):
                return module

        class Child(Base):
            pass

        Child().get(importlib).import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_class_method_parameter_with_non_runtime_argument_is_not_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        class Loader:
            def get(self, module: str) -> str:
                return module

        Loader().get("safe").strip()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_except_handler_instance_alias_is_not_stale(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class SafeError(Exception):
            module = types.SimpleNamespace(environ="safe")

        def read() -> str:
            holder = Holder()
            try:
                raise SafeError()
            except SafeError as holder:
                return holder.module.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_tuple_unpacking_clears_instance_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read() -> str:
            holder = Holder()
            holder, _ = (Safe(), 1)
            return holder.module.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_list_unpacking_clears_instance_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read() -> str:
            holder = Holder()
            [holder] = [Safe()]
            return holder.module.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_match_capture_clears_instance_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read(value) -> str:
            holder = Holder()
            match value:
                case Safe() as holder:
                    return holder.module.environ
            return ""
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_for_target_clears_instance_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read() -> str:
            holder = Holder()
            for holder in [Safe()]:
                return holder.module.environ
            return ""
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_with_target_clears_instance_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read() -> str:
            holder = Holder()
            with Safe() as holder:
                return holder.module.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_if_branches_clear_instance_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read(flag: bool) -> str:
            holder = Holder()
            if flag:
                holder = Safe()
            else:
                holder = Safe()
            return holder.module.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_conditional_instance_alias_still_reports_reachable_runtime_value(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read(flag: bool) -> str:
            holder = Holder()
            if flag:
                holder = Safe()
            return holder.module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_path_parents_non_literal_index_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        index = 0
        Path("x").parents[index].read_text()
        Path("x").parents[-1].read_text()
        Path("x").parents[index + 1].read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 3


def test_path_parents_non_literal_index_pure_operations_are_allowed(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        index = 0
        Path("x").parents[index].name
        Path("x").parents[index].as_posix()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_init_parameter_is_not_resolved_as_other_method_parameter(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Loader:
            def __init__(self, module):
                self.module = module

            def get(self, module):
                return self.module

        Loader("safe").get(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_init_parameter_is_resolved_against_constructor(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, module):
                self.module = module

            def get(self, module):
                return self.module

        Loader(os).get("safe").environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_local_instance_nested_path_field_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class Config:
            general: General

        cfg = Config()
        cfg.general.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_self_nested_path_field_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class User:
            def __init__(self):
                self.general = General()

            def read(self):
                return self.general.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_imported_instance_nested_path_field_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "models.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class Config:
            general: General
        """,
    )
    _write(
        tmp_path / "bad.py",
        """
        from models import Config

        cfg = Config()
        cfg.general.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_imported_package_instance_nested_path_field_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "automl_contracts" / "__init__.py",
        "",
    )
    _write(
        tmp_path / "automl_contracts" / "general.py",
        """
        from pathlib import Path

        class General:
            path_to_model: Path
        """,
    )
    _write(
        tmp_path / "automl_contracts" / "config.py",
        """
        from automl_contracts.general import General

        class Config:
            general: General
        """,
    )
    _write(
        tmp_path / "bad.py",
        """
        from automl_contracts.config import Config

        cfg = Config()
        cfg.general.path_to_model.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_match_without_matching_case_keeps_instance_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read(value) -> str:
            holder = Holder()
            match value:
                case Safe() as holder:
                    pass
            return holder.module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_reassigned_method_parameters_are_not_return_arguments(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Null:
            def __enter__(self):
                return "safe"

            def __exit__(self, *args):
                return False

        class Loader:
            def by_assign(self, module):
                module = "safe"
                return module

            def by_for(self, module):
                for module in ["safe"]:
                    pass
                return module

            def by_with(self, module):
                with Null() as module:
                    pass
                return module

            def by_except(self, module):
                try:
                    pass
                except Exception as module:
                    pass
                return module

            def by_walrus(self, module):
                if (module := "safe"):
                    pass
                return module

        Loader().by_assign(os).environ
        Loader().by_for(os).environ
        Loader().by_with(os).environ
        Loader().by_except(os).environ
        Loader().by_walrus(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_shadowed_identity_helper_is_not_path_identity(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        def identity(value):
            return value

        def read():
            identity = lambda value: 42
            return identity(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_identity_helper_alias_path_result_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def identity(value):
            return value

        f = identity
        f(Path("x")).read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_self_alias_clears_stale_parameter_reference(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Loader:
            module = os

            def get(self, module):
                alias = module
                alias = self
                return alias

        Loader().get(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_init_parameter_is_resolved_for_assigned_instance(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, module):
                self.module = module

            def get(self):
                return self.module

        loader = Loader(os)
        loader.get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_relative_import_instance_nested_path_field_is_reported(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "automl_contracts" / "__init__.py", "")
    _write(
        tmp_path / "automl_contracts" / "models.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class Config:
            general: General
        """,
    )
    _write(
        tmp_path / "automl_contracts" / "config.py",
        """
        from .models import Config

        cfg = Config()
        cfg.general.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_local_instance_path_alias_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class Config:
            path: Path

        cfg = Config()
        p = cfg.path
        p.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_intermediate_instance_alias_path_field_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class Config:
            general: General

        cfg = Config()
        g = cfg.general
        g.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_wildcard_match_branch_clears_stale_instance_alias(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        class Holder:
            module = os

        class Safe:
            module = types.SimpleNamespace(environ="safe")

        def read(value) -> str:
            holder = Holder()
            match value:
                case _:
                    holder = Safe()
            return holder.module.environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_identity_shadowing_covers_binding_forms(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        def identity(value):
            return value

        class Manager:
            def __enter__(self):
                return lambda value: 42

            def __exit__(self, *args):
                return False

        def by_for():
            for identity in [lambda value: 42]:
                pass
            return identity(os).environ

        def by_with():
            with Manager() as identity:
                pass
            return identity(os).environ

        def by_except():
            try:
                pass
            except Exception as identity:
                pass
            return identity(os).environ

        def by_nested_def():
            def identity(value):
                return 42
            return identity(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_method_alias_invalidated_by_local_bindings(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Null:
            def __enter__(self):
                return "safe"

            def __exit__(self, *args):
                return False

        class Loader:
            def by_for(self, module):
                alias = module
                for alias in ["safe"]:
                    pass
                return alias

            def by_with(self, module):
                alias = module
                with Null() as alias:
                    pass
                return alias

            def by_except(self, module):
                alias = module
                try:
                    pass
                except Exception as alias:
                    pass
                return alias

            def by_walrus(self, module):
                alias = module
                if (alias := "safe"):
                    pass
                return alias

            def by_unpacking(self, module):
                alias = module
                alias, _ = ("safe", 1)
                return alias

        Loader().by_for(os).environ
        Loader().by_with(os).environ
        Loader().by_except(os).environ
        Loader().by_walrus(os).environ
        Loader().by_unpacking(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_identity_shadowing_inside_bound_bodies(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        def identity(value):
            return value

        class Manager:
            def __enter__(self):
                return lambda value: 42

            def __exit__(self, *args):
                return False

        def by_for():
            for identity in [lambda value: 42]:
                return identity(os).environ
            return ""

        def by_with():
            with Manager() as identity:
                return identity(os).environ
            return ""

        def by_except():
            try:
                pass
            except Exception as identity:
                return identity(os).environ
            return ""
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_shadowed_identity_path_call_is_not_path(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from pathlib import Path

        def identity(value):
            return value

        def read():
            identity = lambda value: "safe"
            return identity(Path("x")).read_text()
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_local_identity_that_returns_argument_keeps_path_taint(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        def identity(value):
            return value

        def read():
            identity = lambda value: value
            return identity(Path("x")).read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_annotated_parameter_alias_nested_path_field_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class Config:
            general: General

        def read(cfg: Config) -> str:
            g = cfg.general
            return g.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_instance_constructor_is_copied_through_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, module):
                self.module = module

            def get(self):
                return self.module

        loader = Loader(os)
        other = loader
        other.get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_augmented_assignment_invalidates_method_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os

        class Loader:
            def get(self, module):
                alias = module
                alias += "safe"
                return alias

        Loader().get(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_direct_init_parameter_attribute_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, module):
                self.module = module

        loader = Loader(os)
        loader.module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_direct_chained_init_parameter_attribute_is_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, module):
                self.module = module

        Loader(os).module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_keyword_init_parameter_attribute_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, extra, module):
                self.module = module

        Loader("x", module=os).module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_default_init_parameter_attribute_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, module=os):
                self.module = module

        Loader().module.environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_parameter_types_are_scoped_per_function_node(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from pathlib import Path

        class General:
            path: Path

        class Config:
            general: General

        class Safe:
            pass

        class Reader:
            def read(self, cfg: Safe) -> None:
                pass

        def read(cfg: Config) -> str:
            g = cfg.general
            return g.path.read_text()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_identity_alias_runtime_access_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def identity(value):
            return value

        same = identity
        same(os).environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_identity_alias_dynamic_import_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import importlib

        def identity(value):
            return value

        same = identity
        same(importlib).import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]


def test_identity_alias_merged_from_both_branches(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def identity(value):
            return value

        def read(flag):
            if flag:
                same = identity
            else:
                same = identity
            return same(os).environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_identity_alias_overwritten_safely_in_both_branches(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        def identity(value):
            return value

        def read(flag):
            same = identity
            if flag:
                same = lambda value: types.SimpleNamespace(environ="safe")
            else:
                same = lambda value: types.SimpleNamespace(environ="safe")
            return same(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_equal_constructors_in_branches_are_merged(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        class Loader:
            def __init__(self, module):
                self.module = module

            def get(self):
                return self.module

        def read(flag):
            if flag:
                loader = Loader(os)
            else:
                loader = Loader(os)
            return loader.get().environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"]


def test_nested_definition_clears_identity_alias(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        import os
        import types

        def identity(value):
            return value

        same = identity

        def same(value):
            return types.SimpleNamespace(environ="safe")

        same(os).environ
        """,
    )
    assert scan_contract_runtime_access(tmp_path) == []


def test_global_and_closure_facts_are_visible_in_functions(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        import os

        def identity(value):
            return value

        same = identity

        class Loader:
            def __init__(self, module):
                self.module = module

        loader = Loader(os)

        def read_global():
            first = same(os).environ
            second = loader.module.environ
            return first + second

        def outer():
            local_same = identity
            local_loader = Loader(os)

            def inner():
                first = local_same(os).environ
                second = local_loader.module.environ
                return first + second

            return inner()
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 4


def test_imported_constructor_and_identity_facts_are_reported(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(
        tmp_path / "pkg" / "models.py",
        """
        class Library:
            def __init__(self, module):
                self.module = module
        """,
    )
    _write(
        tmp_path / "pkg" / "helpers.py",
        """
        def identity(value):
            return value
        """,
    )
    _write(
        tmp_path / "pkg" / "use.py",
        """
        import os
        from .models import Library
        from .helpers import identity

        Library(os).module.environ
        identity(os).environ
        """,
    )
    violations = scan_contract_runtime_access(tmp_path)
    assert [violation.rule for violation in violations] == ["runtime-access"] * 2


def test_imported_identity_alias_dynamic_import_is_reported(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(
        tmp_path / "pkg" / "helpers.py",
        """
        def identity(value):
            return value
        """,
    )
    _write(
        tmp_path / "pkg" / "use.py",
        """
        import importlib
        from .helpers import identity

        identity(importlib).import_module("os")
        """,
    )
    violations = scan_contract_dynamic_imports(tmp_path)
    assert [violation.rule for violation in violations] == ["dynamic-import"]

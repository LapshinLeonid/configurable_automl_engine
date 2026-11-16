"""Release tag, registry, publication and consumer checks for the contract."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Self

import automl_contracts
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
REFERENCE_CONFIG = Path(__file__).resolve().parent / "data" / "reference_config.json"
IMPORT_MATRIX = (
    Path(__file__).resolve().parents[1] / "engine" / "golden" / "import_matrix.json"
)


def require_engine() -> None:
    """Skip checks that exercise the installed engine."""
    if importlib.util.find_spec("configurable_automl_engine") is None:
        pytest.skip("the engine is not installed in this environment")


def load_script(name: str) -> ModuleType:
    """Load a repository script as a module."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"release_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release() -> ModuleType:
    return load_script("contract_release")


@pytest.fixture(scope="module")
def consumer() -> ModuleType:
    return load_script("e2e_contract_consumer")


@pytest.fixture(scope="module")
def engine_smoke() -> ModuleType:
    return load_script("smoke_engine_install")


def write_pyproject(tmp_path: Path, version: str = "0.1.0") -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "automl-contracts"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    return path


def test_pyproject_version_is_read(release: ModuleType, tmp_path: Path) -> None:
    pyproject = write_pyproject(tmp_path, "1.2.3")
    assert release.read_distribution(pyproject) == ("automl-contracts", "1.2.3")
    assert release.read_distribution_version(pyproject) == "1.2.3"


def test_pyproject_without_version_is_rejected(
    release: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "automl-contracts"\n', encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="static project name and version"):
        release.read_distribution(path)


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("contracts-v0.1.0", "0.1.0"),
        ("contracts-v1.2.3", "1.2.3"),
        ("contracts-v10.20.30", "10.20.30"),
    ],
)
def test_release_tag_is_parsed(release: ModuleType, tag: str, version: str) -> None:
    assert release.parse_release_tag(tag) == version


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "v0.1.0",
        "contracts-0.1.0",
        "contracts-v0.1",
        "contracts-v0.1.0.0",
        "contracts-v0.1.0+local",
        "release-v0.1.0",
        "contracts-vabc",
        "contracts-v0.2.0-rc1",
        " contracts-v0.1.0",
    ],
)
def test_tags_outside_the_release_mask_are_rejected(
    release: ModuleType, tag: str
) -> None:
    with pytest.raises(release.ReleaseError):
        release.parse_release_tag(tag)


def test_matching_tag_returns_the_distribution_version(
    release: ModuleType, tmp_path: Path
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    assert release.verify_release_tag("contracts-v0.1.0", pyproject) == "0.1.0"


def test_tag_and_distribution_version_mismatch_is_rejected(
    release: ModuleType, tmp_path: Path
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    with pytest.raises(
        release.ReleaseError, match="does not match|publication is not allowed"
    ):
        release.verify_release_tag("contracts-v0.2.0", pyproject)


def test_foreign_distribution_is_rejected(release: ModuleType, tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "other-package"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    with pytest.raises(release.ReleaseError, match="declares distribution"):
        release.verify_release_tag("contracts-v0.1.0", pyproject)


def test_registry_settings_use_the_environment(release: ModuleType) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": " registry123 ",
            "PYPI_REGISTRY_PAT": "secret-token",
        }
    )
    assert settings.registry_id == "registry123"
    assert settings.upload_url == (
        "https://pkg.sourcecraft.tech/pypi/ooo-ainod/registry123/legacy/"
    )
    assert settings.index_url == (
        "https://pkg.sourcecraft.tech/pypi/ooo-ainod/registry123/simple/"
    )
    assert "secret-token" not in repr(settings)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "SOURCECRAFT_PYPI_REGISTRY_ID"),
        ({"SOURCECRAFT_PYPI_REGISTRY_ID": "  "}, "SOURCECRAFT_PYPI_REGISTRY_ID"),
        (
            {"SOURCECRAFT_PYPI_REGISTRY_ID": "bad/id", "PYPI_REGISTRY_PAT": "t"},
            "invalid registry identifier",
        ),
        ({"SOURCECRAFT_PYPI_REGISTRY_ID": "registry123"}, "PYPI_REGISTRY_PAT"),
        (
            {"SOURCECRAFT_PYPI_REGISTRY_ID": "registry123", "PYPI_REGISTRY_PAT": " "},
            "PYPI_REGISTRY_PAT",
        ),
        (
            {"SOURCECRAFT_PYPI_REGISTRY_ID": "registry123", "PYPI_REGISTRY_PAT": " t "},
            "whitespace",
        ),
    ],
)
def test_registry_configuration_fails_with_a_clear_message(
    release: ModuleType, environment: dict[str, str], message: str
) -> None:
    with pytest.raises(release.ReleaseError, match=message):
        release.registry_settings(environment)


def test_show_endpoints_does_not_require_the_pat(
    release: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOURCECRAFT_PYPI_REGISTRY_ID", "registry123")
    monkeypatch.delenv("PYPI_REGISTRY_PAT", raising=False)
    assert release.main(["show-endpoints"]) == 0
    output = capsys.readouterr().out
    assert "registry123/legacy/" in output
    assert "registry123/simple/" in output


def test_authenticated_url_encodes_the_pat(release: ModuleType) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "a@b:c/%",
        }
    )
    assert settings.authenticated_index_url == (
        "https://iam:a%40b%3Ac%2F%25@pkg.sourcecraft.tech/"
        "pypi/ooo-ainod/registry123/simple/"
    )
    assert "a@b:c/%" not in settings.authenticated_index_url


def test_redaction_removes_tokens_and_authenticated_urls(release: ModuleType) -> None:
    token = "super-secret-token"
    text = (
        f"password={token} url=https://iam:{token}@pkg.sourcecraft.tech/simple/ "
        f"encoded={token}"
    )
    redacted = release.redact(text, token)
    assert token not in redacted
    assert "iam:***@" in redacted


def make_artifacts(tmp_path: Path, version: str = "0.1.0") -> list[Path]:
    wheel = tmp_path / f"automl_contracts-{version}-py3-none-any.whl"
    sdist = tmp_path / f"automl_contracts-{version}.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    return [wheel, sdist]


def test_artifacts_must_match_the_release_version(
    release: ModuleType, tmp_path: Path
) -> None:
    artifacts = make_artifacts(tmp_path, "0.1.0")
    assert release.validate_artifacts(artifacts, "0.1.0") == tuple(artifacts)
    with pytest.raises(release.ReleaseError, match="carries version"):
        release.validate_artifacts(artifacts, "0.2.0")


def test_publication_requires_a_wheel_and_an_sdist(
    release: ModuleType, tmp_path: Path
) -> None:
    wheel = tmp_path / "automl_contracts-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(release.ReleaseError, match="both wheel and sdist"):
        release.validate_artifacts([wheel], "0.1.0")
    with pytest.raises(release.ReleaseError, match="does not exist"):
        release.validate_artifacts(
            [wheel, tmp_path / "automl_contracts-0.1.0.tar.gz"], "0.1.0"
        )


def test_publication_rejects_unexpected_artifacts(
    release: ModuleType, tmp_path: Path
) -> None:
    artifacts = make_artifacts(tmp_path)
    foreign = tmp_path / "other-0.1.0.tar.gz"
    foreign.write_bytes(b"other")
    with pytest.raises(release.ReleaseError, match="not an automl-contracts"):
        release.validate_artifacts([artifacts[0], foreign], "0.1.0")


def test_registry_has_version_reads_the_simple_index(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {"SOURCECRAFT_PYPI_REGISTRY_ID": "registry123", "PYPI_REGISTRY_PAT": "s3cr3t"}
    )

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'<a href="automl_contracts-0.1.0-py3-none-any.whl">'

    monkeypatch.setattr(
        release.urllib.request,
        "urlopen",
        lambda request, timeout: Response(),
    )
    assert release.registry_has_version(settings, "0.1.0") is True
    assert release.registry_has_version(settings, "0.2.0") is False


def test_publish_rejects_a_version_before_upload(
    release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = make_artifacts(tmp_path)
    settings = release.registry_settings(
        {"SOURCECRAFT_PYPI_REGISTRY_ID": "registry123", "PYPI_REGISTRY_PAT": "s3cr3t"}
    )
    monkeypatch.setattr(release, "registry_has_version", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("twine must not run for a duplicate"),
    )
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.publish(artifacts, settings, "0.1.0")


def test_publish_passes_the_pat_through_the_environment(
    release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = make_artifacts(tmp_path)
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    release.publish(
        artifacts, settings, "0.1.0", python="python3.11", check_existing=False
    )

    command = captured["command"]
    assert command[:4] == ["python3.11", "-m", "twine", "upload"]
    assert "s3cr3t-value" not in " ".join(command)
    assert "--skip-existing" not in command
    assert settings.upload_url in command
    assert captured["env"]["TWINE_USERNAME"] == "iam"
    assert captured["env"]["TWINE_PASSWORD"] == "s3cr3t-value"


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("400 File already exists", "already exists"),
        ("400 Bad Request", "HTTP 400"),
        ("500 Internal Server Error", "HTTP 500"),
        ("401 Unauthorized", "PYPI_REGISTRY_PAT"),
        ("403 Forbidden", "PYPI_REGISTRY_PAT"),
        ("404 Not Found", "SOURCECRAFT_PYPI_REGISTRY_ID"),
    ],
)
def test_upload_failures_are_explained(
    release: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    message: str,
) -> None:
    artifacts = make_artifacts(tmp_path)
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(release.ReleaseError, match=message) as error:
        release.publish(artifacts, settings, "0.1.0", check_existing=False)
    assert "s3cr3t-value" not in str(error.value)


def test_install_uses_the_authenticated_index_without_exposing_the_token(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(
        command: list[str], *, env: dict[str, str] | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, env))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(release, "_run", fake_run)
    release.install_from_registry(
        settings, "0.1.0", python="/tmp/venv/bin/python", with_jsonschema=True
    )

    public_command, public_env = calls[0]
    assert public_env is None
    assert "--index-url" in public_command
    assert "https://pypi.org/simple" in public_command
    assert "jsonschema>=4.18" in public_command

    install_command, install_env = calls[1]
    assert install_env is not None
    assert "s3cr3t-value" not in " ".join(install_command)
    assert "--no-deps" in install_command
    assert "automl-contracts==0.1.0" in install_command
    assert "iam:s3cr3t-value@" in install_env["PIP_INDEX_URL"]
    assert calls[2][0][-2:] == ["pip", "check"]


def test_install_reports_a_missing_version_without_the_token(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    def fake_run(
        command: list[str], *, env: dict[str, str] | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        if "install" in command and "--no-deps" in command:
            return subprocess.CompletedProcess(
                command, 1, "No matching distribution", ""
            )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(release, "_run", fake_run)
    with pytest.raises(release.ReleaseError, match="is not available") as error:
        release.install_from_registry(settings, "9.9.9", python="python")
    assert "s3cr3t-value" not in str(error.value)


def test_release_cli_prints_the_checked_version(
    release: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    assert (
        release.main(
            [
                "check-tag",
                "--tag",
                "contracts-v0.1.0",
                "--pyproject",
                str(pyproject),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_release_cli_fails_on_a_mismatched_tag(
    release: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    assert (
        release.main(
            [
                "check-tag",
                "--tag",
                "v0.1.0",
                "--pyproject",
                str(pyproject),
            ]
        )
        == 1
    )
    assert "publication is not allowed" in capsys.readouterr().err


def test_consumer_validates_the_reference_config(consumer: ModuleType) -> None:
    payload = consumer.load_reference_config(REFERENCE_CONFIG)
    report = consumer.validate_reference_config(payload)
    assert report["model"] == "Config"
    assert report["schema"] == "schemas/config.schema.json"


def test_consumer_rejects_an_invalid_reference_config(consumer: ModuleType) -> None:
    payload = consumer.load_reference_config(REFERENCE_CONFIG)
    payload["unknown_section"] = {}
    with pytest.raises(consumer.ConsumerError, match="reject"):
        consumer.validate_reference_config(payload)


def test_consumer_requires_the_engine_to_be_absent(consumer: ModuleType) -> None:
    require_engine()
    with pytest.raises(consumer.ConsumerError, match="must not need"):
        consumer.ensure_modules_absent(("configurable_automl_engine",))


def test_consumer_rejects_a_version_mismatch(consumer: ModuleType) -> None:
    current = consumer.automl_contracts.__version__
    report = consumer.check_metadata(current)
    assert report["name"] == "automl-contracts"
    assert report["version"] == current
    with pytest.raises(consumer.ConsumerError, match="does not match"):
        consumer.check_metadata("9.9.9")


def test_consumer_checks_package_data_and_public_surface(
    consumer: ModuleType,
) -> None:
    package = consumer.check_package_data()
    assert package["py_typed"] is True
    surface = consumer.check_public_surface()
    assert surface["protocols"] == 2
    assert surface["public_names"] > 0


def test_consumer_rejects_an_import_from_the_source_checkout(
    consumer: ModuleType,
) -> None:
    with pytest.raises(consumer.ConsumerError, match="imported from"):
        consumer.ensure_import_origin(REPO_ROOT)


def test_engine_smoke_covers_every_documented_legacy_path(
    engine_smoke: ModuleType,
) -> None:
    require_engine()
    matrix = engine_smoke.load_import_matrix(IMPORT_MATRIX)
    assert "configurable_automl_engine" in matrix
    checked = engine_smoke.check_import_matrix(matrix)
    assert checked == sum(len(symbols) for symbols in matrix.values())


def test_engine_smoke_checks_the_contract_dependency(
    engine_smoke: ModuleType,
) -> None:
    require_engine()
    current = automl_contracts.__version__
    report = engine_smoke.check_contract_dependency(current)
    assert report["installed"] == current
    assert report["declared"]
    with pytest.raises(engine_smoke.EngineSmokeError, match="does not match"):
        engine_smoke.check_contract_dependency("9.9.9")


def test_engine_smoke_checks_contract_identity(engine_smoke: ModuleType) -> None:
    require_engine()
    assert engine_smoke.check_contract_identity() == 6


def test_engine_smoke_reports_a_missing_symbol(
    engine_smoke: ModuleType,
) -> None:
    require_engine()
    with pytest.raises(engine_smoke.EngineSmokeError, match="disappeared"):
        engine_smoke.check_import_matrix(
            {"configurable_automl_engine": ["definitely_missing_symbol"]}
        )


def test_import_matrix_is_json(engine_smoke: ModuleType) -> None:
    payload = json.loads(IMPORT_MATRIX.read_text(encoding="utf-8"))
    assert payload["paths"]
    assert sys.version_info >= (3, 11)

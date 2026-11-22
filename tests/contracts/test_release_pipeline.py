"""Release tag, registry, publication and consumer checks for the contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        f'[project]\nname = "configurable-automl-contracts"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    return path


def test_pyproject_version_is_read(release: ModuleType, tmp_path: Path) -> None:
    pyproject = write_pyproject(tmp_path, "1.2.3")
    assert release.read_distribution(pyproject) == (
        "configurable-automl-contracts",
        "1.2.3",
    )
    assert release.read_distribution_version(pyproject) == "1.2.3"


def test_pyproject_without_version_is_rejected(
    release: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "configurable-automl-contracts"\n', encoding="utf-8"
    )
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
    wheel = tmp_path / f"configurable_automl_contracts-{version}-py3-none-any.whl"
    sdist = tmp_path / f"configurable_automl_contracts-{version}.tar.gz"
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


def test_artifact_names_use_the_new_distribution(
    release: ModuleType, tmp_path: Path
) -> None:
    artifacts = make_artifacts(tmp_path)
    assert [release.artifact_version(path) for path in artifacts] == [
        "0.1.0",
        "0.1.0",
    ]
    old = tmp_path / "automl_contracts-0.1.0-py3-none-any.whl"
    with pytest.raises(release.ReleaseError, match="not a configurable"):
        release.artifact_version(old)


def test_publication_requires_a_wheel_and_an_sdist(
    release: ModuleType, tmp_path: Path
) -> None:
    wheel = tmp_path / "configurable_automl_contracts-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(release.ReleaseError, match="both wheel and sdist"):
        release.validate_artifacts([wheel], "0.1.0")
    with pytest.raises(release.ReleaseError, match="does not exist"):
        release.validate_artifacts(
            [wheel, tmp_path / "configurable_automl_contracts-0.1.0.tar.gz"], "0.1.0"
        )


def test_publication_rejects_unexpected_artifacts(
    release: ModuleType, tmp_path: Path
) -> None:
    artifacts = make_artifacts(tmp_path)
    foreign = tmp_path / "other-0.1.0.tar.gz"
    foreign.write_bytes(b"other")
    with pytest.raises(
        release.ReleaseError, match="not a configurable-automl-contracts"
    ):
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
            return b'<a href="configurable_automl_contracts-0.1.0-py3-none-any.whl">'

    monkeypatch.setattr(
        release,
        "_open_registry_url",
        lambda settings_arg, url, *, timeout, auth_origin=None: Response(),
    )
    assert release.registry_has_version(settings, "0.1.0") is True
    assert release.registry_has_version(settings, "0.2.0") is False


def test_registry_has_version_ignores_the_old_distribution(
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
        release,
        "_open_registry_url",
        lambda settings_arg, url, *, timeout, auth_origin=None: Response(),
    )
    assert release.registry_has_version(settings, "0.1.0") is False


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
    assert "configurable-automl-contracts==0.1.0" in install_command
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


SIMPLE_INDEX_PAGE = (
    "<html><body><h1>Links for configurable-automl-contracts</h1>"
    '<a href="./0.1.0/-/configurable_automl_contracts-0.1.0-py3-none-any.whl#sha256=aa11" '
    'data-core-metadata="sha256=bb22">configurable_automl_contracts-0.1.0-py3-none-any.whl</a><br>'
    '<a href="./0.0.9/-/configurable_automl_contracts-0.0.9-py3-none-any.whl#sha256=cc33">'
    "configurable_automl_contracts-0.0.9-py3-none-any.whl</a><br>"
    '<a href="./0.1.0/-/configurable_automl_contracts-0.1.0.tar.gz#sha256=dd44">'
    "configurable_automl_contracts-0.1.0.tar.gz</a><br></body></html>"
)
SIMPLE_INDEX_BASE = "https://pkg.sourcecraft.tech/pypi/ooo-ainod/registry123/simple/configurable-automl-contracts/"
MISSING_METADATA_OUTPUT = (
    "Collecting configurable-automl-contracts==0.1.0\n"
    "  ERROR: HTTP error 404 while getting "
    "https://pkg.sourcecraft.tech/pypi/ooo-ainod/registry123/simple/"
    "configurable-automl-contracts/0.1.0/-/configurable_automl_contracts-0.1.0-py3-none-any.whl.metadata\n"
    "ERROR: 404 Client Error: Not Found"
)


def test_parse_index_artifacts_extracts_links_and_hashes(release: ModuleType) -> None:
    artifacts = release.parse_index_artifacts(SIMPLE_INDEX_PAGE, SIMPLE_INDEX_BASE)

    assert [artifact.filename for artifact in artifacts] == [
        "configurable_automl_contracts-0.1.0-py3-none-any.whl",
        "configurable_automl_contracts-0.0.9-py3-none-any.whl",
        "configurable_automl_contracts-0.1.0.tar.gz",
    ]
    wheel = artifacts[0]
    assert wheel.url == (
        "https://pkg.sourcecraft.tech/pypi/ooo-ainod/registry123/simple/"
        "configurable-automl-contracts/0.1.0/-/configurable_automl_contracts-0.1.0-py3-none-any.whl"
    )
    assert wheel.sha256 == "aa11"


def test_select_artifact_prefers_the_wheel_of_the_requested_version(
    release: ModuleType,
) -> None:
    artifacts = release.parse_index_artifacts(SIMPLE_INDEX_PAGE, SIMPLE_INDEX_BASE)

    selected = release.select_artifact(artifacts, "0.1.0")
    assert selected.filename == "configurable_automl_contracts-0.1.0-py3-none-any.whl"

    sdist_only = tuple(
        artifact for artifact in artifacts if artifact.filename.endswith(".tar.gz")
    )
    assert release.select_artifact(sdist_only, "0.1.0").filename.endswith(".tar.gz")
    with pytest.raises(release.ReleaseError, match="does not list an artifact"):
        release.select_artifact(artifacts, "9.9.9")


def test_parse_index_artifacts_rejects_encoded_path_traversal(
    release: ModuleType,
) -> None:
    page = (
        '<a href="./0.1.0/-/configurable_automl_contracts-0.1.0-'
        '%2e%2e%2f%2e%2e%2f%2e%2e%2ftmp%2fpwned.whl#sha256=aa11">traversal</a>'
    )
    assert release.parse_index_artifacts(page, SIMPLE_INDEX_BASE) == ()


def test_select_artifact_prefers_a_universal_wheel_among_several(
    release: ModuleType,
) -> None:
    page = (
        '<a href="./0.1.0/-/configurable_automl_contracts-0.1.0-'
        'cp311-cp311-manylinux_2_17_x86_64.whl#sha256=aa11">platform</a>'
        '<a href="./0.1.0/-/configurable_automl_contracts-0.1.0-py3-none-any.whl#sha256=bb22">'
        "universal</a>"
    )
    artifacts = release.parse_index_artifacts(page, SIMPLE_INDEX_BASE)

    selected = release.select_artifact(artifacts, "0.1.0")
    assert selected.filename == "configurable_automl_contracts-0.1.0-py3-none-any.whl"


def test_select_artifact_rejects_ambiguous_wheels(release: ModuleType) -> None:
    page = (
        '<a href="./0.1.0/-/configurable_automl_contracts-0.1.0-'
        'cp311-cp311-manylinux_2_17_x86_64.whl#sha256=aa11">cp311</a>'
        '<a href="./0.1.0/-/configurable_automl_contracts-0.1.0-'
        'cp312-cp312-manylinux_2_17_x86_64.whl#sha256=bb22">cp312</a>'
    )
    artifacts = release.parse_index_artifacts(page, SIMPLE_INDEX_BASE)

    with pytest.raises(release.ReleaseError, match="several wheels"):
        release.select_artifact(artifacts, "0.1.0")


def test_download_artifact_verifies_the_advertised_hash(
    release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"payload"

    monkeypatch.setattr(
        release,
        "_open_registry_url",
        lambda settings_arg, url, *, timeout, auth_origin=None: Response(),
    )
    matching = release.IndexArtifact(
        filename="configurable_automl_contracts-0.1.0-py3-none-any.whl",
        url="https://pkg.example/configurable_automl_contracts-0.1.0-py3-none-any.whl",
        sha256="239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
    )
    destination = release.download_artifact(
        settings,
        matching,
        tmp_path / matching.filename,
        registry_url=matching.url,
    )
    assert destination.read_bytes() == b"payload"

    mismatched = release.IndexArtifact(
        filename=matching.filename, url=matching.url, sha256="0" * 64
    )
    with pytest.raises(release.ReleaseError, match="advertises") as error:
        release.download_artifact(
            settings,
            mismatched,
            tmp_path / "other.whl",
            registry_url=mismatched.url,
        )
    assert "s3cr3t-value" not in str(error.value)


def test_download_artifact_wraps_network_errors(
    release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    def raise_timeout(
        settings_arg: Any, url: str, *, timeout: int, auth_origin: str | None = None
    ) -> Any:
        raise TimeoutError("timed out")

    monkeypatch.setattr(release, "_open_registry_url", raise_timeout)
    artifact = release.IndexArtifact(
        filename="configurable_automl_contracts-0.1.0-py3-none-any.whl",
        url="https://pkg.example/configurable_automl_contracts-0.1.0-py3-none-any.whl",
    )
    with pytest.raises(
        release.ReleaseError, match="cannot reach the registry artifact"
    ) as error:
        release.download_artifact(
            settings,
            artifact,
            tmp_path / artifact.filename,
            registry_url=artifact.url,
        )
    assert "s3cr3t-value" not in str(error.value)


def test_download_artifact_rejects_a_missing_hash(
    release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"payload"

    monkeypatch.setattr(
        release,
        "_open_registry_url",
        lambda settings_arg, url, *, timeout, auth_origin=None: Response(),
    )
    artifact = release.IndexArtifact(
        filename="configurable_automl_contracts-0.1.0-py3-none-any.whl",
        url="https://pkg.example/configurable_automl_contracts-0.1.0-py3-none-any.whl",
        sha256=None,
    )
    destination = tmp_path / artifact.filename
    with pytest.raises(
        release.ReleaseError, match="does not advertise a sha256"
    ) as error:
        release.download_artifact(
            settings,
            artifact,
            destination,
            registry_url=artifact.url,
        )
    assert "s3cr3t-value" not in str(error.value)
    assert not destination.exists()


def test_download_artifact_wraps_truncated_responses(
    release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            raise release.http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(
        release,
        "_open_registry_url",
        lambda settings_arg, url, *, timeout, auth_origin=None: Response(),
    )
    artifact = release.IndexArtifact(
        filename="configurable_automl_contracts-0.1.0-py3-none-any.whl",
        url="https://pkg.example/configurable_automl_contracts-0.1.0-py3-none-any.whl",
        sha256="0" * 64,
    )
    with pytest.raises(
        release.ReleaseError, match="cannot reach the registry artifact"
    ) as error:
        release.download_artifact(
            settings,
            artifact,
            tmp_path / artifact.filename,
            registry_url=artifact.url,
        )
    assert "s3cr3t-value" not in str(error.value)


def test_redirect_handler_does_not_forward_credentials_to_another_origin(
    release: ModuleType,
) -> None:
    handler = release._CredentialSafeRedirectHandler()
    request = release.urllib.request.Request(
        "https://pkg.sourcecraft.tech/pypi/o/r/simple/configurable_automl_contracts-0.1.0.whl",
        headers={"Authorization": "Basic s3cr3t-value"},
    )

    redirected = handler.redirect_request(
        request,
        None,
        307,
        "Temporary Redirect",
        None,
        "https://storage.yandexcloud.net/artifact-registry/configurable_automl_contracts-0.1.0.whl",
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None
    assert "s3cr3t-value" not in "".join(redirected.headers.values())

    same_origin = handler.redirect_request(
        request,
        None,
        307,
        "Temporary Redirect",
        None,
        "https://pkg.sourcecraft.tech/pypi/o/r/simple/configurable_automl_contracts-0.1.0.whl",
    )
    assert same_origin is not None
    assert same_origin.get_header("Authorization") == "Basic s3cr3t-value"


def test_download_artifact_does_not_send_the_pat_to_a_redirect_host(
    release: ModuleType, tmp_path: Path
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )
    payload = b"wheel-bytes"
    received: dict[str, str | None] = {}

    class TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received["authorization"] = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_url = f"http://127.0.0.1:{target.server_port}/configurable_automl_contracts-0.1.0-py3-none-any.whl"

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(307)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            return None

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    redirect_url = (
        f"http://127.0.0.1:{redirect.server_port}/"
        "configurable_automl_contracts-0.1.0-py3-none-any.whl"
    )
    servers = (redirect, target)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True) for server in servers
    ]
    for thread in threads:
        thread.start()
    try:
        artifact = release.IndexArtifact(
            filename="configurable_automl_contracts-0.1.0-py3-none-any.whl",
            url=redirect_url,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        destination = release.download_artifact(
            settings,
            artifact,
            tmp_path / artifact.filename,
            registry_url=redirect_url,
        )
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()

    assert destination.read_bytes() == payload
    assert received["authorization"] is None


def test_artifact_index_install_does_not_authenticate_external_links(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )
    payload = b"wheel-bytes"
    received: dict[str, str | None] = {}

    class ArtifactHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received["authorization"] = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            return None

    artifact_server = ThreadingHTTPServer(("127.0.0.1", 0), ArtifactHandler)
    digest = hashlib.sha256(payload).hexdigest()
    artifact_url = (
        f"http://127.0.0.1:{artifact_server.server_port}/"
        f"configurable_automl_contracts-0.1.0-py3-none-any.whl#sha256={digest}"
    )
    page = (
        "<html><body>"
        f'<a href="{artifact_url}">configurable_automl_contracts-0.1.0-py3-none-any.whl</a>'
        "</body></html>"
    ).encode()

    class IndexHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *args: object) -> None:
            return None

    index_server = ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler)
    index_url = f"http://127.0.0.1:{index_server.server_port}/simple/"
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, env: dict[str, str] | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    servers = (index_server, artifact_server)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True) for server in servers
    ]
    for thread in threads:
        thread.start()
    try:
        monkeypatch.setattr(release, "_run", fake_run)
        release.install_artifact_from_index(
            settings,
            "0.1.0",
            python="python",
            index_url=index_url,
        )
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()

    assert received["authorization"] is None
    assert commands[0][-1].endswith(
        "configurable_automl_contracts-0.1.0-py3-none-any.whl"
    )


def test_artifact_index_install_wraps_index_http_errors(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    def raise_unauthorized(settings_arg: Any, url: str, *, timeout: int = 30) -> str:
        raise release.urllib.error.HTTPError(url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(release, "_read_registry_page", raise_unauthorized)
    with pytest.raises(release.ReleaseError, match="PYPI_REGISTRY_PAT") as error:
        release.install_artifact_from_index(settings, "0.1.0", python="python")
    assert "s3cr3t-value" not in str(error.value)


def test_artifact_index_install_wraps_index_network_errors(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )

    def raise_timeout(settings_arg: Any, url: str, *, timeout: int = 30) -> str:
        raise TimeoutError("timed out")

    monkeypatch.setattr(release, "_read_registry_page", raise_timeout)
    with pytest.raises(
        release.ReleaseError, match="cannot reach the registry index"
    ) as error:
        release.install_artifact_from_index(settings, "0.1.0", python="python")
    assert "s3cr3t-value" not in str(error.value)


def test_install_falls_back_to_the_index_artifact_when_metadata_is_missing(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )
    commands: list[list[str]] = []
    fallback: dict[str, Any] = {}

    def fake_run(
        command: list[str], *, env: dict[str, str] | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "configurable-automl-contracts==0.1.0" in command:
            return subprocess.CompletedProcess(command, 1, "", MISSING_METADATA_OUTPUT)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def fake_artifact_install(
        passed_settings: Any,
        version: str,
        *,
        python: str,
        index_url: str | None,
        timeout: int,
    ) -> None:
        fallback.update(
            {
                "settings": passed_settings,
                "version": version,
                "python": python,
                "index_url": index_url,
            }
        )

    monkeypatch.setattr(release, "_run", fake_run)
    monkeypatch.setattr(release, "install_artifact_from_index", fake_artifact_install)
    release.install_from_registry(settings, "0.1.0", python="/tmp/venv/bin/python")

    assert fallback["version"] == "0.1.0"
    assert fallback["python"] == "/tmp/venv/bin/python"
    assert commands[-1][-2:] == ["pip", "check"]


def test_install_reports_both_failures_when_the_fallback_fails(
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
        if "configurable-automl-contracts==0.9.0" in command:
            return subprocess.CompletedProcess(
                command, 1, "", MISSING_METADATA_OUTPUT.replace("0.1.0", "0.9.0")
            )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def failing_artifact_install(*args: Any, **kwargs: Any) -> None:
        raise release.ReleaseError("the registry did not serve the wheel")

    monkeypatch.setattr(release, "_run", fake_run)
    monkeypatch.setattr(
        release, "install_artifact_from_index", failing_artifact_install
    )
    with pytest.raises(
        release.ReleaseError, match="artifact link also failed"
    ) as error:
        release.install_from_registry(settings, "0.9.0", python="python")

    message = str(error.value)
    assert "configurable-automl-contracts==0.9.0" in message
    assert "the registry did not serve the wheel" in message
    assert "s3cr3t-value" not in message


def test_install_wraps_os_errors_from_the_fallback(
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
        if "configurable-automl-contracts==0.1.0" in command:
            return subprocess.CompletedProcess(command, 1, "", MISSING_METADATA_OUTPUT)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def failing_artifact_install(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError("connection timed out")

    monkeypatch.setattr(release, "_run", fake_run)
    monkeypatch.setattr(
        release, "install_artifact_from_index", failing_artifact_install
    )
    with pytest.raises(
        release.ReleaseError, match="artifact link also failed"
    ) as error:
        release.install_from_registry(settings, "0.1.0", python="python")

    message = str(error.value)
    assert "connection timed out" in message
    assert "s3cr3t-value" not in message


def test_install_does_not_fall_back_for_unrelated_failures(
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
        if "configurable-automl-contracts==0.1.0" in command:
            return subprocess.CompletedProcess(command, 1, "", "504 Gateway Timeout")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(release, "_run", fake_run)
    monkeypatch.setattr(
        release,
        "install_artifact_from_index",
        lambda *args, **kwargs: pytest.fail("the fallback must not run"),
    )
    with pytest.raises(release.ReleaseError, match="504"):
        release.install_from_registry(settings, "0.1.0", python="python")


def test_artifact_index_install_downloads_and_installs_the_wheel(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release.registry_settings(
        {
            "SOURCECRAFT_PYPI_REGISTRY_ID": "registry123",
            "PYPI_REGISTRY_PAT": "s3cr3t-value",
        }
    )
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, env: dict[str, str] | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def fake_download(
        passed_settings: Any,
        artifact: Any,
        destination: Path,
        *,
        registry_url: str,
        timeout: int,
    ) -> Path:
        destination.write_bytes(b"wheel")
        return destination

    monkeypatch.setattr(
        release,
        "_read_registry_page",
        lambda settings_arg, url, *, timeout=30: SIMPLE_INDEX_PAGE,
    )
    monkeypatch.setattr(release, "download_artifact", fake_download)
    monkeypatch.setattr(release, "_run", fake_run)
    release.install_artifact_from_index(settings, "0.1.0", python="python")

    command = commands[0]
    assert command[:6] == [
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-cache-dir",
    ]
    assert command[-1].endswith("configurable_automl_contracts-0.1.0-py3-none-any.whl")
    assert "s3cr3t-value" not in " ".join(command)


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
    assert report["name"] == "configurable-automl-contracts"
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


def test_invalid_static_version_is_reported_as_a_release_error(
    release: ModuleType, tmp_path: Path
) -> None:
    pyproject = write_pyproject(tmp_path, "not-a-version")
    with pytest.raises(release.ReleaseError, match="invalid version") as error:
        release.verify_release_tag("contracts-v0.1.0", pyproject)
    assert str(pyproject) in str(error.value)
    assert "not-a-version" in str(error.value)


def test_release_cli_reports_an_invalid_static_version_without_a_traceback(
    release: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = write_pyproject(tmp_path, "not-a-version")
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
        == 1
    )
    error = capsys.readouterr().err
    assert "invalid version" in error
    assert "Traceback" not in error


@pytest.mark.parametrize(
    ("ref", "tag"),
    [
        ("refs/tags/contracts-v0.1.0", "contracts-v0.1.0"),
        ("refs/tags/contracts-v10.20.30", "contracts-v10.20.30"),
    ],
)
def test_full_tag_ref_is_parsed(release: ModuleType, ref: str, tag: str) -> None:
    assert release.tag_from_ref(ref) == tag


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "   ",
        "contracts-v0.1.0",
        "refs/heads/contracts-v0.1.0",
        "refs/tags/v0.1.0",
        "refs/tags/contracts-v0.1",
        "refs/tags/contracts-v0.2.0-rc1",
        "refs/tags/contracts-v0.1.0+local",
        " refs/tags/contracts-v0.1.0",
    ],
)
def test_incomplete_or_non_tag_refs_are_rejected(release: ModuleType, ref: str) -> None:
    with pytest.raises(release.ReleaseError):
        release.tag_from_ref(ref)


def test_full_tag_ref_on_head_returns_the_version(
    release: ModuleType, tmp_path: Path
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    assert (
        release.verify_release_ref(
            "refs/tags/contracts-v0.1.0",
            "contracts-v0.1.0",
            pyproject,
            head_tags=["contracts-v0.1.0"],
        )
        == "0.1.0"
    )


def test_branch_with_a_release_name_is_rejected(
    release: ModuleType, tmp_path: Path
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    with pytest.raises(release.ReleaseError, match="full tag ref"):
        release.verify_release_ref(
            "refs/heads/contracts-v0.1.0",
            "contracts-v0.1.0",
            pyproject,
            head_tags=["contracts-v0.1.0"],
        )


def test_empty_ref_is_rejected(release: ModuleType, tmp_path: Path) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    with pytest.raises(release.ReleaseError, match="empty"):
        release.verify_release_ref(
            "",
            "contracts-v0.1.0",
            pyproject,
            head_tags=["contracts-v0.1.0"],
        )


def test_tag_that_does_not_point_at_head_is_rejected(
    release: ModuleType, tmp_path: Path
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    with pytest.raises(release.ReleaseError, match="does not point at HEAD"):
        release.verify_release_ref(
            "refs/tags/contracts-v0.1.0",
            "contracts-v0.1.0",
            pyproject,
            head_tags=["contracts-v0.2.0"],
        )


def test_ref_and_short_tag_must_agree(release: ModuleType, tmp_path: Path) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    with pytest.raises(release.ReleaseError, match="does not match the tag ref"):
        release.verify_release_ref(
            "refs/tags/contracts-v0.1.0",
            "contracts-v0.2.0",
            pyproject,
            head_tags=["contracts-v0.1.0"],
        )


def test_tags_pointing_at_head_reads_git(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        command: list[str], *, env: dict[str, str] | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(
            command, 0, "contracts-v0.1.0\ncontracts-v0.2.0\n", ""
        )

    monkeypatch.setattr(release, "_run", fake_run)
    assert release.tags_pointing_at_head() == ["contracts-v0.1.0", "contracts-v0.2.0"]
    assert captured["command"][:3] == ["git", "tag", "--points-at"]


def test_tags_pointing_at_head_reports_a_git_failure(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        command: list[str], *, env: dict[str, str] | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 128, "", "fatal: not a git repo")

    monkeypatch.setattr(release, "_run", fake_run)
    with pytest.raises(release.ReleaseError, match="cannot list tags"):
        release.tags_pointing_at_head()


@pytest.mark.parametrize(
    ("ref", "tag", "head_tags", "exit_code"),
    [
        (
            "refs/tags/contracts-v0.1.0",
            "contracts-v0.1.0",
            ["contracts-v0.1.0"],
            0,
        ),
        (
            "refs/heads/contracts-v0.1.0",
            "contracts-v0.1.0",
            ["contracts-v0.1.0"],
            1,
        ),
        ("", "contracts-v0.1.0", ["contracts-v0.1.0"], 1),
        (
            "refs/tags/contracts-v0.1.0",
            "contracts-v0.1.0",
            ["contracts-v0.2.0"],
            1,
        ),
    ],
)
def test_release_cli_enforces_the_full_tag_ref(
    release: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    ref: str,
    tag: str,
    head_tags: list[str],
    exit_code: int,
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    monkeypatch.setattr(release, "tags_pointing_at_head", lambda: head_tags)
    assert (
        release.main(
            [
                "check-tag",
                "--ref",
                ref,
                "--tag",
                tag,
                "--pyproject",
                str(pyproject),
            ]
        )
        == exit_code
    )
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_invalid_expected_version_is_rejected_before_upload(
    release: ModuleType, tmp_path: Path
) -> None:
    artifacts = make_artifacts(tmp_path)
    with pytest.raises(release.ReleaseError, match="not a valid PEP 440"):
        release.validate_artifacts(artifacts, "not-a-version")


def test_release_cli_reports_an_invalid_expected_version_without_a_traceback(
    release: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_artifacts(tmp_path)
    monkeypatch.setenv("SOURCECRAFT_PYPI_REGISTRY_ID", "registry123")
    monkeypatch.setenv("PYPI_REGISTRY_PAT", "s3cr3t")
    assert (
        release.main(
            [
                "publish",
                "--dist",
                str(tmp_path),
                "--expected-version",
                "not-a-version",
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "not a valid PEP 440" in error
    assert "Traceback" not in error


def test_missing_local_tags_at_head_are_reported(
    release: ModuleType, tmp_path: Path
) -> None:
    pyproject = write_pyproject(tmp_path, "0.1.0")
    with pytest.raises(release.ReleaseError, match="no local tags point at HEAD"):
        release.verify_release_ref(
            "refs/tags/contracts-v0.1.0",
            "contracts-v0.1.0",
            pyproject,
            head_tags=[],
        )

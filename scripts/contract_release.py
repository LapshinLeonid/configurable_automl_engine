"""Release helpers for the ``automl-contracts`` SourceCraft PyPI registry.

The module keeps the release rules in one testable place: the ``contracts-v*``
tag format, the version synchronization check, the registry endpoint and the
``twine``/``pip`` invocations.  Credentials are read from the environment and
are never written to the command line or to the output.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from packaging.version import InvalidVersion, Version

DISTRIBUTION_NAME = "automl-contracts"
TAG_PREFIX = "contracts-v"
TAG_REF_PREFIX = "refs/tags/"
DEFAULT_ORGANIZATION = "ooo-ainod"
REGISTRY_ID_ENV = "SOURCECRAFT_PYPI_REGISTRY_ID"
PAT_ENV = "PYPI_REGISTRY_PAT"
UPLOAD_KIND = "legacy"
INDEX_KIND = "simple"
PUBLIC_INDEX_URL = "https://pypi.org/simple"
ENDPOINT_TEMPLATE = (
    "https://pkg.sourcecraft.tech/pypi/{organization}/{registry_id}/{kind}/"
)
TWINE_USERNAME = "iam"

_SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)
_REGISTRY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ARTIFACT = re.compile(
    r"^automl[_-]contracts-(?P<version>[^-]+)(?:-.+)?\.(?:whl|tar\.gz)$"
)
_AUTHENTICATED_URL = re.compile(r"(iam:)[^@\s]+@")


class ReleaseError(RuntimeError):
    """A release rule is violated or a registry operation failed."""


@dataclass(frozen=True)
class RegistrySettings:
    """Resolved registry configuration.

    Args:
        registry_id: Non-secret identifier of the SourceCraft PyPI registry.
        token: SourceCraft personal access token used as the ``iam`` password.
        organization: SourceCraft organization slug.
    """

    registry_id: str
    token: str = field(repr=False)
    organization: str = DEFAULT_ORGANIZATION

    def endpoint(self, kind: str) -> str:
        """Return the registry endpoint for ``kind`` (``legacy`` or ``simple``)."""
        return ENDPOINT_TEMPLATE.format(
            organization=self.organization,
            registry_id=self.registry_id,
            kind=kind,
        )

    @property
    def upload_url(self) -> str:
        """Return the ``twine`` upload endpoint."""
        return self.endpoint(UPLOAD_KIND)

    @property
    def index_url(self) -> str:
        """Return the ``pip`` installation index without credentials."""
        return self.endpoint(INDEX_KIND)

    @property
    def authenticated_index_url(self) -> str:
        """Return the installation index with a quoted PAT for pip."""
        return authenticated_url(self.index_url, self.token)


def read_distribution(pyproject: Path) -> tuple[str, str]:
    """Read the distribution name and version from ``pyproject.toml``.

    Args:
        pyproject: Path to the contract ``pyproject.toml``.

    Returns:
        The distribution name and its static version.

    Raises:
        ReleaseError: The file is unreadable or lacks the required fields.
    """
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"cannot read {pyproject}: {error}") from error
    project = document.get("project")
    if not isinstance(project, dict):
        raise ReleaseError(f"{pyproject} has no [project] table")
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ReleaseError(
            f"{pyproject} must declare a static project name and version"
        )
    return name, version


def read_distribution_version(pyproject: Path) -> str:
    """Return the static contract version from ``pyproject.toml``.

    Args:
        pyproject: Path to the contract ``pyproject.toml``.

    Returns:
        The static distribution version.

    Raises:
        ReleaseError: The file is unreadable or has no static version.
    """
    _, version = read_distribution(pyproject)
    return version


def parse_release_tag(tag: str) -> str:
    """Extract the version from a ``contracts-v<version>`` tag.

    Args:
        tag: Tag name that triggered the publication.

    Returns:
        The version encoded in the tag.

    Raises:
        ReleaseError: The tag does not match the release mask or is not SemVer.
    """
    if not tag or tag != tag.strip():
        raise ReleaseError(f"release tag {tag!r} is empty or contains whitespace")
    if not tag.startswith(TAG_PREFIX):
        raise ReleaseError(
            f"tag {tag!r} does not match the {TAG_PREFIX}<version> mask; "
            "publication is not allowed"
        )
    version = tag[len(TAG_PREFIX) :]
    if _SEMVER.fullmatch(version) is None:
        raise ReleaseError(
            f"tag {tag!r} does not encode a MAJOR.MINOR.PATCH version; "
            "publication is not allowed"
        )
    try:
        Version(version)
    except InvalidVersion as error:
        raise ReleaseError(
            f"tag {tag!r} contains invalid version {version!r}"
        ) from error
    return version


def verify_release_tag(tag: str, pyproject: Path) -> str:
    """Check that the release tag matches the built distribution version.

    Args:
        tag: Tag name that triggered the publication.
        pyproject: Path to ``automl-contracts/pyproject.toml``.

    Returns:
        The distribution version that must be published.

    Raises:
        ReleaseError: The tag mask is wrong, the distribution is not
            ``automl-contracts``, the static version is invalid, or the
            versions differ.
    """
    name, version = read_distribution(pyproject)
    if name != DISTRIBUTION_NAME:
        raise ReleaseError(f"{pyproject} declares distribution {name!r}")
    tag_version = parse_release_tag(tag)
    try:
        distribution_version = Version(version)
    except InvalidVersion as error:
        raise ReleaseError(
            f"{pyproject} declares invalid version {version!r}; publication is "
            "not allowed"
        ) from error
    if Version(tag_version) != distribution_version:
        raise ReleaseError(
            f"tag {tag!r} resolves to {tag_version}, but {pyproject} declares "
            f"version {version}; publication is not allowed"
        )
    return version


def tag_from_ref(ref: str) -> str:
    """Extract the release tag from a full ``refs/tags/`` reference.

    Args:
        ref: Full Git reference that triggered the publication.

    Returns:
        The validated ``contracts-v<version>`` tag.

    Raises:
        ReleaseError: The reference is empty, is not a tag reference, or does
            not encode a valid release tag.
    """
    if not ref or ref != ref.strip():
        raise ReleaseError(
            f"release ref {ref!r} is empty or contains whitespace; publication "
            f"requires {TAG_REF_PREFIX}{TAG_PREFIX}<version>"
        )
    if not ref.startswith(TAG_REF_PREFIX):
        raise ReleaseError(
            f"ref {ref!r} is not a full tag ref; publication requires "
            f"{TAG_REF_PREFIX}{TAG_PREFIX}<version>"
        )
    tag = ref[len(TAG_REF_PREFIX) :]
    parse_release_tag(tag)
    return tag


def tags_pointing_at_head(*, timeout: int = 30) -> list[str]:
    """Return every Git tag that points at ``HEAD``.

    Args:
        timeout: Git invocation timeout in seconds.

    Returns:
        Tag names reported by ``git tag --points-at HEAD``.

    Raises:
        ReleaseError: Git cannot be invoked or the repository is unavailable.
    """
    result = _run(["git", "tag", "--points-at", "HEAD"], env=None, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ReleaseError(
            "cannot list tags pointing at HEAD" + (f": {detail}" if detail else "")
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def verify_release_ref(
    ref: str,
    tag: str,
    pyproject: Path,
    *,
    head_tags: Sequence[str] | None = None,
) -> str:
    """Require a full tag ref that points at ``HEAD`` and matches the version.

    Args:
        ref: Full Git reference from the CI environment.
        tag: Short tag name reported by the CI environment.
        pyproject: Path to ``automl-contracts/pyproject.toml``.
        head_tags: Tags pointing at ``HEAD``; discovered with Git when omitted.

    Returns:
        The distribution version that must be published.

    Raises:
        ReleaseError: The ref is not a full release tag, the short tag differs,
            the tag is not on ``HEAD``, or the tag and version disagree.
    """
    ref_tag = tag_from_ref(ref)
    if tag != ref_tag:
        raise ReleaseError(
            f"tag {tag!r} does not match the tag ref {ref!r}; publication is "
            "not allowed"
        )
    version = verify_release_tag(tag, pyproject)
    tags = list(head_tags) if head_tags is not None else tags_pointing_at_head()
    if not tags:
        raise ReleaseError(
            "no local tags point at HEAD; fetch the release tag ref in the CI "
            "checkout before publishing"
        )
    if tag not in tags:
        raise ReleaseError(
            f"tag {tag!r} does not point at HEAD; local tags at HEAD are "
            f"{sorted(tags)!r}"
        )
    return version


def authenticated_url(url: str, token: str) -> str:
    """Embed an URL-encoded token into an HTTPS index URL.

    Args:
        url: Registry index URL without credentials.
        token: PAT that pip must send as the ``iam`` password.

    Returns:
        The URL with an encoded basic-auth userinfo component.
    """
    encoded = quote(token, safe="")
    return url.replace("https://", f"https://{TWINE_USERNAME}:{encoded}@", 1)


def registry_id_from_environment(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read and validate the non-secret registry identifier.

    Args:
        environ: Environment mapping; defaults to ``os.environ``.

    Returns:
        The validated registry identifier.

    Raises:
        ReleaseError: The identifier is missing or malformed.
    """
    values = os.environ if environ is None else environ
    registry_id = values.get(REGISTRY_ID_ENV, "").strip()
    if not registry_id:
        raise ReleaseError(
            f"{REGISTRY_ID_ENV} is not configured. Set the registry identifier "
            "from the SourceCraft registry page as a CI variable."
        )
    if _REGISTRY_ID.fullmatch(registry_id) is None:
        raise ReleaseError(
            f"{REGISTRY_ID_ENV} contains an invalid registry identifier; "
            "use the value from the SourceCraft registry page."
        )
    return registry_id


def registry_settings(
    environ: Mapping[str, str] | None = None,
    *,
    organization: str = DEFAULT_ORGANIZATION,
) -> RegistrySettings:
    """Build registry settings from the CI environment.

    Args:
        environ: Environment mapping; defaults to ``os.environ``.
        organization: SourceCraft organization slug.

    Returns:
        The validated registry settings.

    Raises:
        ReleaseError: The registry identifier or the PAT is missing or malformed.
    """
    values = os.environ if environ is None else environ
    registry_id = registry_id_from_environment(values)
    token = values.get(PAT_ENV, "")
    if not token.strip():
        raise ReleaseError(
            f"{PAT_ENV} is not configured. Add the SourceCraft PAT as a CI "
            "secret with publish rights."
        )
    if token != token.strip():
        raise ReleaseError(
            f"{PAT_ENV} contains leading or trailing whitespace; copy the token "
            "again without surrounding spaces"
        )
    return RegistrySettings(
        registry_id=registry_id,
        token=token,
        organization=organization,
    )


def redact(value: str, token: str) -> str:
    """Remove a token and authenticated URLs from diagnostic output.

    Args:
        value: Text that may contain the token.
        token: Secret that must never be printed.

    Returns:
        The text with the token and authenticated URLs masked.
    """
    redacted = value.replace(token, "***") if token else value
    encoded = quote(token, safe="")
    if encoded and encoded != token:
        redacted = redacted.replace(encoded, "***")
    return _AUTHENTICATED_URL.sub(r"\1***@", redacted)


def artifact_version(path: Path) -> str:
    """Return the distribution version encoded in an artifact file name.

    Args:
        path: Wheel or source distribution path.

    Returns:
        The version component of the file name.

    Raises:
        ReleaseError: The file is not an ``automl-contracts`` artifact.
    """
    match = _ARTIFACT.fullmatch(path.name)
    if match is None:
        raise ReleaseError(
            f"{path.name} is not an automl-contracts wheel or source distribution"
        )
    return match.group("version")


def validate_artifacts(
    artifacts: Sequence[Path],
    expected_version: str,
) -> tuple[Path, ...]:
    """Validate the files that are about to be uploaded.

    Args:
        artifacts: Built wheel and sdist paths.
        expected_version: Version taken from ``pyproject.toml``.

    Returns:
        The validated artifact paths.

    Raises:
        ReleaseError: An artifact is missing, duplicated or carries another
            version, or the expected version is not valid PEP 440.
    """
    if not artifacts:
        raise ReleaseError("no distribution artifacts were provided for publication")
    try:
        expected = Version(expected_version)
    except InvalidVersion as error:
        raise ReleaseError(
            f"release version {expected_version!r} is not a valid PEP 440 "
            "version; publication is not allowed"
        ) from error
    suffixes: set[str] = set()
    checked: list[Path] = []
    for path in artifacts:
        if not path.is_file():
            raise ReleaseError(f"artifact {path} does not exist")
        version = artifact_version(path)
        try:
            artifact = Version(version)
        except InvalidVersion as error:
            raise ReleaseError(
                f"artifact {path.name} carries invalid version {version!r}"
            ) from error
        if artifact != expected:
            raise ReleaseError(
                f"artifact {path.name} carries version {version}, but the "
                f"release version is {expected_version}"
            )
        suffix = ".whl" if path.suffix == ".whl" else ".tar.gz"
        if suffix in suffixes:
            raise ReleaseError(f"more than one {suffix} artifact was provided")
        suffixes.add(suffix)
        checked.append(path)
    missing = {".whl", ".tar.gz"} - suffixes
    if missing:
        names = ", ".join(sorted(missing))
        raise ReleaseError(
            f"publication requires both wheel and sdist; missing {names}"
        )
    return tuple(checked)


def _run(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=None if env is None else dict(env),
    )


def _upload_failure(message: str, output: str, version: str) -> ReleaseError:
    lowered = output.lower()
    if "already exists" in lowered or "file already exists" in lowered:
        return _duplicate_error(version)
    if any(
        marker in lowered
        for marker in ("401", "403", "unauthorized", "forbidden", "authentication")
    ):
        return ReleaseError(
            "Authentication to the SourceCraft registry failed. "
            f"{PAT_ENV} is missing, expired or lacks upload rights; rotate the "
            "CI secret and rerun. The token was not printed."
        )
    if "404" in lowered or "not found" in lowered:
        return ReleaseError(
            "The SourceCraft registry upload endpoint was not found. Check "
            f"{REGISTRY_ID_ENV} and that the registry is connected to this "
            "repository."
        )
    status = re.search(r"\b(?:HTTP\s*)?([45]\d{2})\b", output)
    detail = f"HTTP {status.group(1)}" if status else "an HTTP error"
    return ReleaseError(
        f"{message}\n{output}\nThe registry rejected the upload with {detail}; "
        "check the registry service, credentials and artifact metadata."
    )


def _duplicate_error(version: str) -> ReleaseError:
    return ReleaseError(
        f"Publication failed: {DISTRIBUTION_NAME} {version} already exists in "
        "the registry. A published version is immutable; bump the contract "
        "version and create a new contracts-v* tag."
    )


def registry_has_version(
    settings: RegistrySettings,
    version: str,
    *,
    index_url: str | None = None,
    timeout: int = 30,
) -> bool:
    """Check the registry simple index for an already published version.

    Args:
        settings: Registry settings with the PAT.
        version: Exact contract version to look for.
        index_url: Optional registry index override for local verification.
        timeout: HTTP timeout in seconds.

    Returns:
        ``True`` when the index already lists the version.

    Raises:
        ReleaseError: The index is unreachable or rejects the credentials.
    """
    endpoint = (index_url or settings.index_url).rstrip("/")
    url = f"{endpoint}/{DISTRIBUTION_NAME}/"
    credentials = base64.b64encode(
        f"{TWINE_USERNAME}:{settings.token}".encode()
    ).decode()
    request = urllib.request.Request(
        url, headers={"Authorization": f"Basic {credentials}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        if error.code in (401, 403):
            raise ReleaseError(
                "Authentication to the SourceCraft registry failed. "
                f"{PAT_ENV} is missing, expired or lacks read rights; the token "
                "was not printed."
            ) from error
        raise ReleaseError(
            f"cannot read the registry index: HTTP {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise ReleaseError(
            f"cannot reach the registry index: {error.reason}"
        ) from error
    pattern = re.compile(rf"automl_contracts-{re.escape(version)}(?:[-.]|\.tar\.gz)")
    return pattern.search(body) is not None


def publish(
    artifacts: Sequence[Path],
    settings: RegistrySettings,
    expected_version: str,
    *,
    python: str = sys.executable,
    upload_url: str | None = None,
    index_url: str | None = None,
    check_existing: bool = True,
    timeout: int = 900,
) -> None:
    """Upload validated artifacts with ``twine`` and fail on duplicates.

    Args:
        artifacts: Built wheel and sdist.
        settings: Registry settings with the PAT.
        expected_version: Version that every artifact must carry.
        python: Interpreter that has ``twine`` installed.
        upload_url: Optional endpoint override for local verification.
        index_url: Optional index override for local verification.
        check_existing: Reject an already published version before uploading.
        timeout: Upload timeout in seconds.

    Raises:
        ReleaseError: The artifact set is invalid or the upload fails.
    """
    paths = validate_artifacts(artifacts, expected_version)
    if check_existing and registry_has_version(
        settings, expected_version, index_url=index_url
    ):
        raise _duplicate_error(expected_version)
    endpoint = upload_url or settings.upload_url
    environment = dict(os.environ)
    environment.update(
        {
            "TWINE_USERNAME": TWINE_USERNAME,
            "TWINE_PASSWORD": settings.token,
            "TWINE_NON_INTERACTIVE": "1",
        }
    )
    command = [
        python,
        "-m",
        "twine",
        "upload",
        "--non-interactive",
        "--repository-url",
        endpoint,
        *(str(path) for path in paths),
    ]
    result = _run(command, env=environment, timeout=timeout)
    output = redact(f"{result.stdout}\n{result.stderr}", settings.token).strip()
    if result.returncode != 0:
        raise _upload_failure(
            f"twine upload failed with exit code {result.returncode}",
            output,
            expected_version,
        )
    names = ", ".join(path.name for path in paths)
    print(f"Published {DISTRIBUTION_NAME} {expected_version}: {names}")


def _install_failure(output: str, version: str) -> ReleaseError:
    lowered = output.lower()
    if any(
        marker in lowered
        for marker in ("401", "403", "unauthorized", "forbidden", "authentication")
    ):
        return ReleaseError(
            "Authentication to the SourceCraft registry failed. "
            f"{PAT_ENV} is missing, expired or lacks read rights; the token was "
            "not printed."
        )
    if "no matching distribution" in lowered or "not find a version" in lowered:
        return ReleaseError(
            f"{DISTRIBUTION_NAME} {version} is not available from the SourceCraft "
            f"registry. Upload it first from a contracts-v{version} tag and check "
            f"{REGISTRY_ID_ENV}."
        )
    return ReleaseError(
        f"installing {DISTRIBUTION_NAME} from the registry failed\n{output}"
    )


def install_from_registry(
    settings: RegistrySettings,
    version: str,
    *,
    python: str,
    with_jsonschema: bool = False,
    index_url: str | None = None,
    timeout: int = 900,
) -> None:
    """Install the released contract into a clean interpreter.

    Runtime dependencies come from the public index because the SourceCraft
    registry serves only the contract. The contract itself is installed with
    ``--no-deps`` from the authenticated registry index.

    Args:
        settings: Registry settings with the PAT.
        version: Exact contract version to install.
        python: Interpreter of the clean environment.
        with_jsonschema: Also install ``jsonschema`` for the consumer checks.
        index_url: Optional registry index override for local verification.
        timeout: Timeout for each pip invocation in seconds.

    Raises:
        ReleaseError: pip fails; the message never contains the PAT.
    """
    public_packages = ["pydantic>=2.12.5,<3", "pydantic-core>=2,<3"]
    if with_jsonschema:
        public_packages.append("jsonschema>=4.18")
    public_command = [
        python,
        "-m",
        "pip",
        "install",
        "--index-url",
        PUBLIC_INDEX_URL,
        "--disable-pip-version-check",
        *public_packages,
    ]
    public_result = _run(public_command, env=None, timeout=timeout)
    if public_result.returncode != 0:
        output = redact(public_result.stdout + public_result.stderr, settings.token)
        raise ReleaseError(f"installing runtime dependencies failed\n{output}")

    endpoint = index_url or settings.index_url
    environment = dict(os.environ)
    environment.update(
        {
            "PIP_INDEX_URL": authenticated_url(endpoint, settings.token),
            "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    install_command = [
        python,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-cache-dir",
        "--retries",
        "5",
        "--timeout",
        "60",
        f"{DISTRIBUTION_NAME}=={version}",
    ]
    result = _run(install_command, env=environment, timeout=timeout)
    output = redact(f"{result.stdout}\n{result.stderr}", settings.token).strip()
    if result.returncode != 0:
        raise _install_failure(output, version)

    check_result = _run(
        [python, "-m", "pip", "check"],
        env=None,
        timeout=timeout,
    )
    if check_result.returncode != 0:
        raise ReleaseError(
            f"pip check failed after installing {DISTRIBUTION_NAME}\n"
            f"{redact(check_result.stdout + check_result.stderr, settings.token)}"
        )
    print(f"Installed {DISTRIBUTION_NAME} {version} from the SourceCraft registry")


def _add_registry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--organization",
        default=DEFAULT_ORGANIZATION,
        help="SourceCraft organization slug.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_tag = subparsers.add_parser(
        "check-tag", help="Check the release tag against pyproject.toml."
    )
    check_tag.add_argument("--tag", required=True)
    check_tag.add_argument(
        "--ref",
        help="Full Git reference; enforces refs/tags/ and tag-on-HEAD.",
    )
    check_tag.add_argument(
        "--pyproject",
        type=Path,
        default=Path("automl-contracts/pyproject.toml"),
    )

    endpoints = subparsers.add_parser(
        "show-endpoints", help="Print registry endpoints without credentials."
    )
    _add_registry_arguments(endpoints)

    upload = subparsers.add_parser("publish", help="Upload built artifacts with twine.")
    upload.add_argument("--dist", type=Path, required=True)
    upload.add_argument("--expected-version", required=True)
    upload.add_argument("--python", default=sys.executable)
    upload.add_argument("--upload-url")
    upload.add_argument("--index-url")
    _add_registry_arguments(upload)

    install = subparsers.add_parser(
        "install", help="Install the contract from the SourceCraft registry."
    )
    install.add_argument("--python", required=True)
    install.add_argument("--version", required=True)
    install.add_argument("--with-jsonschema", action="store_true")
    install.add_argument("--index-url")
    _add_registry_arguments(install)
    return parser


def _distribution_artifacts(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ReleaseError(f"distribution directory {directory} does not exist")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )


def main(argv: list[str] | None = None) -> int:
    """Run the release command selected on the command line.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` when a release rule or registry operation fails.
    """
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "check-tag":
            if arguments.ref is not None:
                print(
                    verify_release_ref(
                        arguments.ref, arguments.tag, arguments.pyproject
                    )
                )
            else:
                print(verify_release_tag(arguments.tag, arguments.pyproject))
            return 0
        if arguments.command == "show-endpoints":
            registry_id = registry_id_from_environment()
            settings = RegistrySettings(
                registry_id=registry_id,
                token="",
                organization=arguments.organization,
            )
            print(f"upload: {settings.upload_url}")
            print(f"install: {settings.index_url}")
            return 0
        settings = registry_settings(organization=arguments.organization)
        if arguments.command == "publish":
            publish(
                _distribution_artifacts(arguments.dist),
                settings,
                arguments.expected_version,
                python=arguments.python,
                upload_url=arguments.upload_url,
                index_url=arguments.index_url,
            )
            return 0
        if arguments.command == "install":
            install_from_registry(
                settings,
                arguments.version,
                python=arguments.python,
                with_jsonschema=arguments.with_jsonschema,
                index_url=arguments.index_url,
            )
            return 0
        raise ReleaseError(f"unknown command {arguments.command!r}")
    except ReleaseError as error:
        print(f"contract release failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

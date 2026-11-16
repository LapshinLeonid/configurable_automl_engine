"""Guards for the CI publication and consumer-check configuration."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_guard() -> ModuleType:
    path = REPO_ROOT / "scripts" / "check_release_configuration.py"
    spec = importlib.util.spec_from_file_location("release_configuration_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard() -> ModuleType:
    return load_guard()


@pytest.fixture(scope="module")
def sourcecraft_text() -> str:
    return (REPO_ROOT / ".sourcecraft" / "ci.yaml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def github_publish_text() -> str:
    path = REPO_ROOT / ".github" / "workflows" / "publish-contracts.yaml"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def github_ci_text() -> str:
    path = REPO_ROOT / ".github" / "workflows" / "CI.yaml"
    return path.read_text(encoding="utf-8")


def messages(violations: list[object]) -> str:
    return "\n".join(violation.message for violation in violations)


def mutate_cubes(text: str, function: object) -> str:
    """Apply a mutation to the publish-contracts cubes and dump valid YAML."""
    document = yaml.safe_load(text)
    cubes = document["workflows"]["publish-contracts"]["tasks"][0]["cubes"]
    function(cubes)
    return yaml.safe_dump(document, sort_keys=False)


def inside(text: str, block: str, old: str, new: str, *, count: int = 1) -> str:
    """Replace ``old`` occurrences after the ``block`` marker."""
    start = text.index(block)
    head, tail = text[:start], text[start:]
    assert old in tail, (block, old)
    return head + tail.replace(old, new, count)


def test_repository_release_configuration_is_wired(guard: ModuleType) -> None:
    assert guard.run_checks(REPO_ROOT) == []


def test_check_scripts_rejects_a_modified_uploader(
    guard: ModuleType, tmp_path: Path
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in (
        "contract_release.py",
        "e2e_contract_consumer.py",
        "smoke_engine_install.py",
        "check_release_configuration.py",
    ):
        shutil.copy(REPO_ROOT / "scripts" / name, scripts / name)
    release = scripts / "contract_release.py"
    text = release.read_text(encoding="utf-8")
    release.write_text(
        text.replace(
            '"twine",\n        "upload",',
            '"twine",\n        "upload",\n        "--skip-existing",',
            1,
        ),
        encoding="utf-8",
    )
    violations = guard.check_scripts(tmp_path)
    assert any("skip it" in violation.message for violation in violations)


def test_sourcecraft_requires_the_contracts_tag_trigger(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace('tags: ["contracts-v*"]', 'tags: ["v*"]')
    assert "contracts-v*" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_a_publish_tag_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        '    - workflows: [publish-contracts]\n      filter:\n        tags: ["contracts-v*"]',
        '    - workflows: [publish-contracts]\n      filter:\n        branches: ["main"]',
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_a_sync_tag_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        '    - workflows: [sync-to-github]\n      filter:\n        tags: ["contracts-v*"]\n',
        "",
        1,
    )
    assert "sync-to-github" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_commented_out_gate(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "python3.11 scripts/check_release_configuration.py",
        "# python3.11 scripts/check_release_configuration.py",
    )
    assert "check_release_configuration.py" in messages(
        guard.check_sourcecraft_ci(text)
    )


def test_sourcecraft_requires_build_before_publish(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def move_publish_first(cubes: list[dict[str, object]]) -> None:
        publish = next(cube for cube in cubes if cube["name"] == "publish")
        cubes.remove(publish)
        build = next(cube for cube in cubes if cube["name"] == "build-and-verify")
        cubes.insert(cubes.index(build), publish)

    text = mutate_cubes(sourcecraft_text, move_publish_first)
    assert "order is" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_registry_verification_after_publish(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def move_verification_first(cubes: list[dict[str, object]]) -> None:
        verification = next(
            cube for cube in cubes if cube["name"] == "verify-from-registry"
        )
        cubes.remove(verification)
        publish = next(cube for cube in cubes if cube["name"] == "publish")
        cubes.insert(cubes.index(publish), verification)

    text = mutate_cubes(sourcecraft_text, move_verification_first)
    assert "order is" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_missing_quality_gates_cube(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def remove_gates(cubes: list[dict[str, object]]) -> None:
        cubes[:] = [cube for cube in cubes if cube["name"] != "quality-gates"]

    text = mutate_cubes(sourcecraft_text, remove_gates)
    assert "quality-gates" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_flow_style_branch_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\n  pull_request:",
        "\n    - {workflows: [publish-contracts], filter: {branches: [main]}}\n\n  pull_request:",
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_bare_dash_branch_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\n  pull_request:",
        "\n    -\n      workflows: [publish-contracts]\n      filter:\n        branches: [main]\n\n  pull_request:",
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_broad_tag_mask(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        'tags: ["contracts-v*"]',
        'tags: ["contracts-v*", "v*"]',
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_the_uploader_in_preflight(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_uploader(cubes: list[dict[str, object]]) -> None:
        preflight = next(cube for cube in cubes if cube["name"] == "preflight")
        script = preflight["script"]
        assert isinstance(script, list)
        script.append("python3.11 scripts/contract_release.py publish --dist dist")

    text = mutate_cubes(sourcecraft_text, add_uploader)
    assert "runs the uploader" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_second_branch_publish_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    tag_entry = (
        "    - workflows: [publish-contracts]\n"
        "      filter:\n"
        '        tags: ["contracts-v*"]\n'
    )
    branch_entry = (
        "    - workflows: [publish-contracts]\n"
        "      filter:\n"
        '        branches: ["main"]\n'
    )
    assert tag_entry in sourcecraft_text
    text = sourcecraft_text.replace(tag_entry, tag_entry + branch_entry, 1)
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_pull_request_quality_workflow(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  pull_request:\n",
        "workflows: [quality-checks]",
        "workflows: [ci-static-analysis-only]",
    )
    assert "quality-checks" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_synchronizes_tags_to_github(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace("refs/tags/", "refs/other/")
    assert "tag ref" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_pull_request_checks(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace("  pull_request:\n", "  manual_only:\n", 1)
    assert "pull requests" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_the_publication_workflow(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "  publish-contracts:\n", "  missing-workflow:\n", 1
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "token",
    [
        "scripts/contract_release.py check-tag",
        "scripts/check_contract_boundaries.py",
        "python3.11 scripts/check_release_configuration.py",
        "python3.11 -m ruff check src/ automl-contracts/src/",
        "python3.11 -m mypy scripts/contract_release.py",
        "importlinter.cli",
        "pytest tests/contracts",
        "python3.11 -m build automl-contracts",
        "twine check",
        "scripts/verify_contract_distribution.py",
        "scripts/contract_release.py publish",
        "scripts/contract_release.py install",
        '"$REPO_DIR/scripts/e2e_contract_consumer.py"',
        "SOURCECRAFT_PYPI_REGISTRY_ID",
        "PYPI_REGISTRY_PAT",
    ],
)
def test_sourcecraft_publication_requires_each_gate(
    guard: ModuleType, sourcecraft_text: str, token: str
) -> None:
    text = inside(sourcecraft_text, "  publish-contracts:\n", token, "removed-gate")
    assert token in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_accepts_a_configured_registry_id(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        'SOURCECRAFT_PYPI_REGISTRY_ID: ""',
        'SOURCECRAFT_PYPI_REGISTRY_ID: "registry123"',
    )
    assert guard.check_sourcecraft_ci(text) == []


def test_sourcecraft_rejects_a_literal_pat(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "PYPI_REGISTRY_PAT: ${{ secrets.PYPI_REGISTRY_PAT }}",
        'PYPI_REGISTRY_PAT: "ghp_literal_secret_value"',
    )
    assert "literal secret" in messages(guard.check_sourcecraft_ci(text))


def test_order_helper_reports_a_publish_before_the_gates(guard: ModuleType) -> None:
    order = ("preflight", "quality-gates", "build-and-verify", "publish")
    actual = ["preflight", "publish", "quality-gates", "build-and-verify"]
    assert guard._order_violation(Path("ci.yaml"), order, actual, "release")


def test_sourcecraft_rejects_skip_existing(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "scripts/contract_release.py publish",
        "scripts/contract_release.py publish --skip-existing",
    )
    assert "skip it" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_literal_credential(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "python3.11 -m twine check",
        "python3.11 -m twine upload https://iam:hardcoded-token@pkg.sourcecraft.tech/simple/",
    )
    assert "literal credential" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_password_argument(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "python3.11 -m twine check",
        "python3.11 -m twine upload --password hardcoded-token",
    )
    assert "password literal" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_regular_checks_need_the_release_guard(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  quality-checks:\n",
        "python3.11 scripts/check_release_configuration.py",
        "removed-guard",
        count=-1,
    )
    assert "check_release_configuration.py" in messages(
        guard.check_sourcecraft_ci(text)
    )


def test_github_publication_is_limited_to_the_contract_tag(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace('"contracts-v*"', '"v*"')
    assert "contracts-v*" in messages(guard.check_github_publish(text))


def test_github_publication_requires_the_release_gates(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace("pytest tests/contracts", "removed-gate")
    assert "pytest tests/contracts" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_a_branch_filter(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        '      - "contracts-v*"',
        '      - "contracts-v*"\n    branches:\n      - "main"',
        1,
    )
    assert "limited to contracts-v*" in messages(guard.check_github_publish(text))


def test_github_publication_requires_tags_not_branches(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        '    tags:\n      - "contracts-v*"',
        '    branches:\n      - "main"',
        1,
    )
    assert "limited to contracts-v*" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_a_commented_out_gate(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "pytest tests/contracts", "# pytest tests/contracts", 1
    )
    assert "pytest tests/contracts" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_skip_existing(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "pytest tests/contracts",
        "pytest tests/contracts\n          python scripts/contract_release.py publish --skip-existing",
        1,
    )
    assert "existing version" in messages(guard.check_github_publish(text))


def test_github_publication_requires_the_quality_step(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace("Run release quality gates", "removed-gates")
    assert "Run release quality gates" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_a_literal_pat(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "jobs:",
        'env:\n  PYPI_REGISTRY_PAT: "ghp_literal_secret_value"\njobs:',
        1,
    )
    assert "literal" in messages(guard.check_github_publish(text))


def test_uploader_cannot_use_skip_existing_or_a_password(
    guard: ModuleType,
) -> None:
    path = REPO_ROOT / "scripts" / "contract_release.py"
    text = path.read_text(encoding="utf-8")
    assert guard._check_uploader(path, text) == []
    mutated = text.replace(
        '        "twine",\n        "upload",',
        '        "twine",\n        "upload",\n        "--skip-existing",\n        "--password",\n        "hardcoded-token",',
        1,
    )
    assert mutated != text
    messages = "\n".join(
        violation.message for violation in guard._check_uploader(path, mutated)
    )
    assert "skip it" in messages
    assert "password literal" in messages


def test_violation_paths_follow_the_repo_root(
    guard: ModuleType, sourcecraft_text: str, github_publish_text: str, tmp_path: Path
) -> None:
    ci_path = tmp_path / "ci.yaml"
    github_path = tmp_path / "publish.yaml"
    ci_violations = guard.check_sourcecraft_ci(
        sourcecraft_text.replace("  pull_request:\n", "  no_pr:\n", 1), ci_path
    )
    github_violations = guard.check_github_publish(
        github_publish_text.replace('"contracts-v*"', '"v*"'), github_path
    )
    assert ci_violations
    assert github_violations
    assert {violation.path for violation in ci_violations} == {ci_path}
    assert {violation.path for violation in github_violations} == {github_path}


def test_github_ci_requires_pull_request_checks(
    guard: ModuleType, github_ci_text: str
) -> None:
    text = github_ci_text.replace("  pull_request:\n", "", 1)
    assert "pull requests" in messages(guard.check_github_ci(text))


def test_github_ci_path_follows_the_repo_root(
    guard: ModuleType, github_ci_text: str, tmp_path: Path
) -> None:
    path = tmp_path / "CI.yaml"
    violations = guard.check_github_ci(
        github_ci_text.replace("scripts/smoke_engine_install.py", "removed"), path
    )
    assert violations
    assert {violation.path for violation in violations} == {path}


def test_github_ci_rejects_a_commented_out_guard(
    guard: ModuleType, github_ci_text: str
) -> None:
    text = github_ci_text.replace(
        "          python scripts/check_release_configuration.py",
        "          # python scripts/check_release_configuration.py",
        1,
    )
    assert "check_release_configuration.py" in messages(guard.check_github_ci(text))


def test_github_ci_requires_engine_integration(
    guard: ModuleType, github_ci_text: str
) -> None:
    text = github_ci_text.replace("scripts/smoke_engine_install.py", "removed")
    assert "smoke_engine_install.py" in messages(guard.check_github_ci(text))


def test_missing_repository_files_are_reported(
    guard: ModuleType, tmp_path: Path
) -> None:
    violations = guard.run_checks(tmp_path)
    text = messages(violations)
    assert "SourceCraft CI file is missing" in text
    assert "GitHub release workflow is missing" in text
    assert "GitHub CI file is missing" in text
    assert "release helper script is missing" in text


def test_guard_cli_reports_success(
    guard: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert guard.main(["--repo-root", str(REPO_ROOT)]) == 0
    assert "release configuration is wired" in capsys.readouterr().out


def test_guard_cli_reports_failures(
    guard: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert guard.main(["--repo-root", str(tmp_path)]) == 1
    assert "release configuration violation" in capsys.readouterr().err

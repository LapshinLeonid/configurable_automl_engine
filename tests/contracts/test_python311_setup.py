"""Behaviour checks for the shared Python 3.11 installation script."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_python311.sh"

STUB_HEADER = "#!/usr/bin/env bash\nset -u\n"

APT_STUB = """\
printf 'apt-get %s\\n' "$*" >> "$CALL_LOG"
if [ "${1:-}" = "-c" ] && [ -n "${2:-}" ]; then
  cp "$2" "$APT_CONFIG_LOG" 2>/dev/null || true
fi
case "$*" in
  *install*python3.11*)
    counter="$APT_INSTALL_COUNTER"
    limit="$APT_INSTALL_FAIL_TIMES"
    ;;
  *update*)
    if [ "${APT_FAIL_ON_DIRTY_PYTHON:-0}" = "1" ] &&
      [ "$(cat "$PYTHON3_STATE" 2>/dev/null)" != "system" ]; then
      echo "ModuleNotFoundError: No module named 'apt_pkg'" >&2
      echo "E: Problem executing scripts APT::Update::Post-Invoke-Success '/usr/lib/cnf-update-db'" >&2
      exit 100
    fi
    counter="$APT_UPDATE_COUNTER"
    limit="$APT_UPDATE_FAIL_TIMES"
    ;;
  *)
    exit 0
    ;;
esac
count=$(cat "$counter" 2>/dev/null || echo 0)
count=$((count + 1))
printf '%s\\n' "$count" > "$counter"
if [ "$count" -le "$limit" ]; then
  echo "E: Failed to fetch https://ppa.launchpadcontent.net/deadsnakes 503" >&2
  exit 100
fi
exit 0
"""

PPA_STUB = """\
printf 'add-apt-repository %s\\n' "$*" >> "$CALL_LOG"
count=$(cat "$PPA_COUNTER" 2>/dev/null || echo 0)
count=$((count + 1))
printf '%s\\n' "$count" > "$PPA_COUNTER"
if [ "$count" -le "$PPA_FAIL_TIMES" ]; then
  echo "E: Failed to fetch https://ppa.launchpadcontent.net/deadsnakes 503" >&2
  exit 100
fi
exit 0
"""

GREP_STUB = """\
printf 'grep %s\\n' "$*" >> "$CALL_LOG"
exit "${GREP_EXIT_CODE:-1}"
"""

PYTHON311_STUB = """\
printf 'python3.11 %s\\n' "$*" >> "$CALL_LOG"
if [ "${1:-}" = "--version" ]; then
  echo "Python 3.11.99"
  exit 0
fi
if [ "${1:-}" = "-c" ]; then
  exit 1
fi
cat >/dev/null
"""

PYTHON3_STUB = """\
printf 'python3 %s\\n' "$*" >> "$CALL_LOG"
if [ "${1:-}" = "-c" ]; then
  if [ "$(cat "$PYTHON3_STATE" 2>/dev/null)" = "system" ]; then
    exit 0
  fi
  echo "ModuleNotFoundError: No module named 'apt_pkg'" >&2
  exit 1
fi
exit 0
"""

APT_CAPABLE_PYTHON_STUB = """\
printf 'python3.10 %s\\n' "$*" >> "$CALL_LOG"
exit 0
"""

BROKEN_PYTHON_STUB = """\
printf 'python3.10 %s\\n' "$*" >> "$CALL_LOG"
exit 1
"""

BROKEN_VERSION_STUB = """\
printf '%s %s\\n' "$0" "$*" >> "$CALL_LOG"
exit 1
"""

UPDATE_ALTERNATIVES_STUB = """\
printf 'update-alternatives %s\\n' "$*" >> "$CALL_LOG"
if [ "${1:-}" = "--set" ]; then
  case "${3:-}" in
    *python3.11) printf 'python311\\n' > "$PYTHON3_STATE" ;;
    *) printf 'system\\n' > "$PYTHON3_STATE" ;;
  esac
fi
"""


def write_stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"{STUB_HEADER}{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def run_setup(
    tmp_path: Path,
    *,
    attempts: int = 3,
    install_failures: int = 0,
    update_failures: int = 0,
    ppa_failures: int = 0,
    python3_state: str = "system",
    apt_fail_on_dirty_python: bool = False,
    system_python_has_apt_pkg: bool = True,
    ppa_configured: bool = False,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    state_file = tmp_path / "python3-state"
    state_file.write_text(f"{python3_state}\n", encoding="utf-8")
    write_stub(bin_dir, "apt-get", APT_STUB)
    write_stub(bin_dir, "add-apt-repository", PPA_STUB)
    write_stub(bin_dir, "grep", GREP_STUB)
    write_stub(bin_dir, "update-alternatives", UPDATE_ALTERNATIVES_STUB)
    write_stub(bin_dir, "curl", 'cat <<"PIP"\nprint("pip")\nPIP\n')
    write_stub(bin_dir, "python3", PYTHON3_STUB)
    write_stub(bin_dir, "python3.11", PYTHON311_STUB)
    for version in ("3.14", "3.13", "3.12", "3.9", "3.8", "3.7", "3.6"):
        write_stub(bin_dir, f"python{version}", BROKEN_VERSION_STUB)
    write_stub(
        bin_dir,
        "python3.10",
        APT_CAPABLE_PYTHON_STUB
        if system_python_has_apt_pkg
        else BROKEN_PYTHON_STUB,
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CALL_LOG"] = str(call_log)
    env["APT_CONFIG_LOG"] = str(tmp_path / "apt-config.log")
    env["PYTHON3_STATE"] = str(state_file)
    env["APT_FAIL_ON_DIRTY_PYTHON"] = "1" if apt_fail_on_dirty_python else "0"
    env["GREP_EXIT_CODE"] = "0" if ppa_configured else "1"
    env["APT_INSTALL_COUNTER"] = str(tmp_path / "apt-install-count")
    env["APT_UPDATE_COUNTER"] = str(tmp_path / "apt-update-count")
    env["APT_INSTALL_FAIL_TIMES"] = str(install_failures)
    env["APT_UPDATE_FAIL_TIMES"] = str(update_failures)
    env["PPA_COUNTER"] = str(tmp_path / "ppa-count")
    env["PPA_FAIL_TIMES"] = str(ppa_failures)
    env["PYTHON311_SETUP_ATTEMPTS"] = str(attempts)
    env["PYTHON311_SETUP_RETRY_DELAY"] = "0"
    return subprocess.run(
        ["bash", str(SETUP_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def counter(tmp_path: Path, name: str) -> str:
    return (tmp_path / name).read_text(encoding="utf-8").strip()


def calls_text(tmp_path: Path) -> str:
    return (tmp_path / "calls.log").read_text(encoding="utf-8")


def test_installs_once_when_the_mirror_is_healthy(tmp_path: Path) -> None:
    result = run_setup(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Python 3.11 is ready (attempt 1/3)." in result.stdout
    assert "PYTHON311_SETUP_RETRY" not in result.stdout
    assert counter(tmp_path, "apt-install-count") == "1"
    assert counter(tmp_path, "ppa-count") == "1"
    calls = calls_text(tmp_path)
    assert "Acquire::Retries=3" in calls
    assert "--fix-missing" in calls
    assert "ppa:deadsnakes/ppa" in calls


def test_disables_apt_post_invoke_hooks(tmp_path: Path) -> None:
    result = run_setup(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = calls_text(tmp_path)
    assert "apt-get -c " in calls
    apt_config = (tmp_path / "apt-config.log").read_text(encoding="utf-8")
    assert "#clear APT::Update::Post-Invoke-Success;" in apt_config
    assert "#clear DPkg::Post-Invoke;" in apt_config


def test_recovers_when_python3_already_points_to_311(tmp_path: Path) -> None:
    result = run_setup(
        tmp_path,
        python3_state="python311",
        apt_fail_on_dirty_python=True,
        ppa_configured=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "python3 cannot import apt_pkg; restoring" in result.stdout
    assert "Python 3.11 is ready (attempt 1/3)." in result.stdout
    calls = calls_text(tmp_path)
    assert "update-alternatives --install /usr/bin/python3 python3" in calls
    assert "python3.10" in calls
    assert "update-alternatives --set python3" in calls
    assert counter(tmp_path, "apt-install-count") == "1"


def test_skips_add_apt_repository_when_the_ppa_is_already_configured(
    tmp_path: Path,
) -> None:
    result = run_setup(tmp_path, ppa_configured=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "add-apt-repository" not in calls_text(tmp_path)
    assert counter(tmp_path, "apt-install-count") == "1"


def test_reports_a_worker_without_an_apt_capable_python(tmp_path: Path) -> None:
    result = run_setup(
        tmp_path,
        attempts=2,
        python3_state="python311",
        system_python_has_apt_pkg=False,
    )

    assert result.returncode == 1
    assert "PYTHON311_SETUP_FAILED" in result.stdout
    assert "Last failed step: restore the system python3 for apt" in result.stdout
    assert "Python 3.11 is ready" not in result.stdout


def test_retries_after_a_transient_503_on_the_packages(tmp_path: Path) -> None:
    result = run_setup(tmp_path, install_failures=2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PYTHON311_SETUP_RETRY") == 2
    assert "failed at 'install Python 3.11 packages'" in result.stdout
    assert "Python 3.11 is ready (attempt 3/3)." in result.stdout
    assert counter(tmp_path, "apt-install-count") == "3"


def test_retries_after_a_transient_503_on_the_ppa_index(tmp_path: Path) -> None:
    result = run_setup(tmp_path, update_failures=2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PYTHON311_SETUP_RETRY") == 2
    assert "failed at 'apt-get update'" in result.stdout
    assert "Python 3.11 is ready (attempt 3/3)." in result.stdout


def test_retries_after_a_transient_add_apt_repository_failure(
    tmp_path: Path,
) -> None:
    result = run_setup(tmp_path, ppa_failures=2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PYTHON311_SETUP_RETRY") == 2
    assert "failed at 'add the deadsnakes PPA'" in result.stdout
    assert "Python 3.11 is ready (attempt 3/3)." in result.stdout
    assert counter(tmp_path, "ppa-count") == "3"


def test_reports_exhausted_package_installs(tmp_path: Path) -> None:
    result = run_setup(tmp_path, install_failures=10)

    assert result.returncode == 1
    assert result.stdout.count("PYTHON311_SETUP_RETRY") == 2
    assert "PYTHON311_SETUP_FAILED: Python 3.11 setup stopped after 3 attempt(s)." in (
        result.stdout
    )
    assert "Last failed step: install Python 3.11 packages" in result.stdout
    assert "Python 3.11 is ready" not in result.stdout
    assert counter(tmp_path, "apt-install-count") == "3"


def test_reports_exhausted_ppa_registration(tmp_path: Path) -> None:
    result = run_setup(tmp_path, ppa_failures=10)

    assert result.returncode == 1
    assert result.stdout.count("PYTHON311_SETUP_RETRY") == 2
    assert "Last failed step: add the deadsnakes PPA" in result.stdout
    assert "Python 3.11 is ready" not in result.stdout
    assert counter(tmp_path, "ppa-count") == "3"

#!/usr/bin/env bash
#
# Install Python 3.11 on the native CI worker.
#
# The deadsnakes PPA serves the packages from launchpad.net, which sometimes
# answers with HTTP 503 while apt fetches the archives. The whole installation
# sequence is therefore retried, and every attempt starts with a fresh
# ``apt-get update`` so that a stale mirror index is not reused.
#
# The worker switches ``/usr/bin/python3`` to 3.11 for the build cubes, but
# APT itself keeps using the system Python: the ``command-not-found``
# post-invoke hook imports ``apt_pkg`` from ``/usr/lib/python3/dist-packages``
# and ``add-apt-repository`` does the same. When a previous cube left
# ``python3`` on 3.11, every APT operation fails with
# ``ModuleNotFoundError: No module named 'apt_pkg'``. Every attempt therefore
# restores the system interpreter before touching APT, and the informational
# post-invoke hooks are disabled so that a hook failure can never turn a
# successful package download into a failed build. The 3.11 alternative is
# selected again once the installation is complete.
#
# When all attempts fail the script prints ``PYTHON311_SETUP_FAILED`` together
# with the step that failed and exits non-zero, so an infrastructure failure is
# distinguishable from a build error.
#
# Environment variables:
#   PYTHON311_SETUP_ATTEMPTS: number of full installation attempts (default 5).
#   PYTHON311_SETUP_RETRY_DELAY: first pause between attempts in seconds; it
#     doubles up to 120 seconds (default 20).
#   PYTHON311_APT_RETRIES: retries performed by apt itself (default 3).

set -euo pipefail

readonly ATTEMPTS="${PYTHON311_SETUP_ATTEMPTS:-5}"
readonly RETRY_DELAY="${PYTHON311_SETUP_RETRY_DELAY:-20}"
readonly APT_RETRIES="${PYTHON311_APT_RETRIES:-3}"
readonly PACKAGES=(python3.11 python3.11-distutils python3.11-venv)
# Only retries are tuned: a custom transfer timeout would abort the slow but
# working downloads that the native worker sometimes sees.
readonly APT_OPTIONS=(-o "Acquire::Retries=${APT_RETRIES}")
# APT runs post-invoke hooks even when the package lists were fetched
# successfully. The command-not-found hook is informational but it breaks the
# whole update when the system Python cannot import apt_pkg, so clear the hooks
# for the commands issued here.
readonly APT_CONFIG_FILE="$(mktemp)"
readonly PYTHON_VERSIONS=(3.14 3.13 3.12 3.11 3.10 3.9 3.8 3.7 3.6)

cleanup() {
  rm -f "${APT_CONFIG_FILE}"
}
trap cleanup EXIT

cat >"${APT_CONFIG_FILE}" <<'EOF'
#clear APT::Update::Post-Invoke;
#clear APT::Update::Post-Invoke-Success;
#clear DPkg::Post-Invoke;
#clear DPkg::Post-Invoke-Success;
EOF

CURRENT_STEP="initialising"
PREPARED=0

log() {
  printf '%s\n' "$*"
}

apt_get() {
  DEBIAN_FRONTEND=noninteractive apt-get -c "${APT_CONFIG_FILE}" "$@"
}

python3_can_import_apt_pkg() {
  python3 -c 'import apt_pkg' >/dev/null 2>&1
}

deadsnakes_ppa_configured() {
  # A previous cube usually added the PPA already. Re-adding it is not only
  # unnecessary but also depends on the Python packages of
  # software-properties-common, which a project pip install may have replaced.
  grep -Rqs -E 'ppa\.launchpadcontent\.net/deadsnakes|deadsnakes/ppa' \
    /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null
}

system_python_for_apt() {
  local version candidate
  for version in "${PYTHON_VERSIONS[@]}"; do
    candidate="$(command -v "python${version}" || true)"
    if [[ -n "${candidate}" ]] &&
      "${candidate}" -c 'import apt_pkg' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

ensure_apt_python() {
  if python3_can_import_apt_pkg; then
    return 0
  fi

  CURRENT_STEP="restore the system python3 for apt"
  local candidate
  if ! candidate="$(system_python_for_apt)"; then
    log "PYTHON311_SETUP_ERROR: python3 cannot import apt_pkg and no system Python with apt_pkg was found."
    return 1
  fi

  log "python3 cannot import apt_pkg; restoring ${candidate} for APT."
  update-alternatives --install /usr/bin/python3 python3 "${candidate}" 10 ||
    return 1
  update-alternatives --set python3 "${candidate}" || return 1

  if ! python3_can_import_apt_pkg; then
    log "PYTHON311_SETUP_ERROR: ${candidate} was selected for python3 but apt_pkg is still unavailable."
    return 1
  fi
}

prepare_apt() {
  # A previous attempt or cube may have left python3 on 3.11, so repair the
  # interpreter before every apt invocation, including retries.
  ensure_apt_python || return 1
  if ((PREPARED == 1)); then
    return 0
  fi
  CURRENT_STEP="apt-get update"
  apt_get update -qq "${APT_OPTIONS[@]}" || return 1

  if ! deadsnakes_ppa_configured; then
    CURRENT_STEP="install software-properties-common"
    apt_get install -y -qq software-properties-common || return 1
  fi

  PREPARED=1
}

install_python() {
  if ! deadsnakes_ppa_configured; then
    CURRENT_STEP="add the deadsnakes PPA"
    add-apt-repository -y ppa:deadsnakes/ppa || return 1

    CURRENT_STEP="apt-get update after adding the deadsnakes PPA"
    apt_get update -qq "${APT_OPTIONS[@]}" || return 1
  fi

  CURRENT_STEP="install Python 3.11 packages"
  apt_get install -y -qq "${APT_OPTIONS[@]}" --fix-missing "${PACKAGES[@]}" ||
    return 1

  CURRENT_STEP="configure the python3 alternatives"
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 ||
    return 1
  update-alternatives --set python3 /usr/bin/python3.11 || return 1

  CURRENT_STEP="bootstrap pip for Python 3.11"
  curl -fsSL --retry "${APT_RETRIES}" --retry-delay 5 \
    https://bootstrap.pypa.io/get-pip.py | python3.11 || return 1

  CURRENT_STEP="verify the Python 3.11 interpreter"
  python3.11 --version
}

attempt=1
status=0
delay="${RETRY_DELAY}"
while ((attempt <= ATTEMPTS)); do
  log "──── Python 3.11 setup (attempt ${attempt}/${ATTEMPTS}) ────"
  if prepare_apt && install_python; then
    log "Python 3.11 is ready (attempt ${attempt}/${ATTEMPTS})."
    exit 0
  else
    status=$?
  fi
  if ((attempt >= ATTEMPTS)); then
    break
  fi
  log "PYTHON311_SETUP_RETRY: attempt ${attempt}/${ATTEMPTS} failed at '${CURRENT_STEP}' with status ${status}; retrying in ${delay}s."
  sleep "${delay}"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
  if ((delay > 120)); then
    delay=120
  fi
done

log "PYTHON311_SETUP_FAILED: Python 3.11 setup stopped after ${ATTEMPTS} attempt(s)."
log "Last failed step: ${CURRENT_STEP} (status ${status})."
log "Check the failing step above and rerun the workflow after resolving it."
exit 1

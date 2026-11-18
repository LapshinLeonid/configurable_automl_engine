#!/usr/bin/env bash
#
# Install Python 3.11 on the native CI worker.
#
# The deadsnakes PPA serves the packages from launchpad.net, which sometimes
# answers with HTTP 503 while apt fetches the archives. The whole installation
# sequence is therefore retried, and every attempt starts with a fresh
# ``apt-get update`` so that a stale mirror index is not reused.
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

CURRENT_STEP="initialising"
PREPARED=0

log() {
  printf '%s\n' "$*"
}

apt_get() {
  DEBIAN_FRONTEND=noninteractive apt-get "$@"
}

prepare_apt() {
  if ((PREPARED == 1)); then
    return 0
  fi
  CURRENT_STEP="apt-get update"
  apt_get update -qq || return 1

  CURRENT_STEP="install software-properties-common"
  apt_get install -y -qq software-properties-common || return 1

  PREPARED=1
}

install_python() {
  CURRENT_STEP="add the deadsnakes PPA"
  add-apt-repository -y ppa:deadsnakes/ppa || return 1

  CURRENT_STEP="apt-get update after adding the deadsnakes PPA"
  apt_get update -qq "${APT_OPTIONS[@]}" || return 1

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

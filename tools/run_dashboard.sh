#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${IPFS_ACCELERATE_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VIRTUAL_ENV:-${REPO_DIR}/.venv}"

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

cd "${REPO_DIR}"

if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

exec python -c "from utils.performance_dashboard import start_performance_dashboard; start_performance_dashboard(host='${HOST}', port=int('${PORT}'), background=False)"

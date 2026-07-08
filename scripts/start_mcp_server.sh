#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${IPFS_ACCELERATE_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VIRTUAL_ENV:-${REPO_DIR}/.venv}"

cd "${REPO_DIR}"

if [ -f "${VENV_DIR}/bin/activate" ]; then
	# shellcheck disable=SC1091
	source "${VENV_DIR}/bin/activate"
fi

export PYTHONPATH="${REPO_DIR}"

HOST="${MCP_HOST:-0.0.0.0}"
PORT="${MCP_PORT:-9000}"

exec "${VENV_DIR}/bin/ipfs-accelerate" mcp start --host "${HOST}" --port "${PORT}"
#!/bin/bash
# Auto-update script for IPFS Accelerate MCP Service
# Pulls latest changes from main branch and restarts the service

set -e

LOG_FILE="/tmp/ipfs-accelerate-update.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_DIR="${IPFS_ACCELERATE_REPO_DIR:-${DEFAULT_WORK_DIR}}"

echo "=== IPFS Accelerate Auto-Update ===" | tee -a "$LOG_FILE"
echo "$(date): Starting update process" | tee -a "$LOG_FILE"

cd "$WORK_DIR"

# Check if there are any uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "$(date): Warning - Uncommitted changes detected, stashing them" | tee -a "$LOG_FILE"
    git stash save "Auto-stash before update $(date +%Y-%m-%d-%H:%M:%S)"
fi

# Fetch and pull latest changes
echo "$(date): Fetching latest changes from origin/main" | tee -a "$LOG_FILE"
git fetch origin

# Check if there are updates
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "$(date): Already up to date (no changes)" | tee -a "$LOG_FILE"
    exit 0
fi

echo "$(date): Updates available, pulling changes" | tee -a "$LOG_FILE"
git pull origin main | tee -a "$LOG_FILE"

# Install/update dependencies
echo "$(date): Checking and updating dependencies" | tee -a "$LOG_FILE"
VENV_DIR="${VIRTUAL_ENV:-${WORK_DIR}/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# Install requirements.txt if it exists
if [ -f "$WORK_DIR/requirements.txt" ]; then
    echo "$(date): Installing dependencies from requirements.txt" | tee -a "$LOG_FILE"
    pip install -q -r "$WORK_DIR/requirements.txt" 2>&1 | tee -a "$LOG_FILE"
fi

# Install package in editable mode (checks for setup.py or pyproject.toml)
if [ -f "$WORK_DIR/setup.py" ] || [ -f "$WORK_DIR/pyproject.toml" ]; then
    echo "$(date): Installing package in editable mode" | tee -a "$LOG_FILE"
    pip install -q -e . 2>&1 | tee -a "$LOG_FILE"
fi

echo "$(date): Dependencies update complete" | tee -a "$LOG_FILE"

# Restart the systemd service
echo "$(date): Restarting services (best-effort)" | tee -a "$LOG_FILE"

# Prefer system services when available; fall back to user services.
restart_ok=0

for unit in ipfs-accelerate.service ipfs-accelerate-mcp.service; do
    if systemctl restart "${unit}" >/dev/null 2>&1; then
        echo "$(date): Restarted system unit ${unit}" | tee -a "$LOG_FILE"
        restart_ok=1
        continue
    fi

    if systemctl --user restart "${unit}" >/dev/null 2>&1; then
        echo "$(date): Restarted user unit ${unit}" | tee -a "$LOG_FILE"
        restart_ok=1
        continue
    fi
done

if [ "${restart_ok}" -ne 1 ]; then
    echo "$(date): NOTE - Could not restart services automatically (insufficient permissions or unit not installed)." | tee -a "$LOG_FILE"
fi

echo "$(date): Update complete" | tee -a "$LOG_FILE"

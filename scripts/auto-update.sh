#!/bin/bash
# Auto-update script for ipfs_accelerate_py
# Pulls from main branch and installs Python packages

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="${IPFS_ACCELERATE_REPO_DIR:-${DEFAULT_REPO_DIR}}"
VENV_DIR="${VIRTUAL_ENV:-${REPO_DIR}/.venv}"
LOG_DIR="${IPFS_ACCELERATE_LOG_DIR:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_DIR}/auto-update.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo "[${TIMESTAMP}] $1" | tee -a "${LOG_FILE}"
}

log "Starting auto-update process..."

# Change to repository directory
cd "${REPO_DIR}" || {
    log "ERROR: Failed to change to repository directory"
    exit 1
}

# Fetch latest changes
log "Fetching latest changes from origin..."
if ! git fetch origin; then
    log "ERROR: Failed to fetch from origin"
    exit 1
fi

# Check if we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "${CURRENT_BRANCH}" != "main" ]; then
    log "WARNING: Not on main branch (currently on ${CURRENT_BRANCH}), switching to main..."
    git checkout main || {
        log "ERROR: Failed to checkout main branch"
        exit 1
    }
fi

# Get current commit
CURRENT_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/main)

# Check if update is needed
if [ "${CURRENT_COMMIT}" = "${REMOTE_COMMIT}" ]; then
    log "Already up to date (commit: ${CURRENT_COMMIT})"
    log "Continuing to ensure Python dependencies are up to date..."
else
    log "Update available: ${CURRENT_COMMIT} -> ${REMOTE_COMMIT}"

    # Stash any local changes
    if ! git diff-index --quiet HEAD --; then
        log "Stashing local changes..."
        git stash save "Auto-stash before update at ${TIMESTAMP}"
    fi

    # Pull latest changes
    log "Pulling latest changes from main..."
    if ! git pull origin main; then
        log "ERROR: Failed to pull from main"
        exit 1
    fi

    log "Successfully updated to commit: $(git rev-parse HEAD)"
fi

# Activate virtual environment if it exists, otherwise create it
if [ -d "${VENV_DIR}" ]; then
    log "Activating virtual environment..."
    source "${VENV_DIR}/bin/activate"
else
    log "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
fi

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip

# Install/update requirements
if [ -f "${REPO_DIR}/requirements.txt" ]; then
    log "Installing/updating requirements from requirements.txt..."
    pip install -r "${REPO_DIR}/requirements.txt"
fi

# Keep py-libp2p synced to upstream main (latest commit)
# NOTE: VCS installs can be slow; avoid forcing a full reinstall on every restart.
LIBP2P_VCS_SPEC="libp2p @ git+https://github.com/libp2p/py-libp2p@main"
LIBP2P_STATE_DIR="${IPFS_ACCELERATE_STATE_DIR:-/tmp/ipfs_accelerate_state}"
LIBP2P_STAMP_FILE="${LIBP2P_STATE_DIR}/last_libp2p_reinstall.epoch"
LIBP2P_REINSTALL_INTERVAL_SECONDS="${IPFS_ACCELERATE_LIBP2P_REINSTALL_INTERVAL_SECONDS:-86400}"

mkdir -p "${LIBP2P_STATE_DIR}"

now_epoch=$(date +%s)
last_epoch=0
if [ -f "${LIBP2P_STAMP_FILE}" ]; then
    last_epoch=$(cat "${LIBP2P_STAMP_FILE}" 2>/dev/null || echo 0)
fi

should_force_reinstall=0
if [ "${IPFS_ACCELERATE_FORCE_LIBP2P_REINSTALL:-}" = "1" ]; then
    should_force_reinstall=1
else
    age=$((now_epoch - last_epoch))
    if [ "${last_epoch}" -le 0 ] || [ "${age}" -ge "${LIBP2P_REINSTALL_INTERVAL_SECONDS}" ]; then
        should_force_reinstall=1
    fi
fi

if [ "${should_force_reinstall}" -eq 1 ]; then
    log "Ensuring py-libp2p is installed from GitHub main (forced refresh)..."
    pip install --upgrade --force-reinstall "${LIBP2P_VCS_SPEC}"
    echo "${now_epoch}" > "${LIBP2P_STAMP_FILE}" || true
else
    log "Ensuring py-libp2p is installed from GitHub main (no force-reinstall)..."
    pip install --upgrade "${LIBP2P_VCS_SPEC}"
fi

# Install package in editable mode
if [ -f "${REPO_DIR}/setup.py" ] || [ -f "${REPO_DIR}/pyproject.toml" ]; then
    log "Installing package in editable mode..."
    pip install -e .
fi

log "Auto-update completed successfully!"

# Check if services need restarting
if systemctl is-active --quiet ipfs-accelerate.service; then
    log "Service ipfs-accelerate.service is running. Consider restarting it."
fi

if systemctl is-active --quiet ipfs-accelerate-mcp.service; then
    log "Service ipfs-accelerate-mcp.service is running. Consider restarting it."
fi

exit 0

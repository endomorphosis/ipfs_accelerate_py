#!/bin/bash
# Auto-update script for ipfs_accelerate_py
# Pulls from main branch and installs Python packages

set -e

# Configuration
REPO_DIR="/home/barberb/ipfs_accelerate_py"
VENV_DIR="${REPO_DIR}/.venv"
LOG_FILE="${REPO_DIR}/logs/auto-update.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create logs directory if it doesn't exist
mkdir -p "${REPO_DIR}/logs"

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
    exit 0
fi

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

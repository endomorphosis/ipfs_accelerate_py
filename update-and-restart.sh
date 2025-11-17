#!/bin/bash
# Auto-update script for IPFS Accelerate MCP Service
# Pulls latest changes from main branch and restarts the service

set -e

LOG_FILE="/tmp/ipfs-accelerate-update.log"
WORK_DIR="/home/barberb/ipfs_accelerate_py"

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

# Restart the systemd service
echo "$(date): Restarting ipfs-accelerate-mcp.service" | tee -a "$LOG_FILE"
systemctl --user restart ipfs-accelerate-mcp.service

# Wait a few seconds and check status
sleep 5
if systemctl --user is-active --quiet ipfs-accelerate-mcp.service; then
    echo "$(date): Service restarted successfully" | tee -a "$LOG_FILE"
else
    echo "$(date): ERROR - Service failed to restart!" | tee -a "$LOG_FILE"
    systemctl --user status ipfs-accelerate-mcp.service --no-pager | tee -a "$LOG_FILE"
    exit 1
fi

echo "$(date): Update complete" | tee -a "$LOG_FILE"

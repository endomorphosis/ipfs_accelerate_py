#!/bin/bash

# Quick Setup Script for Backup GitHub Actions Runner
# This script sets up a backup runner alongside an existing primary runner

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if main setup script exists
MAIN_SCRIPT="$SCRIPT_DIR/setup-github-runner.sh"
if [ ! -f "$MAIN_SCRIPT" ]; then
    error "Main setup script not found: $MAIN_SCRIPT"
fi

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Quick Setup Script for Backup GitHub Actions Runner"
    echo ""
    echo "Usage: $0 <github-token> [runner-name]"
    echo ""
    echo "Arguments:"
    echo "  github-token    GitHub Actions runner registration token (required)"
    echo "  runner-name     Optional custom name for the backup runner"
    echo ""
    echo "This script will:"
    echo "  - Install a backup GitHub Actions runner"
    echo "  - Set up the runner as a systemd service"
    echo "  - Configure monitoring"
    echo "  - Use the same labels as the primary runner with backup designation"
    echo ""
    echo "Example:"
    echo "  $0 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    echo "  $0 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX my-backup-runner"
    echo ""
    exit 0
fi

# Check arguments
if [ $# -lt 1 ]; then
    error "GitHub token is required. Use --help for usage information."
fi

TOKEN="$1"
RUNNER_NAME="${2:-$(hostname)-backup}"

log "Setting up backup GitHub Actions runner..."
log "Runner name: $RUNNER_NAME"
log "Using main setup script: $MAIN_SCRIPT"

# Check if primary runner exists
if [ -d "$HOME/actions-runner" ]; then
    log "Primary runner detected at ~/actions-runner"
else
    warn "Primary runner not found. This script is intended for backup runners."
    warn "Consider running the main setup script first for your primary runner."
fi

# Determine appropriate labels by checking existing runner or using defaults
LABELS="self-hosted,linux,x64,backup"

# Add Docker label if available
if command -v docker &> /dev/null && docker ps &> /dev/null; then
    LABELS="$LABELS,docker"
fi

# Add GPU labels if available
if command -v nvidia-smi &> /dev/null; then
    LABELS="$LABELS,cuda,gpu"
elif command -v rocm-smi &> /dev/null; then
    LABELS="$LABELS,rocm,gpu"
elif command -v intel_gpu_top &> /dev/null; then
    LABELS="$LABELS,openvino,gpu"
else
    LABELS="$LABELS,cpu-only"
fi

log "Using labels: $LABELS"

# Run the main setup script with backup configuration
log "Executing main setup script..."
"$MAIN_SCRIPT" \
    --token "$TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$LABELS" \
    --additional \
    --verbose

log "Backup runner setup completed!"

# Show status of both runners
echo ""
log "Runner Status Summary:"
echo "======================"

# Check primary runner
if systemctl is-active --quiet github-actions-runner 2>/dev/null; then
    log "✓ Primary runner (github-actions-runner) is running"
else
    warn "✗ Primary runner (github-actions-runner) is not running"
fi

# Check backup runner
if systemctl is-active --quiet github-actions-runner-backup 2>/dev/null; then
    log "✓ Backup runner (github-actions-runner-backup) is running"
else
    warn "✗ Backup runner (github-actions-runner-backup) is not running"
fi

echo ""
log "Useful commands for managing runners:"
log "  Primary runner status:  sudo systemctl status github-actions-runner"
log "  Backup runner status:   sudo systemctl status github-actions-runner-backup"
log "  Primary runner logs:    sudo journalctl -u github-actions-runner -f"
log "  Backup runner logs:     sudo journalctl -u github-actions-runner-backup -f"
log "  Monitor all runners:    $SCRIPT_DIR/monitor-runners.sh"

echo ""
log "Both runners should now appear in your GitHub repository settings under Actions > Runners."
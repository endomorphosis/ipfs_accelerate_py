#!/bin/bash
# Install updated systemd service files with auto-update support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Installing updated systemd service files..."
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script must be run with sudo"
    echo "Usage: sudo $0"
    exit 1
fi

# Copy service files
echo "Copying service files to /etc/systemd/system/..."
cp "${REPO_DIR}/ipfs-accelerate.service" /etc/systemd/system/
cp "${REPO_DIR}/ipfs-accelerate-mcp.service" /etc/systemd/system/
cp "${REPO_DIR}/containerized-runner-launcher.service" /etc/systemd/system/

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo ""
echo "Service files installed successfully!"
echo ""
echo "Available services:"
echo "  - ipfs-accelerate.service"
echo "  - ipfs-accelerate-mcp.service"
echo "  - containerized-runner-launcher.service"
echo ""
echo "To enable and start a service:"
echo "  sudo systemctl enable <service-name>"
echo "  sudo systemctl start <service-name>"
echo ""
echo "To check service status:"
echo "  sudo systemctl status <service-name>"
echo ""
echo "Note: Services will now automatically update from main branch before starting."

exit 0

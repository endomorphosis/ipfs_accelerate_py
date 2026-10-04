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

INSTALL_USER="${SUDO_USER:-}"
if [[ -z "${INSTALL_USER}" || "${INSTALL_USER}" == "root" ]]; then
    INSTALL_USER="$(logname 2>/dev/null || true)"
fi
if [[ -z "${INSTALL_USER}" || "${INSTALL_USER}" == "root" ]]; then
    echo "ERROR: Could not infer the target non-root user. Re-run as: sudo -u <user> sudo $0" >&2
    exit 2
fi

echo "Installing service files to /etc/systemd/system/ for user: ${INSTALL_USER}"

"${REPO_DIR}/deployments/systemd/install.sh" \
    --unit ipfs-accelerate.service \
    --unit ipfs-accelerate-mcp.service \
    --unit containerized-runner-launcher.service \
    --user "${INSTALL_USER}" \
    --skip-python-deps \
    --no-start

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

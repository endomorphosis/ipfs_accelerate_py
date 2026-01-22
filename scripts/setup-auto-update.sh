#!/bin/bash
# Quick setup script for auto-update system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

echo "======================================"
echo "IPFS Accelerate Auto-Update Setup"
echo "======================================"
echo ""

# Step 1: Test the auto-update script
echo "Step 1: Testing auto-update script..."
"${SCRIPT_DIR}/auto-update.sh"
echo "✓ Auto-update script works!"
echo ""

# Step 2: Setup cron job
echo "Step 2: Setting up cron job (every 6 hours)..."
"${SCRIPT_DIR}/setup-auto-update-cron.sh"
echo "✓ Cron job installed!"
echo ""

# Step 3: Install systemd services
echo "Step 3: Installing systemd service files..."
if [ "$EUID" -ne 0 ]; then
    echo "⚠ Need sudo privileges to install systemd services"
    echo "Please run: sudo ${SCRIPT_DIR}/install-updated-services.sh"
else
    "${SCRIPT_DIR}/install-updated-services.sh"
    echo "✓ Systemd services installed!"
fi
echo ""

echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Summary:"
echo "  ✓ Auto-update script: ${SCRIPT_DIR}/auto-update.sh"
echo "  ✓ Cron job: Every 6 hours"
echo "  ✓ Service integration: Updates before each service start"
echo ""
echo "Logs:"
echo "  - Auto-update: ${REPO_DIR}/logs/auto-update.log"
echo "  - Cron: ${REPO_DIR}/logs/auto-update-cron.log"
echo ""
echo "Next steps:"
if [ "$EUID" -ne 0 ]; then
    echo "  1. Run: sudo ${SCRIPT_DIR}/install-updated-services.sh"
    echo "  2. Enable and start services:"
else
    echo "  1. Enable and start services:"
fi
echo "     sudo systemctl enable ipfs-accelerate.service"
echo "     sudo systemctl start ipfs-accelerate.service"
echo ""
echo "Documentation: ${REPO_DIR}/AUTO_UPDATE_SYSTEM.md"
echo ""

exit 0

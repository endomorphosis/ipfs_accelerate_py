#!/bin/bash
#
# IPFS Accelerate Complete Setup
# This script installs the service and sets up monitoring
#

set -e

PROJECT_DIR="/home/barberb/ipfs_accelerate_py"
cd "$PROJECT_DIR"

echo "=========================================="
echo "IPFS Accelerate Complete Setup"
echo "=========================================="
echo

# Check if running as current user initially
if [[ $EUID -eq 0 ]]; then
   echo "Please run this script as your normal user (not root)"
   echo "The script will ask for sudo permissions when needed"
   exit 1
fi

# Step 1: Install the systemd service
echo "Step 1: Installing systemd service..."
sudo bash install-service.sh

# Step 2: Setup cron monitoring
echo
echo "Step 2: Setting up cron monitoring..."
sudo bash setup-cron.sh

# Step 3: Start the service
echo
echo "Step 3: Starting the service..."
sudo systemctl start ipfs-accelerate

# Step 4: Check service status
echo
echo "Step 4: Checking service status..."
sleep 3
sudo systemctl status ipfs-accelerate --no-pager -l

echo
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo
echo "Your IPFS Accelerate service is now:"
echo "  ✓ Installed as a systemd service"
echo "  ✓ Enabled to start on boot"
echo "  ✓ Monitored by a cron job (every 5 minutes)"
echo "  ✓ Running and accessible at http://localhost:9000"
echo
echo "Service management:"
echo "  Status:  sudo systemctl status ipfs-accelerate"
echo "  Stop:    sudo systemctl stop ipfs-accelerate"
echo "  Start:   sudo systemctl start ipfs-accelerate"
echo "  Restart: sudo systemctl restart ipfs-accelerate"
echo "  Logs:    sudo journalctl -u ipfs-accelerate -f"
echo
echo "Dashboard: http://localhost:9000/dashboard"
echo "Health:    http://localhost:9000/health"
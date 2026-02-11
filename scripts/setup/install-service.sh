#!/bin/bash
#
# IPFS Accelerate Service Installation Script
# This script installs and configures the IPFS Accelerate service to start on boot
#

set -e

# Configuration
SERVICE_NAME="ipfs-accelerate"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PROJECT_DIR="/home/barberb/ipfs_accelerate_py"
USER="barberb"
UNIT_SOURCE_FILE="${PROJECT_DIR}/deployments/systemd/${SERVICE_NAME}.service"

echo "Installing IPFS Accelerate Service..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory $PROJECT_DIR does not exist"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_DIR/.venv"
    echo "Please run: cd $PROJECT_DIR && python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Check if ipfs-accelerate command exists
if [ ! -f "$PROJECT_DIR/.venv/bin/ipfs-accelerate" ]; then
    echo "Error: ipfs-accelerate command not found in virtual environment"
    echo "Please run: cd $PROJECT_DIR && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Copy service file
echo "Installing systemd service file..."
if [ ! -f "$UNIT_SOURCE_FILE" ]; then
    echo "Error: Unit file not found at $UNIT_SOURCE_FILE"
    exit 1
fi

cp "$UNIT_SOURCE_FILE" "$SERVICE_FILE"

# Set correct permissions
chown root:root "$SERVICE_FILE"
chmod 644 "$SERVICE_FILE"

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service
echo "Enabling $SERVICE_NAME service..."
systemctl enable "$SERVICE_NAME"

# Create logs directory
mkdir -p /var/log/ipfs-accelerate
chown "$USER:$USER" /var/log/ipfs-accelerate

echo ""
echo "✓ IPFS Accelerate service installed successfully!"
echo ""
echo "Service management commands:"
echo "  Start service:    sudo systemctl start $SERVICE_NAME"
echo "  Stop service:     sudo systemctl stop $SERVICE_NAME"
echo "  Restart service:  sudo systemctl restart $SERVICE_NAME"
echo "  Service status:   sudo systemctl status $SERVICE_NAME"
echo "  View logs:        sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "The service will automatically start on boot."
echo "To start it now, run: sudo systemctl start $SERVICE_NAME"
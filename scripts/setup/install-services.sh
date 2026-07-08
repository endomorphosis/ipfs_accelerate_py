#!/bin/bash
# Install and configure IPFS Accelerate MCP Server and GitHub Autoscaler as systemd services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SYSTEMD_DIR="${REPO_DIR}/deployments/systemd"
USER=$(whoami)

echo "🚀 Installing IPFS Accelerate Services"
echo "========================================"
echo ""

# Check if running with sudo
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should NOT be run with sudo"
   echo "   Run as your normal user: ./install-services.sh"
   exit 1
fi

echo "📋 Pre-installation checks..."

# Check if virtual environment exists
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "❌ Virtual environment not found at $REPO_DIR/.venv"
    echo "   Please run: cd $REPO_DIR && python -m venv .venv && source .venv/bin/activate && pip install -e .[minimal,mcp]"
    exit 1
fi

# Check if ipfs-accelerate CLI is available
if [ ! -f "$REPO_DIR/.venv/bin/ipfs-accelerate" ]; then
    echo "❌ ipfs-accelerate CLI not found in virtual environment"
    echo "   Please reinstall the package: pip install -e .[minimal,mcp]"
    exit 1
fi

# Check if GitHub CLI is authenticated (for autoscaler)
if ! gh auth status &>/dev/null; then
    echo "⚠️  GitHub CLI not authenticated"
    echo "   Autoscaler will not work without authentication"
    echo "   Run: gh auth login"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Pre-installation checks passed"
echo ""

# Install MCP Server service
echo "📦 Installing IPFS Accelerate MCP Server service..."
sudo cp "$SYSTEMD_DIR/ipfs-accelerate.service" /etc/systemd/system/
sudo sed -i "s/^User=.*/User=${USER}/" /etc/systemd/system/ipfs-accelerate.service || true
sudo sed -i "s/^Group=.*/Group=${USER}/" /etc/systemd/system/ipfs-accelerate.service || true
sudo sed -i "s#%h/ipfs_accelerate_py#${REPO_DIR}#g" /etc/systemd/system/ipfs-accelerate.service || true
sudo sed -i "s#/root/ipfs_accelerate_py#${REPO_DIR}#g" /etc/systemd/system/ipfs-accelerate.service || true
sudo sed -i "s#/home/devel/ipfs_accelerate_py#${REPO_DIR}#g" /etc/systemd/system/ipfs-accelerate.service || true
sudo systemctl daemon-reload
sudo systemctl enable ipfs-accelerate.service
echo "✅ MCP Server service installed"

# Install GitHub Autoscaler service
echo "📦 Installing GitHub Actions Autoscaler service..."
sudo cp "$SYSTEMD_DIR/github-autoscaler.service" /etc/systemd/system/
sudo sed -i "s/^User=.*/User=${USER}/" /etc/systemd/system/github-autoscaler.service || true
sudo sed -i "s/^Group=.*/Group=${USER}/" /etc/systemd/system/github-autoscaler.service || true
sudo sed -i "s#%h/ipfs_accelerate_py#${REPO_DIR}#g" /etc/systemd/system/github-autoscaler.service || true
sudo sed -i "s#/root/ipfs_accelerate_py#${REPO_DIR}#g" /etc/systemd/system/github-autoscaler.service || true
sudo sed -i "s#/home/devel/ipfs_accelerate_py#${REPO_DIR}#g" /etc/systemd/system/github-autoscaler.service || true
sudo systemctl daemon-reload
sudo systemctl enable github-autoscaler.service
echo "✅ Autoscaler service installed"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📝 Service Management Commands:"
echo "   MCP Server:"
echo "     sudo systemctl start ipfs-accelerate"
echo "     sudo systemctl stop ipfs-accelerate"
echo "     sudo systemctl status ipfs-accelerate"
echo "     sudo journalctl -u ipfs-accelerate -f"
echo ""
echo "   GitHub Autoscaler:"
echo "     sudo systemctl start github-autoscaler"
echo "     sudo systemctl stop github-autoscaler"
echo "     sudo systemctl status github-autoscaler"
echo "     sudo journalctl -u github-autoscaler -f"
echo ""
echo "🚀 To start both services now:"
echo "   sudo systemctl start ipfs-accelerate"
echo "   sudo systemctl start github-autoscaler"
echo ""
echo "🌐 MCP Dashboard will be available at: http://localhost:9000/dashboard"
echo ""

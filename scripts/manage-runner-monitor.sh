#!/bin/bash

# Runner Coverage Monitor - Setup and Management Script
# Works cooperatively with github_autoscaler.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_FILE="$REPO_ROOT/deployments/systemd/runner-coverage-monitor.service"
MONITOR_SCRIPT="$REPO_ROOT/.github/scripts/monitor_runner_coverage.sh"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Runner Coverage Monitor - Setup and Management

This monitor ensures repositories updated in the last 24 hours have at least
one active runner. It works cooperatively with github_autoscaler.py:
  - If autoscaler is running: Monitor reports status but lets autoscaler manage
  - If autoscaler is not running: Monitor will start runners as needed

Usage: $0 [COMMAND]

Commands:
  install       Install the monitoring service
  uninstall     Remove the monitoring service
  start         Start the monitoring service
  stop          Stop the monitoring service
  restart       Restart the monitoring service
  status        Check service status
  logs          View service logs
  test          Run a single check (no changes)
  enable        Enable service to start on boot
  disable       Disable service auto-start

Examples:
  $0 install      # Install and start the service
  $0 status       # Check current status
  $0 test         # Run a one-time check

EOF
}

install_service() {
    echo -e "${BLUE}Installing Runner Coverage Monitor...${NC}"
    
    # Check if monitor script exists
    if [ ! -f "$MONITOR_SCRIPT" ]; then
        echo -e "${RED}ERROR: Monitor script not found: $MONITOR_SCRIPT${NC}"
        exit 1
    fi
    
    # Make script executable
    chmod +x "$MONITOR_SCRIPT"
    echo -e "${GREEN}✓${NC} Monitor script is executable"
    
    # Check if systemd user directory exists
    SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SYSTEMD_USER_DIR"
    
    # Copy service file
    if [ -f "$SERVICE_FILE" ]; then
        # Replace %i with actual username
        USERNAME=$(whoami)
        sed "s/%i/$USERNAME/g" "$SERVICE_FILE" > "$SYSTEMD_USER_DIR/runner-coverage-monitor.service"
        echo -e "${GREEN}✓${NC} Service file installed"
    else
        echo -e "${RED}ERROR: Service file not found: $SERVICE_FILE${NC}"
        exit 1
    fi
    
    # Reload systemd
    systemctl --user daemon-reload
    echo -e "${GREEN}✓${NC} Systemd daemon reloaded"
    
    # Enable and start service
    systemctl --user enable runner-coverage-monitor.service
    systemctl --user start runner-coverage-monitor.service
    echo -e "${GREEN}✓${NC} Service enabled and started"
    
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "The monitor is now running and will:"
    echo "  • Check runner coverage every 5 minutes"
    echo "  • Work cooperatively with github_autoscaler.py if running"
    echo "  • Start runners only if autoscaler is not active"
    echo ""
    echo "View status: $0 status"
    echo "View logs:   $0 logs"
}

uninstall_service() {
    echo -e "${BLUE}Uninstalling Runner Coverage Monitor...${NC}"
    
    # Stop and disable service
    systemctl --user stop runner-coverage-monitor.service 2>/dev/null || true
    systemctl --user disable runner-coverage-monitor.service 2>/dev/null || true
    
    # Remove service file
    rm -f "$HOME/.config/systemd/user/runner-coverage-monitor.service"
    
    # Reload systemd
    systemctl --user daemon-reload
    
    echo -e "${GREEN}✓${NC} Service uninstalled"
}

case "${1:-}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        systemctl --user start runner-coverage-monitor.service
        echo -e "${GREEN}✓${NC} Service started"
        ;;
    stop)
        systemctl --user stop runner-coverage-monitor.service
        echo -e "${GREEN}✓${NC} Service stopped"
        ;;
    restart)
        systemctl --user restart runner-coverage-monitor.service
        echo -e "${GREEN}✓${NC} Service restarted"
        ;;
    status)
        systemctl --user status runner-coverage-monitor.service
        ;;
    logs)
        journalctl --user -u runner-coverage-monitor.service -f
        ;;
    test)
        echo -e "${BLUE}Running one-time coverage check...${NC}"
        "$MONITOR_SCRIPT" --once
        ;;
    enable)
        systemctl --user enable runner-coverage-monitor.service
        echo -e "${GREEN}✓${NC} Service enabled (will start on boot)"
        ;;
    disable)
        systemctl --user disable runner-coverage-monitor.service
        echo -e "${GREEN}✓${NC} Service disabled"
        ;;
    *)
        usage
        exit 1
        ;;
esac

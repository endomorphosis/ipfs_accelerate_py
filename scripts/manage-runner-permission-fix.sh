#!/bin/bash

# Management script for runner permission fix timer
# This installs and manages the systemd timer for automatic runner cleanup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_FILE="${PROJECT_ROOT}/deployments/systemd/runner-permission-fix.service"
TIMER_FILE="${PROJECT_ROOT}/deployments/systemd/runner-permission-fix.timer"

show_usage() {
    cat << EOF
Usage: $0 {install|uninstall|status|enable|disable|restart|logs}

Manage GitHub Actions runner permission fix timer

Commands:
  install    Install the systemd timer (runs every 15 minutes)
  uninstall  Remove the systemd timer
  status     Show timer and service status
  enable     Enable timer to start on boot
  disable    Disable timer
  restart    Restart the timer
  logs       Show recent logs from permission fix runs
  test       Run permission fix manually

Examples:
  $0 install    # Install and start the timer
  $0 status     # Check if timer is running
  $0 logs       # View recent fix activity

EOF
    exit 1
}

install_timer() {
    echo "📦 Installing runner permission fix timer..."
    
    # Check files exist
    if [ ! -f "$SERVICE_FILE" ] || [ ! -f "$TIMER_FILE" ]; then
        echo "❌ Service or timer file not found!"
        exit 1
    fi
    
    # Copy files to systemd directory
    sudo cp "$SERVICE_FILE" /etc/systemd/system/
    sudo cp "$TIMER_FILE" /etc/systemd/system/
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable and start timer
    sudo systemctl enable runner-permission-fix.timer
    sudo systemctl start runner-permission-fix.timer
    
    echo "✅ Timer installed and started"
    echo ""
    status_timer
}

uninstall_timer() {
    echo "🗑️  Uninstalling runner permission fix timer..."
    
    # Stop and disable timer
    sudo systemctl stop runner-permission-fix.timer 2>/dev/null || true
    sudo systemctl disable runner-permission-fix.timer 2>/dev/null || true
    
    # Remove files
    sudo rm -f /etc/systemd/system/runner-permission-fix.service
    sudo rm -f /etc/systemd/system/runner-permission-fix.timer
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    echo "✅ Timer uninstalled"
}

status_timer() {
    echo "📊 Timer Status:"
    sudo systemctl status runner-permission-fix.timer --no-pager || true
    echo ""
    echo "⏰ Next scheduled runs:"
    sudo systemctl list-timers runner-permission-fix.timer --no-pager || true
    echo ""
    echo "📋 Recent service runs:"
    sudo systemctl status runner-permission-fix.service --no-pager -l || true
}

show_logs() {
    echo "📜 Recent permission fix logs:"
    sudo journalctl -u runner-permission-fix.service -n 50 --no-pager -l
}

enable_timer() {
    echo "🔄 Enabling timer..."
    sudo systemctl enable runner-permission-fix.timer
    sudo systemctl start runner-permission-fix.timer
    echo "✅ Timer enabled"
    status_timer
}

disable_timer() {
    echo "⏸️  Disabling timer..."
    sudo systemctl stop runner-permission-fix.timer
    sudo systemctl disable runner-permission-fix.timer
    echo "✅ Timer disabled"
}

restart_timer() {
    echo "🔄 Restarting timer..."
    sudo systemctl restart runner-permission-fix.timer
    echo "✅ Timer restarted"
    status_timer
}

test_run() {
    echo "🧪 Running permission fix manually..."
    bash "${PROJECT_ROOT}/.github/scripts/fix_runner_permissions.sh"
}

# Main execution
case "${1:-}" in
    install)
        install_timer
        ;;
    uninstall)
        uninstall_timer
        ;;
    status)
        status_timer
        ;;
    enable)
        enable_timer
        ;;
    disable)
        disable_timer
        ;;
    restart)
        restart_timer
        ;;
    logs)
        show_logs
        ;;
    test)
        test_run
        ;;
    *)
        show_usage
        ;;
esac

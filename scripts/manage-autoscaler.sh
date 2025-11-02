#!/bin/bash
#
# GitHub Actions Runner Autoscaler Service Management Script
#
# This script helps install, start, stop, and manage the GitHub Actions
# autoscaler service with proper Docker isolation and architecture filtering.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_NAME="github-autoscaler"
SERVICE_FILE="${REPO_DIR}/deployments/systemd/github-autoscaler.service"
USER_SERVICE_DIR="$HOME/.config/systemd/user"
SYSTEM_SERVICE_DIR="/etc/systemd/system"

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Function to show usage
show_usage() {
    echo "GitHub Actions Runner Autoscaler Service Manager"
    echo ""
    echo "Usage: $0 {install|start|stop|restart|status|uninstall|logs} [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install     Install the autoscaler service"
    echo "  start       Start the autoscaler service"
    echo "  stop        Stop the autoscaler service"
    echo "  restart     Restart the autoscaler service"
    echo "  status      Show service status"
    echo "  uninstall   Uninstall the autoscaler service"
    echo "  logs        Show service logs (tail -f)"
    echo ""
    echo "Options:"
    echo "  --user      Install as user service (default)"
    echo "  --system    Install as system service (requires sudo)"
    echo ""
    echo "Examples:"
    echo "  $0 install --user"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 logs"
    echo ""
}

# Check if GitHub CLI is authenticated
check_gh_auth() {
    info "Checking GitHub CLI authentication..."
    if ! command -v gh &> /dev/null; then
        error "GitHub CLI (gh) is not installed. Install it first: https://cli.github.com/"
    fi
    
    if ! gh auth status &> /dev/null; then
        error "Not authenticated with GitHub CLI. Run: gh auth login"
    fi
    
    log "✓ GitHub CLI is authenticated"
}

# Check system architecture
check_architecture() {
    local arch=$(uname -m)
    info "System architecture: $arch"
    
    case $arch in
        x86_64|amd64)
            log "✓ Detected x64 architecture - will filter out ARM64-specific workflows"
            ;;
        aarch64|arm64)
            log "✓ Detected ARM64 architecture - will filter out x64-specific workflows"
            ;;
        *)
            warn "Unknown architecture: $arch - architecture filtering may not work correctly"
            ;;
    esac
}

# Check Docker availability
check_docker() {
    info "Checking Docker availability..."
    if ! command -v docker &> /dev/null; then
        warn "Docker is not installed - runners will need Docker for isolation"
        warn "Install Docker: https://docs.docker.com/engine/install/"
        return 1
    fi
    
    if ! docker ps &> /dev/null; then
        warn "Docker is installed but not accessible - check permissions"
        warn "Add user to docker group: sudo usermod -aG docker $USER"
        return 1
    fi
    
    log "✓ Docker is available and accessible"
    return 0
}

# Install service
install_service() {
    local use_system=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --system)
                use_system=true
                shift
                ;;
            --user)
                use_system=false
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log "Installing GitHub Actions Runner Autoscaler service..."
    
    # Pre-installation checks
    check_gh_auth
    check_architecture
    check_docker || warn "Docker not available - continue anyway"
    
    if [ "$use_system" = true ]; then
        info "Installing as system service..."
        
        # Copy service file to system directory
        sudo cp "$SERVICE_FILE" "$SYSTEM_SERVICE_DIR/$SERVICE_NAME.service"
        
        # Update service file to use actual user
        sudo sed -i "s/%i/$USER/g" "$SYSTEM_SERVICE_DIR/$SERVICE_NAME.service"
        
        # Reload systemd
        sudo systemctl daemon-reload
        
        # Enable service
        sudo systemctl enable "$SERVICE_NAME"
        
        log "✓ Service installed as system service: $SERVICE_NAME"
        log "  Start with: sudo systemctl start $SERVICE_NAME"
    else
        info "Installing as user service..."
        
        # Create user service directory if it doesn't exist
        mkdir -p "$USER_SERVICE_DIR"
        
        # Copy service file to user directory
        cp "$SERVICE_FILE" "$USER_SERVICE_DIR/$SERVICE_NAME.service"
        
        # Update service file to use actual user
        sed -i "s/%i/$USER/g" "$USER_SERVICE_DIR/$SERVICE_NAME.service"
        
        # Reload systemd
        systemctl --user daemon-reload
        
        # Enable service
        systemctl --user enable "$SERVICE_NAME"
        
        log "✓ Service installed as user service: $SERVICE_NAME"
        log "  Start with: systemctl --user start $SERVICE_NAME"
    fi
    
    info "Service configuration:"
    info "  Architecture filtering: enabled"
    info "  Docker isolation: required for runners"
    info "  Poll interval: 60 seconds"
    info "  Max runners: system CPU cores"
}

# Start service
start_service() {
    log "Starting GitHub Actions Runner Autoscaler service..."
    
    if systemctl --user is-enabled "$SERVICE_NAME" &> /dev/null; then
        systemctl --user start "$SERVICE_NAME"
        log "✓ User service started"
    elif sudo systemctl is-enabled "$SERVICE_NAME" &> /dev/null 2>&1; then
        sudo systemctl start "$SERVICE_NAME"
        log "✓ System service started"
    else
        error "Service is not installed. Run: $0 install"
    fi
    
    # Wait a moment and check status
    sleep 2
    status_service
}

# Stop service
stop_service() {
    log "Stopping GitHub Actions Runner Autoscaler service..."
    
    if systemctl --user is-active "$SERVICE_NAME" &> /dev/null; then
        systemctl --user stop "$SERVICE_NAME"
        log "✓ User service stopped"
    elif sudo systemctl is-active "$SERVICE_NAME" &> /dev/null 2>&1; then
        sudo systemctl stop "$SERVICE_NAME"
        log "✓ System service stopped"
    else
        warn "Service is not running"
    fi
}

# Restart service
restart_service() {
    log "Restarting GitHub Actions Runner Autoscaler service..."
    stop_service
    sleep 1
    start_service
}

# Show service status
status_service() {
    log "GitHub Actions Runner Autoscaler service status:"
    echo ""
    
    if systemctl --user is-enabled "$SERVICE_NAME" &> /dev/null; then
        systemctl --user status "$SERVICE_NAME" --no-pager || true
    elif sudo systemctl is-enabled "$SERVICE_NAME" &> /dev/null 2>&1; then
        sudo systemctl status "$SERVICE_NAME" --no-pager || true
    else
        warn "Service is not installed"
    fi
}

# Show service logs
show_logs() {
    log "Showing GitHub Actions Runner Autoscaler service logs..."
    log "Press Ctrl+C to stop"
    echo ""
    
    if systemctl --user is-enabled "$SERVICE_NAME" &> /dev/null; then
        journalctl --user -u "$SERVICE_NAME" -f
    elif sudo systemctl is-enabled "$SERVICE_NAME" &> /dev/null 2>&1; then
        sudo journalctl -u "$SERVICE_NAME" -f
    else
        error "Service is not installed"
    fi
}

# Uninstall service
uninstall_service() {
    log "Uninstalling GitHub Actions Runner Autoscaler service..."
    
    # Stop service first
    stop_service || true
    
    if systemctl --user is-enabled "$SERVICE_NAME" &> /dev/null; then
        systemctl --user disable "$SERVICE_NAME"
        rm -f "$USER_SERVICE_DIR/$SERVICE_NAME.service"
        systemctl --user daemon-reload
        log "✓ User service uninstalled"
    elif sudo systemctl is-enabled "$SERVICE_NAME" &> /dev/null 2>&1; then
        sudo systemctl disable "$SERVICE_NAME"
        sudo rm -f "$SYSTEM_SERVICE_DIR/$SERVICE_NAME.service"
        sudo systemctl daemon-reload
        log "✓ System service uninstalled"
    else
        warn "Service is not installed"
    fi
}

# Main script
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    local command=$1
    shift
    
    case $command in
        install)
            install_service "$@"
            ;;
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            status_service
            ;;
        logs)
            show_logs
            ;;
        uninstall)
            uninstall_service
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"

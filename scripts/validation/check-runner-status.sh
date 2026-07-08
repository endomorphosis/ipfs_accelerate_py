#!/bin/bash

# GitHub Actions Runner and Autoscaler Status Check
# Provides comprehensive status of both services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Resolve repo root so this script works from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GH_API_CACHED=(python3 "$SCRIPT_DIR/scripts/utils/gh_api_cached.py")

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  GitHub Actions Runner & Autoscaler Status${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check GitHub Actions Runner
echo -e "${BLUE}[1] GitHub Actions Runner${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RUNNER_SERVICE="actions.runner.endomorphosis-ipfs_accelerate_py.fent-reactor.service"

if systemctl is-active --quiet "$RUNNER_SERVICE" 2>/dev/null; then
    echo -e "Status: ${GREEN}✓ ACTIVE${NC}"
    
    # Get runner details
    if [ -f "$HOME/actions-runner/.runner" ]; then
        echo -e "\nRunner Details:"
        RUNNER_NAME=$(jq -r '.agentName' < "$HOME/actions-runner/.runner" 2>/dev/null)
        RUNNER_ID=$(jq -r '.agentId' < "$HOME/actions-runner/.runner" 2>/dev/null)
        echo "  Name: $RUNNER_NAME"
        echo "  Agent ID: $RUNNER_ID"
    fi
    
    # Show recent activity
    echo -e "\nRecent Activity:"
    journalctl -u "$RUNNER_SERVICE" --since "5 minutes ago" --no-pager -n 3 2>/dev/null | grep "Listening for Jobs" | tail -1 || echo "  Waiting for jobs..."
else
    echo -e "Status: ${RED}✗ INACTIVE${NC}"
    echo "  To start: sudo systemctl start $RUNNER_SERVICE"
fi

echo ""

# Check GitHub Autoscaler
echo -e "${BLUE}[2] GitHub Actions Autoscaler${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

AUTOSCALER_SERVICE="github-autoscaler.service"

if systemctl --user is-active --quiet "$AUTOSCALER_SERVICE" 2>/dev/null; then
    echo -e "Status: ${GREEN}✓ ACTIVE${NC}"
    
    # Get autoscaler configuration
    echo -e "\nConfiguration:"
    journalctl --user -u "$AUTOSCALER_SERVICE" --no-pager | grep "Poll interval" | tail -1 | sed 's/.*INFO - /  /'
    journalctl --user -u "$AUTOSCALER_SERVICE" --no-pager | grep "Max runners" | tail -1 | sed 's/.*INFO - /  /'
    journalctl --user -u "$AUTOSCALER_SERVICE" --no-pager | grep "System architecture" | tail -1 | sed 's/.*INFO - /  /'
    journalctl --user -u "$AUTOSCALER_SERVICE" --no-pager | grep "Architecture filtering" | tail -1 | sed 's/.*INFO - /  /'
    
    # Show recent monitoring activity
    echo -e "\nRecent Activity:"
    journalctl --user -u "$AUTOSCALER_SERVICE" --since "2 minutes ago" --no-pager | grep -E "(Check #|Found.*repositories|No repositories)" | tail -3 | sed 's/.*INFO - /  /'
else
    echo -e "Status: ${RED}✗ INACTIVE${NC}"
    echo "  To start: systemctl --user start $AUTOSCALER_SERVICE"
fi

echo ""

# Check GitHub CLI authentication
echo -e "${BLUE}[3] GitHub CLI Authentication${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if gh auth status &> /dev/null; then
    echo -e "Status: ${GREEN}✓ AUTHENTICATED${NC}"
    GH_USER=$(${GH_API_CACHED[@]} user --jq .login 2>/dev/null)
    echo "  User: $GH_USER"
else
    echo -e "Status: ${YELLOW}⚠ AUTHENTICATION ISSUES${NC}"
    echo "  Note: Token may need refresh but services are still running"
    echo "  To re-authenticate: gh auth login"
fi

echo ""

# System Resources
echo -e "${BLUE}[4] System Resources${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3 " / " $2}')"
echo "Disk (root): $(df -h / | awk 'NR==2 {print $3 " / " $2 " (" $5 " used)"}')"

echo ""

# Service Management Commands
echo -e "${BLUE}[5] Quick Commands${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Runner:"
echo "  Status:  sudo systemctl status $RUNNER_SERVICE"
echo "  Restart: sudo systemctl restart $RUNNER_SERVICE"
echo "  Logs:    sudo journalctl -u $RUNNER_SERVICE -f"
echo ""
echo "Autoscaler:"
echo "  Status:  systemctl --user status $AUTOSCALER_SERVICE"
echo "  Restart: systemctl --user restart $AUTOSCALER_SERVICE"
echo "  Logs:    journalctl --user -u $AUTOSCALER_SERVICE -f"
echo "  Manage:  ./scripts/manage-autoscaler.sh"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

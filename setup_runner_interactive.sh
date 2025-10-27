#!/bin/bash

# GitHub Actions Runner Quick Setup
# This script provides an interactive setup experience

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                GitHub Actions Runner Setup                  ║
║                  for ipfs_accelerate_py                     ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo "This script will help you set up a backup GitHub Actions runner."
echo

# Check if already installed
if systemctl list-units --type=service --state=active | grep -q "actions.runner"; then
    echo -e "${YELLOW}⚠️ GitHub Actions runner service is already running.${NC}"
    echo
    echo "Existing runners:"
    systemctl list-units --type=service --state=active | grep "actions.runner" || true
    echo
    read -p "Do you want to continue and install another runner? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 0
    fi
fi

# Check if token is already set
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo -e "${GREEN}✅ GITHUB_TOKEN is already set${NC}"
else
    echo -e "${YELLOW}Please set up your GitHub Personal Access Token:${NC}"
    echo
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Click 'Generate new token (classic)'"
    echo "3. Select scopes: 'repo' and 'admin:repo_hook'"
    echo "4. Copy the generated token"
    echo
    read -p "Enter your GitHub token: " -s github_token
    echo
    export GITHUB_TOKEN="$github_token"
fi

# Confirm repository
echo -e "${BLUE}Repository: ${NC}endomorphosis/ipfs_accelerate_py"
read -p "Is this correct? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    read -p "Enter repository (owner/repo): " repo
    export GITHUB_REPOSITORY="$repo"
fi

echo
echo -e "${GREEN}Starting installation...${NC}"
echo

# Run the main setup script
if [[ -f "./setup_github_actions_runner.sh" ]]; then
    ./setup_github_actions_runner.sh
else
    echo -e "${RED}❌ setup_github_actions_runner.sh not found${NC}"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

echo
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo
echo -e "${CYAN}What's next?${NC}"
echo "1. Check your runner status on GitHub:"
echo "   https://github.com/${GITHUB_REPOSITORY:-endomorphosis/ipfs_accelerate_py}/settings/actions/runners"
echo
echo "2. Test your runner with a workflow"
echo
echo "3. Monitor logs with:"
echo "   sudo journalctl -u actions.runner.* -f"
echo
echo -e "${GREEN}Happy CI/CD! 🚀${NC}"
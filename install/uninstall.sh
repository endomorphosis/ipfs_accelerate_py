#!/usr/bin/env bash
#
# Uninstaller for ipfs_accelerate_py Cache Infrastructure
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CACHE_DIR="${CACHE_DIR:-$HOME/.cache/ipfs_accelerate_py}"

echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║                                                              ║${NC}"
echo -e "${YELLOW}║   ipfs_accelerate_py Cache Infrastructure Uninstaller        ║${NC}"
echo -e "${YELLOW}║                                                              ║${NC}"
echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Ask for confirmation
read -p "Are you sure you want to uninstall ipfs_accelerate_py cache infrastructure? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}✓${NC} Uninstall cancelled"
    exit 0
fi

echo -e "${YELLOW}⚠${NC} Uninstalling..."

# Uninstall Python package
if command -v pip >/dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Uninstalling Python package..."
    pip uninstall -y ipfs_accelerate_py 2>/dev/null || true
fi

# Remove cache directory
if [ -d "$CACHE_DIR" ]; then
    read -p "Remove cache directory ($CACHE_DIR)? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}→${NC} Removing cache directory..."
        rm -rf "$CACHE_DIR"
        echo -e "${GREEN}✓${NC} Cache directory removed"
    fi
fi

# Remove environment variables from shell configs
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rc" ] && grep -q "IPFS_ACCELERATE_CACHE_DIR" "$rc"; then
        echo -e "${YELLOW}→${NC} Removing environment variables from $rc..."
        sed -i.bak '/# ipfs_accelerate_py cache configuration/,/export IPFS_ACCELERATE_CACHE_TTL/d' "$rc"
        echo -e "${GREEN}✓${NC} Environment variables removed from $rc"
    fi
done

# Remove virtual environment
if [ -d ".venv" ]; then
    read -p "Remove virtual environment (.venv)? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}→${NC} Removing virtual environment..."
        rm -rf ".venv"
        echo -e "${GREEN}✓${NC} Virtual environment removed"
    fi
fi

echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║   ✓ Uninstallation Complete!                                 ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

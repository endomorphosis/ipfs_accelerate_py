#!/bin/bash

# Script to update GitHub CLI to the latest version from official releases
# and re-authenticate if needed

set -e

echo "🔄 GitHub CLI Update Script"
echo "=============================="
echo ""

# Get current version
if command -v gh &> /dev/null; then
    CURRENT_VERSION=$(gh --version 2>/dev/null | head -1 | awk '{print $3}' || echo "unknown")
    echo "Current version: $CURRENT_VERSION"
else
    CURRENT_VERSION="not installed"
    echo "GitHub CLI is not currently installed"
fi

# Get latest version from GitHub
echo ""
echo "📡 Checking for latest version..."
LATEST_VERSION=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | grep '"tag_name"' | cut -d'"' -f4 | sed 's/v//')

if [ -z "$LATEST_VERSION" ]; then
    echo "❌ Failed to fetch latest version from GitHub"
    exit 1
fi

echo "Latest version: $LATEST_VERSION"
echo ""

# Check if update is needed
if [ "$CURRENT_VERSION" = "$LATEST_VERSION" ]; then
    echo "✅ GitHub CLI is already up to date!"
    gh auth status 2>&1
    exit 0
fi

# Download and install latest version
echo "📥 Downloading gh CLI v$LATEST_VERSION..."
cd /tmp
rm -rf gh_*

DOWNLOAD_URL="https://github.com/cli/cli/releases/download/v${LATEST_VERSION}/gh_${LATEST_VERSION}_linux_amd64.tar.gz"
echo "   URL: $DOWNLOAD_URL"

if ! curl -L -o gh_latest.tar.gz "$DOWNLOAD_URL"; then
    echo "❌ Failed to download gh CLI"
    exit 1
fi

echo "📦 Extracting..."
tar -xzf gh_latest.tar.gz

echo "📥 Installing to ~/.local/bin/gh..."
mkdir -p ~/.local/bin
install -m 755 "gh_${LATEST_VERSION}_linux_amd64/bin/gh" ~/.local/bin/gh

# Cleanup
rm -rf gh_*

# Verify installation
echo ""
echo "✅ Installation complete!"
NEW_VERSION=$(~/.local/bin/gh --version 2>/dev/null | head -1 || echo "Installation failed")
echo "   $NEW_VERSION"
echo ""

# Check authentication status
echo "🔐 Checking authentication status..."
if gh auth status &>/dev/null; then
    echo "✅ GitHub CLI is authenticated"
    gh auth status 2>&1
else
    echo "⚠️  GitHub CLI is not authenticated or token is invalid"
    echo ""
    echo "To authenticate, run one of:"
    echo "  1. gh auth login                    # Interactive login"
    echo "  2. gh auth login --with-token < token.txt  # Token from file"
    echo "  3. export GH_TOKEN=ghp_xxxxx        # Set environment variable"
    echo ""
    echo "To generate a token:"
    echo "  https://github.com/settings/tokens/new?scopes=repo,workflow,admin:org"
fi

echo ""
echo "📚 Useful commands:"
echo "  gh --version       # Check version"
echo "  gh auth status     # Check authentication"
echo "  gh auth login      # Re-authenticate"
echo "  gh auth refresh    # Refresh token"

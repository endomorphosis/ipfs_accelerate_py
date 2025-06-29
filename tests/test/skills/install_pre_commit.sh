#!/bin/bash
# Script to install the pre-commit hook for the HuggingFace test generator

# Directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Git hooks directory
GIT_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$GIT_ROOT/.git/hooks"

# Pre-commit hook source and destination
HOOK_SOURCE="$SCRIPT_DIR/pre-commit"
HOOK_DEST="$HOOKS_DIR/pre-commit"

# Check if git repository exists
if [ ! -d "$GIT_ROOT/.git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install the pre-commit hook
echo "📋 Installing pre-commit hook for HuggingFace test generator..."

if [ -f "$HOOK_DEST" ]; then
    echo "⚠️ Existing pre-commit hook found. Creating backup..."
    cp "$HOOK_DEST" "$HOOK_DEST.bak"
    echo "✅ Backup created: $HOOK_DEST.bak"
fi

cp "$HOOK_SOURCE" "$HOOK_DEST"
chmod +x "$HOOK_DEST"

echo "✅ Pre-commit hook installed successfully!"
echo "🧪 The hook will run test generator validation before each commit."

# Output the installed hook path
echo "📌 Hook installed at: $HOOK_DEST"
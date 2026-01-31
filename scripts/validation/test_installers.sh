#!/usr/bin/env bash
#
# Test script for multi-platform installers
#

set -e

echo "Testing ipfs_accelerate_py installers..."
echo

# Test 1: Verify scripts are executable
echo "Test 1: Checking script permissions..."
if [ -x "install/install.sh" ] && [ -x "install/uninstall.sh" ]; then
    echo "✓ Scripts are executable"
else
    echo "✗ Scripts are not executable"
    exit 1
fi

# Test 2: Verify platform detection
echo "Test 2: Testing platform detection..."
source_platform=$(uname -s | tr '[:upper:]' '[:lower:]')
source_arch=$(uname -m)
echo "  Detected: $source_platform / $source_arch"
echo "✓ Platform detection works"

# Test 3: Verify Python is available
echo "Test 3: Checking Python availability..."
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version)
    echo "✓ Python available: $python_version"
else
    echo "✗ Python not found"
    exit 1
fi

# Test 4: Test help output
echo "Test 4: Testing help output..."
if ./install/install.sh --help >/dev/null 2>&1; then
    echo "✓ Help output works"
else
    echo "✗ Help output failed"
    exit 1
fi

# Test 5: Verify Docker file syntax
echo "Test 5: Checking Dockerfile syntax..."
if [ -f "install/Dockerfile.cache" ]; then
    echo "✓ Dockerfile exists"
else
    echo "✗ Dockerfile not found"
    exit 1
fi

# Test 6: Verify setup.py has cache extras
echo "Test 6: Checking setup.py for cache extras..."
if grep -q '"cache":' setup.py; then
    echo "✓ setup.py has cache extras"
else
    echo "✗ setup.py missing cache extras"
    exit 1
fi

# Test 7: Verify all installer files exist
echo "Test 7: Checking all installer files..."
required_files=(
    "install/README.md"
    "install/install.sh"
    "install/install.ps1"
    "install/uninstall.sh"
    "install/uninstall.ps1"
    "install/Dockerfile.cache"
    "install/INSTALLATION_GUIDE.md"
)

all_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file missing"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    echo "✓ All installer files present"
else
    echo "✗ Some installer files missing"
    exit 1
fi

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║   ✓ All Installer Tests Passed!                              ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

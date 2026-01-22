#!/bin/bash
# Install P2P Cache Dependencies
# This script installs all required dependencies for P2P cache functionality

set -e

echo "=============================================="
echo "P2P Cache Dependencies Installer"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher required (found $PYTHON_VERSION)"
    exit 1
fi
print_success "Python $PYTHON_VERSION detected"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 not found. Please install pip first."
    exit 1
fi
print_success "pip3 available"

# Install dependencies
echo ""
print_info "Installing P2P cache dependencies..."
echo ""

# Core dependencies
DEPENDENCIES=(
    "libp2p>=0.4.0"
    "pymultihash>=0.8.2"
    "py-multiformats-cid"
    "cryptography"
)

for dep in "${DEPENDENCIES[@]}"; do
    print_info "Installing $dep..."
    if pip3 install --quiet "$dep"; then
        print_success "$dep installed"
    else
        print_error "Failed to install $dep"
        print_warning "Continuing with other dependencies..."
    fi
done

# Verify installations
echo ""
print_info "Verifying installations..."
echo ""

# Check libp2p
if python3 -c "import libp2p" 2>/dev/null; then
    LIBP2P_VERSION=$(python3 -c "import libp2p; print(getattr(libp2p, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    print_success "libp2p installed (version: $LIBP2P_VERSION)"
else
    print_error "libp2p not found"
    INSTALL_FAILED=1
fi

# Check pymultihash
if python3 -c "import multihash" 2>/dev/null || python3 -c "import pymultihash" 2>/dev/null; then
    print_success "pymultihash installed"
else
    print_error "pymultihash not found"
    INSTALL_FAILED=1
fi

# Check multiformats
if python3 -c "from multiformats import CID" 2>/dev/null; then
    print_success "py-multiformats-cid installed"
else
    print_error "py-multiformats-cid not found"
    INSTALL_FAILED=1
fi

# Check cryptography
if python3 -c "import cryptography" 2>/dev/null; then
    CRYPTO_VERSION=$(python3 -c "import cryptography; print(cryptography.__version__)" 2>/dev/null)
    print_success "cryptography installed (version: $CRYPTO_VERSION)"
else
    print_error "cryptography not found"
    INSTALL_FAILED=1
fi

# Run diagnostic test if available
echo ""
if [ -f "test_docker_runner_cache_connectivity.py" ]; then
    print_info "Running diagnostic test..."
    echo ""
    if python3 test_docker_runner_cache_connectivity.py; then
        print_success "Diagnostic test passed!"
    else
        print_warning "Diagnostic test failed. Review errors above."
    fi
else
    print_warning "Diagnostic test not found (test_docker_runner_cache_connectivity.py)"
    print_info "You can run it manually when available."
fi

# Summary
echo ""
echo "=============================================="
if [ -n "$INSTALL_FAILED" ]; then
    print_error "Installation completed with errors"
    echo ""
    print_info "To fix installation issues:"
    echo "  1. Check pip3 is up to date: pip3 install --upgrade pip"
    echo "  2. Install build dependencies: sudo apt-get install build-essential python3-dev"
    echo "  3. Try installing failed packages individually with verbose output:"
    echo "     pip3 install -v <package-name>"
    exit 1
else
    print_success "All dependencies installed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Configure environment variables:"
    echo "     export CACHE_ENABLE_P2P=true"
    echo "     export CACHE_LISTEN_PORT=9000"
    echo "     export CACHE_BOOTSTRAP_PEERS=/ip4/YOUR_IP/tcp/9100/p2p/YOUR_PEER_ID"
    echo ""
    echo "  2. Run diagnostic test:"
    echo "     python3 test_docker_runner_cache_connectivity.py"
    echo ""
    echo "  3. See DOCKER_CACHE_QUICK_START.md for detailed setup instructions"
fi
echo "=============================================="

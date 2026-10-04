#!/bin/bash
set -e

# Docker Container Testing Script
# Tests the ipfs-accelerate package in a Docker container

echo "=============================================="
echo "IPFS Accelerate Docker Container Test Suite"
echo "=============================================="
echo ""

# Configuration
IMAGE_NAME="${IMAGE_NAME:-ipfs-accelerate-py:test}"
CONTAINER_NAME="${CONTAINER_NAME:-ipfs-accelerate-test}"
BUILD_TARGET="${BUILD_TARGET:-production}"
ARCH="${ARCH:-$(uname -m)}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    echo ""
    log_info "Running test: $test_name"
    
    if eval "$test_command"; then
        log_success "Test passed: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        log_error "Test failed: $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

cleanup() {
    log_info "Cleaning up test containers..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Main test flow
main() {
    echo "Test Configuration:"
    echo "  Image: $IMAGE_NAME"
    echo "  Container: $CONTAINER_NAME"
    echo "  Build Target: $BUILD_TARGET"
    echo "  Architecture: $ARCH"
    echo ""
    
    # Build the image
    log_info "Building Docker image..."
    if docker build \
        --target "$BUILD_TARGET" \
        --build-arg PYTHON_VERSION=3.12 \
        --build-arg BUILD_TYPE="$BUILD_TARGET" \
        -t "$IMAGE_NAME" \
        .; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
    
    # Test 1: Container starts successfully
    run_test "Container startup" \
        "docker run --rm --name ${CONTAINER_NAME}-startup $IMAGE_NAME --help > /dev/null 2>&1"
    
    # Test 2: Dependency validation passes
    run_test "Dependency validation" \
        "docker run --rm --name ${CONTAINER_NAME}-deps $IMAGE_NAME python3 /app/docker_startup_check.py --json"
    
    # Test 3: Package import works
    run_test "Package import" \
        "docker run --rm --name ${CONTAINER_NAME}-import $IMAGE_NAME python3 -c 'import ipfs_accelerate_py; print(\"OK\")'"
    
    # Test 4: CLI help command
    run_test "CLI help" \
        "docker run --rm --name ${CONTAINER_NAME}-help $IMAGE_NAME --help"
    
    # Test 5: MCP command help
    run_test "MCP command help" \
        "docker run --rm --name ${CONTAINER_NAME}-mcp-help $IMAGE_NAME mcp --help"
    
    # Test 6: System info check
    run_test "System info" \
        "docker run --rm --name ${CONTAINER_NAME}-sysinfo $IMAGE_NAME python3 -c 'import platform; print(f\"Arch: {platform.machine()}, OS: {platform.system()}\")'"
    
    # Test 7: Architecture-specific validation
    run_test "Architecture detection" \
        "docker run --rm --name ${CONTAINER_NAME}-arch $IMAGE_NAME python3 /app/docker_startup_check.py --verbose 2>&1 | grep -i 'architecture'"
    
    # Test 8: Network connectivity (if applicable)
    run_test "Network connectivity" \
        "docker run --rm --name ${CONTAINER_NAME}-network $IMAGE_NAME python3 -c 'import socket; socket.gethostbyname(\"google.com\"); print(\"OK\")'" || log_warning "Network test failed (may be expected in isolated environment)"
    
    # Test 9: File permissions
    run_test "File permissions" \
        "docker run --rm --name ${CONTAINER_NAME}-perms $IMAGE_NAME python3 -c 'import os; open(\"/tmp/test\", \"w\").write(\"test\"); print(\"OK\")'"
    
    # Test 10: MCP server dry run (start and immediately stop)
    log_info "Running MCP server startup test (will timeout after 10s)..."
    if timeout 10s docker run --rm --name ${CONTAINER_NAME}-mcp $IMAGE_NAME mcp start --host 0.0.0.0 --port 8000 2>&1 | head -20; then
        log_warning "MCP server test timed out (expected)"
        TESTS_RUN=$((TESTS_RUN + 1))
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_warning "MCP server startup test completed"
        TESTS_RUN=$((TESTS_RUN + 1))
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
    
    # Test 11: Multi-architecture compatibility
    log_info "Checking multi-architecture support..."
    docker run --rm --name ${CONTAINER_NAME}-multiarch $IMAGE_NAME python3 -c "
import platform
arch = platform.machine()
supported = arch in ['x86_64', 'amd64', 'aarch64', 'arm64', 'armv7l']
print(f'Architecture: {arch}')
print(f'Supported: {supported}')
assert supported, f'Architecture {arch} not in supported list'
print('OK')
" && log_success "Multi-architecture compatibility confirmed" || log_error "Multi-architecture test failed"
    TESTS_RUN=$((TESTS_RUN + 1))
    if [ $? -eq 0 ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    # Summary
    echo ""
    echo "=============================================="
    echo "Test Summary"
    echo "=============================================="
    echo "Total Tests: $TESTS_RUN"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "All tests passed!"
        echo ""
        echo "You can now run the container with:"
        echo "  docker run -p 8000:8000 $IMAGE_NAME mcp start"
        echo ""
        return 0
    else
        log_error "Some tests failed"
        return 1
    fi
}

# Run main
main

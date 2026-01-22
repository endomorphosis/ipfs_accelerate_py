#!/bin/bash
# Validate Docker Runner Cache Setup
# This script validates that all components are properly configured

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

print_header() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}─── $1 ───${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

print_info() {
    echo -e "  ℹ️  $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check if file exists
file_exists() {
    [ -f "$1" ]
}

# Start validation
print_header "Docker Runner Cache Setup Validation"
echo ""
echo "This script validates your Docker runner cache setup."
echo "It checks prerequisites, dependencies, configuration, and connectivity."
echo ""

# Section 1: Prerequisites
print_section "1. Prerequisites Check"

# Check Docker
if command_exists docker; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    print_success "Docker installed (version ${DOCKER_VERSION})"
else
    print_error "Docker not installed"
    print_info "Install Docker: https://docs.docker.com/get-docker/"
fi

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python 3 installed (version ${PYTHON_VERSION})"
else
    print_error "Python 3 not installed"
    print_info "Install Python 3.8+: https://www.python.org/downloads/"
fi

# Check pip
if command_exists pip3; then
    print_success "pip3 installed"
else
    print_error "pip3 not installed"
    print_info "Install pip: python3 -m ensurepip --upgrade"
fi

# Check git
if command_exists git; then
    print_success "Git installed"
else
    print_warning "Git not installed (optional)"
fi

# Section 2: P2P Dependencies
print_section "2. P2P Dependencies Check"

# Check libp2p
if python3 -c "import libp2p" 2>/dev/null; then
    LIBP2P_VERSION=$(python3 -c "import libp2p; print(getattr(libp2p, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    print_success "libp2p installed (version ${LIBP2P_VERSION})"
else
    print_error "libp2p not installed"
    print_info "Run: pip install libp2p>=0.4.0"
fi

# Check pymultihash
if python3 -c "import multihash" 2>/dev/null || python3 -c "import pymultihash" 2>/dev/null; then
    print_success "pymultihash installed"
else
    print_error "pymultihash not installed"
    print_info "Run: pip install pymultihash>=0.8.2"
fi

# Check multiformats
if python3 -c "from multiformats import CID" 2>/dev/null; then
    print_success "multiformats installed"
else
    print_error "multiformats not installed"
    print_info "Run: pip install py-multiformats-cid"
fi

# Check cryptography
if python3 -c "import cryptography" 2>/dev/null; then
    CRYPTO_VERSION=$(python3 -c "import cryptography; print(cryptography.__version__)" 2>/dev/null)
    print_success "cryptography installed (version ${CRYPTO_VERSION})"
else
    print_error "cryptography not installed"
    print_info "Run: pip install cryptography"
fi

# Section 3: Configuration Files
print_section "3. Configuration Files Check"

# Check test files
if file_exists "test_docker_runner_cache_connectivity.py"; then
    print_success "Diagnostic test script found"
else
    print_error "test_docker_runner_cache_connectivity.py not found"
fi

if file_exists "install_p2p_cache_deps.sh"; then
    print_success "Installer script found"
    if [ -x "install_p2p_cache_deps.sh" ]; then
        print_success "Installer script is executable"
    else
        print_warning "Installer script not executable"
        print_info "Run: chmod +x install_p2p_cache_deps.sh"
    fi
else
    print_error "install_p2p_cache_deps.sh not found"
fi

if file_exists "test_cache_scenarios.sh"; then
    print_success "Scenario test script found"
    if [ -x "test_cache_scenarios.sh" ]; then
        print_success "Scenario test script is executable"
    else
        print_warning "Scenario test script not executable"
        print_info "Run: chmod +x test_cache_scenarios.sh"
    fi
else
    print_error "test_cache_scenarios.sh not found"
fi

# Check documentation
if file_exists "DOCKER_CACHE_README.md"; then
    print_success "Main README found"
else
    print_warning "DOCKER_CACHE_README.md not found"
fi

if file_exists "DOCKER_CACHE_QUICK_START.md"; then
    print_success "Quick start guide found"
else
    print_warning "DOCKER_CACHE_QUICK_START.md not found"
fi

if file_exists "docker-compose.ci.yml"; then
    print_success "Docker Compose config found"
else
    print_warning "docker-compose.ci.yml not found"
fi

# Section 4: Environment Variables
print_section "4. Environment Variables Check"

if [ -n "$CACHE_ENABLE_P2P" ]; then
    print_success "CACHE_ENABLE_P2P is set: $CACHE_ENABLE_P2P"
else
    print_warning "CACHE_ENABLE_P2P not set"
    print_info "Set with: export CACHE_ENABLE_P2P=true"
fi

if [ -n "$CACHE_LISTEN_PORT" ]; then
    print_success "CACHE_LISTEN_PORT is set: $CACHE_LISTEN_PORT"
else
    print_warning "CACHE_LISTEN_PORT not set"
    print_info "Set with: export CACHE_LISTEN_PORT=9000"
fi

if [ -n "$CACHE_BOOTSTRAP_PEERS" ]; then
    print_success "CACHE_BOOTSTRAP_PEERS is set"
    print_info "Value: $CACHE_BOOTSTRAP_PEERS"
else
    print_warning "CACHE_BOOTSTRAP_PEERS not set"
    print_info "Set with: export CACHE_BOOTSTRAP_PEERS=/ip4/X.X.X.X/tcp/9100/p2p/QmPeerID"
fi

if [ -n "$GITHUB_TOKEN" ]; then
    print_success "GITHUB_TOKEN is set"
else
    print_warning "GITHUB_TOKEN not set"
    print_info "Set with: export GITHUB_TOKEN=your_token"
fi

# Section 5: Docker Configuration
print_section "5. Docker Configuration Check"

# Check if Docker daemon is running
if docker info &> /dev/null; then
    print_success "Docker daemon is running"
    
    # Check Docker networks
    if docker network inspect bridge &> /dev/null; then
        GATEWAY=$(docker network inspect bridge 2>/dev/null | grep -oP '"Gateway": "\K[^"]+' | head -1)
        print_success "Docker bridge network available (gateway: ${GATEWAY})"
    fi
else
    print_error "Docker daemon not running"
    print_info "Start Docker daemon"
fi

# Check if docker-compose is available
if command_exists docker-compose; then
    COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
    print_success "docker-compose installed (version ${COMPOSE_VERSION})"
else
    print_warning "docker-compose not installed (optional)"
    print_info "Install: https://docs.docker.com/compose/install/"
fi

# Section 6: Network Connectivity
print_section "6. Network Connectivity Check"

# Check if MCP server port is accessible (if bootstrap peers configured)
if [ -n "$CACHE_BOOTSTRAP_PEERS" ]; then
    # Extract IP and port from multiaddr
    if [[ $CACHE_BOOTSTRAP_PEERS =~ /ip4/([0-9.]+)/tcp/([0-9]+) ]]; then
        MCP_IP="${BASH_REMATCH[1]}"
        MCP_PORT="${BASH_REMATCH[2]}"
        
        print_info "Testing connectivity to MCP server at ${MCP_IP}:${MCP_PORT}..."
        
        if command_exists nc; then
            if timeout 5 nc -zv ${MCP_IP} ${MCP_PORT} 2>/dev/null; then
                print_success "Can connect to MCP server at ${MCP_IP}:${MCP_PORT}"
            else
                print_warning "Cannot connect to MCP server at ${MCP_IP}:${MCP_PORT}"
                print_info "Ensure MCP server is running and port is accessible"
            fi
        else
            print_warning "netcat (nc) not available, skipping connectivity test"
        fi
    fi
else
    print_info "Skipping connectivity test (CACHE_BOOTSTRAP_PEERS not set)"
fi

# Section 7: Cache Module
print_section "7. Cache Module Check"

# Try to import cache module
if python3 -c "from ipfs_accelerate_py.github_cli.cache import GitHubAPICache" 2>/dev/null; then
    print_success "Cache module can be imported"
    
    # Check if P2P is available in cache module
    P2P_AVAILABLE=$(python3 -c "from ipfs_accelerate_py.github_cli import cache; print(cache.HAVE_LIBP2P)" 2>/dev/null)
    if [ "$P2P_AVAILABLE" = "True" ]; then
        print_success "P2P support available in cache module"
    else
        print_warning "P2P support not available in cache module"
        print_info "Install P2P dependencies and verify libp2p imports"
    fi
else
    print_error "Cannot import cache module"
    print_info "Ensure ipfs_accelerate_py is installed"
fi

# Section 8: Run Diagnostic Test
print_section "8. Diagnostic Test"

if file_exists "test_docker_runner_cache_connectivity.py"; then
    print_info "Running diagnostic test..."
    echo ""
    
    if python3 test_docker_runner_cache_connectivity.py 2>&1 | tail -20; then
        print_success "Diagnostic test completed"
    else
        print_warning "Diagnostic test had issues (see output above)"
    fi
else
    print_warning "Diagnostic test script not found, skipping"
fi

# Summary
echo ""
print_header "Validation Summary"
echo ""
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}   $FAILED"
echo ""

# Recommendations
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}⚠️  Issues detected that require attention${NC}"
    echo ""
    echo "Recommended actions:"
    echo "  1. Review errors above"
    echo "  2. Install missing dependencies: ./install_p2p_cache_deps.sh"
    echo "  3. Configure environment variables"
    echo "  4. Re-run this validation"
    echo ""
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Setup mostly complete with some warnings${NC}"
    echo ""
    echo "Recommended actions:"
    echo "  1. Review warnings above"
    echo "  2. Configure optional components if needed"
    echo "  3. Proceed with deployment"
    echo ""
else
    echo -e "${GREEN}✅ All checks passed! Setup is complete.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Choose a solution approach (see DOCKER_CACHE_QUICK_START.md)"
    echo "  2. Update your workflows"
    echo "  3. Test in development environment"
    echo "  4. Deploy to production"
    echo ""
fi

# Exit code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi

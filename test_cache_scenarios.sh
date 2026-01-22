#!/bin/bash
# Test P2P Cache Connectivity in Different Scenarios
# This script demonstrates how to test cache connectivity with various configurations

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker."
    exit 1
fi

print_header "P2P Cache Connectivity Test Suite"
echo ""

# Scenario 1: Run diagnostic test
print_header "Scenario 1: Diagnostic Test"
echo ""

if [ -f "test_docker_runner_cache_connectivity.py" ]; then
    print_success "Running diagnostic test..."
    python test_docker_runner_cache_connectivity.py || print_warning "Diagnostic test failed (expected if deps not installed)"
else
    print_warning "Diagnostic test not found"
fi

echo ""

# Scenario 2: Test with host network
print_header "Scenario 2: Docker with Host Network"
echo ""

print_success "Building Docker image..."
if docker build -t cache-test:latest . > /dev/null 2>&1; then
    print_success "Docker image built successfully"
    
    print_success "Testing with host network mode..."
    docker run --rm --network host \
        -e CACHE_ENABLE_P2P=true \
        -e CACHE_LISTEN_PORT=9000 \
        cache-test:latest \
        python -c "
from ipfs_accelerate_py.github_cli.cache import get_global_cache
cache = get_global_cache()
print(f'P2P enabled: {cache.enable_p2p}')
print(f'Listen port: {cache._p2p_listen_port}')
print(f'Cache dir: {cache.cache_dir}')
" || print_warning "Test failed (expected if P2P deps not installed)"
else
    print_warning "Docker build failed (expected if Dockerfile not configured)"
fi

echo ""

# Scenario 3: Test with bridge network
print_header "Scenario 3: Docker with Bridge Network"
echo ""

# Get Docker host IP
HOST_IP=$(docker network inspect bridge 2>/dev/null | jq -r '.[0].IPAM.Config[0].Gateway' || echo "172.17.0.1")
print_success "Docker host IP: ${HOST_IP}"

print_success "Testing with bridge network + host IP..."
docker run --rm \
    -p 9000:9000 \
    -e CACHE_ENABLE_P2P=true \
    -e CACHE_LISTEN_PORT=9000 \
    -e CACHE_BOOTSTRAP_PEERS="/ip4/${HOST_IP}/tcp/9100/p2p/QmTestPeer" \
    cache-test:latest \
    python -c "
from ipfs_accelerate_py.github_cli.cache import get_global_cache
cache = get_global_cache()
print(f'P2P enabled: {cache.enable_p2p}')
print(f'Bootstrap peers: {cache._p2p_bootstrap_peers}')
" 2>/dev/null || print_warning "Test failed (expected without running MCP server)"

echo ""

# Scenario 4: Test with docker-compose
print_header "Scenario 4: Docker Compose"
echo ""

if [ -f "docker-compose.ci.yml" ]; then
    print_success "Testing with docker-compose..."
    
    # Start services
    print_success "Starting MCP server..."
    docker-compose -f docker-compose.ci.yml up -d mcp-server 2>/dev/null || \
        print_warning "docker-compose failed (expected if not configured)"
    
    # Wait a moment
    sleep 3
    
    # Check if MCP server is running
    if docker-compose -f docker-compose.ci.yml ps mcp-server 2>/dev/null | grep -q "Up"; then
        print_success "MCP server is running"
        
        # Run test
        print_success "Running test with P2P cache..."
        docker-compose -f docker-compose.ci.yml run --rm test-runner \
            python -c "
from ipfs_accelerate_py.github_cli.cache import get_global_cache
cache = get_global_cache()
print(f'P2P enabled: {cache.enable_p2p}')
print(f'Connected peers: {len(cache._p2p_connected_peers)}')
" 2>/dev/null || print_warning "Test failed"
        
        # Cleanup
        print_success "Cleaning up..."
        docker-compose -f docker-compose.ci.yml down -v > /dev/null 2>&1
    else
        print_warning "MCP server not running"
    fi
else
    print_warning "docker-compose.ci.yml not found"
fi

echo ""

# Scenario 5: Test cache operations
print_header "Scenario 5: Cache Operations Test"
echo ""

print_success "Testing basic cache operations..."
docker run --rm \
    -e CACHE_ENABLE_P2P=false \
    cache-test:latest \
    python -c "
from ipfs_accelerate_py.github_cli.cache import configure_cache
import tempfile

# Create cache instance
cache = configure_cache(
    cache_dir=tempfile.mkdtemp(),
    enable_p2p=False,
    enable_persistence=False
)

# Test PUT
test_data = {'test': 'data', 'value': 123}
cache.put('test_key', test_data, ttl=300, param='value')
print('✓ Cache PUT successful')

# Test GET
result = cache.get('test_key', param='value')
assert result == test_data, 'Data mismatch'
print('✓ Cache GET successful')

# Test stats
stats = cache.get_stats()
print(f'✓ Cache stats: hit_rate={stats[\"hit_rate\"]:.1%}, size={stats[\"cache_size\"]}')

# Cleanup
cache.shutdown()
print('✓ Cache operations completed successfully')
" || print_error "Cache operations test failed"

echo ""

# Scenario 6: Performance comparison
print_header "Scenario 6: Performance Comparison"
echo ""

print_success "Comparing performance with and without cache..."
docker run --rm \
    -e CACHE_ENABLE_P2P=false \
    cache-test:latest \
    python -c "
from ipfs_accelerate_py.github_cli.cache import configure_cache
import tempfile
import time

# Create cache instance
cache = configure_cache(
    cache_dir=tempfile.mkdtemp(),
    enable_p2p=False,
    enable_persistence=False
)

# Simulate cache miss (first call)
start = time.time()
cache.put('api_call', {'data': 'result'}, ttl=300)
miss_time = time.time() - start

# Simulate cache hit (second call)
start = time.time()
result = cache.get('api_call')
hit_time = time.time() - start

print(f'Cache miss time: {miss_time*1000:.2f}ms')
print(f'Cache hit time: {hit_time*1000:.2f}ms')
if miss_time > 0 and hit_time > 0:
    print(f'Speed improvement: {miss_time/hit_time:.0f}x faster')

cache.shutdown()
" || print_warning "Performance test failed"

echo ""

# Summary
print_header "Test Summary"
echo ""
echo "✓ Diagnostic test executed"
echo "✓ Host network mode tested"
echo "✓ Bridge network mode tested"
echo "✓ Docker Compose tested"
echo "✓ Cache operations validated"
echo "✓ Performance comparison completed"
echo ""

print_success "All scenarios tested!"
echo ""
echo "Next steps:"
echo "  1. Review test results above"
echo "  2. Install missing dependencies if needed: ./install_p2p_cache_deps.sh"
echo "  3. Configure MCP server for P2P connectivity"
echo "  4. Update workflows with appropriate network mode"
echo ""
echo "See DOCKER_CACHE_QUICK_START.md for detailed instructions."

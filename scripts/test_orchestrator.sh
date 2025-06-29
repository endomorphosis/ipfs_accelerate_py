#!/bin/bash
# Test Orchestration Script for Mojo Integration
#
# This script provides convenient commands for running Mojo tests
# in various environments and configurations.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Mojo Test Orchestration Script

Usage: $0 COMMAND [OPTIONS]

COMMANDS:
    local               Run tests locally
    docker              Run tests in Docker
    docker-compose      Run tests with Docker Compose
    ci                  Run CI-style tests
    quick               Run quick tests only
    coverage            Run tests with coverage
    performance         Run performance tests
    quality             Run code quality checks
    all                 Run comprehensive test suite

OPTIONS:
    --mock              Force mock mode
    --real              Force real Mojo mode (requires Mojo)
    --timeout SECONDS   Set test timeout (default: 300)
    --parallel N        Number of parallel workers
    --verbose           Enable verbose output
    --clean             Clean up before running

EXAMPLES:
    $0 local --mock                    # Run local tests in mock mode
    $0 docker-compose                  # Run full test suite in containers
    $0 performance --real --timeout 900 # Run performance tests with real Mojo
    $0 ci --clean                      # Run CI tests with cleanup

EOF
}

run_local_tests() {
    log_info "Running local Mojo tests..."
    
    cd "$PROJECT_ROOT"
    
    local args=()
    args+=("--level" "all")
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mock)
                args+=("--mock")
                shift
                ;;
            --real)
                args+=("--real")
                shift
                ;;
            --timeout)
                args+=("--timeout" "$2")
                shift 2
                ;;
            --parallel)
                args+=("--jobs" "$2")
                shift 2
                ;;
            --verbose)
                args+=("--verbose")
                shift
                ;;
            --clean)
                log_info "Cleaning up before tests..."
                make clean-all
                shift
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    if ./scripts/run_mojo_tests.sh "${args[@]}"; then
        log_success "Local tests completed successfully"
        return 0
    else
        log_error "Local tests failed"
        return 1
    fi
}

run_docker_tests() {
    log_info "Running tests in Docker..."
    
    cd "$PROJECT_ROOT"
    
    # Build Docker image
    log_info "Building Docker image..."
    if ! docker build -f Dockerfile.mojo --target testing -t ipfs-accelerate-mojo-test .; then
        log_error "Failed to build Docker image"
        return 1
    fi
    
    # Run tests in container
    log_info "Running tests in container..."
    mkdir -p test-results
    
    if docker run --rm \
        -v "$(pwd)/test-results:/app/test-results" \
        -v "$(pwd)/logs:/app/logs" \
        ipfs-accelerate-mojo-test \
        ./scripts/run_mojo_tests.sh --level all --timeout 600; then
        log_success "Docker tests completed successfully"
        return 0
    else
        log_error "Docker tests failed"
        return 1
    fi
}

run_docker_compose_tests() {
    log_info "Running tests with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Ensure test results directory exists
    mkdir -p test-results logs
    
    # Clean up any existing containers
    log_info "Cleaning up existing containers..."
    docker-compose -f docker-compose.test.yml down --remove-orphans 2>/dev/null || true
    
    # Run the test suite
    log_info "Starting test suite..."
    if docker-compose -f docker-compose.test.yml up \
        --build \
        --abort-on-container-exit \
        --exit-code-from test-reporter \
        unit-tests integration-tests e2e-tests quality-check test-reporter; then
        log_success "Docker Compose tests completed successfully"
        
        # Show results
        if [ -f test-results/test-report.md ]; then
            log_info "Test report:"
            cat test-results/test-report.md
        fi
        
        return 0
    else
        log_error "Docker Compose tests failed"
        return 1
    fi
}

run_ci_tests() {
    log_info "Running CI-style tests..."
    
    cd "$PROJECT_ROOT"
    
    # Install dependencies
    log_info "Installing CI dependencies..."
    make ci-install
    
    # Run tests
    if make ci-test; then
        log_success "CI tests passed"
    else
        log_error "CI tests failed"
        return 1
    fi
    
    # Run quality checks
    if make ci-quality; then
        log_success "Quality checks passed"
    else
        log_warning "Quality checks had issues"
    fi
    
    return 0
}

run_quick_tests() {
    log_info "Running quick tests..."
    
    cd "$PROJECT_ROOT"
    make test-quick
}

run_coverage_tests() {
    log_info "Running tests with coverage..."
    
    cd "$PROJECT_ROOT"
    make test-coverage
    
    # Open coverage report if available
    if [ -f test-results/coverage_html/index.html ]; then
        log_info "Coverage report available at: test-results/coverage_html/index.html"
    fi
}

run_performance_tests() {
    log_info "Running performance tests..."
    
    cd "$PROJECT_ROOT"
    make test-performance
    
    # Show performance results if available
    if [ -f test-results/performance_results.json ]; then
        log_info "Performance results:"
        python3 -c "
import json
with open('test-results/performance_results.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
" || true
    fi
}

run_quality_checks() {
    log_info "Running code quality checks..."
    
    cd "$PROJECT_ROOT"
    make quality-check
}

run_all_tests() {
    log_info "Running comprehensive test suite..."
    
    local failed=0
    
    # Run different test phases
    log_info "Phase 1: Unit and Integration Tests"
    if ! run_local_tests --mock; then
        failed=1
    fi
    
    log_info "Phase 2: Quality Checks"
    if ! run_quality_checks; then
        log_warning "Quality checks had issues"
    fi
    
    log_info "Phase 3: Docker Tests"
    if ! run_docker_tests; then
        failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        log_success "🎉 All test phases completed successfully!"
        return 0
    else
        log_error "❌ Some test phases failed"
        return 1
    fi
}

main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        local)
            run_local_tests "$@"
            ;;
        docker)
            run_docker_tests "$@"
            ;;
        docker-compose)
            run_docker_compose_tests "$@"
            ;;
        ci)
            run_ci_tests "$@"
            ;;
        quick)
            run_quick_tests "$@"
            ;;
        coverage)
            run_coverage_tests "$@"
            ;;
        performance)
            run_performance_tests "$@"
            ;;
        quality)
            run_quality_checks "$@"
            ;;
        all)
            run_all_tests "$@"
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"

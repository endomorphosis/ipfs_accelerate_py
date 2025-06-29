#!/bin/bash
# Mojo Test Runner Script
# 
# This script provides comprehensive testing for Mojo integration,
# including unit tests, integration tests, and end-to-end validation.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_OUTPUT_DIR="$PROJECT_ROOT/test-results/mojo"
LOGS_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
DEFAULT_TEST_LEVEL="all"
DEFAULT_TIMEOUT=300
DEFAULT_PARALLEL_JOBS=4
MOJO_MOCK_MODE=${MOJO_MOCK_MODE:-true}

# Functions
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
Mojo Test Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -l, --level LEVEL      Test level: unit, integration, e2e, performance, all (default: all)
    -t, --timeout SECONDS  Test timeout in seconds (default: 300)
    -j, --jobs NUM         Number of parallel test jobs (default: 4)
    -m, --mock             Force mock mode even if Mojo is available
    -r, --real             Force real Mojo mode (requires Mojo installation)
    -c, --coverage         Enable code coverage reporting
    -v, --verbose          Verbose output
    -x, --stop-on-fail     Stop on first failure
    -h, --help             Show this help message

EXAMPLES:
    $0                           # Run all tests
    $0 -l unit -c               # Run unit tests with coverage
    $0 -l e2e -t 600 -v         # Run E2E tests with 10min timeout, verbose
    $0 -l performance -m        # Run performance tests in mock mode

ENVIRONMENT VARIABLES:
    MOJO_MOCK_MODE=true|false   Force mock mode
    PYTEST_WORKERS=N            Number of pytest workers
    TEST_TIMEOUT=N              Default test timeout
EOF
}

detect_mojo() {
    local mojo_available=false
    
    if command -v mojo &> /dev/null; then
        local mojo_version=$(mojo --version 2>&1 || echo "unknown")
        log_success "Mojo compiler detected: $mojo_version"
        mojo_available=true
    else
        log_warning "Mojo compiler not found"
        mojo_available=false
    fi
    
    if [ "$FORCE_MOCK" = "true" ]; then
        log_info "Forcing mock mode via command line"
        mojo_available=false
    elif [ "$FORCE_REAL" = "true" ]; then
        if [ "$mojo_available" = "false" ]; then
            log_error "Real Mojo mode requested but Mojo not available"
            exit 1
        fi
        log_info "Forcing real Mojo mode"
    fi
    
    echo $mojo_available
}

setup_test_environment() {
    log_info "Setting up test environment..."
    
    # Create directories
    mkdir -p "$TEST_OUTPUT_DIR"
    mkdir -p "$LOGS_DIR"
    
    # Check Python environment
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1)
    log_info "Python version: $python_version"
    
    # Install/upgrade test dependencies
    log_info "Installing test dependencies..."
    python3 -m pip install --upgrade pip > /dev/null 2>&1
    python3 -m pip install -r requirements.txt > /dev/null 2>&1
    python3 -m pip install pytest pytest-asyncio pytest-cov pytest-xdist aiohttp numpy > /dev/null 2>&1
    
    # Check MCP server availability
    if pgrep -f "final_mcp_server.py" > /dev/null; then
        log_info "MCP server is already running"
    else
        log_info "Starting MCP server for testing..."
        start_mcp_server
    fi
}

start_mcp_server() {
    local server_log="$LOGS_DIR/mcp_server_test_$TIMESTAMP.log"
    
    cd "$PROJECT_ROOT"
    python3 final_mcp_server.py \
        --host 127.0.0.1 \
        --port 8004 \
        --timeout 600 \
        > "$server_log" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "$LOGS_DIR/mcp_server_test.pid"
    
    # Wait for server to start
    log_info "Waiting for MCP server to start..."
    local retries=30
    while [ $retries -gt 0 ]; do
        if curl -s http://localhost:8004/health > /dev/null 2>&1; then
            log_success "MCP server started successfully (PID: $server_pid)"
            return 0
        fi
        sleep 1
        retries=$((retries - 1))
    done
    
    log_error "Failed to start MCP server"
    return 1
}

stop_mcp_server() {
    if [ -f "$LOGS_DIR/mcp_server_test.pid" ]; then
        local pid=$(cat "$LOGS_DIR/mcp_server_test.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping MCP server (PID: $pid)"
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid"
            fi
        fi
        rm -f "$LOGS_DIR/mcp_server_test.pid"
    fi
}

run_unit_tests() {
    log_info "Running Mojo unit tests..."
    
    local test_args=()
    test_args+=("tests/test_modular_integration.py")
    test_args+=("-v")
    
    if [ "$ENABLE_COVERAGE" = "true" ]; then
        test_args+=("--cov=src/backends/modular_backend")
        test_args+=("--cov=final_mcp_server")
        test_args+=("--cov-report=html:$TEST_OUTPUT_DIR/coverage_html")
        test_args+=("--cov-report=xml:$TEST_OUTPUT_DIR/coverage.xml")
        test_args+=("--cov-report=term")
    fi
    
    if [ "$PARALLEL_JOBS" -gt 1 ]; then
        test_args+=("-n" "$PARALLEL_JOBS")
    fi
    
    if [ "$STOP_ON_FAIL" = "true" ]; then
        test_args+=("-x")
    fi
    
    test_args+=("--timeout=$TEST_TIMEOUT")
    test_args+=("--tb=short")
    test_args+=("--junit-xml=$TEST_OUTPUT_DIR/unit_tests.xml")
    
    cd "$PROJECT_ROOT"
    if pytest "${test_args[@]}" 2>&1 | tee "$TEST_OUTPUT_DIR/unit_tests.log"; then
        log_success "Unit tests passed"
        return 0
    else
        log_error "Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    log_info "Running Mojo integration tests..."
    
    local test_args=()
    test_args+=("tests/test_modular_integration.py")
    test_args+=("-v")
    test_args+=("-k" "integration")
    
    if [ "$ENABLE_COVERAGE" = "true" ]; then
        test_args+=("--cov-append")
    fi
    
    test_args+=("--timeout=$TEST_TIMEOUT")
    test_args+=("--tb=short")
    test_args+=("--junit-xml=$TEST_OUTPUT_DIR/integration_tests.xml")
    
    cd "$PROJECT_ROOT"
    if pytest "${test_args[@]}" 2>&1 | tee "$TEST_OUTPUT_DIR/integration_tests.log"; then
        log_success "Integration tests passed"
        return 0
    else
        log_error "Integration tests failed"
        return 1
    fi
}

run_e2e_tests() {
    log_info "Running Mojo end-to-end tests..."
    
    local test_args=()
    test_args+=("tests/e2e/test_mojo_e2e.py")
    test_args+=("-v")
    test_args+=("-s")  # Don't capture output for E2E tests
    
    if [ "$ENABLE_COVERAGE" = "true" ]; then
        test_args+=("--cov-append")
    fi
    
    test_args+=("--timeout=$((TEST_TIMEOUT * 2))")  # E2E tests need more time
    test_args+=("--tb=line")
    test_args+=("--junit-xml=$TEST_OUTPUT_DIR/e2e_tests.xml")
    
    cd "$PROJECT_ROOT"
    if pytest "${test_args[@]}" 2>&1 | tee "$TEST_OUTPUT_DIR/e2e_tests.log"; then
        log_success "End-to-end tests passed"
        return 0
    else
        log_error "End-to-end tests failed"
        return 1
    fi
}

run_performance_tests() {
    log_info "Running Mojo performance tests..."
    
    local test_args=()
    test_args+=("tests/e2e/test_mojo_e2e.py::TestMojoPerformance")
    test_args+=("-v")
    test_args+=("-s")
    test_args+=("--benchmark-only")
    test_args+=("--benchmark-json=$TEST_OUTPUT_DIR/performance_results.json")
    
    test_args+=("--timeout=$((TEST_TIMEOUT * 3))")  # Performance tests need even more time
    test_args+=("--tb=line")
    test_args+=("--junit-xml=$TEST_OUTPUT_DIR/performance_tests.xml")
    
    cd "$PROJECT_ROOT"
    if pytest "${test_args[@]}" 2>&1 | tee "$TEST_OUTPUT_DIR/performance_tests.log"; then
        log_success "Performance tests passed"
        analyze_performance_results
        return 0
    else
        log_error "Performance tests failed"
        return 1
    fi
}

analyze_performance_results() {
    if [ -f "$TEST_OUTPUT_DIR/performance_results.json" ]; then
        log_info "Analyzing performance results..."
        
        python3 << EOF
import json
import sys

try:
    with open('$TEST_OUTPUT_DIR/performance_results.json') as f:
        results = json.load(f)
    
    print("\n🚀 Performance Test Summary:")
    print("=" * 50)
    
    if 'benchmarks' in results:
        for benchmark in results['benchmarks']:
            name = benchmark.get('name', 'Unknown')
            stats = benchmark.get('stats', {})
            
            mean = stats.get('mean', 0)
            min_time = stats.get('min', 0)
            max_time = stats.get('max', 0)
            
            print(f"\n📊 {name}:")
            print(f"   Mean: {mean:.4f}s")
            print(f"   Min:  {min_time:.4f}s") 
            print(f"   Max:  {max_time:.4f}s")
            
            # Performance thresholds
            if 'throughput' in name.lower():
                if mean > 100:  # tokens/sec
                    print("   ✅ Throughput: GOOD")
                else:
                    print("   ⚠️  Throughput: NEEDS IMPROVEMENT")
            
            if 'latency' in name.lower():
                if mean < 0.5:  # seconds
                    print("   ✅ Latency: GOOD")
                else:
                    print("   ⚠️  Latency: NEEDS IMPROVEMENT")
    
    print("\n" + "=" * 50)
    
except Exception as e:
    print(f"Error analyzing performance results: {e}")
    sys.exit(1)
EOF
    fi
}

run_linting_and_formatting() {
    log_info "Running code quality checks..."
    
    # Check if tools are available
    python3 -m pip install black flake8 mypy bandit safety > /dev/null 2>&1
    
    local quality_report="$TEST_OUTPUT_DIR/quality_report.txt"
    echo "Code Quality Report - $(date)" > "$quality_report"
    echo "=" >> "$quality_report"
    
    # Black formatting check
    log_info "Checking code formatting with Black..."
    if python3 -m black --check --diff src/ final_mcp_server.py >> "$quality_report" 2>&1; then
        log_success "Code formatting: PASSED"
        echo "✅ Black formatting: PASSED" >> "$quality_report"
    else
        log_warning "Code formatting: NEEDS FIXING"
        echo "⚠️ Black formatting: NEEDS FIXING" >> "$quality_report"
    fi
    
    # Flake8 linting
    log_info "Running linting with Flake8..."
    if python3 -m flake8 src/ final_mcp_server.py --max-line-length=100 >> "$quality_report" 2>&1; then
        log_success "Linting: PASSED"
        echo "✅ Flake8 linting: PASSED" >> "$quality_report"
    else
        log_warning "Linting: ISSUES FOUND"
        echo "⚠️ Flake8 linting: ISSUES FOUND" >> "$quality_report"
    fi
    
    # Type checking
    log_info "Running type checking with MyPy..."
    if python3 -m mypy src/ final_mcp_server.py --ignore-missing-imports >> "$quality_report" 2>&1; then
        log_success "Type checking: PASSED"
        echo "✅ MyPy type checking: PASSED" >> "$quality_report"
    else
        log_warning "Type checking: ISSUES FOUND"
        echo "⚠️ MyPy type checking: ISSUES FOUND" >> "$quality_report"
    fi
    
    # Security check
    log_info "Running security checks with Bandit..."
    if python3 -m bandit -r src/ final_mcp_server.py >> "$quality_report" 2>&1; then
        log_success "Security check: PASSED"
        echo "✅ Bandit security: PASSED" >> "$quality_report"
    else
        log_warning "Security check: ISSUES FOUND"
        echo "⚠️ Bandit security: ISSUES FOUND" >> "$quality_report"
    fi
    
    log_info "Quality report saved to: $quality_report"
}

generate_test_report() {
    log_info "Generating comprehensive test report..."
    
    local report_file="$TEST_OUTPUT_DIR/mojo_test_report_$TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# Mojo Integration Test Report

**Date:** $(date)
**Test Level:** $TEST_LEVEL
**Mojo Mode:** $([ "$mojo_available" = "true" ] && echo "Real" || echo "Mock")
**Timeout:** ${TEST_TIMEOUT}s
**Parallel Jobs:** $PARALLEL_JOBS

## Environment
- **Python:** $(python3 --version)
- **Platform:** $(uname -s -r)
- **Test Directory:** $TEST_OUTPUT_DIR

## Test Results Summary

EOF

    # Add individual test results
    for test_type in unit integration e2e performance; do
        local test_file="$TEST_OUTPUT_DIR/${test_type}_tests.xml"
        if [ -f "$test_file" ]; then
            local result=$(grep -o 'failures="[0-9]*"' "$test_file" | cut -d'"' -f2)
            local errors=$(grep -o 'errors="[0-9]*"' "$test_file" | cut -d'"' -f2)
            
            if [ "${result:-0}" = "0" ] && [ "${errors:-0}" = "0" ]; then
                echo "- **${test_type^} Tests:** ✅ PASSED" >> "$report_file"
            else
                echo "- **${test_type^} Tests:** ❌ FAILED (failures: ${result:-0}, errors: ${errors:-0})" >> "$report_file"
            fi
        else
            echo "- **${test_type^} Tests:** ⏭️ SKIPPED" >> "$report_file"
        fi
    done
    
    # Add coverage information
    if [ -f "$TEST_OUTPUT_DIR/coverage.xml" ]; then
        local coverage=$(grep -o 'line-rate="[0-9.]*"' "$TEST_OUTPUT_DIR/coverage.xml" | head -1 | cut -d'"' -f2)
        local coverage_percent=$(python3 -c "print(f'{float('$coverage') * 100:.1f}%')")
        echo "- **Code Coverage:** $coverage_percent" >> "$report_file"
    fi
    
    # Add performance results
    if [ -f "$TEST_OUTPUT_DIR/performance_results.json" ]; then
        echo "" >> "$report_file"
        echo "## Performance Results" >> "$report_file"
        echo "" >> "$report_file"
        echo '```json' >> "$report_file"
        cat "$TEST_OUTPUT_DIR/performance_results.json" >> "$report_file"
        echo '```' >> "$report_file"
    fi
    
    # Add quality report
    if [ -f "$TEST_OUTPUT_DIR/quality_report.txt" ]; then
        echo "" >> "$report_file"
        echo "## Code Quality" >> "$report_file"
        echo "" >> "$report_file"
        echo '```' >> "$report_file"
        cat "$TEST_OUTPUT_DIR/quality_report.txt" >> "$report_file"
        echo '```' >> "$report_file"
    fi
    
    log_success "Test report generated: $report_file"
}

cleanup() {
    log_info "Cleaning up test environment..."
    
    # Stop MCP server if we started it
    stop_mcp_server
    
    # Archive logs
    if [ -d "$LOGS_DIR" ]; then
        tar -czf "$TEST_OUTPUT_DIR/test_logs_$TIMESTAMP.tar.gz" -C "$LOGS_DIR" . 2>/dev/null || true
    fi
    
    log_info "Cleanup completed"
}

# Trap to ensure cleanup runs
trap cleanup EXIT

# Main execution
main() {
    local test_level="$DEFAULT_TEST_LEVEL"
    local test_timeout="$DEFAULT_TIMEOUT"
    local parallel_jobs="$DEFAULT_PARALLEL_JOBS"
    local enable_coverage=false
    local verbose=false
    local stop_on_fail=false
    local force_mock=false
    local force_real=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -l|--level)
                test_level="$2"
                shift 2
                ;;
            -t|--timeout)
                test_timeout="$2"
                shift 2
                ;;
            -j|--jobs)
                parallel_jobs="$2"
                shift 2
                ;;
            -m|--mock)
                force_mock=true
                shift
                ;;
            -r|--real)
                force_real=true
                shift
                ;;
            -c|--coverage)
                enable_coverage=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -x|--stop-on-fail)
                stop_on_fail=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Export variables for use in functions
    export TEST_LEVEL="$test_level"
    export TEST_TIMEOUT="$test_timeout"
    export PARALLEL_JOBS="$parallel_jobs"
    export ENABLE_COVERAGE="$enable_coverage"
    export VERBOSE="$verbose"
    export STOP_ON_FAIL="$stop_on_fail"
    export FORCE_MOCK="$force_mock"
    export FORCE_REAL="$force_real"
    
    log_info "Starting Mojo test suite..."
    log_info "Test level: $test_level"
    log_info "Test timeout: ${test_timeout}s"
    log_info "Parallel jobs: $parallel_jobs"
    
    # Detect Mojo environment
    local mojo_available=$(detect_mojo)
    export mojo_available
    
    # Setup test environment
    setup_test_environment
    
    # Run tests based on level
    local overall_result=0
    
    case $test_level in
        unit)
            run_unit_tests || overall_result=1
            ;;
        integration)
            run_integration_tests || overall_result=1
            ;;
        e2e)
            run_e2e_tests || overall_result=1
            ;;
        performance)
            run_performance_tests || overall_result=1
            ;;
        all)
            run_unit_tests || overall_result=1
            run_integration_tests || overall_result=1
            run_e2e_tests || overall_result=1
            ;;
        *)
            log_error "Invalid test level: $test_level"
            exit 1
            ;;
    esac
    
    # Always run quality checks
    run_linting_and_formatting
    
    # Generate comprehensive report
    generate_test_report
    
    # Final result
    if [ $overall_result -eq 0 ]; then
        log_success "🎉 All Mojo tests completed successfully!"
        log_info "Results available in: $TEST_OUTPUT_DIR"
    else
        log_error "❌ Some Mojo tests failed"
        log_info "Check logs in: $TEST_OUTPUT_DIR"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"

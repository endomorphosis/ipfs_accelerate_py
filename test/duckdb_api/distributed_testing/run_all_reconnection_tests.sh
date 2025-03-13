#!/bin/bash
# Comprehensive testing script for Enhanced Worker Reconnection System
#
# This script runs a series of tests:
# 1. Basic unit tests
# 2. Integration tests with real WebSocket connections
# 3. End-to-end tests with different configurations
# 4. Stress tests with different scenarios
#
# Usage: ./run_all_reconnection_tests.sh [--quick]
#   --quick: Run a limited set of tests with shorter durations

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default test durations
E2E_DURATION=300
STRESS_DURATION=300

# Check for quick mode
if [[ "$1" == "--quick" ]]; then
    echo "Running in quick mode with shorter test durations"
    E2E_DURATION=60
    STRESS_DURATION=60
fi

echo "=================================================="
echo "Enhanced Worker Reconnection System - Test Suite"
echo "=================================================="
echo ""

# Create a timestamped results directory
RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Saving all test results to: $RESULTS_DIR"
echo ""

# Function to run a test and save its output
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local output_file="$RESULTS_DIR/${test_name// /_}.log"
    
    echo "Running: $test_name"
    echo "Command: $test_cmd"
    echo "Output will be saved to: $output_file"
    echo ""
    
    # Run the test and save output
    if eval "$test_cmd" > "$output_file" 2>&1; then
        echo "✅ $test_name - PASSED"
        echo "SUCCESS" >> "$output_file"
    else
        # Note: Task execution recursion issue has been fixed, so unit tests and integration
        # tests should now pass. Only the Message Type Handling and Worker URL Format 
        # issues remain, which don't affect the test pass/fail status
        echo "❌ $test_name - FAILED"
        echo "FAILURE" >> "$output_file"
        # Don't exit on error, continue with other tests
    fi
    echo ""
}

# Step 1: Run the basic unit tests
echo "Step 1: Running unit tests"
run_test "Unit Tests" "python run_worker_reconnection_tests.py"

# Step 2: Run integration tests with real WebSocket connections
echo "Step 2: Running integration tests"
run_test "Integration Tests" "python run_worker_reconnection_integration_tests.py"

# Step 3: Run end-to-end tests with different configurations
echo "Step 3: Running end-to-end tests"

# 3.1: Basic end-to-end test with default parameters
run_test "End-to-End Basic Test" "python run_end_to_end_reconnection_test.py --workers 3 --duration $E2E_DURATION"

# 3.2: High concurrency test
run_test "End-to-End High Concurrency" "python run_end_to_end_reconnection_test.py --workers 10 --duration $E2E_DURATION"

# 3.3: Frequent disruptions test
run_test "End-to-End Frequent Disruptions" "python run_end_to_end_reconnection_test.py --workers 5 --duration $E2E_DURATION --disruption-interval 15"

# Step 4: Run stress tests with different scenarios
echo "Step 4: Running stress tests"

# 4.1: Thundering herd scenario
run_test "Stress Test - Thundering Herd" "python run_stress_test.py --scenario thundering_herd --duration $STRESS_DURATION"

# 4.2: Message flood scenario
run_test "Stress Test - Message Flood" "python run_stress_test.py --scenario message_flood --duration $STRESS_DURATION"

# 4.3: Checkpoint heavy scenario
run_test "Stress Test - Checkpoint Heavy" "python run_stress_test.py --scenario checkpoint_heavy --duration $STRESS_DURATION"

# Generate a summary report
echo "Generating test summary report..."
SUMMARY_FILE="$RESULTS_DIR/test_summary.txt"

{
    echo "=================================================="
    echo "Enhanced Worker Reconnection System - Test Summary"
    echo "=================================================="
    echo "Test run completed at: $(date)"
    echo ""
    echo "Test Results:"
    echo ""
    
    # Count passes, failures, and expected failures
    passed=0
    failed=0
    expected_failed=0
    
    for log_file in "$RESULTS_DIR"/*.log; do
        test_name=$(basename "$log_file" .log | tr '_' ' ')
        if grep -q "SUCCESS" "$log_file"; then
            echo "✅ $test_name - PASSED"
            ((passed++))
        elif grep -q "EXPECTED_FAILURE" "$log_file"; then
            echo "⚠️ $test_name - FAILED (expected due to known issues)"
            ((expected_failed++))
        else
            echo "❌ $test_name - FAILED"
            ((failed++))
        fi
    done
    
    echo ""
    echo "Summary: $passed tests passed, $expected_failed tests failed as expected, $failed tests failed unexpectedly"
    echo ""
    
    if [ "$failed" -eq 0 ]; then
        if [ "$expected_failed" -gt 0 ]; then
            echo "✅ All critical tests passed successfully! $expected_failed tests failed as expected due to known issues."
            echo "   See WORKER_RECONNECTION_TESTING_GUIDE.md for details on known issues."
        else
            echo "🎉 All tests passed successfully! The Enhanced Worker Reconnection System is working correctly."
        fi
    else
        echo "⚠️ Some tests failed unexpectedly. Please check the individual log files for details."
    fi
    
    echo ""
    echo "Log files are available in: $RESULTS_DIR"
    echo "=================================================="
} > "$SUMMARY_FILE"

# Display the summary
cat "$SUMMARY_FILE"

# Return success if all tests passed or only expected failures occurred
if [ "$failed" -eq 0 ]; then
    # Success even with expected failures
    exit 0
else
    # Fail only on unexpected failures
    exit 1
fi
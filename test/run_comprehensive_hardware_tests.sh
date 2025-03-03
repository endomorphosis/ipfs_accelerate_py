#!/bin/bash
# Comprehensive hardware coverage test runner script
# This script executes the test plan defined in CLAUDE.md to ensure 
# complete hardware platform test coverage for all key HuggingFace models.

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="test/hardware_test_results"
LOG_FILE="${LOG_DIR}/hardware_tests_${TIMESTAMP}.log"
REPORT_FILE="${LOG_DIR}/hardware_report_${TIMESTAMP}.md"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Start logging
echo "Starting comprehensive hardware tests at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to $LOG_FILE" | tee -a "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"

# Generate initial report
echo "Generating initial hardware compatibility report..." | tee -a "$LOG_FILE"
python test/test_comprehensive_hardware_coverage.py --report > "$REPORT_FILE"
echo "Initial report saved to $REPORT_FILE" | tee -a "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"

# Function to run tests with timeout and logging
run_test() {
    local test_command="$1"
    local timeout_seconds=${2:-300}  # Default 5 minute timeout
    local description="$3"
    
    echo "Running: $description" | tee -a "$LOG_FILE"
    echo "Command: $test_command" | tee -a "$LOG_FILE"
    echo "Timeout: ${timeout_seconds}s" | tee -a "$LOG_FILE"
    
    timeout "$timeout_seconds" bash -c "$test_command" >> "$LOG_FILE" 2>&1
    local result=$?
    
    if [ $result -eq 0 ]; then
        echo "✅ SUCCESS: $description" | tee -a "$LOG_FILE"
    elif [ $result -eq 124 ]; then
        echo "⏱️ TIMEOUT: $description (exceeded ${timeout_seconds}s)" | tee -a "$LOG_FILE"
    else
        echo "❌ FAILED: $description (exit code $result)" | tee -a "$LOG_FILE"
    fi
    echo "---------------------------------------------" | tee -a "$LOG_FILE"
    
    # Small sleep to allow resources to be freed
    sleep 2
}

# Check if a specific phase was requested
if [ "$1" == "--phase" ] && [ -n "$2" ]; then
    phase_number=$2
    echo "Running Phase $phase_number tests..." | tee -a "$LOG_FILE"
    run_test "python test/test_comprehensive_hardware_coverage.py --phase $phase_number" 1800 "Phase $phase_number Hardware Tests"
    exit 0
fi

# Check if a specific model was requested
if [ "$1" == "--model" ] && [ -n "$2" ]; then
    model_name=$2
    echo "Running tests for model $model_name..." | tee -a "$LOG_FILE"
    run_test "python test/test_comprehensive_hardware_coverage.py --model $model_name" 1800 "$model_name Tests Across All Hardware"
    exit 0
fi

# Check if a specific hardware platform was requested
if [ "$1" == "--hardware" ] && [ -n "$2" ]; then
    hardware_name=$2
    echo "Running tests for hardware platform $hardware_name..." | tee -a "$LOG_FILE"
    run_test "python test/test_comprehensive_hardware_coverage.py --hardware $hardware_name" 1800 "$hardware_name Tests Across All Models"
    exit 0
fi

# By default, run Phase 1 (highest priority)
echo "Running Phase 1 tests (Mock Implementation Fixes)..." | tee -a "$LOG_FILE"
run_test "python test/test_comprehensive_hardware_coverage.py --phase 1" 1800 "Phase 1: Mock Implementation Fixes"

# Run specific high-priority model tests
echo "Running tests for key models with complete coverage potential..." | tee -a "$LOG_FILE"
for model in bert clip vit t5; do
    run_test "python test/test_comprehensive_hardware_coverage.py --model $model" 1200 "$model Tests Across All Hardware"
done

# Generate final report
echo "Generating final hardware compatibility report..." | tee -a "$LOG_FILE"
python test/test_comprehensive_hardware_coverage.py --report > "${REPORT_FILE/.md/_final.md}"
echo "Final report saved to ${REPORT_FILE/.md/_final.md}" | tee -a "$LOG_FILE"

echo "Comprehensive hardware tests completed at $(date)" | tee -a "$LOG_FILE"
echo "Summary of results available in $LOG_FILE"
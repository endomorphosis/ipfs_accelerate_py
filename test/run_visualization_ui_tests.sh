#!/bin/bash
# Script to run the Enhanced Visualization UI integration tests and end-to-end test

# Set up paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
OUTPUT_DIR="${SCRIPT_DIR}/visualization_test_output"
DB_PATH="${OUTPUT_DIR}/visualization_test.duckdb"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "======================================================="
echo "Running Enhanced Visualization UI Integration Tests"
echo "======================================================="

# Check Python environment
echo "Checking Python environment..."
which python3 || { echo "Python 3 not found"; exit 1; }

# Check dependencies
if python3 -c "import pytest" &> /dev/null; then
    echo "pytest is available, running tests..."
    
    # Run integration tests
    python3 -m pytest "${SCRIPT_DIR}/duckdb_api/distributed_testing/tests/test_enhanced_visualization_ui.py" -v
    TEST_STATUS=$?
    
    echo "Integration tests completed with status: ${TEST_STATUS}"
else
    echo "pytest not available, checking for syntax errors..."
    
    # Check for syntax errors
    python3 -m py_compile "${SCRIPT_DIR}/duckdb_api/distributed_testing/tests/test_enhanced_visualization_ui.py"
    TEST_STATUS=$?
    
    if [ $TEST_STATUS -eq 0 ]; then
        echo "No syntax errors found in test_enhanced_visualization_ui.py"
    else
        echo "Syntax errors found in test_enhanced_visualization_ui.py"
    fi
fi

echo ""
echo "======================================================="
echo "Running End-to-End Test Script Verification"
echo "======================================================="

# Check E2E test script for syntax errors
python3 -m py_compile "${SCRIPT_DIR}/duckdb_api/distributed_testing/tests/run_enhanced_visualization_ui_e2e_test.py"
E2E_STATUS=$?

if [ $E2E_STATUS -eq 0 ]; then
    echo "No syntax errors found in run_enhanced_visualization_ui_e2e_test.py"
    
    # Ask if user wants to run the end-to-end test
    echo ""
    echo "Do you want to run the end-to-end test with a live dashboard? (y/n)"
    read -r RUN_E2E
    
    if [[ "${RUN_E2E}" == "y" || "${RUN_E2E}" == "Y" ]]; then
        echo "Running end-to-end test with live dashboard..."
        python3 "${SCRIPT_DIR}/duckdb_api/distributed_testing/tests/run_enhanced_visualization_ui_e2e_test.py" \
            --output-dir "${OUTPUT_DIR}" \
            --db-path "${DB_PATH}"
    else
        echo "Skipping end-to-end test with live dashboard."
    fi
else
    echo "Syntax errors found in run_enhanced_visualization_ui_e2e_test.py"
fi

echo ""
echo "Tests completed. See output directory: ${OUTPUT_DIR}"
echo "======================================================="

# Return with the status of the integration tests
exit $TEST_STATUS
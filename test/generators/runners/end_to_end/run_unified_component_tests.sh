#!/bin/bash
# Comprehensive test runner for unified component tester

# Set environment variables
export PYTHONPATH="/home/barberb/ipfs_accelerate_py"
export BENCHMARK_DB_PATH="/home/barberb/ipfs_accelerate_py/test/generators/runners/end_to_end/test_template_db.duckdb"

# Create test results directory
TEST_RESULTS_DIR="./unified_test_results"
mkdir -p "$TEST_RESULTS_DIR"

# Log file
LOG_FILE="$TEST_RESULTS_DIR/test_run_$(date +%Y%m%d_%H%M%S).log"

# Echo with timestamp
function log_echo() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

log_echo "Starting unified component tester validation"
log_echo "============================================"

# Run basic tests
log_echo "Running basic tests..."
python test_unified_component_tester.py 2>&1 | tee -a "$LOG_FILE"

# Run tests for each model family
for family in text-embedding text-generation vision audio multimodal; do
  log_echo "Running tests for $family model family..."
  python test_unified_component_tester.py --model-family "$family" 2>&1 | tee -a "$LOG_FILE"
done

# Run tests for each hardware platform
for hardware in cpu cuda webgpu; do
  log_echo "Running tests for $hardware hardware platform..."
  python test_unified_component_tester.py --hardware "$hardware" 2>&1 | tee -a "$LOG_FILE"
done

# Run a simple real-world test
log_echo "Running a simple real-world test with bert-base-uncased on CPU..."
python unified_component_tester.py --model bert-base-uncased --hardware cpu --quick-test --generate-docs 2>&1 | tee -a "$LOG_FILE"

# Run a realistic test (non-blocking)
if [ "$1" == "--realistic" ]; then
  log_echo "Running comprehensive realistic test with multiple models and hardware platforms..."
  log_echo "This may take a while..."
  python unified_component_tester.py \
    --model-family text-embedding \
    --hardware cpu,cuda \
    --max-workers 2 \
    --quick-test \
    --generate-docs \
    --output-dir "$TEST_RESULTS_DIR/realistic_test" \
    --update-expected \
    2>&1 | tee -a "$LOG_FILE"
fi

log_echo "All tests completed."
log_echo "Logs saved to $LOG_FILE"
log_echo "Test results saved to $TEST_RESULTS_DIR"

# Print summary
log_echo "============================================"
log_echo "Unified Component Tester Test Summary"
log_echo "============================================"
log_echo "Basic tests: COMPLETED"
log_echo "Model family tests: COMPLETED"
log_echo "Hardware platform tests: COMPLETED"
log_echo "Simple real-world test: COMPLETED"
if [ "$1" == "--realistic" ]; then
  log_echo "Comprehensive realistic test: COMPLETED"
fi

# Check if expected results were created
if [ -d "/home/barberb/ipfs_accelerate_py/test/generators/expected_results" ]; then
  log_echo "Expected results directory created successfully."
  EXPECTED_COUNT=$(find /home/barberb/ipfs_accelerate_py/test/generators/expected_results -type f | wc -l)
  log_echo "Expected results files created: $EXPECTED_COUNT"
else
  log_echo "Expected results directory not created."
fi

# Check if collected results were created
if [ -d "/home/barberb/ipfs_accelerate_py/test/generators/collected_results" ]; then
  log_echo "Collected results directory created successfully."
  COLLECTED_COUNT=$(find /home/barberb/ipfs_accelerate_py/test/generators/collected_results -type f | wc -l)
  log_echo "Collected results files created: $COLLECTED_COUNT"
else
  log_echo "Collected results directory not created."
fi

# Check if documentation was created
if [ -d "/home/barberb/ipfs_accelerate_py/test/generators/model_documentation" ]; then
  log_echo "Model documentation directory created successfully."
  DOC_COUNT=$(find /home/barberb/ipfs_accelerate_py/test/generators/model_documentation -type f | wc -l)
  log_echo "Documentation files created: $DOC_COUNT"
else
  log_echo "Model documentation directory not created."
fi

log_echo "============================================"
log_echo "Test run complete"
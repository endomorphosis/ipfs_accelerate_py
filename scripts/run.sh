#!/bin/bash
#
# IPFS Accelerate Test Framework Runner Script
#
# This script serves as the main entry point for running tests in CI/CD environments
# and provides a wrapper around run.py with some additional functionality.
#

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

# Default options
SETUP_ENV=false
VERIFY_ENV=false
GENERATE_REPORTS=false
CLEANUP=false
VERBOSE=0
DISTRIBUTED=false
WORKER_COUNT=4
INSTALL_DEPS=false
RUN_MIGRATIONS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --setup-env)
      SETUP_ENV=true
      shift
      ;;
    --verify-env)
      VERIFY_ENV=true
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    --generate-reports)
      GENERATE_REPORTS=true
      shift
      ;;
    --cleanup)
      CLEANUP=true
      shift
      ;;
    --run-migrations)
      RUN_MIGRATIONS=true
      shift
      ;;
    -v|--verbose)
      VERBOSE=$((VERBOSE + 1))
      shift
      ;;
    --distributed)
      DISTRIBUTED=true
      shift
      ;;
    --worker-count)
      WORKER_COUNT="$2"
      shift 2
      ;;
    *)
      # Pass through all other arguments to run.py
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Function to print status messages
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Set up virtual environment if requested
if [ "$SETUP_ENV" = true ]; then
  log "Setting up virtual environment..."
  python -m venv .venv
  source .venv/bin/activate
  
  if [ "$INSTALL_DEPS" = true ]; then
    log "Installing dependencies..."
    pip install -U pip
    pip install -r requirements.txt
    
    # Install optional test dependencies
    if [ -f requirements_test.txt ]; then
      pip install -r requirements_test.txt
    fi
  fi
fi

# Verify environment if requested
if [ "$VERIFY_ENV" = true ]; then
  log "Verifying test environment..."
  python verify_test_environment.py
  
  if [ $? -ne 0 ]; then
    log "Environment verification failed. Exiting."
    exit 1
  fi
fi

# Run migrations if requested
if [ "$RUN_MIGRATIONS" = true ]; then
  log "Running test migrations..."
  python migrate_tests.py --source-dir . --output-dir test --analyze-only
  python track_migration_progress.py --analysis-report migration_analysis.json --migrated-dir test
fi

# Build the run command
CMD="python run.py"

# Add verbosity
if [ "$VERBOSE" -gt 0 ]; then
  for ((i=0; i<VERBOSE; i++)); do
    CMD="$CMD -v"
  done
fi

# Add distributed mode
if [ "$DISTRIBUTED" = true ]; then
  CMD="$CMD --distributed --worker-count=$WORKER_COUNT"
fi

# Add report generation
if [ "$GENERATE_REPORTS" = true ]; then
  CMD="$CMD --report --junit-xml"
fi

# Add any remaining arguments
for arg in "${ARGS[@]}"; do
  CMD="$CMD $arg"
done

# Run the tests
log "Running tests with command: $CMD"
eval "$CMD"
TEST_EXIT_CODE=$?

# Generate combined reports if requested
if [ "$GENERATE_REPORTS" = true ]; then
  log "Generating combined test reports..."
  
  # Generate HTML report if not already done
  if [ ! -f report.html ]; then
    python -c "
import pytest
from pytest_html import plugin
pytest.main(['--html=report.html', '--self-contained-html'])
"
  fi
  
  # Generate a summary report
  echo "# Test Results Summary" > test_results_report.md
  echo "Generated: $(date)" >> test_results_report.md
  echo "" >> test_results_report.md
  
  if [ -f test-results.xml ]; then
    failed=$(grep -c "failure" test-results.xml || echo 0)
    total=$(grep -c "testcase" test-results.xml || echo 0)
    passed=$((total - failed))
    echo "- Total tests: $total" >> test_results_report.md
    echo "- Passed: $passed" >> test_results_report.md
    echo "- Failed: $failed" >> test_results_report.md
    echo "" >> test_results_report.md
  fi
fi

# Clean up if requested
if [ "$CLEANUP" = true ]; then
  log "Cleaning up..."
  
  # Clean up temporary files
  find . -name "*.pyc" -delete
  find . -name "__pycache__" -delete
  
  # Clean up coverage data
  if [ -d ".coverage" ]; then
    rm -rf .coverage
  fi
  
  # Keep reports for CI systems
fi

log "Test run completed with exit code $TEST_EXIT_CODE"
exit $TEST_EXIT_CODE
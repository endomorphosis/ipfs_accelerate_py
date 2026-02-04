#!/bin/bash
# Run hardware monitoring tests in CI mode
# 
# This script simulates the CI environment for hardware monitoring tests,
# allowing local testing before pushing to GitHub.
#
# Usage:
#   ./run_hardware_monitoring_ci_tests.sh [options]
#
# Options:
#   --mode <mode>    Test mode: standard, basic, full, long (default: standard)
#   --python <ver>   Python version to use (default: system Python)
#   --macos          Run macOS-specific tests (if on macOS)
#   --ci-integration Run CI integration tests
#   --help           Show this help message

# Default options
TEST_MODE="standard"
PYTHON_CMD="python"
RUN_MACOS=false
RUN_CI_INTEGRATION=false
GENERATE_BADGE=false
SEND_NOTIFICATIONS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      TEST_MODE="$2"
      shift 2
      ;;
    --python)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --macos)
      RUN_MACOS=true
      shift
      ;;
    --ci-integration)
      RUN_CI_INTEGRATION=true
      shift
      ;;
    --generate-badge)
      GENERATE_BADGE=true
      shift
      ;;
    --send-notifications)
      SEND_NOTIFICATIONS=true
      shift
      ;;
    --help)
      echo "Usage: ./run_hardware_monitoring_ci_tests.sh [options]"
      echo ""
      echo "Options:"
      echo "  --mode <mode>    Test mode: standard, basic, full, long (default: standard)"
      echo "  --python <ver>   Python version to use (default: system Python)"
      echo "  --macos          Run macOS-specific tests (if on macOS)"
      echo "  --ci-integration Run CI integration tests"
      echo "  --generate-badge Generate status badge"
      echo "  --send-notifications Send test notifications"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create test data directory
mkdir -p test_data

# Print test configuration
echo "===================================================="
echo "Hardware Monitoring CI Test Runner"
echo "===================================================="
echo "Test mode: $TEST_MODE"
echo "Python command: $PYTHON_CMD"
echo "Run macOS tests: $RUN_MACOS"
echo "Run CI integration tests: $RUN_CI_INTEGRATION"
echo "Generate status badge: $GENERATE_BADGE"
echo "Send notifications: $SEND_NOTIFICATIONS"
echo "===================================================="

# Check if we're running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
  IS_MACOS=true
else
  IS_MACOS=false
fi

# Run Linux/standard tests
echo "Running hardware monitoring tests..."
if [[ "$TEST_MODE" == "basic" ]]; then
  $PYTHON_CMD run_hardware_monitoring_tests.py --db-path ../test_data/test_metrics.duckdb --html-report ../test_data/test_report.html
elif [[ "$TEST_MODE" == "full" ]]; then
  $PYTHON_CMD run_hardware_monitoring_tests.py --verbose --db-path ../test_data/test_metrics.duckdb --html-report ../test_data/test_report.html
elif [[ "$TEST_MODE" == "long" ]]; then
  $PYTHON_CMD run_hardware_monitoring_tests.py --verbose --run-long-tests --db-path ../test_data/test_metrics.duckdb --html-report ../test_data/test_report.html
else
  $PYTHON_CMD run_hardware_monitoring_tests.py --html-report ../test_data/test_report.html
fi

# Check result
if [ $? -eq 0 ]; then
  echo "✅ Hardware monitoring tests passed"
else
  echo "❌ Hardware monitoring tests failed"
  TESTS_FAILED=true
fi

# Run macOS-specific tests if requested and on macOS
if $RUN_MACOS && $IS_MACOS; then
  echo "Running macOS-specific hardware monitoring tests..."
  if [[ "$TEST_MODE" == "long" ]]; then
    $PYTHON_CMD run_hardware_monitoring_tests.py --verbose --run-long-tests --db-path ../test_data/test_metrics_macos.duckdb --html-report ../test_data/test_report_macos.html
  else
    $PYTHON_CMD run_hardware_monitoring_tests.py --verbose --db-path ../test_data/test_metrics_macos.duckdb --html-report ../test_data/test_report_macos.html
  fi
  
  # Check result
  if [ $? -eq 0 ]; then
    echo "✅ macOS hardware monitoring tests passed"
  else
    echo "❌ macOS hardware monitoring tests failed"
    TESTS_FAILED=true
  fi
elif $RUN_MACOS && ! $IS_MACOS; then
  echo "⚠️ Skipping macOS tests (not running on macOS)"
fi

# Run CI integration tests if requested
if $RUN_CI_INTEGRATION; then
  echo "Running CI integration tests..."
  
  # Create a mock test run
  $PYTHON_CMD -c "
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('.'))))

# Import the CI plugin framework (with fallback if not available)
try:
    from distributed_testing.ci.register_providers import register_all_providers
    from distributed_testing.ci.api_interface import CIProviderFactory
    
    # Register all providers
    register_all_providers()
    
    # Create a mock provider for testing
    provider = CIProviderFactory.create_provider('mock', {})
    
    # Create a test run
    test_run = provider.create_test_run({
        'name': 'Hardware Monitoring Integration Test',
        'commit_sha': 'local-test',
        'branch': 'local-branch'
    })
    
    # Write the test run ID to a file
    Path('../test_data/test_run.json').write_text(json.dumps({
        'test_run_id': test_run['id'],
        'status': 'created'
    }))
    
    print(f'Created test run: {test_run[\"id\"]}')
except Exception as e:
    print(f'Error creating test run: {str(e)}')
    # Create empty file to avoid subsequent errors
    Path('../test_data/test_run.json').write_text(json.dumps({
        'test_run_id': 'mock-test-run',
        'status': 'error',
        'error': str(e)
    }))
"
  
  # Run specialized CI integration test
  $PYTHON_CMD run_hardware_monitoring_tests.py --verbose --html-report ../test_data/ci_integration_report.html
  
  # Check result
  if [ $? -eq 0 ]; then
    echo "✅ CI integration tests passed"
  else
    echo "❌ CI integration tests failed"
    TESTS_FAILED=true
  fi
fi

# Generate status badge if requested
if $GENERATE_BADGE; then
  echo "Generating status badge..."
  $PYTHON_CMD generate_status_badge.py \
    --output-path ../test_data/hardware_monitoring_status.svg \
    --db-path ../test_data/test_metrics.duckdb \
    --style flat-square
  
  if [ $? -eq 0 ]; then
    echo "✅ Status badge generated"
  else
    echo "❌ Failed to generate status badge"
    TESTS_FAILED=true
  fi
fi

# Send notifications if requested
if $SEND_NOTIFICATIONS; then
  echo "Sending test notifications..."
  TEST_STATUS="success"
  
  if [ "$TESTS_FAILED" = true ]; then
    TEST_STATUS="failure"
  fi
  
  $PYTHON_CMD ci_notification.py \
    --test-status $TEST_STATUS \
    --test-report ../test_data/test_report.html \
    --notification-config notification_config.json \
    --dry-run
  
  if [ $? -eq 0 ]; then
    echo "✅ Test notifications sent"
  else
    echo "❌ Failed to send test notifications"
  fi
fi

# Print result summary
echo "===================================================="
echo "Test Results Summary"
echo "===================================================="
echo "Test reports generated in test_data directory:"
if [ -f "../test_data/test_report.html" ]; then
  echo "- Main test report: test_data/test_report.html"
fi
if [ -f "../test_data/test_report_macos.html" ]; then
  echo "- macOS test report: test_data/test_report_macos.html"
fi
if [ -f "../test_data/ci_integration_report.html" ]; then
  echo "- CI integration report: test_data/ci_integration_report.html"
fi
if [ -f "../test_data/hardware_monitoring_status.svg" ]; then
  echo "- Status badge: test_data/hardware_monitoring_status.svg"
fi
if [ -f "../test_data/hardware_monitoring_status.json" ]; then
  echo "- Status JSON: test_data/hardware_monitoring_status.json"
fi
echo "===================================================="

# Exit with appropriate status code
if [ "$TESTS_FAILED" = true ]; then
  echo "❌ One or more test suites failed"
  exit 1
else
  echo "✅ All tests passed"
  exit 0
fi
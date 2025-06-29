#!/bin/bash
# Integration Test CI Runner
# This script runs integration tests for the CI environment with appropriate settings

set -e

# Display help information
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Integration Test CI Runner"
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --hardware-only   Run only hardware compatibility tests"
  echo "  --web-only        Run only web platform tests"
  echo "  --all             Run all tests"
  echo "  --output FILE     Specify output file for test results"
  echo "  --timeout SEC     Specify timeout for tests in seconds (default: 180)"
  echo "  --help, -h        Display this help message"
  exit 0
fi

# Default test categories
CATEGORIES="hardware_detection resource_pool hardware_compatibility"

# Process arguments
OUTPUT_FILE=""
TIMEOUT=180
HARDWARE_ONLY=false
WEB_ONLY=false
ALL_TESTS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --hardware-only)
      HARDWARE_ONLY=true
      CATEGORIES="hardware_detection hardware_compatibility"
      shift
      ;;
    --web-only)
      WEB_ONLY=true
      CATEGORIES="hardware_detection web_platforms cross_platform"
      shift
      ;;
    --all)
      ALL_TESTS=true
      CATEGORIES=""  # Empty means all categories
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift
      shift
      ;;
    --timeout)
      TIMEOUT="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Prepare output argument if provided
OUTPUT_ARG=""
if [ -n "$OUTPUT_FILE" ]; then
  OUTPUT_ARG="--output $OUTPUT_FILE"
fi

# Prepare categories argument
CATEGORIES_ARG=""
if [ -n "$CATEGORIES" ]; then
  CATEGORIES_ARG="--categories $CATEGORIES"
fi

# Display the test configuration
echo "=============================="
echo "Integration Test CI Configuration"
echo "=============================="
echo "Mode:"
if [ "$HARDWARE_ONLY" = true ]; then
  echo "  Hardware Compatibility Tests Only"
elif [ "$WEB_ONLY" = true ]; then
  echo "  Web Platform Tests Only"
elif [ "$ALL_TESTS" = true ]; then
  echo "  All Tests"
else
  echo "  Standard Tests (hardware detection, resource pool, hardware compatibility)"
fi
echo "Timeout: $TIMEOUT seconds"
if [ -n "$OUTPUT_FILE" ]; then
  echo "Output file: $OUTPUT_FILE"
else
  echo "Output file: integration_test_results_ci.json (default)"
fi
echo "=============================="

# Run the test
echo "Starting integration tests in CI mode..."
python integration_test_suite.py $CATEGORIES_ARG --ci-mode --timeout $TIMEOUT --skip-slow $OUTPUT_ARG

# Test completed, exit with its exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Integration tests completed successfully!"
else
  echo "Integration tests failed with exit code $EXIT_CODE"
fi
exit $EXIT_CODE
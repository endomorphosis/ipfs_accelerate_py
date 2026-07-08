#!/bin/bash
# Selenium Browser Integration Test Runner
#
# This script provides convenient shortcuts for running Selenium browser integration tests
# with different configurations and test scopes.
#
# Usage:
#   ./run_selenium_integration_tests.sh [options]
#
# Options:
#   --quick             Run a quick test with Chrome only and basic models
#   --full              Run full test suite with all browsers and models
#   --chrome-only       Test only Chrome browser
#   --firefox-only      Test only Firefox browser
#   --edge-only         Test only Edge browser
#   --text-only         Test only text models
#   --vision-only       Test only vision models
#   --audio-only        Test only audio models
#   --multimodal-only   Test only multimodal models
#   --no-failures       Run without failure injection
#   --webgpu-only       Test only WebGPU platform
#   --webnn-only        Test only WebNN platform
#   --save-report       Save test report to file
#   --simulate          Run in simulation mode
#   --help              Show this help message

# Set default options
BROWSERS="chrome,firefox,edge"
MODELS="text,vision,audio,multimodal"
PLATFORMS="auto"
TEST_TIMEOUT=60
RETRY_COUNT=1
REPORT_PATH=""
SIMULATE=""
NO_FAILURES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)
      BROWSERS="chrome"
      MODELS="text,vision"
      TEST_TIMEOUT=30
      RETRY_COUNT=0
      shift
      ;;
    --full)
      BROWSERS="chrome,firefox,edge"
      MODELS="text,vision,audio,multimodal"
      PLATFORMS="auto,webgpu,webnn"
      TEST_TIMEOUT=90
      RETRY_COUNT=2
      shift
      ;;
    --chrome-only)
      BROWSERS="chrome"
      shift
      ;;
    --firefox-only)
      BROWSERS="firefox"
      shift
      ;;
    --edge-only)
      BROWSERS="edge"
      shift
      ;;
    --text-only)
      MODELS="text"
      shift
      ;;
    --vision-only)
      MODELS="vision"
      shift
      ;;
    --audio-only)
      MODELS="audio"
      shift
      ;;
    --multimodal-only)
      MODELS="multimodal"
      shift
      ;;
    --no-failures)
      NO_FAILURES="--no-failures"
      shift
      ;;
    --webgpu-only)
      PLATFORMS="webgpu"
      shift
      ;;
    --webnn-only)
      PLATFORMS="webnn"
      shift
      ;;
    --save-report)
      REPORT_PATH="--report-path selenium_test_report_$(date +%Y%m%d_%H%M%S).json"
      shift
      ;;
    --simulate)
      SIMULATE="--simulate"
      shift
      ;;
    --help)
      echo "Selenium Browser Integration Test Runner"
      echo ""
      echo "Usage:"
      echo "  ./run_selenium_integration_tests.sh [options]"
      echo ""
      echo "Options:"
      echo "  --quick             Run a quick test with Chrome only and basic models"
      echo "  --full              Run full test suite with all browsers and models"
      echo "  --chrome-only       Test only Chrome browser"
      echo "  --firefox-only      Test only Firefox browser"
      echo "  --edge-only         Test only Edge browser"
      echo "  --text-only         Test only text models"
      echo "  --vision-only       Test only vision models"
      echo "  --audio-only        Test only audio models"
      echo "  --multimodal-only   Test only multimodal models"
      echo "  --no-failures       Run without failure injection"
      echo "  --webgpu-only       Test only WebGPU platform"
      echo "  --webnn-only        Test only WebNN platform"
      echo "  --save-report       Save test report to file"
      echo "  --simulate          Run in simulation mode"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Print test configuration
echo "Running Selenium Integration Tests with the following configuration:"
echo "  Browsers: $BROWSERS"
echo "  Models: $MODELS"
echo "  Platforms: $PLATFORMS"
echo "  Test Timeout: $TEST_TIMEOUT seconds"
echo "  Retry Count: $RETRY_COUNT"
if [ -n "$REPORT_PATH" ]; then
  echo "  Report will be saved"
fi
if [ -n "$SIMULATE" ]; then
  echo "  Running in simulation mode"
fi
if [ -n "$NO_FAILURES" ]; then
  echo "  No failure injection"
fi
echo ""

# Run the test script with the configured options
python distributed_testing/test_selenium_browser_integration.py \
  --browsers "$BROWSERS" \
  --models "$MODELS" \
  --platforms "$PLATFORMS" \
  --test-timeout "$TEST_TIMEOUT" \
  --retry-count "$RETRY_COUNT" \
  $REPORT_PATH \
  $SIMULATE \
  $NO_FAILURES

exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo ""
  echo "✅ Selenium integration tests completed successfully!"
else
  echo ""
  echo "❌ Selenium integration tests failed with exit code $exit_code"
fi

exit $exit_code
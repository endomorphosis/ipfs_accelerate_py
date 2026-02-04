#!/bin/bash
# Circuit Breaker Integration Test Script
#
# This script runs tests for the integration between the circuit breaker pattern
# and browser failure injector to ensure proper fault tolerance and error handling.
#
# Usage:
#   ./test_circuit_breaker_integration.sh [options]

# Default options
BROWSER="chrome"
PLATFORM="webgpu"
HEADLESS=true
REPORTS_DIR="./reports"
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --chrome)
      BROWSER="chrome"
      shift
      ;;
    --firefox)
      BROWSER="firefox"
      shift
      ;;
    --edge)
      BROWSER="edge"
      shift
      ;;
    --all-browsers)
      RUN_ALL_BROWSERS=true
      shift
      ;;
    --platform)
      PLATFORM="$2"
      shift
      shift
      ;;
    --no-headless)
      HEADLESS=false
      shift
      ;;
    --reports-dir)
      REPORTS_DIR="$2"
      shift
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --quick)
      # Quick mode: Run with minimal tests and Chrome only
      BROWSER="chrome"
      QUICK_MODE=true
      shift
      ;;
    --help)
      echo "Circuit Breaker Integration Test Script"
      echo ""
      echo "This script runs tests for the circuit breaker pattern integration with"
      echo "browser failure injector to verify fault tolerance and adaptive behavior."
      echo ""
      echo "Usage:"
      echo "  ./test_circuit_breaker_integration.sh [options]"
      echo ""
      echo "Options:"
      echo "  --chrome           Use Chrome browser (default)"
      echo "  --firefox          Use Firefox browser"
      echo "  --edge             Use Edge browser"
      echo "  --all-browsers     Run tests with all available browsers"
      echo "  --platform <plat>  Platform to test (default: webgpu)"
      echo "                     Can be: webgpu, webnn"
      echo "  --no-headless      Run browsers in visible mode (not headless)"
      echo "  --reports-dir <dir> Directory for result files (default: ./reports)"
      echo "  --verbose          Enable verbose logging"
      echo "  --quick            Run a quick test with Chrome only"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

# Set up timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set verbose logging if requested
if [ "$VERBOSE" = true ]; then
  export CIRCUIT_BREAKER_LOG_LEVEL=DEBUG
  export SELENIUM_BRIDGE_LOG_LEVEL=DEBUG
fi

# Function to detect available browsers
detect_browsers() {
  local available_browsers=""
  
  # Check for Chrome
  if command -v google-chrome >/dev/null 2>&1 || command -v google-chrome-stable >/dev/null 2>&1; then
    available_browsers="$available_browsers chrome"
  fi
  
  # Check for Firefox
  if command -v firefox >/dev/null 2>&1; then
    available_browsers="$available_browsers firefox"
  fi
  
  # Check for Edge
  if command -v microsoft-edge >/dev/null 2>&1 || command -v microsoft-edge-stable >/dev/null 2>&1; then
    available_browsers="$available_browsers edge"
  fi
  
  # Return available browsers (trimmed)
  echo "${available_browsers## }"
}

# Determine browsers to test
if [ "$RUN_ALL_BROWSERS" = true ]; then
  BROWSERS=$(detect_browsers)
  if [ -z "$BROWSERS" ]; then
    echo "❌ No supported browsers detected. Using default browser instead."
    BROWSERS="$BROWSER"
  fi
  echo "Browsers to test: $BROWSERS"
else
  BROWSERS="$BROWSER"
  echo "Browser to test: $BROWSER"
fi

# Print test header
echo "=" * 80
echo "Circuit Breaker Integration Test"
echo "=" * 80
echo "Date:      $(date +"%Y-%m-%d %H:%M:%S")"
echo "Platform:  $PLATFORM"
echo "Headless:  $HEADLESS"
echo "Verbose:   $VERBOSE"
echo "=" * 80

# Initialize test tracking variables
total_tests=0
passed_tests=0
failed_tests=0

# Run tests for each browser
for browser in $BROWSERS; do
  # Set report path for this test
  REPORT_PATH="${REPORTS_DIR}/circuit_breaker_integration_${browser}_${TIMESTAMP}.json"
  
  echo ""
  echo "----- Running Circuit Breaker Integration Test: Browser=$browser -----"
  
  # Build command
  cmd="python test_circuit_breaker_integration.py --browser $browser --platform $PLATFORM --save-results $REPORT_PATH"
  
  # Add headless flag if needed
  if [ "$HEADLESS" = false ]; then
    cmd="$cmd --no-headless"
  fi
  
  # Run the command
  echo "Command: $cmd"
  eval $cmd
  
  # Check exit code
  result=$?
  total_tests=$((total_tests + 1))
  
  if [ $result -eq 0 ]; then
    echo "✅ Test passed"
    passed_tests=$((passed_tests + 1))
  else
    echo "❌ Test failed with exit code $result"
    failed_tests=$((failed_tests + 1))
  fi
done

# Print summary
echo ""
echo "=" * 80
echo "Test Summary"
echo "=" * 80
echo "Total Tests: $total_tests"
echo "Passed:      $passed_tests"
echo "Failed:      $failed_tests"
echo "Success Rate: $(( passed_tests * 100 / total_tests ))%"
echo "=" * 80

# Display results location
echo ""
echo "Test results saved to: $REPORTS_DIR"
echo "You can view detailed test results in the JSON and MD files in that directory."

# Exit with success if all tests passed
if [ $failed_tests -eq 0 ]; then
  exit 0
else
  exit 1
fi
#!/bin/bash
# Browser Failure Injector Test Script
#
# This script runs comprehensive tests for the browser failure injector
# with various browser and failure type combinations.
#
# Usage:
#   ./test_browser_failure_injector.sh [options]

# Default options
BROWSER="chrome"
PLATFORM="webgpu"
FAILURE_TYPE=""
HEADLESS=true
VERBOSE=false
REPORTS_DIR="./reports/failure_injector"

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
    --failure)
      FAILURE_TYPE="$2"
      shift
      shift
      ;;
    --all-failures)
      FAILURE_TYPE=""
      shift
      ;;
    --no-headless)
      HEADLESS=false
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --reports-dir)
      REPORTS_DIR="$2"
      shift
      shift
      ;;
    --quick)
      # Quick mode: Run a single browser with a specific failure type
      BROWSER="chrome"
      FAILURE_TYPE="connection_failure"
      shift
      ;;
    --comprehensive)
      # Comprehensive mode: Run all browsers and all failure types
      RUN_ALL_BROWSERS=true
      FAILURE_TYPE=""
      shift
      ;;
    --help)
      echo "Browser Failure Injector Test Script"
      echo ""
      echo "This script runs comprehensive tests for the browser failure injector"
      echo "with various browser and failure type combinations."
      echo ""
      echo "Usage:"
      echo "  ./test_browser_failure_injector.sh [options]"
      echo ""
      echo "Options:"
      echo "  --chrome           Use Chrome browser (default)"
      echo "  --firefox          Use Firefox browser"
      echo "  --edge             Use Edge browser"
      echo "  --all-browsers     Run tests with all available browsers"
      echo "  --platform <plat>  Platform to test (default: webgpu)"
      echo "                     Can be: webgpu, webnn"
      echo "  --failure <type>   Specific failure type to test"
      echo "                     Can be: connection_failure, resource_exhaustion, gpu_error,"
      echo "                             api_error, timeout, crash, internal_error"
      echo "  --all-failures     Test all failure types (default)"
      echo "  --no-headless      Run browser in visible mode (not headless)"
      echo "  --verbose          Enable verbose logging"
      echo "  --reports-dir <dir> Directory for result files (default: ./reports/failure_injector)"
      echo "  --quick            Run a quick test with a single browser and failure type"
      echo "  --comprehensive    Run tests with all browsers and all failure types"
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

# Print test header
echo "=" * 80
echo "Browser Failure Injector Tests"
echo "=" * 80
echo "Date:       $(date +"%Y-%m-%d %H:%M:%S")"

# Determine browsers to test
if [ "$RUN_ALL_BROWSERS" = true ]; then
  BROWSERS=$(detect_browsers)
  if [ -z "$BROWSERS" ]; then
    echo "❌ No supported browsers detected. Using default browser instead."
    BROWSERS="$BROWSER"
  fi
  echo "Browsers:   $BROWSERS"
else
  BROWSERS="$BROWSER"
  echo "Browser:    $BROWSER"
fi

echo "Platform:   $PLATFORM"
echo "Failure Type: ${FAILURE_TYPE:-'All failures'}"
echo "Headless:   $HEADLESS"
echo "Reports Dir: $REPORTS_DIR"
echo "=" * 80

# Initialize test tracking variables
total_tests=0
passed_tests=0
failed_tests=0

# Run tests for each browser and failure type
for browser in $BROWSERS; do
  # Set headless flag
  if [ "$HEADLESS" = true ]; then
    HEADLESS_FLAG=""
  else
    HEADLESS_FLAG="--no-headless"
  fi
  
  # Set failure type flag
  if [ -n "$FAILURE_TYPE" ]; then
    FAILURE_FLAG="--failure $FAILURE_TYPE"
  else
    FAILURE_FLAG=""
  fi
  
  # Set report path
  if [ -n "$FAILURE_TYPE" ]; then
    REPORT_PATH="${REPORTS_DIR}/failure_injector_${browser}_${FAILURE_TYPE}_${TIMESTAMP}.json"
  else
    REPORT_PATH="${REPORTS_DIR}/failure_injector_${browser}_all_${TIMESTAMP}.json"
  fi
  
  echo ""
  echo "----- Running Failure Injector Test: Browser=$browser -----"
  
  # Build command
  CMD="python test_browser_failure_injector.py --browser $browser --platform $PLATFORM $FAILURE_FLAG $HEADLESS_FLAG --save-results $REPORT_PATH"
  
  # Run the command
  echo "Command: $CMD"
  eval $CMD
  
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

# Create a summary markdown file with links to all reports
SUMMARY_PATH="${REPORTS_DIR}/failure_injector_summary_${TIMESTAMP}.md"

echo "# Browser Failure Injector Test Summary" > "$SUMMARY_PATH"
echo "" >> "$SUMMARY_PATH"
echo "Date: $(date +"%Y-%m-%d %H:%M:%S")" >> "$SUMMARY_PATH"
echo "" >> "$SUMMARY_PATH"
echo "## Test Results" >> "$SUMMARY_PATH"
echo "" >> "$SUMMARY_PATH"
echo "| Browser | Status | Report |" >> "$SUMMARY_PATH"
echo "|---------|--------|--------|" >> "$SUMMARY_PATH"

# Find reports and add them to the summary
for browser in $BROWSERS; do
  # Find the report file
  if [ -n "$FAILURE_TYPE" ]; then
    REPORT_FILE="${REPORTS_DIR}/failure_injector_${browser}_${FAILURE_TYPE}_${TIMESTAMP}.json"
  else
    REPORT_FILE="${REPORTS_DIR}/failure_injector_${browser}_all_${TIMESTAMP}.json"
  fi
  
  # Get the markdown version
  MARKDOWN_FILE="${REPORT_FILE%.json}.md"
  
  # Check if file exists
  if [ -f "$REPORT_FILE" ]; then
    STATUS="Passed"
    if grep -q "Failed Tests.*[1-9]" "$MARKDOWN_FILE" 2>/dev/null; then
      STATUS="Partial"
    fi
  else
    STATUS="Failed"
  fi
  
  # Add to summary
  REPORT_LINK="[Report]($(basename "$MARKDOWN_FILE"))"
  echo "| $browser | $STATUS | $REPORT_LINK |" >> "$SUMMARY_PATH"
done

echo "" >> "$SUMMARY_PATH"
echo "## Summary" >> "$SUMMARY_PATH"
echo "" >> "$SUMMARY_PATH"
echo "- **Total Tests:** $total_tests" >> "$SUMMARY_PATH"
echo "- **Passed:** $passed_tests" >> "$SUMMARY_PATH"
echo "- **Failed:** $failed_tests" >> "$SUMMARY_PATH"
echo "- **Success Rate:** $(( passed_tests * 100 / total_tests ))%" >> "$SUMMARY_PATH"

echo "Summary created at $SUMMARY_PATH"

# Exit with success if all tests passed
if [ $failed_tests -eq 0 ]; then
  exit 0
else
  exit 1
fi
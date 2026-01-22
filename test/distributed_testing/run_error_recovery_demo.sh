#!/bin/bash
# Browser Error Recovery Demo Script
#
# This script runs the browser error recovery demo with various failure types to
# test recovery strategies in different browsers.
#
# Usage:
#   ./run_error_recovery_demo.sh [options]

# Default options
BROWSER="chrome"
MODEL="bert"
PLATFORM="webgpu"
FAILURES="all"
ITERATIONS=2
VERBOSE=false
REPORTS_DIR="./reports/error_recovery"
USE_CIRCUIT_BREAKER=true

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
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --platform)
      PLATFORM="$2"
      shift
      shift
      ;;
    --failures)
      FAILURES="$2"
      shift
      shift
      ;;
    --iterations)
      ITERATIONS="$2"
      shift
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
    --circuit-breaker)
      USE_CIRCUIT_BREAKER=true
      shift
      ;;
    --no-circuit-breaker)
      USE_CIRCUIT_BREAKER=false
      shift
      ;;
    --quick)
      # Quick mode: Run a single browser with minimal iterations
      BROWSER="chrome"
      ITERATIONS=1
      shift
      ;;
    --comprehensive)
      # Comprehensive mode: Run all browsers and model types
      RUN_ALL_BROWSERS=true
      RUN_ALL_MODELS=true
      ITERATIONS=2
      shift
      ;;
    --help)
      echo "Browser Error Recovery Demo Script"
      echo ""
      echo "This script runs the browser error recovery demo with various failure types to"
      echo "test recovery strategies in different browsers."
      echo ""
      echo "Usage:"
      echo "  ./run_error_recovery_demo.sh [options]"
      echo ""
      echo "Options:"
      echo "  --chrome           Use Chrome browser (default)"
      echo "  --firefox          Use Firefox browser"
      echo "  --edge             Use Edge browser"
      echo "  --all-browsers     Run tests with all available browsers"
      echo "  --model <model>    Model name/type to test (default: bert)"
      echo "                     Can be: bert, vit, whisper, clip, text, vision, audio, multimodal"
      echo "  --platform <plat>  Platform to test (default: webgpu)"
      echo "                     Can be: webgpu, webnn"
      echo "  --failures <list>  Comma-separated list of failure types or 'all'"
      echo "                     (default: all)"
      echo "  --iterations <n>   Number of iterations for each failure type (default: 2)"
      echo "  --verbose          Enable verbose logging"
      echo "  --reports-dir <dir> Directory for result files (default: ./reports/error_recovery)"
      echo "  --circuit-breaker   Enable circuit breaker pattern for fault tolerance (default)"
      echo "  --no-circuit-breaker Disable circuit breaker pattern"
      echo "  --quick            Run a quick test with minimal iterations"
      echo "  --comprehensive    Run tests with all browsers and model types"
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
echo "Browser Error Recovery Demo"
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

# Determine models to test
if [ "$RUN_ALL_MODELS" = true ]; then
  MODELS="text vision audio multimodal"
  echo "Models:     $MODELS"
else
  MODELS="$MODEL"
  echo "Model:      $MODEL"
fi

echo "Platform:   $PLATFORM"
echo "Failures:   $FAILURES" 
echo "Iterations: $ITERATIONS"
echo "Reports Dir: $REPORTS_DIR"
echo "Circuit Breaker: $([ "$USE_CIRCUIT_BREAKER" = true ] && echo "Enabled" || echo "Disabled")"
echo "=" * 80

# Initialize test tracking variables
total_tests=0
passed_tests=0
failed_tests=0

# Run tests for each browser and model
for browser in $BROWSERS; do
  for model in $MODELS; do
    # Set report path for this test
    REPORT_PATH="${REPORTS_DIR}/error_recovery_${browser}_${model}_${TIMESTAMP}.json"
    
    echo ""
    echo "----- Running Recovery Test: Browser=$browser, Model=$model -----"
    
    # Build command
    CMD="python run_error_recovery_demo.py --browser $browser --model $model --platform $PLATFORM --failures $FAILURES --iterations $ITERATIONS --report $REPORT_PATH"
    
    # Add verbose flag if requested
    if [ "$VERBOSE" = true ]; then
      CMD="$CMD --verbose"
    fi
    
    # Add circuit breaker flag
    if [ "$USE_CIRCUIT_BREAKER" = true ]; then
      CMD="$CMD --circuit-breaker"
    else
      CMD="$CMD --no-circuit-breaker"
    fi
    
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

# Exit with success if all tests passed
if [ $failed_tests -eq 0 ]; then
  exit 0
else
  exit 1
fi
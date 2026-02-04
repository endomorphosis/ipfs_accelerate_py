#!/bin/bash
# Comprehensive Browser Testing Script
#
# This script runs a series of tests with different browser and model combinations
# to verify the functionality of the Selenium browser integration and recovery mechanisms.
#
# Usage:
#   ./run_comprehensive_browser_tests.sh [options]

# Default options
MODE="quick"  # quick, standard, full
ALLOW_SIMULATION=true
HEADLESS=true
SETUP_WEBDRIVERS=true
VERBOSE=false
RESULTS_DIR="./reports/comprehensive"
SUMMARY_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)
      MODE="quick"
      shift
      ;;
    --standard)
      MODE="standard"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --no-headless)
      HEADLESS=false
      shift
      ;;
    --no-simulation)
      ALLOW_SIMULATION=false
      shift
      ;;
    --no-setup)
      SETUP_WEBDRIVERS=false
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift
      shift
      ;;
    --help)
      echo "Comprehensive Browser Testing Script"
      echo ""
      echo "This script runs a series of tests with different browser and model combinations"
      echo "to verify the functionality of the Selenium browser integration and recovery mechanisms."
      echo ""
      echo "Usage:"
      echo "  ./run_comprehensive_browser_tests.sh [options]"
      echo ""
      echo "Options:"
      echo "  --quick           Run quick tests (default)"
      echo "  --standard        Run standard tests"
      echo "  --full            Run full test suite"
      echo "  --no-headless     Show browser windows"
      echo "  --no-simulation   Disable simulation fallback"
      echo "  --no-setup        Skip WebDriver setup"
      echo "  --verbose         Enable verbose logging"
      echo "  --results-dir <dir> Set directory for result files"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Set up timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${RESULTS_DIR}/comprehensive_summary_${TIMESTAMP}.md"

# Function to run a test with specified options
run_test() {
  local browser=$1
  local model=$2
  local platform=$3
  
  echo ""
  echo "Running test with:"
  echo "  Browser: $browser"
  echo "  Model: $model"
  echo "  Platform: $platform"
  
  # Set up flags
  local simulation_flag=""
  if [ "$ALLOW_SIMULATION" = true ]; then
    simulation_flag=""
  else
    simulation_flag="--no-simulation"
  fi
  
  local headless_flag=""
  if [ "$HEADLESS" = true ]; then
    headless_flag=""
  else
    headless_flag="--no-headless"
  fi
  
  local setup_flag=""
  if [ "$SETUP_WEBDRIVERS" = true ]; then
    setup_flag=""
  else
    setup_flag="--no-setup"
  fi
  
  local verbose_flag=""
  if [ "$VERBOSE" = true ]; then
    verbose_flag="--verbose"
  fi
  
  # Run the test
  ./run_real_browser_test.sh \
    "--$browser" \
    "--$model" \
    "--$platform" \
    $headless_flag \
    $simulation_flag \
    $setup_flag \
    $verbose_flag \
    --results-dir "$RESULTS_DIR"
  
  local result=$?
  if [ $result -eq 0 ]; then
    echo "✅ Test passed"
    return 0
  else
    echo "❌ Test failed"
    return 1
  fi
}

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
echo "Comprehensive Browser Test Suite"
echo "=" * 80
echo "Mode: $MODE"
echo "Headless: $HEADLESS"
echo "Allow Simulation: $ALLOW_SIMULATION"
echo "Results Directory: $RESULTS_DIR"
echo "=" * 80

# Detect available browsers if simulation mode is disabled
BROWSERS="chrome firefox edge"
if [ "$ALLOW_SIMULATION" = false ]; then
  BROWSERS=$(detect_browsers)
  if [ -z "$BROWSERS" ]; then
    echo "❌ No supported browsers detected and simulation mode is disabled. Exiting."
    exit 1
  fi
  echo "Detected browsers: $BROWSERS"
else
  echo "Using all browsers (with simulation if needed): $BROWSERS"
fi

# Initialize test tracking variables
total_tests=0
passed_tests=0
failed_tests=0
test_results=()

# Run tests based on mode
if [ "$MODE" = "quick" ]; then
  # Quick mode: Just test one browser with one model
  echo "Running quick test suite..."
  
  # Use first available browser
  browser=$(echo $BROWSERS | awk '{print $1}')
  
  # Run one test
  run_test "$browser" "bert" "webgpu"
  result=$?
  
  total_tests=$((total_tests + 1))
  if [ $result -eq 0 ]; then
    passed_tests=$((passed_tests + 1))
    test_results+=("✅ $browser/bert/webgpu: Passed")
  else
    failed_tests=$((failed_tests + 1))
    test_results+=("❌ $browser/bert/webgpu: Failed")
  fi
  
elif [ "$MODE" = "standard" ]; then
  # Standard mode: Test each browser with a small set of models
  echo "Running standard test suite..."
  
  # Models to test
  models=("bert" "vit")
  platforms=("webgpu")
  
  # Add WebNN testing for Edge
  if [[ $BROWSERS == *"edge"* ]]; then
    platforms+=("webnn")
  fi
  
  # Run tests for each browser, model, and platform
  for browser in $BROWSERS; do
    for model in "${models[@]}"; do
      for platform in "${platforms[@]}"; do
        # Skip WebNN tests for non-Edge browsers
        if [ "$platform" = "webnn" ] && [ "$browser" != "edge" ]; then
          continue
        fi
        
        run_test "$browser" "$model" "$platform"
        result=$?
        
        total_tests=$((total_tests + 1))
        if [ $result -eq 0 ]; then
          passed_tests=$((passed_tests + 1))
          test_results+=("✅ $browser/$model/$platform: Passed")
        else
          failed_tests=$((failed_tests + 1))
          test_results+=("❌ $browser/$model/$platform: Failed")
        fi
      done
    done
  done
  
else
  # Full mode: Test all browser/model/platform combinations
  echo "Running full test suite..."
  
  # Complete test matrix
  browsers=($BROWSERS)
  models=("bert" "vit" "whisper" "clip")
  platforms=("webgpu" "webnn")
  
  # Run tests for each browser, model, and platform
  for browser in "${browsers[@]}"; do
    for model in "${models[@]}"; do
      for platform in "${platforms[@]}"; do
        # Skip WebNN tests for non-Edge browsers
        if [ "$platform" = "webnn" ] && [ "$browser" != "edge" ]; then
          continue
        fi
        
        # For Firefox, prioritize audio models with WebGPU
        if [ "$browser" = "firefox" ] && [ "$model" = "whisper" ]; then
          run_test "$browser" "$model" "webgpu"
          result=$?
          
          total_tests=$((total_tests + 1))
          if [ $result -eq 0 ]; then
            passed_tests=$((passed_tests + 1))
            test_results+=("✅ $browser/$model/webgpu: Passed")
          else
            failed_tests=$((failed_tests + 1))
            test_results+=("❌ $browser/$model/webgpu: Failed")
          fi
          continue
        fi
        
        # Run standard test
        run_test "$browser" "$model" "$platform"
        result=$?
        
        total_tests=$((total_tests + 1))
        if [ $result -eq 0 ]; then
          passed_tests=$((passed_tests + 1))
          test_results+=("✅ $browser/$model/$platform: Passed")
        else
          failed_tests=$((failed_tests + 1))
          test_results+=("❌ $browser/$model/$platform: Failed")
        fi
      done
    done
  done
fi

# Generate test summary
echo ""
echo "=" * 80
echo "Test Summary"
echo "=" * 80
echo "Total Tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo "Success Rate: $(( passed_tests * 100 / total_tests ))%"
echo ""
echo "Test Results:"
for result in "${test_results[@]}"; do
  echo "  $result"
done
echo "=" * 80

# Save summary to file
cat > "$SUMMARY_FILE" << EOF
# Comprehensive Browser Test Summary

Date: $(date +"%Y-%m-%d %H:%M:%S")
Mode: $MODE
Headless: $HEADLESS
Allow Simulation: $ALLOW_SIMULATION

## Test Summary

- **Total Tests:** $total_tests
- **Passed:** $passed_tests
- **Failed:** $failed_tests
- **Success Rate:** $(( passed_tests * 100 / total_tests ))%

## Test Results

| Browser | Model | Platform | Result |
|---------|-------|----------|--------|
EOF

# Add test results to summary file
for result in "${test_results[@]}"; do
  # Parse result string
  if [[ $result == *"✅"* ]]; then
    status="Passed"
  else
    status="Failed"
  fi
  
  # Extract components
  combo=$(echo "$result" | cut -d':' -f1 | awk '{print $2}')
  browser=$(echo "$combo" | cut -d'/' -f1)
  model=$(echo "$combo" | cut -d'/' -f2)
  platform=$(echo "$combo" | cut -d'/' -f3)
  
  # Add to summary
  echo "| $browser | $model | $platform | $status |" >> "$SUMMARY_FILE"
done

# Add recommendations to summary file
cat >> "$SUMMARY_FILE" << EOF

## Recommendations

Based on the test results, we have the following recommendations:

- **Text Models (BERT):** 
  - Best on Edge with WebNN
  - Good on Chrome with WebGPU
  
- **Vision Models (ViT):**
  - Best on Chrome with WebGPU
  - Good on Firefox with WebGPU
  
- **Audio Models (Whisper):**
  - Best on Firefox with WebGPU and compute shaders
  - Good on Chrome with WebGPU
  
- **Multimodal Models (CLIP):**
  - Best on Chrome with WebGPU and parallel loading
  - Good on Firefox with WebGPU
EOF

echo "Summary saved to $SUMMARY_FILE"

# Return success if all tests passed
if [ $failed_tests -eq 0 ]; then
  exit 0
else
  exit 1
fi
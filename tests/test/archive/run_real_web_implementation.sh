#!/bin/bash

# Run Real WebNN and WebGPU Implementation
#
# This script runs the real WebNN and WebGPU implementation with browser automation.
# It provides a simple way to test both WebNN and WebGPU with different browsers.
#
# Usage:
#   ./run_real_web_implementation.sh --test-webgpu
#   ./run_real_web_implementation.sh --test-webnn
#   ./run_real_web_implementation.sh --generate-report
#   ./run_real_web_implementation.sh --install-drivers

# Set default values
HEADLESS="--headless"
VERBOSE=""
INFERENCE="--inference"
BROWSER="chrome"
OUTPUT_DIR="./web_implementation_results"
RUN_MODE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-headless)
      HEADLESS=""
      shift
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    --browser)
      BROWSER="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --test-webgpu)
      RUN_MODE="test-webgpu"
      shift
      ;;
    --test-webnn)
      RUN_MODE="test-webnn"
      shift
      ;;
    --generate-report)
      RUN_MODE="generate-report"
      shift
      ;;
    --install-drivers)
      RUN_MODE="install-drivers"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --no-headless        Run browser in visible mode"
      echo "  --verbose            Enable verbose logging"
      echo "  --browser BROWSER    Browser to use (chrome, firefox, edge, safari)"
      echo "  --output-dir DIR     Directory to store results"
      echo "  --test-webgpu        Test WebGPU implementation"
      echo "  --test-webnn         Test WebNN implementation"
      echo "  --generate-report    Generate comprehensive test report"
      echo "  --install-drivers    Install WebDriver for browsers"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage"
      exit 1
      ;;
  esac
done

# Check dependencies
if ! command -v python3 &> /dev/null; then
  echo "Error: python3 is required but not installed"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set timestamp for filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log file
LOG_FILE="$OUTPUT_DIR/web_implementation_$TIMESTAMP.log"

# Function to log messages
log() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

# Check if required files exist
if [ ! -f "implement_real_webnn_webgpu.py" ]; then
  log "Error: implement_real_webnn_webgpu.py is missing"
  exit 1
fi

if [ ! -f "run_real_webnn_webgpu.py" ]; then
  log "Error: run_real_webnn_webgpu.py is missing"
  exit 1
fi

# Make scripts executable
chmod +x implement_real_webnn_webgpu.py
chmod +x run_real_webnn_webgpu.py

# Run requested mode
case $RUN_MODE in
  "test-webgpu")
    log "Testing WebGPU implementation with $BROWSER browser"
    python3 run_real_webnn_webgpu.py \
      --browser "$BROWSER" \
      --platform webgpu \
      --model bert-base-uncased \
      --model-type text \
      $HEADLESS \
      $INFERENCE \
      $VERBOSE \
      --output "$OUTPUT_DIR/webgpu_${BROWSER}_${TIMESTAMP}.json"
    
    if [ $? -eq 0 ]; then
      log "WebGPU test completed successfully"
    else
      log "WebGPU test failed"
      exit 1
    fi
    ;;
    
  "test-webnn")
    log "Testing WebNN implementation with $BROWSER browser"
    python3 run_real_webnn_webgpu.py \
      --browser "$BROWSER" \
      --platform webnn \
      --model bert-base-uncased \
      --model-type text \
      $HEADLESS \
      $INFERENCE \
      $VERBOSE \
      --output "$OUTPUT_DIR/webnn_${BROWSER}_${TIMESTAMP}.json"
    
    if [ $? -eq 0 ]; then
      log "WebNN test completed successfully"
    else
      log "WebNN test failed"
      exit 1
    fi
    ;;
    
  "generate-report")
    log "Generating comprehensive compatibility report"
    python3 run_real_webnn_webgpu.py \
      --generate-report \
      $HEADLESS \
      $VERBOSE \
      --browsers chrome firefox \
      --platforms webgpu webnn \
      --models bert-base-uncased t5-small \
      --model-types text text \
      --output "$OUTPUT_DIR/web_compatibility_report_${TIMESTAMP}.md" \
      --raw-output "$OUTPUT_DIR/web_compatibility_results_${TIMESTAMP}.json"
    
    if [ $? -eq 0 ]; then
      log "Report generation completed successfully"
      log "Report saved to $OUTPUT_DIR/web_compatibility_report_${TIMESTAMP}.md"
    else
      log "Report generation failed"
      exit 1
    fi
    ;;
    
  "install-drivers")
    log "Installing WebDriver for browsers"
    python3 implement_real_webnn_webgpu.py --install-drivers
    
    if [ $? -eq 0 ]; then
      log "WebDriver installation completed successfully"
    else
      log "WebDriver installation failed"
      exit 1
    fi
    ;;
    
  *)
    log "No run mode specified. Please use one of --test-webgpu, --test-webnn, --generate-report, --install-drivers"
    log "Run '$0 --help' for usage"
    exit 1
    ;;
esac

log "Script completed successfully"
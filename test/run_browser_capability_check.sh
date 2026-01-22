#!/bin/bash
# Run browser capability check with the correct browser flags

# Default settings
BROWSER="chrome"
OUTPUT_DIR="./browser_capabilities"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --browser)
      BROWSER="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set browser-specific environment variables to enable WebNN/WebGPU
echo "Setting up $BROWSER with WebNN/WebGPU support..."

case "$BROWSER" in
  chrome)
    # Chrome requires WebNN and WebGPU flags
    export CHROME_FLAGS="--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist"
    echo "Using Chrome with flags: $CHROME_FLAGS"
    # Set environment variable to communicate these flags to the benchmark script
    export BROWSER_FLAGS="$CHROME_FLAGS"
    ;;
  edge)
    # Edge requires WebNN and WebGPU flags
    export EDGE_FLAGS="--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist"
    echo "Using Edge with flags: $EDGE_FLAGS"
    # Set environment variable to communicate these flags to the benchmark script
    export BROWSER_FLAGS="$EDGE_FLAGS"
    ;;
  firefox)
    # Firefox WebGPU flags
    export FIREFOX_FLAGS="--MOZ_WEBGPU_FEATURES=dawn --MOZ_ENABLE_WEBGPU=1 --MOZ_WEBGPU_ADVANCED_COMPUTE=1"
    echo "Using Firefox with WebGPU flags: $FIREFOX_FLAGS"
    # Set environment variable to communicate these flags to the benchmark script
    export BROWSER_FLAGS="$FIREFOX_FLAGS"
    ;;
  safari)
    # Safari doesn't support command line flags
    echo "Safari doesn't support command line flags for WebNN/WebGPU. Using default settings."
    export BROWSER_FLAGS=""
    ;;
  *)
    echo "Unsupported browser: $BROWSER"
    echo "Please use chrome, edge, firefox, or safari"
    exit 1
    ;;
esac

# Run the browser capability check
echo "Checking $BROWSER capabilities..."
OUTPUT_FILE="$OUTPUT_DIR/${BROWSER}_capabilities_$(date +%Y%m%d_%H%M%S).json"

# Run the capability check script
python check_browser_capabilities.py --browser "$BROWSER" --no-headless --output "$OUTPUT_FILE" --flags "$BROWSER_FLAGS"

# Check exit code
if [ $? -eq 0 ]; then
  echo "SUCCESS: Browser has WebNN or WebGPU capabilities"
else
  echo "WARNING: Browser doesn't have WebNN or WebGPU capabilities"
fi

echo "Capabilities results saved to: $OUTPUT_FILE"

# Print the WebNN and WebGPU status from the results file
echo -e "\nSummary of WebNN and WebGPU Status:"
if [ -f "$OUTPUT_FILE" ]; then
  # Extract WebNN and WebGPU support status
  WEBNN_SUPPORTED=$(grep -A1 "\"webnn\":" "$OUTPUT_FILE" | grep "supported" | grep -oP 'true|false')
  WEBGPU_SUPPORTED=$(grep -A1 "\"webgpu\":" "$OUTPUT_FILE" | grep "supported" | grep -oP 'true|false')
  
  echo "WebNN Support: ${WEBNN_SUPPORTED:-unknown}"
  echo "WebGPU Support: ${WEBGPU_SUPPORTED:-unknown}"
  
  # Print backends/adapters if available
  WEBNN_BACKENDS=$(grep -A2 "\"backends\":" "$OUTPUT_FILE" | grep -v "backends" | grep -oP '\[\K[^\]]*')
  if [ ! -z "$WEBNN_BACKENDS" ]; then
    echo "WebNN Backends: $WEBNN_BACKENDS"
  fi
  
  WEBGPU_ADAPTER=$(grep -A5 "\"adapter\":" "$OUTPUT_FILE" | grep "vendor" | grep -oP '"vendor": "\K[^"]*')
  if [ ! -z "$WEBGPU_ADAPTER" ]; then
    echo "WebGPU Adapter: $WEBGPU_ADAPTER"
  fi
fi
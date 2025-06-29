#!/bin/bash
# Run WebNN Benchmark with the correct browser flags

# Default settings
BROWSER="chrome"
MODEL="bert-base-uncased"
ITERATIONS=5
BATCH_SIZE=1
OUTPUT_DIR="./benchmark_results"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --browser)
      BROWSER="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
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

# Set browser-specific environment variables to enable WebNN
echo "Setting up $BROWSER with WebNN support..."

case "$BROWSER" in
  chrome)
    # Chrome requires WebNN flags
    export CHROME_WEBNN_FLAGS="--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features"
    echo "Using Chrome with WebNN flags: $CHROME_WEBNN_FLAGS"
    # Set environment variable to communicate these flags to the benchmark script
    export BROWSER_FLAGS="$CHROME_WEBNN_FLAGS"
    ;;
  edge)
    # Edge requires WebNN flags
    export EDGE_WEBNN_FLAGS="--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features"
    echo "Using Edge with WebNN flags: $EDGE_WEBNN_FLAGS"
    # Set environment variable to communicate these flags to the benchmark script
    export BROWSER_FLAGS="$EDGE_WEBNN_FLAGS"
    ;;
  firefox)
    # Firefox doesn't fully support WebNN, but set WebGPU flags for comparison
    export FIREFOX_WEBGPU_FLAGS="--MOZ_WEBGPU_FEATURES=dawn --MOZ_ENABLE_WEBGPU=1 --MOZ_WEBGPU_ADVANCED_COMPUTE=1"
    echo "WARNING: Firefox doesn't fully support WebNN. Using WebGPU flags: $FIREFOX_WEBGPU_FLAGS"
    echo "Consider using Chrome or Edge for WebNN testing."
    # Set environment variable to communicate these flags to the benchmark script
    export BROWSER_FLAGS="$FIREFOX_WEBGPU_FLAGS"
    ;;
  *)
    echo "Unsupported browser: $BROWSER"
    echo "Please use chrome, edge, or firefox"
    exit 1
    ;;
esac

# Run the WebNN benchmark
echo "Running WebNN benchmark with $BROWSER..."
OUTPUT_FILE="$OUTPUT_DIR/webnn_benchmark_${BROWSER}_${MODEL// /_}_$(date +%Y%m%d_%H%M%S).json"

# Set WEBNN_ENABLED environment variable to inform the script
export WEBNN_ENABLED=1

# Run the benchmark script
python test_webnn_benchmark.py --browser "$BROWSER" --model "$MODEL" --iterations "$ITERATIONS" --batch-size "$BATCH_SIZE" --output "$OUTPUT_FILE"

# Check exit code
if [ $? -eq 0 ]; then
  echo "SUCCESS: WebNN is enabled and using real hardware acceleration"
elif [ $? -eq 1 ]; then
  echo "WARNING: WebNN is supported but using simulation mode"
else
  echo "ERROR: WebNN is not supported or failed to initialize"
fi

echo "Benchmark results saved to: $OUTPUT_FILE"
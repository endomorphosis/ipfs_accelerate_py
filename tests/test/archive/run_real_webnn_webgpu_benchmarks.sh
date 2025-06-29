#!/bin/bash
# Run Real WebNN/WebGPU Benchmarks
# This script provides a convenient way to run WebNN/WebGPU benchmarks with different configurations

# Default values
DB_PATH="./benchmark_db.duckdb"
BROWSER=""
PLATFORM=""
MODEL_TYPE=""
MODELS=""
BATCH_SIZES="1,2,4"
BITS=8
MIXED_PRECISION=false
COMPUTE_SHADERS=false
SHADER_PRECOMPILE=false
PARALLEL_LOADING=false
QUICK_TEST=false
VISIBLE=false
COMPREHENSIVE=false
MAX_CONNECTIONS=4

# Print usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --help                 Display this help message"
    echo "  --db-path PATH         Path to database file (default: ./benchmark_db.duckdb)"
    echo "  --chrome               Use Chrome browser"
    echo "  --firefox              Use Firefox browser"
    echo "  --edge                 Use Edge browser"
    echo "  --webnn                Benchmark WebNN platform"
    echo "  --webgpu               Benchmark WebGPU platform"
    echo "  --text                 Benchmark text models"
    echo "  --vision               Benchmark vision models"
    echo "  --audio                Benchmark audio models"
    echo "  --models LIST          Comma-separated list of models to benchmark"
    echo "  --batch-sizes LIST     Comma-separated list of batch sizes (default: 1,2,4)"
    echo "  --bits NUM             Precision level to test (4, 8, 16, 32) (default: 8)"
    echo "  --mixed-precision      Use mixed precision"
    echo "  --compute-shaders      Enable compute shader optimization for audio models"
    echo "  --shader-precompile    Enable shader precompilation for faster startup"
    echo "  --parallel-loading     Enable parallel model loading for multimodal models"
    echo "  --quick-test           Run a quick test with fewer iterations and batch sizes"
    echo "  --visible              Run browsers in visible mode (not headless)"
    echo "  --comprehensive        Run comprehensive benchmarks across all model types"
    echo "  --max-connections NUM  Maximum number of browser connections (default: 4)"
    echo ""
    echo "Examples:"
    echo "  $0 --webgpu --chrome --text"
    echo "  $0 --webnn --edge --models bert-base-uncased,t5-small"
    echo "  $0 --webgpu --firefox --audio --compute-shaders"
    echo "  $0 --comprehensive --db-path ./my_benchmarks.duckdb"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --chrome)
            BROWSER="--chrome"
            shift
            ;;
        --firefox)
            BROWSER="--firefox"
            shift
            ;;
        --edge)
            BROWSER="--edge"
            shift
            ;;
        --webnn)
            PLATFORM="--webnn"
            shift
            ;;
        --webgpu)
            PLATFORM="--webgpu"
            shift
            ;;
        --text)
            MODEL_TYPE="$MODEL_TYPE --text"
            shift
            ;;
        --vision)
            MODEL_TYPE="$MODEL_TYPE --vision"
            shift
            ;;
        --audio)
            MODEL_TYPE="$MODEL_TYPE --audio"
            shift
            ;;
        --models)
            MODELS="--models $2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --bits)
            BITS="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION=true
            shift
            ;;
        --compute-shaders)
            COMPUTE_SHADERS=true
            shift
            ;;
        --shader-precompile)
            SHADER_PRECOMPILE=true
            shift
            ;;
        --parallel-loading)
            PARALLEL_LOADING=true
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --visible)
            VISIBLE=true
            shift
            ;;
        --comprehensive)
            COMPREHENSIVE=true
            shift
            ;;
        --max-connections)
            MAX_CONNECTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Build command
CMD="python run_real_webnn_webgpu_benchmarks.py"

# Add database path
CMD="$CMD --db-path $DB_PATH"

# Add browser selection
if [ ! -z "$BROWSER" ]; then
    CMD="$CMD $BROWSER"
fi

# Add platform selection
if [ ! -z "$PLATFORM" ]; then
    CMD="$CMD $PLATFORM"
fi

# Add model type selection
if [ ! -z "$MODEL_TYPE" ]; then
    CMD="$CMD $MODEL_TYPE"
fi

# Add models
if [ ! -z "$MODELS" ]; then
    CMD="$CMD $MODELS"
fi

# Add batch sizes
CMD="$CMD --batch-sizes $BATCH_SIZES"

# Add precision
CMD="$CMD --bits $BITS"

# Add optimization flags
if [ "$MIXED_PRECISION" = true ]; then
    CMD="$CMD --mixed-precision"
fi

if [ "$COMPUTE_SHADERS" = true ]; then
    CMD="$CMD --compute-shaders"
fi

if [ "$SHADER_PRECOMPILE" = true ]; then
    CMD="$CMD --shader-precompile"
fi

if [ "$PARALLEL_LOADING" = true ]; then
    CMD="$CMD --parallel-loading"
fi

# Add test configuration
if [ "$QUICK_TEST" = true ]; then
    CMD="$CMD --quick-test"
fi

if [ "$VISIBLE" = true ]; then
    CMD="$CMD --visible"
fi

if [ "$COMPREHENSIVE" = true ]; then
    CMD="$CMD --comprehensive"
fi

# Add max connections
CMD="$CMD --max-connections $MAX_CONNECTIONS"

# Print command
echo "Executing: $CMD"

# Run command
eval $CMD
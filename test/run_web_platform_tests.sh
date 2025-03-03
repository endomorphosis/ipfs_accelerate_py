#!/bin/bash
# Unified Web Platform Test Runner with Database Integration
# Supports all March 2025 optimizations: compute shaders, parallel loading, shader precompilation

set -e

# Default values
BROWSER="chrome"
DB_PATH="${BENCHMARK_DB_PATH:-./benchmark_db.duckdb}"
BROWSER_AUTOMATION=false
MODEL="bert"
TEST_SCRIPT="./web_platform_test_runner.py"
ENABLE_COMPUTE_SHADERS=false
ENABLE_PARALLEL_LOADING=false
ENABLE_SHADER_PRECOMPILE=false
ALL_OPTIMIZATIONS=false
WEBNN_ONLY=false
WEBGPU_ONLY=false
GENERATE_REPORT=false
TEST_MODE="standard"
USE_FIREFOX=false
COMPARE_BROWSERS=false

# Display help information
function show_help {
    echo "Unified Web Platform Test Runner"
    echo "--------------------------------"
    echo "Usage: $0 [options] [python_script] [script_args...]"
    echo ""
    echo "Options:"
    echo "  --browser <browser>         Specify browser (chrome, edge, firefox) [default: chrome]"
    echo "  --db-path <path>            Path to benchmark database [default: $DB_PATH]"
    echo "  --use-browser-automation    Use browser automation for testing"
    echo "  --model <model>             Model to test [default: bert]"
    echo "  --models <model1,model2>    Comma-separated list of models to test"
    echo "  --platform <platform>       Platform to test (webnn, webgpu) [default: auto]"
    echo "  --small-models              Use smaller models for faster testing"
    echo "  --webnn-only                Test only WebNN platform"
    echo "  --webgpu-only               Test only WebGPU platform"
    echo "  --enable-compute-shaders    Enable compute shader optimization for audio models"
    echo "  --enable-parallel-loading   Enable parallel loading for multimodal models"
    echo "  --enable-shader-precompile  Enable shader precompilation"
    echo "  --all-optimizations         Enable all optimizations (all three above)"
    echo "  --run-optimizations         Run optimization-specific tests"
    echo "  --generate-report           Generate visual performance report"
    echo "  --firefox                   Use Firefox-specific WebGPU optimizations"
    echo "  --compare-browsers          Compare Firefox vs Chrome performance"
    echo "  --help                      Display this help message"
    echo ""
    echo "Examples:"
    echo "  # Run standard tests for BERT model with WebGPU"
    echo "  $0 --model bert --webgpu-only"
    echo ""
    echo "  # Run tests with all optimizations enabled"
    echo "  $0 --model whisper --all-optimizations"
    echo ""
    echo "  # Run tests with Firefox-specific optimizations"
    echo "  $0 --firefox --enable-compute-shaders --model whisper"
    echo ""
    echo "  # Run browser comparison test"
    echo "  $0 --compare-browsers --model whisper"
    echo ""
    echo "  # Run specific optimization tests with database integration"
    echo "  $0 --run-optimizations --db-path ./benchmark_db.duckdb"
    echo ""
    echo "  # Run tests with browser automation"
    echo "  $0 --use-browser-automation --browser edge --model clip"
    echo ""
    echo "  # Run tests for multiple models with shader precompilation"
    echo "  $0 --models bert,t5,vit --enable-shader-precompile"
    echo ""
    echo "  # Run custom Python script with arguments"
    echo "  $0 --all-optimizations ./my_custom_test.py --custom-arg value"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --browser)
            BROWSER="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --use-browser-automation)
            BROWSER_AUTOMATION=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --small-models)
            SMALL_MODELS=true
            shift
            ;;
        --webnn-only)
            WEBNN_ONLY=true
            WEBGPU_ONLY=false
            shift
            ;;
        --webgpu-only)
            WEBGPU_ONLY=true
            WEBNN_ONLY=false
            shift
            ;;
        --enable-compute-shaders)
            ENABLE_COMPUTE_SHADERS=true
            shift
            ;;
        --enable-parallel-loading)
            ENABLE_PARALLEL_LOADING=true
            shift
            ;;
        --enable-shader-precompile)
            ENABLE_SHADER_PRECOMPILE=true
            shift
            ;;
        --all-optimizations)
            ALL_OPTIMIZATIONS=true
            ENABLE_COMPUTE_SHADERS=true
            ENABLE_PARALLEL_LOADING=true
            ENABLE_SHADER_PRECOMPILE=true
            shift
            ;;
        --run-optimizations)
            TEST_MODE="optimizations"
            TEST_SCRIPT="./test_web_platform_optimizations.py"
            shift
            ;;
        --generate-report)
            GENERATE_REPORT=true
            shift
            ;;
        --firefox)
            USE_FIREFOX=true
            BROWSER="firefox"
            shift
            ;;
        --compare-browsers)
            COMPARE_BROWSERS=true
            TEST_SCRIPT="./test_firefox_webgpu_compute_shaders.py"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            # Check if this is a Python script
            if [[ "$1" == *.py ]]; then
                TEST_SCRIPT="$1"
                shift
                break
            else
                echo "Unknown option: $1"
                show_help
                exit 1
            fi
            ;;
    esac
done

# Set environment variables for testing
export BENCHMARK_DB_PATH="$DB_PATH"
export TEST_BROWSER="$BROWSER"

# Enable WebNN and WebGPU testing
export WEBNN_ENABLED=1
export WEBGPU_ENABLED=1

# Firefox-specific settings
if [ "$USE_FIREFOX" = true ]; then
    echo "Using Firefox-specific WebGPU optimizations"
    export BROWSER_PREFERENCE="firefox"
    export USE_FIREFOX_WEBGPU=1
    export MOZ_WEBGPU_ADVANCED_COMPUTE=1
fi

# Browser comparison mode
if [ "$COMPARE_BROWSERS" = true ]; then
    echo "Running browser comparison between Firefox and Chrome"
    export COMPARE_BROWSERS=1
fi

# Platform-specific settings
if [ "$WEBNN_ONLY" = true ]; then
    echo "Testing with WebNN only"
    unset WEBGPU_ENABLED
    export WEBNN_ENABLED=1
fi

if [ "$WEBGPU_ONLY" = true ]; then
    echo "Testing with WebGPU only"
    unset WEBNN_ENABLED
    export WEBGPU_ENABLED=1
fi

# Set browser automation if requested
if [ "$BROWSER_AUTOMATION" = true ]; then
    echo "Using browser automation with $BROWSER"
    export USE_BROWSER_AUTOMATION=1
    export BROWSER_TYPE="$BROWSER"
fi

# Set optimization environment variables
if [ "$ENABLE_COMPUTE_SHADERS" = true ]; then
    echo "Enabling WebGPU compute shader optimization"
    export WEBGPU_COMPUTE_SHADERS_ENABLED=1
fi

if [ "$ENABLE_PARALLEL_LOADING" = true ]; then
    echo "Enabling parallel model loading optimization"
    export WEB_PARALLEL_LOADING_ENABLED=1
fi

if [ "$ENABLE_SHADER_PRECOMPILE" = true ]; then
    echo "Enabling shader precompilation optimization"
    export WEBGPU_SHADER_PRECOMPILE_ENABLED=1
fi

# No JSON output, only database
export DEPRECATE_JSON_OUTPUT=1

# Determine Python command
PYTHON_CMD=$(which python3 2>/dev/null || which python)

# Build command arguments based on test mode
if [ "$TEST_MODE" = "optimizations" ]; then
    # Running optimization-specific tests
    CMD_ARGS=""
    
    if [ "$ALL_OPTIMIZATIONS" = true ]; then
        CMD_ARGS="--all-optimizations"
    else
        if [ "$ENABLE_COMPUTE_SHADERS" = true ]; then
            CMD_ARGS="$CMD_ARGS --compute-shaders"
        fi
        
        if [ "$ENABLE_PARALLEL_LOADING" = true ]; then
            CMD_ARGS="$CMD_ARGS --parallel-loading"
        fi
        
        if [ "$ENABLE_SHADER_PRECOMPILE" = true ]; then
            CMD_ARGS="$CMD_ARGS --shader-precompile"
        fi
    fi
    
    # Add model if specified
    if [ -n "$MODEL" ]; then
        CMD_ARGS="$CMD_ARGS --model $MODEL"
    fi
    
    # Add database path
    CMD_ARGS="$CMD_ARGS --db-path $DB_PATH"
    
    # Add report generation if requested
    if [ "$GENERATE_REPORT" = true ]; then
        CMD_ARGS="$CMD_ARGS --generate-report"
    fi
    
    # Execute the command
    echo "Running: $PYTHON_CMD $TEST_SCRIPT $CMD_ARGS $@"
    $PYTHON_CMD $TEST_SCRIPT $CMD_ARGS "$@"
else
    # Running standard web platform tests
    CMD_ARGS=""
    
    # Add model or models if specified
    if [ -n "$MODELS" ]; then
        CMD_ARGS="$CMD_ARGS --models $MODELS"
    elif [ -n "$MODEL" ]; then
        CMD_ARGS="$CMD_ARGS --model $MODEL"
    fi
    
    # Add platform if specified
    if [ -n "$PLATFORM" ]; then
        CMD_ARGS="$CMD_ARGS --platform $PLATFORM"
    fi
    
    # Add small models flag if requested
    if [ "$SMALL_MODELS" = true ]; then
        CMD_ARGS="$CMD_ARGS --small-models"
    fi
    
    # Add database-specific arguments if using run_web_platform_tests_with_db.py
    if [[ "$TEST_SCRIPT" == *"_with_db.py" ]]; then
        CMD_ARGS="$CMD_ARGS --db-path $DB_PATH"
    elif [[ "$TEST_SCRIPT" == *"test_firefox_webgpu_compute_shaders.py" ]]; then
        # Special handling for Firefox vs Chrome comparison
        if [ -n "$MODEL" ]; then
            CMD_ARGS="$CMD_ARGS --model $MODEL"
        fi
        CMD_ARGS="$CMD_ARGS --create-charts --output-dir ./firefox_webgpu_results"
    fi
    
    # Execute the command
    echo "Running: $PYTHON_CMD $TEST_SCRIPT $CMD_ARGS $@"
    $PYTHON_CMD $TEST_SCRIPT $CMD_ARGS "$@"
fi

echo "Web platform tests completed successfully!"
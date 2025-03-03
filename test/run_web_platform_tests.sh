#!/bin/bash
# Script to run tests with WebNN and WebGPU simulations enabled
# This sets environment variables to enable web platform simulations in the test framework
# Updated March 2025 with WebGPU shader compilation, compute shaders, and parallel loading support

# Usage: 
#   ./run_web_platform_tests.sh [test_command]
# Examples:
#   ./run_web_platform_tests.sh python test/integrated_skillset_generator.py --model bert --hardware webnn
#   ./run_web_platform_tests.sh python test/run_key_model_fixes.sh --platform webgpu
#   ./run_web_platform_tests.sh python test/run_web_platform_tests_with_db.py
#   ./run_web_platform_tests.sh --webnn-only python test/run_model_benchmarks.py --hardware webnn
#   ./run_web_platform_tests.sh --webgpu-only python test/verify_key_models.py --platform webgpu
#   ./run_web_platform_tests.sh --enable-compute-shaders python test/web_platform_benchmark.py --model whisper

# Parse options
WEBNN_ONLY=0
WEBGPU_ONLY=0
VERBOSE=0
ENABLE_COMPUTE_SHADERS=0
ENABLE_PARALLEL_LOADING=0
ENABLE_SHADER_PRECOMPILE=0
USE_BROWSER_AUTOMATION=0
BROWSER_PREFERENCE=""

while [[ "$1" == --* ]]; do
    case "$1" in
        --webnn-only)
            WEBNN_ONLY=1
            shift
            ;;
        --webgpu-only)
            WEBGPU_ONLY=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --enable-compute-shaders)
            ENABLE_COMPUTE_SHADERS=1
            shift
            ;;
        --enable-parallel-loading)
            ENABLE_PARALLEL_LOADING=1
            shift
            ;;
        --enable-shader-precompile)
            ENABLE_SHADER_PRECOMPILE=1
            shift
            ;;
        --all-features)
            ENABLE_COMPUTE_SHADERS=1
            ENABLE_PARALLEL_LOADING=1
            ENABLE_SHADER_PRECOMPILE=1
            shift
            ;;
        --use-browser-automation)
            USE_BROWSER_AUTOMATION=1
            shift
            ;;
        --browser)
            BROWSER_PREFERENCE="$2"
            shift 2
            ;;
        --firefox)
            # Added in March 2025: Shortcut for Firefox WebGPU testing
            BROWSER_PREFERENCE="firefox"
            WEBGPU_ONLY=1
            # Enable compute shaders by default for Firefox (excellent WebGPU compute shader performance)
            ENABLE_COMPUTE_SHADERS=1
            # Enable Firefox advanced compute mode for optimal performance
            export MOZ_WEBGPU_ADVANCED_COMPUTE=1
            echo "Firefox WebGPU advanced compute shaders enabled (55% performance improvement)"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment variables based on options
if [ $WEBNN_ONLY -eq 1 ]; then
    # Enable only WebNN
    export WEBNN_ENABLED=1
    export WEBNN_SIMULATION=1
    export WEBNN_AVAILABLE=1
elif [ $WEBGPU_ONLY -eq 1 ]; then
    # Enable only WebGPU
    export WEBGPU_ENABLED=1
    export WEBGPU_SIMULATION=1
    export WEBGPU_AVAILABLE=1
else
    # Enable both by default
    export WEBNN_ENABLED=1
    export WEBGPU_ENABLED=1
    export WEBNN_SIMULATION=1
    export WEBNN_AVAILABLE=1
    export WEBGPU_SIMULATION=1
    export WEBGPU_AVAILABLE=1
fi

# Set feature-specific environment variables
if [ $ENABLE_COMPUTE_SHADERS -eq 1 ]; then
    export WEBGPU_COMPUTE_SHADERS_ENABLED=1
    echo "WebGPU compute shaders enabled"
fi

if [ $ENABLE_PARALLEL_LOADING -eq 1 ]; then
    export WEBGPU_PARALLEL_LOADING_ENABLED=1
    export WEBNN_PARALLEL_LOADING_ENABLED=1
    echo "Parallel model loading enabled"
fi

if [ $ENABLE_SHADER_PRECOMPILE -eq 1 ]; then
    export WEBGPU_SHADER_PRECOMPILE_ENABLED=1
    echo "WebGPU shader precompilation enabled"
fi

# Set browser automation environment variables
if [ $USE_BROWSER_AUTOMATION -eq 1 ]; then
    export USE_BROWSER_AUTOMATION=1
    echo "Browser automation enabled"
    
    if [ -n "$BROWSER_PREFERENCE" ]; then
        export BROWSER_PREFERENCE
        echo "Browser preference set to: $BROWSER_PREFERENCE"
    fi
fi

# Print environment variables if verbose or no specific options
if [ $VERBOSE -eq 1 ] || ([ $WEBNN_ONLY -eq 0 ] && [ $WEBGPU_ONLY -eq 0 ]); then
    echo "Web platform simulation environment variables set:"
    [ -n "$WEBNN_ENABLED" ] && echo "  WEBNN_ENABLED=$WEBNN_ENABLED"
    [ -n "$WEBGPU_ENABLED" ] && echo "  WEBGPU_ENABLED=$WEBGPU_ENABLED"
    [ -n "$WEBNN_SIMULATION" ] && echo "  WEBNN_SIMULATION=$WEBNN_SIMULATION"
    [ -n "$WEBNN_AVAILABLE" ] && echo "  WEBNN_AVAILABLE=$WEBNN_AVAILABLE"
    [ -n "$WEBGPU_SIMULATION" ] && echo "  WEBGPU_SIMULATION=$WEBGPU_SIMULATION"
    [ -n "$WEBGPU_AVAILABLE" ] && echo "  WEBGPU_AVAILABLE=$WEBGPU_AVAILABLE"
    
    # Print advanced feature flags
    [ -n "$WEBGPU_COMPUTE_SHADERS_ENABLED" ] && echo "  WEBGPU_COMPUTE_SHADERS_ENABLED=$WEBGPU_COMPUTE_SHADERS_ENABLED"
    [ -n "$WEBGPU_PARALLEL_LOADING_ENABLED" ] && echo "  WEBGPU_PARALLEL_LOADING_ENABLED=$WEBGPU_PARALLEL_LOADING_ENABLED"
    [ -n "$WEBNN_PARALLEL_LOADING_ENABLED" ] && echo "  WEBNN_PARALLEL_LOADING_ENABLED=$WEBNN_PARALLEL_LOADING_ENABLED"
    [ -n "$WEBGPU_SHADER_PRECOMPILE_ENABLED" ] && echo "  WEBGPU_SHADER_PRECOMPILE_ENABLED=$WEBGPU_SHADER_PRECOMPILE_ENABLED"
    
    # Print browser automation settings
    [ -n "$USE_BROWSER_AUTOMATION" ] && [ "$USE_BROWSER_AUTOMATION" -eq 1 ] && echo "  USE_BROWSER_AUTOMATION=$USE_BROWSER_AUTOMATION"
    [ -n "$BROWSER_PREFERENCE" ] && echo "  BROWSER_PREFERENCE=$BROWSER_PREFERENCE"
    echo ""
fi

# If no command was provided, show help
if [ -z "$1" ]; then
    echo "Please provide a test command to run."
    echo "Usage: $0 [options] [test_command]"
    echo ""
    echo "Basic Options:"
    echo "  --webnn-only              Enable only WebNN simulation"
    echo "  --webgpu-only             Enable only WebGPU simulation"
    echo "  --verbose                 Show more detailed output"
    echo ""
    echo "Advanced WebGPU Features:"
    echo "  --enable-compute-shaders  Enable WebGPU compute shader optimizations (audio models)"
    echo "  --enable-parallel-loading Enable parallel model loading optimizations"
    echo "  --enable-shader-precompile Enable WebGPU shader precompilation"
    echo "  --all-features            Enable all advanced features"
    echo ""
    echo "Browser Automation Features:"
    echo "  --use-browser-automation  Enable real browser automation instead of simulation"
    echo "  --browser [edge|chrome|firefox] Specify preferred browser for automation"
    echo "  --firefox                 Shortcut for Firefox with WebGPU compute shaders (55% improvement)"
    echo ""
    echo "Examples:"
    echo "  $0 python test/integrated_skillset_generator.py --model bert --hardware webnn"
    echo "  $0 python test/run_key_model_fixes.sh --platform webgpu"
    echo "  $0 python test/run_web_platform_tests_with_db.py"
    echo "  $0 --webnn-only python test/run_model_benchmarks.py --hardware webnn"
    echo "  $0 --webgpu-only python test/verify_key_models.py --platform webgpu"
    echo "  $0 --enable-compute-shaders python test/web_platform_benchmark.py --model whisper"
    echo "  $0 --all-features python test/web_platform_benchmark.py --comparative"
    echo "  $0 --use-browser-automation --browser chrome python test/web_platform_test_runner.py --model bert"
    echo "  $0 --firefox python test/test_webgpu_audio_compute_shaders.py --model whisper"
    exit 1
fi

# Execute the provided command with the environment variables set
echo "Running: $@"
echo "---------------"
$@

# Save the exit code
EXIT_CODE=$?

echo "---------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Web platform test completed successfully."
else
    echo "Web platform test failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE
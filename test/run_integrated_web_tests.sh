#!/bin/bash
# Integrated Web Platform Test Runner
# Combines all test types with database integration for Phase 16

set -e

# Default values
DB_PATH="${BENCHMARK_DB_PATH:-./benchmark_db.duckdb}"
TEST_TYPE="standard"
MODEL="bert"
HARDWARE="all"
SMALL_MODELS=true
REPORT_DIR="./web_platform_reports"
RUN_ALL=false
MARCH_2025_FEATURES=false
RUN_MAY_2025_FEATURES=false
CROSS_PLATFORM=false
MODE="integration"
TIMEOUT=1800  # 30 minute timeout for comprehensive tests

# Display help information
function show_help {
    echo "Integrated Web Platform Test Runner"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --db-path <path>              Path to benchmark database [default: $DB_PATH]"
    echo "  --test-type <type>            Test type (standard, optimization, hardware, cross-platform)"
    echo "  --model <model>               Model to test [default: bert]"
    echo "  --models <model1,model2>      Comma-separated list of models to test"
    echo "  --hardware <type>             Hardware type to test (all, webnn, webgpu)"
    echo "  --report-dir <dir>            Directory for reports [default: $REPORT_DIR]"
    echo "  --small-models                Use smaller model variants [default: true]"
    echo "  --march-2025-features         Enable March 2025 features (compute shaders, parallel loading, shader precompilation)"
    echo "  --may-2025-features           Enable May 2025 features (4-bit inference, efficient KV-cache, component cache)"
    echo "  --cross-platform              Run cross-platform comparison tests"
    echo "  --run-all                     Run all tests (comprehensive test suite)"
    echo "  --mode <mode>                 Run mode (integration, benchmark, validation)"
    echo "  --timeout <seconds>           Timeout in seconds [default: 1800]"
    echo "  --help                        Display this help message"
    echo ""
    echo "Examples:"
    echo "  # Run standard tests for BERT with database integration"
    echo "  $0 --model bert"
    echo ""
    echo "  # Run optimization tests with all March 2025 features"
    echo "  $0 --test-type optimization --march-2025-features"
    echo ""
    echo "  # Run hardware compatibility tests for all models"
    echo "  $0 --test-type hardware --models all"
    echo ""
    echo "  # Run cross-platform comparison"
    echo "  $0 --cross-platform --model whisper"
    echo ""
    echo "  # Run comprehensive test suite with all features"
    echo "  $0 --run-all"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --hardware)
            HARDWARE="$2"
            shift 2
            ;;
        --report-dir)
            REPORT_DIR="$2"
            shift 2
            ;;
        --small-models)
            SMALL_MODELS=true
            shift
            ;;
        --march-2025-features)
            MARCH_2025_FEATURES=true
            shift
            ;;
        --may-2025-features)
            RUN_MAY_2025_FEATURES=true
            shift
            ;;
        --cross-platform)
            CROSS_PLATFORM=true
            shift
            ;;
        --run-all)
            RUN_ALL=true
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create report directory
mkdir -p "$REPORT_DIR"

# Export database path
export BENCHMARK_DB_PATH="$DB_PATH"

# Set common environment variables
export DEPRECATE_JSON_OUTPUT=1  # Only use database for storage

# Determine Python command
PYTHON_CMD=$(which python3 2>/dev/null || which python)

# Report start time
start_time=$(date +%s)
echo "Starting integrated web tests at $(date)"
echo "Database path: $DB_PATH"
echo "Report directory: $REPORT_DIR"

# Function to run optimization tests
function run_optimization_tests() {
    local model=$1
    local hardware=$2
    
    echo "Running optimization tests for $model on $hardware..."
    
    # Set up environment variables for March 2025 features
    if [ "$MARCH_2025_FEATURES" = true ]; then
        export WEBGPU_COMPUTE_SHADERS_ENABLED=1
        export WEB_PARALLEL_LOADING_ENABLED=1
        export WEBGPU_SHADER_PRECOMPILE_ENABLED=1
        
        echo "Enabled March 2025 features: compute shaders, parallel loading, shader precompilation"
    fi
    
    # Set up environment variables for May 2025 features
    if [ "$RUN_MAY_2025_FEATURES" = true ]; then
        export WEBGPU_4BIT_INFERENCE=1
        export WEBGPU_EFFICIENT_KV_CACHE=1
        export WEB_COMPONENT_CACHE=1
        
        echo "Enabled May 2025 features: 4-bit inference, efficient KV-cache, component cache"
    fi
    
    # Construct the command for optimization tests
    cmd=("$PYTHON_CMD" "test_web_platform_optimizations.py")
    
    # Add features
    if [ "$MARCH_2025_FEATURES" = true ]; then
        cmd+=("--all-optimizations")
    else
        if [ "$hardware" = "webgpu" ]; then
            # Determine which optimization to enable based on model
            if [[ "$model" == "whisper" || "$model" == "wav2vec2" || "$model" == "clap" ]]; then
                cmd+=("--compute-shaders")
            elif [[ "$model" == "clip" || "$model" == "llava" || "$model" == "xclip" ]]; then
                cmd+=("--parallel-loading")
            else
                cmd+=("--shader-precompile")
            fi
        fi
    fi
    
    # Add model
    cmd+=("--model" "$model")
    
    # Add database path
    cmd+=("--db-path" "$DB_PATH")
    
    # Add report generation
    cmd+=("--generate-report")
    
    # Run the command
    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Optimization tests for $model completed successfully."
    else
        echo "Optimization tests for $model failed."
    fi
}

# Function to run hardware compatibility tests
function run_hardware_tests() {
    local model=$1
    
    echo "Running hardware compatibility tests for $model..."
    
    # Construct the command
    cmd=("$PYTHON_CMD" "hardware_compatibility_reporter.py" "--check-model" "$model" "--web-focus")
    
    # Add output options
    cmd+=("--output" "$REPORT_DIR/hardware_${model}_$(date +%Y%m%d_%H%M%S).md")
    
    # Run the command
    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Hardware compatibility tests for $model completed successfully."
    else
        echo "Hardware compatibility tests for $model failed."
    fi
}

# Function to run cross-platform tests
function run_cross_platform_tests() {
    local model=$1
    
    echo "Running cross-platform tests for $model..."
    
    # Construct the command
    cmd=("$PYTHON_CMD" "web_platform_benchmark.py" "--model" "$model" "--compare")
    
    # Add small models flag if requested
    if [ "$SMALL_MODELS" = true ]; then
        cmd+=("--small-models")
    fi
    
    # Add database path
    cmd+=("--db-path" "$DB_PATH")
    
    # Add output options
    cmd+=("--output" "$REPORT_DIR/cross_platform_${model}_$(date +%Y%m%d_%H%M%S).html")
    
    # Run the command
    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Cross-platform tests for $model completed successfully."
    else
        echo "Cross-platform tests for $model failed."
    fi
}

# Function to run standard web platform tests
function run_standard_tests() {
    local model=$1
    local hardware=$2
    
    echo "Running standard web platform tests for $model on $hardware..."
    
    # Construct the command
    cmd=("$PYTHON_CMD" "run_web_platform_tests_with_db.py")
    
    # Add model
    cmd+=("--models" "$model")
    
    # Add hardware
    if [ "$hardware" != "all" ]; then
        if [ "$hardware" = "webnn" ]; then
            cmd+=("--run-webnn")
        elif [ "$hardware" = "webgpu" ]; then
            cmd+=("--run-webgpu")
        fi
    fi
    
    # Add March 2025 features if requested
    if [ "$MARCH_2025_FEATURES" = true ]; then
        cmd+=("--compute-shaders" "--parallel-loading" "--shader-precompile")
    fi
    
    # Add small models flag if requested
    if [ "$SMALL_MODELS" = true ]; then
        cmd+=("--small-models")
    fi
    
    # Add database path
    cmd+=("--db-path" "$DB_PATH")
    
    # Run the command
    echo "Running: ${cmd[*]}"
    timeout "$TIMEOUT" "${cmd[@]}"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Standard tests for $model on $hardware completed successfully."
    else
        echo "Standard tests for $model on $hardware failed."
    fi
}

# Function to process models list
function process_models() {
    local models_arg=$1
    local model_list=()
    
    if [ "$models_arg" = "all" ]; then
        # All high priority models
        model_list=("bert" "t5" "llama" "clip" "vit" "wav2vec2" "whisper" "clap" "llava" "xclip" "qwen2" "detr")
    else
        # Split comma-separated list
        IFS=',' read -ra model_list <<< "$models_arg"
    fi
    
    echo "${model_list[@]}"
}

# Process models
if [ -n "$MODELS" ]; then
    MODELS_LIST=($(process_models "$MODELS"))
else
    MODELS_LIST=("$MODEL")
fi

# Run all tests if requested
if [ "$RUN_ALL" = true ]; then
    echo "Running all tests (comprehensive test suite)..."
    
    # Get all model types
    ALL_MODELS=("bert" "t5" "llama" "clip" "vit" "wav2vec2" "whisper" "clap" "llava" "xclip" "qwen2" "detr")
    
    # Run optimization tests for each model type
    echo "Running optimization tests for all model types..."
    for model in "${ALL_MODELS[@]}"; do
        run_optimization_tests "$model" "webgpu"
    done
    
    # Run hardware compatibility tests for each model type
    echo "Running hardware compatibility tests for all model types..."
    for model in "${ALL_MODELS[@]}"; do
        run_hardware_tests "$model"
    done
    
    # Run cross-platform tests for each model type
    echo "Running cross-platform tests for all model types..."
    for model in "${ALL_MODELS[@]}"; do
        run_cross_platform_tests "$model"
    done
    
    # Run standard tests for each model type on all platforms
    echo "Running standard tests for all model types on all platforms..."
    for model in "${ALL_MODELS[@]}"; do
        for hw in "webnn" "webgpu"; do
            run_standard_tests "$model" "$hw"
        done
    done
    
else
    # Run tests based on test type
    case "$TEST_TYPE" in
        optimization)
            for model in "${MODELS_LIST[@]}"; do
                run_optimization_tests "$model" "$HARDWARE"
            done
            ;;
        hardware)
            for model in "${MODELS_LIST[@]}"; do
                run_hardware_tests "$model"
            done
            ;;
        cross-platform)
            for model in "${MODELS_LIST[@]}"; do
                run_cross_platform_tests "$model"
            done
            ;;
        standard)
            for model in "${MODELS_LIST[@]}"; do
                if [ "$HARDWARE" = "all" ]; then
                    for hw in "webnn" "webgpu"; do
                        run_standard_tests "$model" "$hw"
                    done
                else
                    run_standard_tests "$model" "$HARDWARE"
                fi
            done
            ;;
        *)
            echo "Unknown test type: $TEST_TYPE"
            exit 1
            ;;
    esac
fi

# If cross-platform is enabled, also run cross-platform tests
if [ "$CROSS_PLATFORM" = true ] && [ "$TEST_TYPE" != "cross-platform" ]; then
    for model in "${MODELS_LIST[@]}"; do
        run_cross_platform_tests "$model"
    done
fi

# Report end time and duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Completed integrated web tests at $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"

# Generate comprehensive report
if [ "$MODE" = "integration" ] || [ "$MODE" = "benchmark" ]; then
    echo "Generating comprehensive report from database..."
    "$PYTHON_CMD" "scripts/benchmark_db_query.py" --report web_platform --format html --output "$REPORT_DIR/web_platform_report_$(date +%Y%m%d_%H%M%S).html"
fi

echo "All tests completed successfully! Reports are available in $REPORT_DIR"
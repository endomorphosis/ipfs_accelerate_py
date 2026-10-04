#!/bin/bash
# Script to run the parallel model loading tests with different configurations
# This demonstrates the performance benefits of parallel model loading for multimodal models

# Usage:
#  ./test_run_parallel_model_loading.sh [--all-models | --model MODEL_TYPE | --model-name MODEL_NAME]

MODELS=()
MODEL_NAME=""
ALL_MODELS=0
CREATE_CHART=0
BENCHMARK=0
UPDATE_HANDLER=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-models)
            ALL_MODELS=1
            shift
            ;;
        --model)
            MODELS+=("$2")
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --create-chart)
            CREATE_CHART=1
            shift
            ;;
        --benchmark)
            BENCHMARK=1
            shift
            ;;
        --update-handler)
            UPDATE_HANDLER=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all-models | --model MODEL_TYPE | --model-name MODEL_NAME] [--create-chart] [--benchmark] [--update-handler]"
            exit 1
            ;;
    esac
done

# Print header
echo "========================================================"
echo "  WebGPU Parallel Model Loading Test Suite"
echo "========================================================"
echo ""

# Check if the test file exists
if [ ! -f "test_webgpu_parallel_model_loading.py" ]; then
    echo "Error: test_webgpu_parallel_model_loading.py not found!"
    exit 1
fi

# Update the handler if requested
if [ $UPDATE_HANDLER -eq 1 ]; then
    echo "Updating the web platform handler with enhanced parallel loading..."
    python test_webgpu_parallel_model_loading.py --update-handler
    if [ $? -ne 0 ]; then
        echo "Error: Failed to update handler!"
        exit 1
    fi
    echo ""
fi

# Run tests based on arguments
if [ $ALL_MODELS -eq 1 ]; then
    # Run tests for all models
    echo "Running tests for all supported model types..."
    CMD="python test_webgpu_parallel_model_loading.py --test-all"
    
    if [ $BENCHMARK -eq 1 ]; then
        CMD="$CMD --benchmark"
    fi
    
    if [ $CREATE_CHART -eq 1 ]; then
        CMD="$CMD --create-chart"
    fi
    
    echo "Executing: $CMD"
    $CMD
    
elif [ ${#MODELS[@]} -gt 0 ]; then
    # Run tests for specific model types
    for model in "${MODELS[@]}"; do
        echo "Running tests for model type: $model"
        CMD="python test_webgpu_parallel_model_loading.py --model-type $model"
        
        if [ $BENCHMARK -eq 1 ]; then
            CMD="$CMD --benchmark"
        fi
        
        if [ $CREATE_CHART -eq 1 ]; then
            CMD="$CMD --create-chart"
        fi
        
        echo "Executing: $CMD"
        $CMD
        echo ""
    done
    
elif [ -n "$MODEL_NAME" ]; then
    # Run tests for a specific model name
    echo "Running tests for model: $MODEL_NAME"
    CMD="python test_webgpu_parallel_model_loading.py --model-name \"$MODEL_NAME\""
    
    if [ $BENCHMARK -eq 1 ]; then
        CMD="$CMD --benchmark"
    fi
    
    if [ $CREATE_CHART -eq 1 ]; then
        CMD="$CMD --create-chart"
    fi
    
    echo "Executing: $CMD"
    eval $CMD
    
else
    # No specific models provided, run the default test
    echo "Running default test for multimodal model..."
    CMD="python test_webgpu_parallel_model_loading.py --model-type multimodal"
    
    if [ $BENCHMARK -eq 1 ]; then
        CMD="$CMD --benchmark"
    fi
    
    if [ $CREATE_CHART -eq 1 ]; then
        CMD="$CMD --create-chart"
    fi
    
    echo "Executing: $CMD"
    $CMD
fi

echo ""
echo "========================================================"
echo "  Test Suite Completed"
echo "========================================================"

exit 0
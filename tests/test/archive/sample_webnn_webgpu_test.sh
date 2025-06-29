#!/bin/bash

# Script to test a sample of high priority HuggingFace models with WebNN and WebGPU
# This is a reduced version to avoid timeouts

BASEDIR=$(dirname "$0")
cd $BASEDIR

echo "Starting Sample WebNN/WebGPU model compatibility tests..."
echo "--------------------------------------------------------"

# Define a subset of high priority models and their types
declare -A MODELS
MODELS["bert-base-uncased"]="text"
MODELS["google/vit-base-patch16-224"]="vision"
MODELS["openai/whisper-tiny"]="audio"

# Define quantization configurations (just test 8-bit and 4-bit)
QUANTIZATIONS=(8 4)

# Create output directory
mkdir -p sample_webnn_webgpu_results

# Function to run test and log results
run_test() {
    local platform=$1
    local model=$2
    local model_type=$3
    local bits=$4
    
    echo "Testing $model ($model_type) on $platform with ${bits}-bit precision..."
    
    # Define log file
    local log_file="sample_webnn_webgpu_results/${platform}_${model//\//_}_${bits}bit.log"
    
    # Run the test with simulation mode (to ensure it completes even without hardware)
    python run_real_webgpu_webnn.py \
        --platform $platform \
        --browser chrome \
        --headless \
        --model $model \
        --model-type $model_type \
        --bits $bits \
        --simulation-only \
        --verbose > "$log_file" 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "  ✅ SUCCESS with real hardware"
        echo "SUCCESS_REAL" >> "$log_file"
    elif [ $status -eq 2 ]; then
        echo "  ✓ Success with simulation"
        echo "SUCCESS_SIMULATION" >> "$log_file"
    else
        echo "  ❌ FAILED (see $log_file for details)"
        echo "FAILED" >> "$log_file"
    fi
}

# Test models on both platforms
for platform in webgpu webnn; do
    echo ""
    echo "========== Testing on $platform platform =========="
    
    for model in "${!MODELS[@]}"; do
        model_type=${MODELS[$model]}
        
        for bits in "${QUANTIZATIONS[@]}"; do
            # Test with standard precision
            run_test $platform "$model" $model_type $bits
        done
    done
done

echo ""
echo "Sample tests completed. Results saved in sample_webnn_webgpu_results directory."
echo "--------------------------------------------------------"

# Generate a brief summary report
echo "Generating summary report..."
{
    echo "# WebNN/WebGPU Sample Model Compatibility Report"
    echo ""
    echo "## Results Summary"
    
    # Combined results table
    echo ""
    echo "| Model | Type | WebGPU 8-bit | WebGPU 4-bit | WebNN 8-bit | WebNN 4-bit |"
    echo "|-------|------|-------------|-------------|------------|------------|"
    
    for model in "${!MODELS[@]}"; do
        model_display=$(echo $model | sed 's/.*\///')
        model_type=${MODELS[$model]}
        echo -n "| $model_display | $model_type | "
        
        # Check WebGPU 8-bit
        if grep -q "SUCCESS" "sample_webnn_webgpu_results/webgpu_${model//\//_}_8bit.log"; then
            echo -n "✅ | "
        else
            echo -n "❌ | "
        fi
        
        # Check WebGPU 4-bit
        if grep -q "SUCCESS" "sample_webnn_webgpu_results/webgpu_${model//\//_}_4bit.log"; then
            echo -n "✅ | "
        else
            echo -n "❌ | "
        fi
        
        # Check WebNN 8-bit
        if grep -q "SUCCESS" "sample_webnn_webgpu_results/webnn_${model//\//_}_8bit.log"; then
            echo -n "✅ | "
        else
            echo -n "❌ | "
        fi
        
        # Check WebNN 4-bit
        if grep -q "SUCCESS" "sample_webnn_webgpu_results/webnn_${model//\//_}_4bit.log"; then
            echo -n "✅ |"
        else
            echo -n "❌ |"
        fi
        
        echo ""
    done
    
    echo ""
    echo "## Simulation Detection"
    echo ""
    echo "R = Real hardware, S = Simulation"
    echo ""
    echo "| Model | WebGPU 8-bit | WebGPU 4-bit | WebNN 8-bit | WebNN 4-bit |"
    echo "|-------|-------------|-------------|------------|------------|"
    
    for model in "${!MODELS[@]}"; do
        model_display=$(echo $model | sed 's/.*\///')
        echo -n "| $model_display | "
        
        # Check WebGPU 8-bit
        if grep -q "SUCCESS_REAL" "sample_webnn_webgpu_results/webgpu_${model//\//_}_8bit.log"; then
            echo -n "R | "
        elif grep -q "SUCCESS_SIMULATION" "sample_webnn_webgpu_results/webgpu_${model//\//_}_8bit.log"; then
            echo -n "S | "
        else
            echo -n "- | "
        fi
        
        # Check WebGPU 4-bit
        if grep -q "SUCCESS_REAL" "sample_webnn_webgpu_results/webgpu_${model//\//_}_4bit.log"; then
            echo -n "R | "
        elif grep -q "SUCCESS_SIMULATION" "sample_webnn_webgpu_results/webgpu_${model//\//_}_4bit.log"; then
            echo -n "S | "
        else
            echo -n "- | "
        fi
        
        # Check WebNN 8-bit
        if grep -q "SUCCESS_REAL" "sample_webnn_webgpu_results/webnn_${model//\//_}_8bit.log"; then
            echo -n "R | "
        elif grep -q "SUCCESS_SIMULATION" "sample_webnn_webgpu_results/webnn_${model//\//_}_8bit.log"; then
            echo -n "S | "
        else
            echo -n "- | "
        fi
        
        # Check WebNN 4-bit
        if grep -q "SUCCESS_REAL" "sample_webnn_webgpu_results/webnn_${model//\//_}_4bit.log"; then
            echo -n "R |"
        elif grep -q "SUCCESS_SIMULATION" "sample_webnn_webgpu_results/webnn_${model//\//_}_4bit.log"; then
            echo -n "S |"
        else
            echo -n "- |"
        fi
        
        echo ""
    done
    
} > sample_webnn_webgpu_results/summary_report.md

echo "Summary report generated: sample_webnn_webgpu_results/summary_report.md"
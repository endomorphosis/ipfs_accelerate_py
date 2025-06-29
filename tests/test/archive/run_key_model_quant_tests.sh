#!/bin/bash
# Script to test key HuggingFace models with WebNN and WebGPU at different quantization levels

RESULT_DIR="quant_test_results"
mkdir -p $RESULT_DIR

# Define models to test
declare -A TEXT_MODELS
TEXT_MODELS["BERT"]="bert-base-uncased"
TEXT_MODELS["T5"]="t5-small"
TEXT_MODELS["LLAMA"]="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEXT_MODELS["Qwen2"]="Qwen/Qwen2-7B-Instruct"

declare -A VISION_MODELS
VISION_MODELS["ViT"]="google/vit-base-patch16-224"
VISION_MODELS["DETR"]="facebook/detr-resnet-50"

declare -A MULTIMODAL_MODELS
MULTIMODAL_MODELS["CLIP"]="openai/clip-vit-base-patch32"
MULTIMODAL_MODELS["LLaVA"]="llava-hf/llava-1.5-7b-hf"
MULTIMODAL_MODELS["XCLIP"]="microsoft/xclip-base-patch32"

declare -A AUDIO_MODELS
AUDIO_MODELS["Whisper"]="openai/whisper-tiny"
AUDIO_MODELS["Wav2Vec2"]="facebook/wav2vec2-base"
AUDIO_MODELS["CLAP"]="laion/clap-htsat-unfused"

# Define bit precisions to test
BIT_PRECISIONS=(16 8 4)
EXPERIMENTAL_PRECISIONS=(2)

# Define platforms
PLATFORMS=("webgpu" "webnn")

# Define browsers
BROWSERS=("chrome" "firefox" "edge")

# Test function for a single model
test_model() {
    local model_name=$1
    local model=$2
    local model_type=$3
    local platform=$4
    local browser=$5
    local bits=$6
    local mixed_precision=$7
    local experimental=$8
    
    echo "Testing $model_name ($model) on $platform with $browser at $bits-bit precision..."
    
    # Build command
    cmd="python run_real_webgpu_webnn.py --platform $platform --browser $browser --model $model --model-type $model_type --bits $bits"
    
    # Add mixed precision if enabled
    if [ "$mixed_precision" = true ]; then
        cmd="$cmd --mixed-precision"
    fi
    
    # Add experimental flag if enabled
    if [ "$experimental" = true ]; then
        cmd="$cmd --experimental-precision"
    fi
    
    # Run the command and save output
    output_file="${RESULT_DIR}/${model_name}_${platform}_${browser}_${bits}bit"
    if [ "$mixed_precision" = true ]; then
        output_file="${output_file}_mixed"
    fi
    if [ "$experimental" = true ]; then
        output_file="${output_file}_experimental"
    fi
    output_file="${output_file}.log"
    
    echo "Running: $cmd"
    echo "Saving output to: $output_file"
    $cmd > "$output_file" 2>&1
    
    # Check result
    if grep -q "implementation test completed successfully" "$output_file"; then
        echo "✅ SUCCESS: $model_name on $platform with $browser at $bits-bit precision"
    else
        echo "❌ FAILED: $model_name on $platform with $browser at $bits-bit precision"
    fi
    echo ""
}

# Function to test all models for a specific type
test_models_of_type() {
    local -n models=$1
    local model_type=$2
    
    for model_name in "${!models[@]}"; do
        local model=${models[$model_name]}
        
        # Test on each platform with each browser
        for platform in "${PLATFORMS[@]}"; do
            for browser in "${BROWSERS[@]}"; do
                # Test standard bit precisions
                for bits in "${BIT_PRECISIONS[@]}"; do
                    # Test without mixed precision
                    test_model "$model_name" "$model" "$model_type" "$platform" "$browser" "$bits" false false
                    
                    # Test with mixed precision
                    if [ "$bits" -lt 16 ]; then
                        test_model "$model_name" "$model" "$model_type" "$platform" "$browser" "$bits" true false
                    fi
                done
                
                # Test experimental bit precisions (only for WebGPU)
                if [ "$platform" = "webgpu" ]; then
                    for bits in "${EXPERIMENTAL_PRECISIONS[@]}"; do
                        # Test with mixed precision (required for experimental bits)
                        test_model "$model_name" "$model" "$model_type" "$platform" "$browser" "$bits" true true
                    done
                fi
            done
        done
    done
}

# Generate summary report
generate_summary() {
    local summary_file="${RESULT_DIR}/quant_test_summary.md"
    
    echo "# WebNN and WebGPU Quantization Test Results" > "$summary_file"
    echo "" >> "$summary_file"
    echo "Date: $(date)" >> "$summary_file"
    echo "" >> "$summary_file"
    
    echo "## Text Models" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "| Model | Platform | Browser | Bits | Mixed Precision | Status |" >> "$summary_file"
    echo "|-------|----------|---------|------|----------------|--------|" >> "$summary_file"
    
    for file in "${RESULT_DIR}"/*.log; do
        filename=$(basename "$file" .log)
        IFS='_' read -r model platform browser bits mixed_precision <<< "$filename"
        
        if grep -q "implementation test completed successfully" "$file"; then
            status="✅ Success"
        else
            status="❌ Failed"
        fi
        
        mixed="No"
        if [[ "$mixed_precision" == "mixed" ]]; then
            mixed="Yes"
        fi
        
        # Extract model type from filename to organize in report
        model_type="Unknown"
        if [[ " ${!TEXT_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Text"
        elif [[ " ${!VISION_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Vision"
        elif [[ " ${!MULTIMODAL_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Multimodal"
        elif [[ " ${!AUDIO_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Audio"
        fi
        
        # Only add to the appropriate section
        if [ "$model_type" = "Text" ]; then
            echo "| $model | $platform | $browser | $bits | $mixed | $status |" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "## Vision Models" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "| Model | Platform | Browser | Bits | Mixed Precision | Status |" >> "$summary_file"
    echo "|-------|----------|---------|------|----------------|--------|" >> "$summary_file"
    
    for file in "${RESULT_DIR}"/*.log; do
        filename=$(basename "$file" .log)
        IFS='_' read -r model platform browser bits mixed_precision <<< "$filename"
        
        if grep -q "implementation test completed successfully" "$file"; then
            status="✅ Success"
        else
            status="❌ Failed"
        fi
        
        mixed="No"
        if [[ "$mixed_precision" == "mixed" ]]; then
            mixed="Yes"
        fi
        
        # Extract model type from filename to organize in report
        model_type="Unknown"
        if [[ " ${!TEXT_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Text"
        elif [[ " ${!VISION_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Vision"
        elif [[ " ${!MULTIMODAL_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Multimodal"
        elif [[ " ${!AUDIO_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Audio"
        fi
        
        # Only add to the appropriate section
        if [ "$model_type" = "Vision" ]; then
            echo "| $model | $platform | $browser | $bits | $mixed | $status |" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "## Multimodal Models" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "| Model | Platform | Browser | Bits | Mixed Precision | Status |" >> "$summary_file"
    echo "|-------|----------|---------|------|----------------|--------|" >> "$summary_file"
    
    for file in "${RESULT_DIR}"/*.log; do
        filename=$(basename "$file" .log)
        IFS='_' read -r model platform browser bits mixed_precision <<< "$filename"
        
        if grep -q "implementation test completed successfully" "$file"; then
            status="✅ Success"
        else
            status="❌ Failed"
        fi
        
        mixed="No"
        if [[ "$mixed_precision" == "mixed" ]]; then
            mixed="Yes"
        fi
        
        # Extract model type from filename to organize in report
        model_type="Unknown"
        if [[ " ${!TEXT_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Text"
        elif [[ " ${!VISION_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Vision"
        elif [[ " ${!MULTIMODAL_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Multimodal"
        elif [[ " ${!AUDIO_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Audio"
        fi
        
        # Only add to the appropriate section
        if [ "$model_type" = "Multimodal" ]; then
            echo "| $model | $platform | $browser | $bits | $mixed | $status |" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "## Audio Models" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "| Model | Platform | Browser | Bits | Mixed Precision | Status |" >> "$summary_file"
    echo "|-------|----------|---------|------|----------------|--------|" >> "$summary_file"
    
    for file in "${RESULT_DIR}"/*.log; do
        filename=$(basename "$file" .log)
        IFS='_' read -r model platform browser bits mixed_precision <<< "$filename"
        
        if grep -q "implementation test completed successfully" "$file"; then
            status="✅ Success"
        else
            status="❌ Failed"
        fi
        
        mixed="No"
        if [[ "$mixed_precision" == "mixed" ]]; then
            mixed="Yes"
        fi
        
        # Extract model type from filename to organize in report
        model_type="Unknown"
        if [[ " ${!TEXT_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Text"
        elif [[ " ${!VISION_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Vision"
        elif [[ " ${!MULTIMODAL_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Multimodal"
        elif [[ " ${!AUDIO_MODELS[@]} " =~ " ${model} " ]]; then
            model_type="Audio"
        fi
        
        # Only add to the appropriate section
        if [ "$model_type" = "Audio" ]; then
            echo "| $model | $platform | $browser | $bits | $mixed | $status |" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "## Summary Statistics" >> "$summary_file"
    echo "" >> "$summary_file"
    
    total=$(ls "${RESULT_DIR}"/*.log | wc -l)
    success=$(grep -l "implementation test completed successfully" "${RESULT_DIR}"/*.log | wc -l)
    failed=$((total - success))
    
    echo "- Total tests: $total" >> "$summary_file"
    echo "- Successful tests: $success ($(echo "scale=2; $success*100/$total" | bc)%)" >> "$summary_file"
    echo "- Failed tests: $failed ($(echo "scale=2; $failed*100/$total" | bc)%)" >> "$summary_file"
    
    echo "Summary report generated at: $summary_file"
}

# Main execution

echo "Starting comprehensive test of key HuggingFace models with WebNN and WebGPU..."
echo "Results will be saved in: $RESULT_DIR"

# Test all model types
echo "Testing text models..."
test_models_of_type TEXT_MODELS "text"

echo "Testing vision models..."
test_models_of_type VISION_MODELS "vision"

echo "Testing multimodal models..."
test_models_of_type MULTIMODAL_MODELS "multimodal"

echo "Testing audio models..."
test_models_of_type AUDIO_MODELS "audio"

# Generate summary report
generate_summary

echo "Testing completed. See $RESULT_DIR for detailed logs and summary."
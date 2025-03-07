#!/bin/bash
# Script to test a subset of key models with WebNN and WebGPU at different quantization levels

RESULT_DIR="quant_test_results_targeted"
mkdir -p $RESULT_DIR

# Run key models with both WebNN and WebGPU at different bit precisions
echo "Testing BERT model..."

# Test BERT with WebGPU at 16-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model bert-base-uncased --model-type text --bits 16 > $RESULT_DIR/BERT_webgpu_16bit.log 2>&1
echo "Completed BERT with WebGPU at 16-bit"

# Test BERT with WebGPU at 8-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model bert-base-uncased --model-type text --bits 8 > $RESULT_DIR/BERT_webgpu_8bit.log 2>&1
echo "Completed BERT with WebGPU at 8-bit"

# Test BERT with WebGPU at 4-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model bert-base-uncased --model-type text --bits 4 > $RESULT_DIR/BERT_webgpu_4bit.log 2>&1
echo "Completed BERT with WebGPU at 4-bit"

# Test BERT with WebGPU at 4-bit with mixed precision
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model bert-base-uncased --model-type text --bits 4 --mixed-precision > $RESULT_DIR/BERT_webgpu_4bit_mixed.log 2>&1
echo "Completed BERT with WebGPU at 4-bit with mixed precision"

# Test BERT with WebNN at 16-bit
python run_real_webgpu_webnn.py --platform webnn --browser chrome --model bert-base-uncased --model-type text --bits 16 > $RESULT_DIR/BERT_webnn_16bit.log 2>&1
echo "Completed BERT with WebNN at 16-bit"

# Test BERT with WebNN at 8-bit
python run_real_webgpu_webnn.py --platform webnn --browser chrome --model bert-base-uncased --model-type text --bits 8 > $RESULT_DIR/BERT_webnn_8bit.log 2>&1
echo "Completed BERT with WebNN at 8-bit"

echo "Testing ViT model..."

# Test ViT with WebGPU at 16-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model google/vit-base-patch16-224 --model-type vision --bits 16 > $RESULT_DIR/ViT_webgpu_16bit.log 2>&1
echo "Completed ViT with WebGPU at 16-bit"

# Test ViT with WebGPU at 8-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model google/vit-base-patch16-224 --model-type vision --bits 8 > $RESULT_DIR/ViT_webgpu_8bit.log 2>&1
echo "Completed ViT with WebGPU at 8-bit"

# Test ViT with WebGPU at 4-bit with mixed precision
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model google/vit-base-patch16-224 --model-type vision --bits 4 --mixed-precision > $RESULT_DIR/ViT_webgpu_4bit_mixed.log 2>&1
echo "Completed ViT with WebGPU at 4-bit with mixed precision"

echo "Testing CLIP model..."

# Test CLIP with WebGPU at 16-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model openai/clip-vit-base-patch32 --model-type multimodal --bits 16 > $RESULT_DIR/CLIP_webgpu_16bit.log 2>&1
echo "Completed CLIP with WebGPU at 16-bit"

# Test CLIP with WebGPU at 8-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model openai/clip-vit-base-patch32 --model-type multimodal --bits 8 > $RESULT_DIR/CLIP_webgpu_8bit.log 2>&1
echo "Completed CLIP with WebGPU at 8-bit"

echo "Testing Whisper model..."

# Test Whisper with WebGPU at 16-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model openai/whisper-tiny --model-type audio --bits 16 > $RESULT_DIR/Whisper_webgpu_16bit.log 2>&1
echo "Completed Whisper with WebGPU at 16-bit"

# Test Whisper with WebGPU at 8-bit
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --model openai/whisper-tiny --model-type audio --bits 8 > $RESULT_DIR/Whisper_webgpu_8bit.log 2>&1
echo "Completed Whisper with WebGPU at 8-bit"

# Test Whisper with WebGPU at 4-bit with mixed precision
python run_real_webgpu_webnn.py --platform webgpu --browser firefox --model openai/whisper-tiny --model-type audio --bits 4 --mixed-precision > $RESULT_DIR/Whisper_webgpu_firefox_4bit_mixed.log 2>&1
echo "Completed Whisper with WebGPU (Firefox) at 4-bit with mixed precision"

# Generate summary
echo "Generating summary report..."

echo "# Targeted WebNN and WebGPU Quantization Test Results" > $RESULT_DIR/summary.md
echo "" >> $RESULT_DIR/summary.md
echo "Date: $(date)" >> $RESULT_DIR/summary.md
echo "" >> $RESULT_DIR/summary.md

echo "## Results Table" >> $RESULT_DIR/summary.md
echo "" >> $RESULT_DIR/summary.md
echo "| Model | Platform | Bits | Mixed Precision | Status | Simulation | Time (ms) | Memory (MB) |" >> $RESULT_DIR/summary.md
echo "|-------|----------|------|----------------|--------|------------|-----------|-------------|" >> $RESULT_DIR/summary.md

for file in $RESULT_DIR/*.log; do
    # Extract info from filename
    filename=$(basename "$file" .log)
    IFS='_' read -r model platform bits mixed_precision <<< "$filename"
    
    # Determine status
    if grep -q "implementation test completed successfully" "$file"; then
        status="✅ Success"
    else
        status="❌ Failed"
    fi
    
    # Check if simulation was used
    if grep -q "SIMULATION mode" "$file"; then
        simulation="Yes"
    else
        simulation="No"
    fi
    
    # Extract performance metrics if available
    inference_time="N/A"
    memory_usage="N/A"
    
    time_match=$(grep -o "inference_time_ms\": [0-9.]*" "$file" | grep -o "[0-9.]*")
    if [ ! -z "$time_match" ]; then
        inference_time="$time_match"
    fi
    
    memory_match=$(grep -o "memory_usage_mb\": [0-9.]*" "$file" | grep -o "[0-9.]*")
    if [ ! -z "$memory_match" ]; then
        memory_usage="$memory_match"
    fi
    
    # Format "mixed" field
    mixed="No"
    if [[ "$mixed_precision" == "mixed" ]]; then
        mixed="Yes"
    fi
    
    # Add to summary
    echo "| $model | $platform | $bits | $mixed | $status | $simulation | $inference_time | $memory_usage |" >> $RESULT_DIR/summary.md
done

echo "" >> $RESULT_DIR/summary.md
echo "## Summary" >> $RESULT_DIR/summary.md
echo "" >> $RESULT_DIR/summary.md

total=$(ls "${RESULT_DIR}"/*.log | wc -l)
success=$(grep -l "implementation test completed successfully" "${RESULT_DIR}"/*.log | wc -l)
failed=$((total - success))

echo "- Total tests: $total" >> $RESULT_DIR/summary.md
echo "- Successful tests: $success ($(echo "scale=2; $success*100/$total" | bc)%)" >> $RESULT_DIR/summary.md
echo "- Failed tests: $failed ($(echo "scale=2; $failed*100/$total" | bc)%)" >> $RESULT_DIR/summary.md

echo "Testing completed. See $RESULT_DIR/summary.md for results."
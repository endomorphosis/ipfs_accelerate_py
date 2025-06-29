#!/bin/bash
# Script to demonstrate the model file verification and conversion pipeline

# Set up output directory
OUTPUT_DIR="model_verification_results"
mkdir -p $OUTPUT_DIR

# Get the timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/model_verification_$TIMESTAMP.log"

# Function to run a command and log it
run_and_log() {
    local title="$1"
    local cmd="$2"
    
    echo "===== $title =====" | tee -a "$LOG_FILE"
    echo "Command: $cmd" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "===== End of $title =====" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Basic verification of a model file
run_and_log "Basic ONNX File Verification" "python model_file_verification.py --model prajjwal1/bert-tiny --file-path model.onnx"

# Verify with automatic conversion
run_and_log "Verification with Automatic Conversion" "python model_file_verification.py --model microsoft/deberta-v3-xsmall --file-path model.onnx"

# Check if a model file exists
run_and_log "Check Model File Existence" "python model_file_verification.py --model google/vit-base-patch16-224 --file-path pytorch_model.bin --check-exists"

# Get model metadata
run_and_log "Get Model Metadata" "python model_file_verification.py --model prajjwal1/bert-tiny --get-metadata"

# Batch verification of multiple models
echo '[{"model_id": "prajjwal1/bert-tiny", "file_path": "model.onnx", "model_type": "bert"}, {"model_id": "hf-internal-testing/tiny-random-t5", "file_path": "model.onnx", "model_type": "t5"}]' > $OUTPUT_DIR/batch_models.json
run_and_log "Batch Verification of Multiple Models" "python model_file_verification.py --batch --batch-file $OUTPUT_DIR/batch_models.json --output $OUTPUT_DIR/batch_results.json"

# Run the benchmark integration example
run_and_log "Benchmark Integration Example" "python benchmark_model_verification.py --model prajjwal1/bert-tiny --file-path model.onnx --output-dir $OUTPUT_DIR --batch-sizes 1,2"

# Test with multiple models
run_and_log "Benchmark with Multiple Models" "python benchmark_model_verification.py --models prajjwal1/bert-tiny hf-internal-testing/tiny-random-t5 --file-path model.onnx --output-dir $OUTPUT_DIR --batch-sizes 1,2 --hardware cpu"

# Create a model list file
echo "prajjwal1/bert-tiny" > $OUTPUT_DIR/model_list.txt
echo "hf-internal-testing/tiny-random-t5" >> $OUTPUT_DIR/model_list.txt
echo "google/vit-base-patch16-224" >> $OUTPUT_DIR/model_list.txt

# Run with a model list file
run_and_log "Benchmark with Model List File" "python benchmark_model_verification.py --model-file $OUTPUT_DIR/model_list.txt --file-path model.onnx --output-dir $OUTPUT_DIR --hardware cpu"

echo "All tests completed. Results saved to $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
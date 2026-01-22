#!/bin/bash

# Generate the remaining high-priority models

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="remaining_high_priority_${TIMESTAMP}.log"

echo "Generating remaining high-priority models (logging to $LOG_FILE)..."

# Create a function to generate a test file
generate_test() {
  local model=$1
  local template=$2
  # Replace hyphens with underscores for models with hyphens
  local safe_model=${model//-/_}
  
  echo "Generating test for $safe_model using template $template..." | tee -a "$LOG_FILE"
  python test_toolkit.py generate "$safe_model" --template "$template" 2>&1 | tee -a "$LOG_FILE"
  
  # Check if the file was created
  local test_file="/home/barberb/ipfs_accelerate_py/test/test_hf_${safe_model}.py"
  
  if [ -f "$test_file" ]; then
    # Add a comment if model name was converted from hyphenated
    if [ "$model" != "$safe_model" ]; then
      echo "Adding original model name comment to $test_file" | tee -a "$LOG_FILE"
      sed -i "1s/^/# Original model name: $model\n/" "$test_file"
    fi
    
    # Verify the file has valid syntax
    python -m py_compile "$test_file" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
      echo "✅ Generated $safe_model successfully" | tee -a "$LOG_FILE"
    else
      echo "❌ Failed to verify $safe_model" | tee -a "$LOG_FILE"
    fi
  else
    echo "❌ Could not find generated file $test_file" | tee -a "$LOG_FILE"
  fi
  echo "" | tee -a "$LOG_FILE"
}

# Regenerate files that need template updates
generate_test "gpt-neo" "decoder_only"
generate_test "gpt-neox" "decoder_only"

# Generate remaining multimodal models
generate_test "fuyu" "multimodal"
generate_test "kosmos-2" "multimodal"
generate_test "llava-next" "multimodal"
generate_test "video-llava" "multimodal"

# Generate speech model
generate_test "bark" "speech"

# Generate vision model
generate_test "mobilenet-v2" "vision"

# Generate vision-text models
generate_test "chinese-clip" "vision_text"
generate_test "clipseg" "vision_text"

echo "Remaining high-priority model generation complete. See $LOG_FILE for details."
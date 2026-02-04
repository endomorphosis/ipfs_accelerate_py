#!/bin/bash

# Fix hyphenated model names by using the underscore naming convention

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="hyphenated_fix_${TIMESTAMP}.log"

echo "Fixing hyphenated model names (logging to $LOG_FILE)..."

# Create a function to generate and fix the test file
generate_test() {
  local model=$1
  local template=$2
  # Replace hyphens with underscores for the output filename
  local safe_model=${model//-/_}
  
  echo "Generating test for $safe_model using template $template..." | tee -a "$LOG_FILE"
  python test_toolkit.py generate "$safe_model" --template "$template" 2>&1 | tee -a "$LOG_FILE"
  
  # Check if the file was created with the underscore name
  local test_file="/home/barberb/ipfs_accelerate_py/test/test_hf_${safe_model}.py"
  
  if [ -f "$test_file" ]; then
    # Fix references to underscore in the file
    echo "Fixing references inside $test_file" | tee -a "$LOG_FILE"
    
    # Add a comment indicating the original hyphenated name
    sed -i "1s/^/# Original model name: $model\n/" "$test_file"
    
    # Verify the file has valid syntax
    python -m py_compile "$test_file" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
      echo "✅ Generated $safe_model successfully (for $model)" | tee -a "$LOG_FILE"
    else
      echo "❌ Failed to verify $safe_model" | tee -a "$LOG_FILE"
    fi
  else
    echo "❌ Could not find generated file $test_file" | tee -a "$LOG_FILE"
  fi
  echo "" | tee -a "$LOG_FILE"
}

# Generate tests for the hyphenated models using underscore naming
generate_test "flan-t5" "encoder_decoder"
generate_test "pegasus-x" "encoder_decoder"
generate_test "deberta-v2" "encoder_only" 
generate_test "blip-2" "multimodal"

echo "Hyphenated model fix complete. See $LOG_FILE for details."
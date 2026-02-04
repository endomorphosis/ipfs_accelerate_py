#\!/bin/bash

# Generate tests for all high-priority models identified in the HF_MODEL_COVERAGE_ROADMAP.md

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="high_priority_generation_${TIMESTAMP}.log"

echo "Generating high-priority model tests (logging to $LOG_FILE)..."

# Create a function to run the test toolkit and log results
generate_test() {
  local model=$1
  local template=$2
  echo "Generating test for $model using template $template..." | tee -a "$LOG_FILE"
  python test_toolkit.py generate "$model" --template "$template" 2>&1 | tee -a "$LOG_FILE"
  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Generated $model successfully" | tee -a "$LOG_FILE"
  else
    echo "❌ Failed to generate $model" | tee -a "$LOG_FILE"
  fi
  echo "" | tee -a "$LOG_FILE"
}

# Decoder-only language models
generate_test "mistral" "decoder_only"
generate_test "falcon" "decoder_only"
generate_test "mixtral" "decoder_only"
generate_test "phi" "decoder_only"
generate_test "codellama" "decoder_only"
generate_test "qwen2" "decoder_only"
generate_test "qwen3" "decoder_only"

# Encoder-decoder models
generate_test "flan-t5" "encoder_decoder"
generate_test "longt5" "encoder_decoder"
generate_test "pegasus-x" "encoder_decoder"

# Encoder-only models
generate_test "deberta" "encoder_only"
generate_test "deberta-v2" "encoder_only"
generate_test "luke" "encoder_only"
generate_test "mpnet" "encoder_only"

# Multimodal models
generate_test "blip-2" "multimodal"

echo "High-priority model test generation complete. See $LOG_FILE for details."

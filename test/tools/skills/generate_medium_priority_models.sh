#!/bin/bash

# Generate medium-priority models identified in the HF_MODEL_COVERAGE_ROADMAP.md

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="medium_priority_generation_${TIMESTAMP}.log"

echo "Generating medium-priority models (logging to $LOG_FILE)..."

# Function to generate a batch of model tests
generate_batch() {
  local models=("$@")
  local batch_size=${#models[@]}
  local success_count=0

  echo "Generating batch of $batch_size models..." | tee -a "$LOG_FILE"

  for model_info in "${models[@]}"; do
    # Split model info into model name and template
    IFS=':' read -r model template <<< "$model_info"

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
        ((success_count++))
      else
        echo "❌ Failed to verify $safe_model" | tee -a "$LOG_FILE"
      fi
    else
      echo "❌ Could not find generated file $test_file" | tee -a "$LOG_FILE"
    fi
    echo "" | tee -a "$LOG_FILE"
  done

  echo "Batch generation complete: $success_count/$batch_size successful" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
}

# Define batches of medium-priority models with their templates
declare -a BATCH_1=(
  "bigbird:encoder_only"
  "canine:encoder_only"
  "flaubert:encoder_only"
  "funnel:encoder_only"
  "layoutlm:encoder_only"
  "roformer:encoder_only"
  "xlnet:encoder_only"
)

declare -a BATCH_2=(
  "codegen:decoder_only"
  "mosaic_mpt:decoder_only"
  "stablelm:decoder_only"
  "pythia:decoder_only"
  "xglm:decoder_only"
  "open_llama:decoder_only"
  "olmo:decoder_only"
)

declare -a BATCH_3=(
  "bigbird_pegasus:encoder_decoder"
  "nllb:encoder_decoder"
  "pegasus_x:encoder_decoder"
  "umt5:encoder_decoder"
  "m2m_100:encoder_decoder"
  "plbart:encoder_decoder"
  "speech_to_text:encoder_decoder"
)

declare -a BATCH_4=(
  "mobilevit:vision"
  "cvt:vision"
  "levit:vision"
  "swinv2:vision"
  "perceiver:vision"
  "poolformer:vision"
  "convnextv2:vision"
)

declare -a BATCH_5=(
  "vilt:multimodal"
  "instruct_blip:multimodal"
  "owlvit:multimodal"
  "siglip:multimodal"
  "groupvit:multimodal"
  "blip_2:multimodal"
)

declare -a BATCH_6=(
  "unispeech:speech"
  "wavlm:speech"
  "data2vec_audio:speech"
  "sew:speech"
  "audioldm2:speech"
  "clap:speech"
  "speecht5:speech"
)

# Generate each batch
echo "=== GENERATING BATCH 1: ENCODER-ONLY MODELS ===" | tee -a "$LOG_FILE"
generate_batch "${BATCH_1[@]}"

echo "=== GENERATING BATCH 2: DECODER-ONLY MODELS ===" | tee -a "$LOG_FILE"
generate_batch "${BATCH_2[@]}"

echo "=== GENERATING BATCH 3: ENCODER-DECODER MODELS ===" | tee -a "$LOG_FILE"
generate_batch "${BATCH_3[@]}"

echo "=== GENERATING BATCH 4: VISION MODELS ===" | tee -a "$LOG_FILE"
generate_batch "${BATCH_4[@]}"

echo "=== GENERATING BATCH 5: MULTIMODAL MODELS ===" | tee -a "$LOG_FILE"
generate_batch "${BATCH_5[@]}"

echo "=== GENERATING BATCH 6: SPEECH MODELS ===" | tee -a "$LOG_FILE"
generate_batch "${BATCH_6[@]}"

echo "Medium-priority model generation complete. See $LOG_FILE for details."
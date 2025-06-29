#!/bin/bash
# Script to fix the key model tests for all hardware platforms

# Colors for pretty output
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
RESET="\033[0m"

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_MODELS_DIR="$SCRIPT_DIR/key_models_hardware_fixes"
OUTPUT_DIR="$SCRIPT_DIR/hardware_fix_results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print section header
echo -e "${BLUE}==========================================================${RESET}"
echo -e "${BLUE}     Running Hardware Integration Fixes for Key Models     ${RESET}"
echo -e "${BLUE}==========================================================${RESET}"

# Key models to process
KEY_MODELS=(
    "bert"
    "t5"
    "llama"
    "clip"
    "vit"
    "clap"
    "whisper"
    "wav2vec2"
    "llava"
    "llava_next"
    "xclip"
    "qwen2"
    "detr"
)

# First, analyze all key models
echo -e "\n${YELLOW}Step 1: Analyzing all key models for hardware integration issues${RESET}"
python "$SCRIPT_DIR/fix_hardware_integration.py" --all-key-models --analyze-only --output-json "$OUTPUT_DIR/key_models_analysis.json"

# Now fix each model individually to provide detailed output
echo -e "\n${YELLOW}Step 2: Fixing hardware integration issues for each key model${RESET}"
for model in "${KEY_MODELS[@]}"; do
    echo -e "\n${GREEN}Processing $model model...${RESET}"
    python "$SCRIPT_DIR/fix_hardware_integration.py" --specific-models "$model" --output-json "$OUTPUT_DIR/${model}_fixes.json"
    
    # Check result
    status=$?
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully processed $model${RESET}"
    else
        echo -e "${RED}✗ Error processing $model (exit code: $status)${RESET}"
    fi
done

# Run a final analysis to see remaining issues
echo -e "\n${YELLOW}Step 3: Re-analyzing models after fixes${RESET}"
python "$SCRIPT_DIR/fix_hardware_integration.py" --all-key-models --analyze-only --output-json "$OUTPUT_DIR/key_models_post_fix_analysis.json"

# Output summary
echo -e "\n${BLUE}==========================================================${RESET}"
echo -e "${BLUE}                 Hardware Fix Summary                     ${RESET}"
echo -e "${BLUE}==========================================================${RESET}"
echo -e "All key models processed for hardware platform support."
echo -e "Details saved to $OUTPUT_DIR/"
echo -e "Next steps:"
echo -e "1. Review the generated fixes"
echo -e "2. Run the updated test files to verify functionality"
echo -e "3. Validate hardware detection with run_comprehensive_hardware_tests.sh"

echo -e "\n${GREEN}Done!${RESET}"
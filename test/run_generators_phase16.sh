#!/bin/bash
# Run key models with fixed Phase 16 generators

# Ensure the test output directory exists
mkdir -p test_outputs

# Set of models to test with different modalities
MODELS=(
  "bert-base-uncased"  # Text
  "t5-small"           # Text
  "vit-base"           # Vision
  "clip-vit-base-patch32"  # Multimodal
  "whisper-tiny"       # Audio
)

# Hardware platforms to test
PLATFORMS=(
  "cpu"
  "cpu,cuda"
  "cpu,webnn,webgpu"
  "all"
)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Phase 16 Generator Tests${NC}"
echo "=================================================="

# Test fixed_merged_test_generator.py
echo -e "${GREEN}Testing fixed_merged_test_generator.py${NC}"
for MODEL in "${MODELS[@]}"; do
  echo -e "${YELLOW}Generating tests for model: ${MODEL}${NC}"
  for PLATFORM in "${PLATFORMS[@]}"; do
    echo -n "  Platform $PLATFORM: "
    OUTPUT=$(python fixed_merged_test_generator.py -g $MODEL -p $PLATFORM -o test_outputs/ 2>&1)
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}SUCCESS${NC}"
    else
      echo -e "${RED}FAILED${NC}"
      echo "$OUTPUT"
    fi
  done
done

echo "=================================================="

# Test merged_test_generator.py
echo -e "${GREEN}Testing merged_test_generator.py${NC}"
for MODEL in "${MODELS[@]}"; do
  echo -e "${YELLOW}Generating tests for model: ${MODEL}${NC}"
  for PLATFORM in "${PLATFORMS[@]}"; do
    echo -n "  Platform $PLATFORM: "
    OUTPUT=$(python merged_test_generator.py -g $MODEL -p $PLATFORM -o test_outputs/ 2>&1)
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}SUCCESS${NC}"
    else
      echo -e "${RED}FAILED${NC}"
      echo "$OUTPUT"
    fi
  done
done

echo "=================================================="

# Test integrated_skillset_generator.py
echo -e "${GREEN}Testing integrated_skillset_generator.py${NC}"
for MODEL in "${MODELS[@]}"; do
  echo -e "${YELLOW}Generating skills for model: ${MODEL}${NC}"
  for PLATFORM in "${PLATFORMS[@]}"; do
    echo -n "  Platform $PLATFORM: "
    OUTPUT=$(python integrated_skillset_generator.py -m $MODEL -p $PLATFORM -o test_outputs/ 2>&1)
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}SUCCESS${NC}"
    else
      echo -e "${RED}FAILED${NC}"
      echo "$OUTPUT"
    fi
  done
done

echo "=================================================="
echo -e "${GREEN}All tests completed!${NC}"
echo "Check the test_outputs directory for generated files."
#!/bin/bash
# Script to run tests with Phase 16 generators
# This script runs both the fixed_merged_test_generator.py and integrated_skillset_generator.py
# with the same set of models to ensure compatibility and cross-platform support

# Define color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set environment variables for web platform optimizations
export WEB_ALL_OPTIMIZATIONS=1  # Enable all web platform optimizations
export DEPRECATE_JSON_OUTPUT=1  # Use DuckDB for output

# Define key models to test
KEY_MODELS=("bert" "t5" "llama" "vit" "clip" "detr" "clap" "wav2vec2" "whisper" "llava" "xclip" "qwen2")

# Function to run test generator
run_test_generator() {
    local model=$1
    local hardware=$2
    local output_dir=$3
    
    echo -e "${BLUE}Running test generator for ${model} with hardware: ${hardware}${NC}"
    
    # Create output directory if it doesn't exist
    mkdir -p ${output_dir}
    
    # Run fixed merged test generator with latest improvements
    python fixed_merged_test_generator.py --generate ${model} --platform ${hardware} --output-dir ${output_dir}
    
    # Check if the test was generated successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully generated test for ${model} with ${hardware}${NC}"
        return 0
    else
        echo -e "${RED}Failed to generate test for ${model} with ${hardware}${NC}"
        return 1
    fi
}

# Function to run skillset generator
run_skillset_generator() {
    local model=$1
    local hardware=$2
    local output_dir=$3
    
    echo -e "${BLUE}Running skillset generator for ${model} with hardware: ${hardware}${NC}"
    
    # Create output directory if it doesn't exist
    mkdir -p ${output_dir}
    
    # Run integrated skillset generator
    python integrated_skillset_generator.py --model ${model} --hardware ${hardware} --cross-platform --output-dir ${output_dir}
    
    # Check if the skillset was generated successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully generated skillset for ${model} with hardware: ${hardware}${NC}"
        return 0
    else
        echo -e "${RED}Failed to generate skillset for ${model} with hardware: ${hardware}${NC}"
        return 1
    fi
}

# Function to run a quick test on the generated file
run_quick_test() {
    local test_file=$1
    
    echo -e "${YELLOW}Running quick syntax check on ${test_file}${NC}"
    
    # Check file exists
    if [ ! -f ${test_file} ]; then
        echo -e "${RED}Test file ${test_file} does not exist${NC}"
        return 1
    fi
    
    # Run a syntax check
    python -m py_compile ${test_file} 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Syntax check passed for ${test_file}${NC}"
        return 0
    else
        echo -e "${RED}Syntax check failed for ${test_file}${NC}"
        return 1
    fi
}

# Main test flow
echo -e "${BLUE}====== Starting Phase 16 Generator Tests (Updated March 6, 2025) ======${NC}"

# Create test output directories
TEST_OUTPUT_DIR="./generated_tests"
SKILL_OUTPUT_DIR="./generated_skillsets"

mkdir -p ${TEST_OUTPUT_DIR}
mkdir -p ${SKILL_OUTPUT_DIR}

# Clear existing test files to ensure clean test
rm -f ${TEST_OUTPUT_DIR}/*.py
rm -f ${SKILL_OUTPUT_DIR}/*.py

echo -e "${YELLOW}Testing with latest generator fixes (March 6, 2025)${NC}"

# Array for tracking results
declare -A TEST_RESULTS
declare -A SKILL_RESULTS

# Hardware platforms to test
HARDWARE_PLATFORMS=("all" "cpu,cuda" "cpu,openvino" "cpu,webnn,webgpu")

# Run tests for each key model
for model in "${KEY_MODELS[@]}"; do
    echo -e "\n${BLUE}==== Testing model: ${model} ====${NC}"
    
    # Track results for this model
    TEST_RESULTS[${model}]=0
    SKILL_RESULTS[${model}]=0
    
    # Run generators with different hardware configurations
    for hardware in "${HARDWARE_PLATFORMS[@]}"; do
        echo -e "\n${YELLOW}-- Testing ${model} with hardware: ${hardware} --${NC}"
        
        # Run test generator
        run_test_generator ${model} ${hardware} ${TEST_OUTPUT_DIR}
        if [ $? -eq 0 ]; then
            TEST_RESULTS[${model}]=$((${TEST_RESULTS[${model}]}+1))
        fi
        
        # Run quick test on generated file
        run_quick_test "${TEST_OUTPUT_DIR}/test_hf_${model}.py"
        
        # Run skillset generator
        run_skillset_generator ${model} ${hardware} ${SKILL_OUTPUT_DIR}
        if [ $? -eq 0 ]; then
            SKILL_RESULTS[${model}]=$((${SKILL_RESULTS[${model}]}+1))
        fi
        
        # Run quick test on generated skillset
        run_quick_test "${SKILL_OUTPUT_DIR}/hf_${model//-/_}.py"
        
        echo -e "${YELLOW}-- Completed testing ${model} with hardware: ${hardware} --${NC}"
    done
done

# Print summary
echo -e "\n${BLUE}====== Test Summary ======${NC}"
echo "Model | Test Generator | Skillset Generator"
echo "-----|----------------|-------------------"

total_tests=0
total_skills=0
total_models=${#KEY_MODELS[@]}
total_configs=${#HARDWARE_PLATFORMS[@]}

for model in "${KEY_MODELS[@]}"; do
    test_success=${TEST_RESULTS[${model}]}
    skill_success=${SKILL_RESULTS[${model}]}
    
    # Update totals
    total_tests=$((total_tests + test_success))
    total_skills=$((total_skills + skill_success))
    
    # Calculate percentages
    test_percent=$((100 * test_success / total_configs))
    skill_percent=$((100 * skill_success / total_configs))
    
    # Choose colors based on percentages
    test_color=${GREEN}
    skill_color=${GREEN}
    
    if [ ${test_percent} -lt 100 ]; then
        test_color=${YELLOW}
    fi
    if [ ${test_percent} -lt 50 ]; then
        test_color=${RED}
    fi
    
    if [ ${skill_percent} -lt 100 ]; then
        skill_color=${YELLOW}
    fi
    if [ ${skill_percent} -lt 50 ]; then
        skill_color=${RED}
    fi
    
    echo -e "${model} | ${test_color}${test_success}/${total_configs} (${test_percent}%)${NC} | ${skill_color}${skill_success}/${total_configs} (${skill_percent}%)${NC}"
done

# Calculate overall percentages
test_overall_percent=$((100 * total_tests / (total_models * total_configs)))
skill_overall_percent=$((100 * total_skills / (total_models * total_configs)))

echo -e "\n${BLUE}====== Overall Results ======${NC}"
echo -e "Test Generator: ${GREEN}${total_tests}/${total_models * total_configs} (${test_overall_percent}%)${NC}"
echo -e "Skillset Generator: ${GREEN}${total_skills}/${total_models * total_configs} (${skill_overall_percent}%)${NC}"

# Exit with success if both are at least 90% successful
if [ ${test_overall_percent} -ge 90 ] && [ ${skill_overall_percent} -ge 90 ]; then
    echo -e "\n${GREEN}Success! Phase 16 generators are working well.${NC}"
    exit 0
else
    echo -e "\n${YELLOW}Warning: At least one generator is not reaching 90% success rate.${NC}"
    exit 1
fi
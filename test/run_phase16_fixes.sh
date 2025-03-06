#!/bin/bash
# Script to verify and run the Phase 16 fixes

set -e  # Exit immediately if a command exits with a non-zero status

# Create output directory
mkdir -p hardware_test_results

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}= Phase 16 Hardware Support Verification =${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Verify that Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Verify the existence of updated model files
MODELS_DIR="updated_models"
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${RED}Error: $MODELS_DIR directory not found${NC}"
    exit 1
fi

echo -e "${BLUE}=> Running hardware tests for updated models...${NC}"
echo ""

# Run hardware tests on CPU (should work everywhere)
echo -e "${BLUE}=> Testing on CPU platform...${NC}"
python run_hardware_tests.py --models bert t5 --platforms cpu --output-dir hardware_test_results

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}CUDA detected, testing GPU support...${NC}"
    python run_hardware_tests.py --models bert t5 --platforms cuda --output-dir hardware_test_results
else
    echo -e "${RED}CUDA not available, skipping GPU tests${NC}"
fi

# Check if MPS (Apple Silicon) is available
if python -c "import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}MPS (Apple Silicon) detected, testing MPS support...${NC}"
    python run_hardware_tests.py --models bert t5 --platforms mps --output-dir hardware_test_results
else
    echo -e "${RED}MPS (Apple Silicon) not available, skipping MPS tests${NC}"
fi

# Check if OpenVINO is available
if python -c "import openvino" &>/dev/null; then
    echo -e "${GREEN}OpenVINO detected, testing OpenVINO support...${NC}"
    python run_hardware_tests.py --models bert t5 --platforms openvino --output-dir hardware_test_results
else
    echo -e "${RED}OpenVINO not available, skipping OpenVINO tests${NC}"
fi

# Check if ROCm is available (AMD)
if python -c "import torch; print(torch.cuda.is_available() and hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}ROCm (AMD) detected, testing ROCm support...${NC}"
    python run_hardware_tests.py --models bert t5 --platforms rocm --output-dir hardware_test_results
else
    echo -e "${RED}ROCm (AMD) not available, skipping ROCm tests${NC}"
fi

# Always test WebNN and WebGPU (simulation)
echo -e "${BLUE}=> Testing web platform support (simulation)...${NC}"
python run_hardware_tests.py --models bert t5 --platforms webnn webgpu --output-dir hardware_test_results

echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}Hardware testing complete!${NC}"
echo -e "${BLUE}Results saved to: hardware_test_results/${NC}"
echo -e "${BLUE}==========================================${NC}"

# Copy new implementations to key_models_hardware_fixes
echo ""
echo -e "${BLUE}=> Copying new implementations to key_models_hardware_fixes...${NC}"
cp "$MODELS_DIR/test_hf_bert.py" key_models_hardware_fixes/
cp "$MODELS_DIR/test_hf_t5.py" key_models_hardware_fixes/
echo -e "${GREEN}Files copied successfully!${NC}"

echo ""
echo -e "${BLUE}=> Next steps: ${NC}"
echo -e "1. Implement CUDA and MPS support for the remaining models"
echo -e "2. Run comprehensive tests across all platforms"
echo -e "3. Update the hardware compatibility matrix"
echo -e "4. Update documentation to reflect the current implementation status"
echo ""

# Check if there were any failed tests
if grep -q "Failed" hardware_test_results/*.json 2>/dev/null; then
    echo -e "${RED}Warning: Some tests failed. Check the test results for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests completed successfully!${NC}"
    exit 0
fi
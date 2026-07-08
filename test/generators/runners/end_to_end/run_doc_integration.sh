#!/bin/bash
# 
# Documentation System Integration Script
#
# This script runs all the steps required to integrate the enhanced documentation system
# with the Integrated Component Test Runner.
#

# Exit on any error
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section header
print_header() {
    echo -e "\n${BLUE}========== $1 ==========${NC}\n"
}

print_header "Documentation System Integration"
echo "This script will enhance the documentation system and integrate it with the Integrated Component Test Runner."
echo "Working directory: $SCRIPT_DIR"

# Step 1: Add enhanced documentation templates to the database
print_header "Step 1: Adding enhanced documentation templates"
echo "Running enhance_documentation_templates.py..."
python enhance_documentation_templates.py --verbose || { echo -e "${RED}Failed to add enhanced documentation templates${NC}"; exit 1; }
echo -e "${GREEN}Successfully added enhanced documentation templates${NC}"

# Step 2: Apply patches to fix variable substitution issues
print_header "Step 2: Applying doc_template_fixer.py"
echo "Running doc_template_fixer.py..."
python doc_template_fixer.py || { echo -e "${RED}Failed to apply documentation template fixes${NC}"; exit 1; }
echo -e "${GREEN}Successfully applied documentation template fixes${NC}"

# Step 3: Run the integration script
print_header "Step 3: Running integrate_documentation_system.py"
echo "Running integrate_documentation_system.py..."
python integrate_documentation_system.py --skip-test || { echo -e "${RED}Failed to integrate documentation system${NC}"; exit 1; }
echo -e "${GREEN}Successfully integrated documentation system${NC}"

# Step 4: Run the verification script
print_header "Step 4: Verifying integration"
echo "Running verify_doc_integration.py with a test model..."
echo "This will generate a test document and verify it has all required sections."
python verify_doc_integration.py --model bert-base-uncased --hardware cuda || { echo -e "${YELLOW}Verification yielded warnings or errors, check output${NC}"; }

# Step 5: Final integration test
print_header "Step 5: Final integration test"
echo "Running a full model and hardware test with integrated_component_test_runner.py..."
python integrated_component_test_runner.py --model bert-base-uncased --hardware cuda --generate-docs --quick-test || { echo -e "${YELLOW}Final test yielded warnings or errors, check output${NC}"; }

# Done
print_header "Integration Complete"
echo -e "${GREEN}✓${NC} Documentation system integration completed!"
echo "You can now use the integrated_component_test_runner.py with --generate-docs to create enhanced documentation."
echo "The documentation will include visualization of benchmark results and hardware-specific optimizations."
echo
echo "Try it out with:"
echo "  python integrated_component_test_runner.py --model <model_name> --hardware <hardware> --generate-docs"
echo
echo "Example models: bert-base-uncased, gpt2, vit-base-patch16-224, whisper-tiny"
echo "Example hardware: cpu, cuda, rocm, openvino, webgpu"
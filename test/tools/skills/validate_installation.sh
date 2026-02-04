#!/bin/bash
# Complete validation script for HuggingFace test toolkit and fixed generator
# This script verifies that all components are correctly installed and functional

# Colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
NC="\033[0m" # No Color

echo -e "${CYAN}=== HuggingFace Test Toolkit Validation ===${NC}\n"

# Check test generator
echo -e "${CYAN}Checking test generator...${NC}"
if [ -f "../test_generator.py" ]; then
    echo -e "${GREEN}✅ Test generator found${NC}"
else
    echo -e "${RED}❌ Test generator not found${NC}"
    echo -e "Expected at: ../test_generator.py"
    exit 1
fi

# Check test toolkit
echo -e "\n${CYAN}Checking test toolkit...${NC}"
if [ -f "./test_toolkit.py" ]; then
    echo -e "${GREEN}✅ Test toolkit found${NC}"
else
    echo -e "${RED}❌ Test toolkit not found${NC}"
    echo -e "Expected at: ./test_toolkit.py"
    exit 1
fi

# Check test suite
echo -e "\n${CYAN}Checking test suite...${NC}"
if [ -f "./test_generator_test_suite.py" ]; then
    echo -e "${GREEN}✅ Test suite found${NC}"
else
    echo -e "${RED}❌ Test suite not found${NC}"
    echo -e "Expected at: ./test_generator_test_suite.py"
    exit 1
fi

# Check core test files
echo -e "\n${CYAN}Checking core test files...${NC}"
count=0
missing=""
for model in "bert" "gpt2" "t5" "vit"; do
    if [ -f "../test_hf_${model}.py" ]; then
        echo -e "${GREEN}✅ test_hf_${model}.py found${NC}"
        ((count++))
    else
        echo -e "${RED}❌ test_hf_${model}.py not found${NC}"
        missing="${missing} test_hf_${model}.py"
    fi
done

if [ $count -eq 4 ]; then
    echo -e "${GREEN}All core test files are present${NC}"
else
    echo -e "${RED}Missing core test files: ${missing}${NC}"
fi

# Verify toolkit commands
echo -e "\n${CYAN}Verifying toolkit commands...${NC}"

# Make toolkit executable
chmod +x ./test_toolkit.py

# Check help command
if ./test_toolkit.py help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Help command works${NC}"
else
    echo -e "${RED}❌ Help command failed${NC}"
fi

# Check list command
if ./test_toolkit.py list > /dev/null 2>&1; then
    echo -e "${GREEN}✅ List command works${NC}"
else
    echo -e "${RED}❌ List command failed${NC}"
fi

# Check verify command
if ./test_toolkit.py verify > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Verify command works${NC}"
else
    echo -e "${YELLOW}⚠️ Verify command failed (this might be OK if test files are not yet in the required location)${NC}"
fi

# Create directories
mkdir -p temp_generated coverage_visualizations backups

# Check documentation
echo -e "\n${CYAN}Checking documentation...${NC}"
docs_count=0
docs_missing=""
for doc in "HF_TEST_TOOLKIT_README.md" "HF_TEST_DEVELOPMENT_GUIDE.md" "HF_TEST_IMPLEMENTATION_CHECKLIST.md" "HF_TEST_CICD_INTEGRATION.md" "HF_TEST_TROUBLESHOOTING_GUIDE.md" "HF_MODEL_COVERAGE_ROADMAP.md"; do
    if [ -f "./${doc}" ]; then
        echo -e "${GREEN}✅ ${doc} found${NC}"
        ((docs_count++))
    else
        echo -e "${RED}❌ ${doc} not found${NC}"
        docs_missing="${docs_missing} ${doc}"
    fi
done

if [ $docs_count -eq 6 ]; then
    echo -e "${GREEN}All documentation files are present${NC}"
else
    echo -e "${RED}Missing documentation files: ${docs_missing}${NC}"
fi

# Check CI templates
echo -e "\n${CYAN}Checking CI templates...${NC}"
ci_count=0
ci_missing=""
if [ -d "./ci_templates" ]; then
    echo -e "${GREEN}✅ CI templates directory found${NC}"
    for template in "github-actions-test-validation.yml" "gitlab-ci.yml"; do
        if [ -f "./ci_templates/${template}" ]; then
            echo -e "${GREEN}✅ ${template} found${NC}"
            ((ci_count++))
        else
            echo -e "${RED}❌ ${template} not found${NC}"
            ci_missing="${ci_missing} ${template}"
        fi
    done
else
    echo -e "${RED}❌ CI templates directory not found${NC}"
fi

if [ $ci_count -eq 2 ]; then
    echo -e "${GREEN}All CI template files are present${NC}"
else
    echo -e "${RED}Missing CI template files: ${ci_missing}${NC}"
fi

# Check pre-commit hook
echo -e "\n${CYAN}Checking pre-commit hook...${NC}"
if [ -f "./pre-commit" ]; then
    echo -e "${GREEN}✅ Pre-commit hook found${NC}"
else
    echo -e "${RED}❌ Pre-commit hook not found${NC}"
    echo -e "Expected at: ./pre-commit"
fi

if [ -f "./install_pre_commit.sh" ]; then
    echo -e "${GREEN}✅ Pre-commit installer found${NC}"
else
    echo -e "${RED}❌ Pre-commit installer not found${NC}"
    echo -e "Expected at: ./install_pre_commit.sh"
fi

# Summary
echo -e "\n${CYAN}=== Installation Validation Summary ===${NC}"

# Count issues
issues=0
if [ $count -ne 4 ]; then
    ((issues++))
fi
if [ $docs_count -ne 6 ]; then
    ((issues++))
fi
if [ $ci_count -ne 2 ]; then
    ((issues++))
fi
if [ ! -f "./pre-commit" ] || [ ! -f "./install_pre_commit.sh" ]; then
    ((issues++))
fi

if [ $issues -eq 0 ]; then
    echo -e "${GREEN}✅ All components are properly installed${NC}"
    echo -e "\n${GREEN}Your HuggingFace test toolkit is ready to use!${NC}"
    echo -e "\nKey commands to try:"
    echo -e "  ./test_toolkit.py help"
    echo -e "  ./test_toolkit.py list"
    echo -e "  ./test_toolkit.py verify"
    echo -e "  ./test_toolkit.py test bert --model-id bert-base-uncased --cpu-only"
else
    echo -e "${RED}❌ There are $issues issue(s) with your installation${NC}"
    echo -e "Please fix the issues highlighted above to ensure all components work correctly."
fi
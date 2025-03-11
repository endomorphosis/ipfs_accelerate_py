#!/bin/bash
# Template System Enhancement Runner
# This script runs the template system enhancements

set -e

# Check if jq is installed (needed for pretty-printing JSON output)
if ! command -v jq &> /dev/null; then
    echo "Warning: jq is not installed. JSON output will not be pretty-printed."
    USE_JQ=0
else
    USE_JQ=1
fi

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}IPFS Accelerate Python Framework${NC}"
echo -e "${BLUE}Template System Enhancement Runner${NC}"
echo "-------------------------------------------"

# Check for template database
echo -e "${YELLOW}Checking database...${NC}"
python enhanced_templates/template_system_enhancement.py --check-db

# Apply database schema enhancements if needed
if [ $? -ne 0 ]; then
    echo -e "${RED}Database check failed!${NC}"
    echo -e "${YELLOW}Creating new template database...${NC}"
    python create_template_database.py --create
fi

# Validate all templates before enhancements
echo -e "${YELLOW}Validating templates before enhancements...${NC}"
python enhanced_templates/template_system_enhancement.py --validate-templates

# Apply template inheritance system
echo -e "${YELLOW}Adding template inheritance system...${NC}"
python enhanced_templates/template_system_enhancement.py --add-inheritance

# Enhance placeholder handling
echo -e "${YELLOW}Enhancing placeholder system...${NC}"
python enhanced_templates/template_system_enhancement.py --enhance-placeholders

# Validate templates after enhancements
echo -e "${YELLOW}Validating templates after enhancements...${NC}"
python enhanced_templates/template_system_enhancement.py --validate-templates

# List final templates with validation status
echo -e "${YELLOW}Final template status:${NC}"
python enhanced_templates/template_system_enhancement.py --list-templates

echo -e "${GREEN}Template system enhancements completed!${NC}"
echo -e "${GREEN}Documentation available at: enhanced_templates/TEMPLATE_SYSTEM_ENHANCEMENTS.md${NC}"
#!/bin/bash
#
# Comprehensive HuggingFace Model Test Validation Script
#
# This script runs a complete validation and analysis of HuggingFace model test files,
# checking for syntax correctness, pipeline configurations, and model coverage.
#
# It generates the following reports:
# 1. Validation report - Details about test file quality
# 2. Missing models report - Analysis of test coverage
# 3. Combined summary report - Overall assessment and next steps
#

set -e  # Exit on error

# Define directories and report paths
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
TESTS_DIR="${1:-$SCRIPT_DIR/fixed_tests}"
REPORTS_DIR="$SCRIPT_DIR/reports"
VALIDATION_REPORT="$REPORTS_DIR/validation_report.md"
MISSING_MODELS_REPORT="$REPORTS_DIR/missing_models_$(date +%Y%m%d_%H%M%S).md"
SUMMARY_REPORT="$REPORTS_DIR/test_validation_summary.md"

# Create reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

echo "================================================"
echo "HuggingFace Model Test Validation and Analysis"
echo "================================================"
echo "Tests directory: $TESTS_DIR"
echo "Reports directory: $REPORTS_DIR"
echo

# Step 1: Run syntax and structure validation
echo "[1/3] Running test file validation..."
python "$SCRIPT_DIR/validate_model_tests.py" --directory "$TESTS_DIR" --report "$VALIDATION_REPORT"
VALIDATION_STATUS=$?

# Step 2: Analyze model coverage
echo "[2/3] Analyzing model coverage..."
python "$SCRIPT_DIR/generate_missing_model_report.py" --directory "$TESTS_DIR" --report "$MISSING_MODELS_REPORT"
COVERAGE_STATUS=$?

# Step 3: Generate combined summary report
echo "[3/3] Generating combined summary report..."

cat > "$SUMMARY_REPORT" << EOL
# HuggingFace Model Testing Framework - Validation Summary

**Date:** $(date +"%Y-%m-%d %H:%M:%S")

This report summarizes the validation results and model coverage analysis for the HuggingFace model testing framework.

## Overview

The validation process checks test files for:
- Syntax correctness (using AST parsing)
- Structure validity (required classes and methods)
- Pipeline configuration (appropriate tasks for each model)
- Task input validation (appropriate inputs for tasks)

The coverage analysis tracks:
- Implemented models vs. missing models
- Model implementations by architecture type
- Priority-based implementation roadmap

## Validation Summary

EOL

# Extract summary data from validation report
if [ -f "$VALIDATION_REPORT" ]; then
    # Extract summary section with sed
    sed -n '/^## Summary$/,/^## /p' "$VALIDATION_REPORT" | sed '$d' >> "$SUMMARY_REPORT"
    
    # Add pass/fail status
    if [ $VALIDATION_STATUS -eq 0 ]; then
        echo -e "\n✅ **Validation PASSED** - All files are syntactically correct, properly structured, and have appropriate pipeline configurations." >> "$SUMMARY_REPORT"
    else
        echo -e "\n⚠️ **Validation FAILED** - Some files have issues that need to be addressed. See details in the validation report." >> "$SUMMARY_REPORT"
    fi
else
    echo "❌ Validation report not found! Could not extract summary data." >> "$SUMMARY_REPORT"
fi

# Add coverage summary
cat >> "$SUMMARY_REPORT" << EOL

## Coverage Summary

EOL

if [ -f "$MISSING_MODELS_REPORT" ]; then
    # Extract summary section with sed
    sed -n '/^## Summary$/,/^## /p' "$MISSING_MODELS_REPORT" | sed '$d' >> "$SUMMARY_REPORT"
    
    # Add implementation status
    if [ $COVERAGE_STATUS -eq 0 ]; then
        # Extract critical models missing count
        CRITICAL_MISSING=$(grep -oP "Critical models missing: \K\d+" <<< "$(python "$SCRIPT_DIR/generate_missing_model_report.py" --directory "$TESTS_DIR" 2>/dev/null)")
        
        if [ "$CRITICAL_MISSING" = "0" ]; then
            echo -e "\n✅ **All critical models implemented** - No critical models are missing from the test suite." >> "$SUMMARY_REPORT"
        else
            echo -e "\n⚠️ **Missing critical models** - $CRITICAL_MISSING critical models need to be implemented." >> "$SUMMARY_REPORT"
        fi
    else
        echo -e "\n❌ **Coverage analysis failed** - Could not analyze model coverage properly." >> "$SUMMARY_REPORT"
    fi
else
    echo "❌ Missing models report not found! Could not extract coverage data." >> "$SUMMARY_REPORT"
fi

# Add next steps and recommendations
cat >> "$SUMMARY_REPORT" << EOL

## Next Steps

Based on the validation and coverage analysis, the following actions are recommended:

EOL

# Add different recommendations based on validation and coverage status
if [ $VALIDATION_STATUS -ne 0 ]; then
    cat >> "$SUMMARY_REPORT" << EOL
1. **Fix syntax errors** - Address syntax issues in test files first
   - Use the \`fix_syntax.py\` script or manually correct errors
   - Run validation again to verify fixes

2. **Fix structure issues** - Ensure all test files have the required classes and methods
   - Add missing \`test_pipeline\` methods where needed
   - Ensure proper class naming conventions are followed

3. **Fix pipeline configurations** - Update pipeline configurations for appropriate tasks
   - Run \`add_pipeline_configuration.py\` to add missing configurations
   - Run \`standardize_task_configurations.py\` to fix incorrect tasks

EOL
fi

# Add coverage-based recommendations
if [ -f "$MISSING_MODELS_REPORT" ]; then
    CRITICAL_MISSING=$(grep -oP "Critical models missing: \K\d+" <<< "$(python "$SCRIPT_DIR/generate_missing_model_report.py" --directory "$TESTS_DIR" 2>/dev/null)")
    
    if [ "$CRITICAL_MISSING" -gt 0 ]; then
        cat >> "$SUMMARY_REPORT" << EOL
4. **Implement critical models** - Focus on implementing the missing critical models first
   - Use the implementation roadmap in the missing models report
   - Use the \`simplified_fix_hyphenated.py\` script for hyphenated model names

5. **Implement high priority models** - After critical models are implemented, focus on high priority models
   - These include models with significant usage but not as widespread as critical models

EOL
    else
        cat >> "$SUMMARY_REPORT" << EOL
4. **Implement high priority models** - Focus on implementing high priority models
   - Use the implementation roadmap in the missing models report
   - Use the \`simplified_fix_hyphenated.py\` script for hyphenated model names

5. **Expand test coverage** - Consider implementing medium priority models to improve coverage
   - These include specialized models that have more specific use cases

EOL
    fi
fi

# Add final recommendations for validation and execution
cat >> "$SUMMARY_REPORT" << EOL
6. **Run functional tests** - Verify that test files execute correctly
   - Run a sample of test files with small models to verify functionality
   - Focus on files that have been modified to ensure they work

7. **Integrate with distributed testing** - Connect with the distributed testing framework
   - Add support for hardware-specific configurations
   - Implement results collection and visualization

## Report Links

- [Detailed Validation Report]($(basename "$VALIDATION_REPORT"))
- [Model Coverage Report]($(basename "$MISSING_MODELS_REPORT"))

## Summary

$(if [ $VALIDATION_STATUS -eq 0 ] && [ "$CRITICAL_MISSING" = "0" ]; then
    echo "✅ **EXCELLENT STATUS** - All tests are valid and all critical models are implemented."
elif [ $VALIDATION_STATUS -eq 0 ]; then
    echo "⚠️ **GOOD STATUS** - All tests are valid, but some critical models are missing."
elif [ "$CRITICAL_MISSING" = "0" ]; then
    echo "⚠️ **MODERATE STATUS** - All critical models are implemented, but some tests have validation issues."
else
    echo "⚠️ **NEEDS IMPROVEMENT** - Some tests have validation issues and some critical models are missing."
fi)

EOL

echo "Validation and analysis complete!"
echo
echo "Reports generated:"
echo "- Validation Report: $VALIDATION_REPORT"
echo "- Missing Models Report: $MISSING_MODELS_REPORT"
echo "- Summary Report: $SUMMARY_REPORT"
echo

# Determine overall status
if [ $VALIDATION_STATUS -eq 0 ] && [ "$CRITICAL_MISSING" = "0" ]; then
    echo "✅ EXCELLENT STATUS - All tests are valid and all critical models are implemented."
    exit 0
else
    echo "⚠️ NEEDS IMPROVEMENT - See reports for details and recommendations."
    exit 1
fi
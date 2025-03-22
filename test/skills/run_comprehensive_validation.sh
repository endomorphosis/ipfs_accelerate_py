#!/bin/bash

# Comprehensive Model Test Validation Suite
# This script runs all validation tools to ensure model tests are working correctly.

# Set up logging and output directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="validation_logs"
REPORTS_DIR="reports"
mkdir -p "$LOG_DIR"
mkdir -p "$REPORTS_DIR"

LOG_FILE="${LOG_DIR}/validation_${TIMESTAMP}.log"
SUMMARY_FILE="${REPORTS_DIR}/validation_summary_${TIMESTAMP}.md"

# Initialize log file
echo "# HuggingFace Model Test Validation" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log function that prints to console and log file
log() {
    echo -e "$1"
    echo "$1" | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,3})*)?[mGK]//g" >> "$LOG_FILE"
}

# Create summary header
echo "# HuggingFace Model Test Validation Summary" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Step 1: Run syntax and structure validation
log "${YELLOW}Step 1: Running syntax and structure validation...${NC}"
python validate_model_tests.py --directory fixed_tests --report "${REPORTS_DIR}/validation_report_${TIMESTAMP}.md" --verbose
VALIDATION_STATUS=$?

if [ $VALIDATION_STATUS -eq 0 ]; then
    log "${GREEN}Syntax validation passed!${NC}"
    echo "## Syntax and Structure Validation" >> "$SUMMARY_FILE"
    echo "✅ All tests passed syntax and structure validation" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
else
    log "${RED}Syntax validation failed. See report for details.${NC}"
    echo "## Syntax and Structure Validation" >> "$SUMMARY_FILE"
    echo "❌ Some tests failed syntax validation. See detailed report: [Validation Report](validation_report_${TIMESTAMP}.md)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

# Step 2: Generate missing models report
log "${YELLOW}Step 2: Analyzing missing model implementations...${NC}"
python generate_missing_model_report.py --test-directory fixed_tests --output-report "${REPORTS_DIR}/missing_models_${TIMESTAMP}.md"
MISSING_STATUS=$?

log "${GREEN}Missing models analysis complete!${NC}"
echo "## Missing Models Analysis" >> "$SUMMARY_FILE"
echo "📊 Missing models report generated: [Missing Models Report](missing_models_${TIMESTAMP}.md)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Step 3: Run sample of tests to validate functionality
log "${YELLOW}Step 3: Running functional test validation (sample)...${NC}"
python run_test_validation.py --directory fixed_tests --max-tests 10 --report "${REPORTS_DIR}/test_execution_${TIMESTAMP}.md" --verbose
EXECUTION_STATUS=$?

if [ $EXECUTION_STATUS -eq 0 ]; then
    log "${GREEN}Functional test validation passed!${NC}"
    echo "## Functional Test Validation" >> "$SUMMARY_FILE"
    echo "✅ All sampled tests executed successfully" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
else
    log "${RED}Functional test validation failed. See report for details.${NC}"
    echo "## Functional Test Validation" >> "$SUMMARY_FILE"
    echo "❌ Some tests failed during execution. See detailed report: [Execution Report](test_execution_${TIMESTAMP}.md)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

# Step 4: Check for template consistency
log "${YELLOW}Step 4: Checking template consistency...${NC}"

# Count models by template type
ENCODER_ONLY_COUNT=$(grep -l "encoder-only" fixed_tests/*.py | wc -l)
DECODER_ONLY_COUNT=$(grep -l "decoder-only" fixed_tests/*.py | wc -l)
ENCODER_DECODER_COUNT=$(grep -l "encoder-decoder" fixed_tests/*.py | wc -l)
VISION_COUNT=$(grep -l "\"vision\"" fixed_tests/*.py | wc -l)
VISION_TEXT_COUNT=$(grep -l "vision-text" fixed_tests/*.py | wc -l)
SPEECH_COUNT=$(grep -l "\"speech\"" fixed_tests/*.py | wc -l)
MULTIMODAL_COUNT=$(grep -l "multimodal" fixed_tests/*.py | wc -l)

# Log template counts
log "${GREEN}Template consistency analysis:${NC}"
log "  Encoder-only models: $ENCODER_ONLY_COUNT"
log "  Decoder-only models: $DECODER_ONLY_COUNT"
log "  Encoder-decoder models: $ENCODER_DECODER_COUNT"
log "  Vision models: $VISION_COUNT"
log "  Vision-text models: $VISION_TEXT_COUNT"
log "  Speech models: $SPEECH_COUNT"
log "  Multimodal models: $MULTIMODAL_COUNT"

# Add to summary
echo "## Template Consistency" >> "$SUMMARY_FILE"
echo "Template consistency analysis:" >> "$SUMMARY_FILE"
echo "- Encoder-only models: $ENCODER_ONLY_COUNT" >> "$SUMMARY_FILE"
echo "- Decoder-only models: $DECODER_ONLY_COUNT" >> "$SUMMARY_FILE"
echo "- Encoder-decoder models: $ENCODER_DECODER_COUNT" >> "$SUMMARY_FILE"
echo "- Vision models: $VISION_COUNT" >> "$SUMMARY_FILE"
echo "- Vision-text models: $VISION_TEXT_COUNT" >> "$SUMMARY_FILE"
echo "- Speech models: $SPEECH_COUNT" >> "$SUMMARY_FILE"
echo "- Multimodal models: $MULTIMODAL_COUNT" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Step 5: Check for test generator issues
log "${YELLOW}Step 5: Validating test generator...${NC}"

# Attempt to generate a test file for validation
TEST_GEN_OUTPUT="${LOG_DIR}/test_generator_validation_${TIMESTAMP}.log"
python test_generator_fixed.py bert "${LOG_DIR}/test_gen_output" > "$TEST_GEN_OUTPUT" 2>&1
GEN_STATUS=$?

if [ $GEN_STATUS -eq 0 ]; then
    log "${GREEN}Test generator validation passed!${NC}"
    echo "## Test Generator Validation" >> "$SUMMARY_FILE"
    echo "✅ Test generator successfully created test files" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
else
    log "${RED}Test generator validation failed. See log for details.${NC}"
    echo "## Test Generator Validation" >> "$SUMMARY_FILE"
    echo "❌ Test generator encountered errors. See log: [Generator Log](../validation_logs/test_generator_validation_${TIMESTAMP}.log)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

# Generate final report summary
TOTAL_TESTS=$(find fixed_tests -name "test_hf_*.py" | wc -l)

if [ $VALIDATION_STATUS -eq 0 ] && [ $EXECUTION_STATUS -eq 0 ] && [ $GEN_STATUS -eq 0 ]; then
    log "${GREEN}All validation steps completed successfully!${NC}"
    log "${GREEN}Total test files: $TOTAL_TESTS${NC}"
    echo "## Summary" >> "$SUMMARY_FILE"
    echo "✅ **All validation steps completed successfully!**" >> "$SUMMARY_FILE"
    echo "- Total test files: $TOTAL_TESTS" >> "$SUMMARY_FILE"
    echo "- Log file: [Validation Log](../validation_logs/validation_${TIMESTAMP}.log)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "## Next Steps" >> "$SUMMARY_FILE"
    echo "1. Implement missing high-priority models identified in the missing models report" >> "$SUMMARY_FILE"
    echo "2. Continue integration with the distributed testing framework" >> "$SUMMARY_FILE"
    echo "3. Expand test coverage for multimodal models" >> "$SUMMARY_FILE"
    echo "4. Run comprehensive hardware compatibility tests" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    FINAL_STATUS=0
else
    log "${RED}Validation completed with errors!${NC}"
    log "${YELLOW}Total test files: $TOTAL_TESTS${NC}"
    log "${YELLOW}Please check the reports for details.${NC}"
    echo "## Summary" >> "$SUMMARY_FILE"
    echo "❌ **Validation completed with errors!**" >> "$SUMMARY_FILE"
    echo "- Total test files: $TOTAL_TESTS" >> "$SUMMARY_FILE"
    echo "- Log file: [Validation Log](../validation_logs/validation_${TIMESTAMP}.log)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "## Action Items" >> "$SUMMARY_FILE"
    echo "1. Fix syntax and structure issues in failing tests" >> "$SUMMARY_FILE"
    echo "2. Resolve runtime errors in failing functional tests" >> "$SUMMARY_FILE"
    echo "3. Address any issues with the test generator" >> "$SUMMARY_FILE"
    echo "4. Implement missing high-priority models" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    FINAL_STATUS=1
fi

# Log completion
log "Validation completed at: $(date)"
log "Summary report: $SUMMARY_FILE"
log "Log file: $LOG_FILE"

exit $FINAL_STATUS
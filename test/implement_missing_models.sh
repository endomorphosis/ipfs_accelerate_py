#!/bin/bash
# Script to implement all missing models from the HF_MODEL_COVERAGE_ROADMAP.md

set -e  # Exit on error

# Set up timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="implementation_logs_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "======= HF Model Implementation Enhancement ======="
echo "This script will update MODEL_REGISTRY, generate tests for missing models, and update the coverage report"
echo "Log files will be saved to ${LOG_DIR}"
echo "===================================================="

# Check if we're in the right directory
if [ ! -f "enhanced_generator.py" ]; then
    echo "Error: enhanced_generator.py not found in the current directory!"
    echo "Please run this script from the directory containing enhanced_generator.py"
    exit 1
fi

# 1. Update the MODEL_REGISTRY in enhanced_generator.py
echo "[1/5] Updating MODEL_REGISTRY with additional models..."
python update_model_registry.py 2>&1 | tee "${LOG_DIR}/update_registry.log"

# Copy the consolidated model mapping to the log directory for reference
if [ -f "consolidated_model_mapping.md" ]; then
    cp consolidated_model_mapping.md "${LOG_DIR}/"
    echo "✅ Model registry update completed. See consolidated_model_mapping.md for details."
else
    echo "⚠️ Model registry update completed but consolidated_model_mapping.md not found."
fi

# 2. Generate tests for critical priority models first
echo
echo "[2/5] Generating tests for critical priority models..."
python generate_missing_models.py --output-dir critical_priority_tests --priority critical 2>&1 | tee "${LOG_DIR}/critical_models.log"

# Check if reports exist and copy them
if [ -f "critical_priority_tests/generation_report.md" ]; then
    cp critical_priority_tests/generation_report.md "${LOG_DIR}/critical_models_report.md"
    CRITICAL_SUCCESS=$(grep -c "status: success" "${LOG_DIR}/critical_models.log" || echo "0")
    CRITICAL_TOTAL=$(grep -c "Processing .* critical priority models" "${LOG_DIR}/critical_models.log" | awk '{print $2}')
    echo "✅ Critical model tests: Generated ${CRITICAL_SUCCESS} out of ${CRITICAL_TOTAL}"
else
    echo "⚠️ Critical model generation completed but report not found."
fi

# 3. Generate tests for high priority models
echo
echo "[3/5] Generating tests for high priority models..."
python generate_missing_models.py --output-dir high_priority_tests --priority high 2>&1 | tee "${LOG_DIR}/high_models.log"

if [ -f "high_priority_tests/generation_report.md" ]; then
    cp high_priority_tests/generation_report.md "${LOG_DIR}/high_models_report.md"
    HIGH_SUCCESS=$(grep -c "status: success" "${LOG_DIR}/high_models.log" || echo "0")
    HIGH_TOTAL=$(grep -c "Processing .* high priority models" "${LOG_DIR}/high_models.log" | awk '{print $2}')
    echo "✅ High priority model tests: Generated ${HIGH_SUCCESS} out of ${HIGH_TOTAL}"
else
    echo "⚠️ High priority model generation completed but report not found."
fi

# 4. Generate tests for medium priority models
echo
echo "[4/5] Generating tests for medium priority models..."
python generate_missing_models.py --output-dir medium_priority_tests --priority medium 2>&1 | tee "${LOG_DIR}/medium_models.log"

if [ -f "medium_priority_tests/generation_report.md" ]; then
    cp medium_priority_tests/generation_report.md "${LOG_DIR}/medium_models_report.md"
    MEDIUM_SUCCESS=$(grep -c "status: success" "${LOG_DIR}/medium_models.log" || echo "0")
    MEDIUM_TOTAL=$(grep -c "Processing .* medium priority models" "${LOG_DIR}/medium_models.log" | awk '{print $2}')
    echo "✅ Medium priority model tests: Generated ${MEDIUM_SUCCESS} out of ${MEDIUM_TOTAL}"
else
    echo "⚠️ Medium priority model generation completed but report not found."
fi

# 5. Update the coverage report
echo
echo "[5/5] Updating coverage report..."
python update_coverage_report.py 2>&1 | tee "${LOG_DIR}/update_coverage.log"

# Copy coverage report
COVERAGE_REPORT=$(ls -t reports/model_coverage_report_*.md 2>/dev/null | head -1)
if [ -n "${COVERAGE_REPORT}" ]; then
    cp "${COVERAGE_REPORT}" "${LOG_DIR}/final_coverage_report.md"
    FINAL_COVERAGE=$(grep -o "[0-9]\+\.[0-9]\+%" "${COVERAGE_REPORT}" | head -1)
    echo "✅ Coverage report updated. Current coverage: ${FINAL_COVERAGE}"
else
    echo "⚠️ Coverage report update completed but report file not found."
fi

# 6. Verify all generated models
echo
echo "[6/6] Verifying generated model tests..."

# Verify critical models
echo "Verifying critical priority models..."
python verify_model_tests.py --directory critical_priority_tests --output "${LOG_DIR}/critical_verification.json" 2>&1 | tee "${LOG_DIR}/critical_verification.log"
CRITICAL_VERIFIED=$(grep -o "[0-9]\+ valid" "${LOG_DIR}/critical_verification.log" | awk '{print $1}' || echo "0")

# Verify high priority models
echo "Verifying high priority models..."
python verify_model_tests.py --directory high_priority_tests --output "${LOG_DIR}/high_verification.json" 2>&1 | tee "${LOG_DIR}/high_verification.log"
HIGH_VERIFIED=$(grep -o "[0-9]\+ valid" "${LOG_DIR}/high_verification.log" | awk '{print $1}' || echo "0")

# Verify medium priority models
echo "Verifying medium priority models..."
python verify_model_tests.py --directory medium_priority_tests --output "${LOG_DIR}/medium_verification.json" 2>&1 | tee "${LOG_DIR}/medium_verification.log"
MEDIUM_VERIFIED=$(grep -o "[0-9]\+ valid" "${LOG_DIR}/medium_verification.log" | awk '{print $1}' || echo "0")

# Create a summary report
echo
echo "Creating implementation summary..."
cat > "${LOG_DIR}/implementation_summary.md" << EOF
# HuggingFace Model Implementation Summary

**Date:** $(date)

## Implementation Status

| Priority | Generated | Verified | Total | Success Rate | Verification Rate |
|----------|-----------|----------|-------|-------------|-------------------|
| Critical | ${CRITICAL_SUCCESS:-0} | ${CRITICAL_VERIFIED:-0} | ${CRITICAL_TOTAL:-0} | $([ -n "${CRITICAL_TOTAL}" ] && [ "${CRITICAL_TOTAL}" != "0" ] && echo "$(( ${CRITICAL_SUCCESS} * 100 / ${CRITICAL_TOTAL} ))%" || echo "N/A") | $([ -n "${CRITICAL_SUCCESS}" ] && [ "${CRITICAL_SUCCESS}" != "0" ] && echo "$(( ${CRITICAL_VERIFIED} * 100 / ${CRITICAL_SUCCESS} ))%" || echo "N/A") |
| High     | ${HIGH_SUCCESS:-0} | ${HIGH_VERIFIED:-0} | ${HIGH_TOTAL:-0} | $([ -n "${HIGH_TOTAL}" ] && [ "${HIGH_TOTAL}" != "0" ] && echo "$(( ${HIGH_SUCCESS} * 100 / ${HIGH_TOTAL} ))%" || echo "N/A") | $([ -n "${HIGH_SUCCESS}" ] && [ "${HIGH_SUCCESS}" != "0" ] && echo "$(( ${HIGH_VERIFIED} * 100 / ${HIGH_SUCCESS} ))%" || echo "N/A") |
| Medium   | ${MEDIUM_SUCCESS:-0} | ${MEDIUM_VERIFIED:-0} | ${MEDIUM_TOTAL:-0} | $([ -n "${MEDIUM_TOTAL}" ] && [ "${MEDIUM_TOTAL}" != "0" ] && echo "$(( ${MEDIUM_SUCCESS} * 100 / ${MEDIUM_TOTAL} ))%" || echo "N/A") | $([ -n "${MEDIUM_SUCCESS}" ] && [ "${MEDIUM_SUCCESS}" != "0" ] && echo "$(( ${MEDIUM_VERIFIED} * 100 / ${MEDIUM_SUCCESS} ))%" || echo "N/A") |
| **TOTAL**    | $(( ${CRITICAL_SUCCESS:-0} + ${HIGH_SUCCESS:-0} + ${MEDIUM_SUCCESS:-0} )) | $(( ${CRITICAL_VERIFIED:-0} + ${HIGH_VERIFIED:-0} + ${MEDIUM_VERIFIED:-0} )) | $(( ${CRITICAL_TOTAL:-0} + ${HIGH_TOTAL:-0} + ${MEDIUM_TOTAL:-0} )) | $([ -n "${CRITICAL_TOTAL}" ] && [ "${HIGH_TOTAL}" ] && [ "${MEDIUM_TOTAL}" ] && [ "$(( ${CRITICAL_TOTAL} + ${HIGH_TOTAL} + ${MEDIUM_TOTAL} ))" != "0" ] && echo "$(( (${CRITICAL_SUCCESS:-0} + ${HIGH_SUCCESS:-0} + ${MEDIUM_SUCCESS:-0}) * 100 / (${CRITICAL_TOTAL} + ${HIGH_TOTAL} + ${MEDIUM_TOTAL}) ))%" || echo "N/A") | $([ -n "${CRITICAL_SUCCESS}" ] && [ "${HIGH_SUCCESS}" ] && [ "${MEDIUM_SUCCESS}" ] && [ "$(( ${CRITICAL_SUCCESS} + ${HIGH_SUCCESS} + ${MEDIUM_SUCCESS} ))" != "0" ] && echo "$(( (${CRITICAL_VERIFIED:-0} + ${HIGH_VERIFIED:-0} + ${MEDIUM_VERIFIED:-0}) * 100 / (${CRITICAL_SUCCESS} + ${HIGH_SUCCESS} + ${MEDIUM_SUCCESS}) ))%" || echo "N/A") |

## Overall Coverage

Current model coverage: ${FINAL_COVERAGE:-Unknown}

## Implementation Logs

Detailed logs and reports are available in the \`${LOG_DIR}\` directory.

## Verification Results

Verification metrics for generated test files:

- Critical models: ${CRITICAL_VERIFIED:-0}/${CRITICAL_SUCCESS:-0} verified ($([ -n "${CRITICAL_SUCCESS}" ] && [ "${CRITICAL_SUCCESS}" != "0" ] && echo "$(( ${CRITICAL_VERIFIED} * 100 / ${CRITICAL_SUCCESS} ))%" || echo "N/A"))
- High priority models: ${HIGH_VERIFIED:-0}/${HIGH_SUCCESS:-0} verified ($([ -n "${HIGH_SUCCESS}" ] && [ "${HIGH_SUCCESS}" != "0" ] && echo "$(( ${HIGH_VERIFIED} * 100 / ${HIGH_SUCCESS} ))%" || echo "N/A"))
- Medium priority models: ${MEDIUM_VERIFIED:-0}/${MEDIUM_SUCCESS:-0} verified ($([ -n "${MEDIUM_SUCCESS}" ] && [ "${MEDIUM_SUCCESS}" != "0" ] && echo "$(( ${MEDIUM_VERIFIED} * 100 / ${MEDIUM_SUCCESS} ))%" || echo "N/A"))

For detailed verification results, see:
- [Critical Models Verification](critical_verification.md)
- [High Priority Models Verification](high_verification.md)
- [Medium Priority Models Verification](medium_verification.md)

## Next Steps

1. Review failed model generations in the respective reports
2. Manually implement any critical models that failed automatic generation
3. Fix verification issues in generated but invalid test files
4. Run the test suite to execute verified tests
5. Update documentation to reflect the improved coverage
EOF

echo "===================================================="
echo "Implementation completed!"
echo 
echo "Summary:"
echo "- Log directory: ${LOG_DIR}"
echo "- Critical models: ${CRITICAL_SUCCESS:-0}/${CRITICAL_TOTAL:-0} generated, ${CRITICAL_VERIFIED:-0} verified"
echo "- High priority models: ${HIGH_SUCCESS:-0}/${HIGH_TOTAL:-0} generated, ${HIGH_VERIFIED:-0} verified"
echo "- Medium priority models: ${MEDIUM_SUCCESS:-0}/${MEDIUM_TOTAL:-0} generated, ${MEDIUM_VERIFIED:-0} verified"
echo "- Total: $(( ${CRITICAL_SUCCESS:-0} + ${HIGH_SUCCESS:-0} + ${MEDIUM_SUCCESS:-0} ))/$(( ${CRITICAL_TOTAL:-0} + ${HIGH_TOTAL:-0} + ${MEDIUM_TOTAL:-0} )) generated, $(( ${CRITICAL_VERIFIED:-0} + ${HIGH_VERIFIED:-0} + ${MEDIUM_VERIFIED:-0} )) verified"
echo "- Final coverage: ${FINAL_COVERAGE:-Unknown}"
echo
echo "Check the implementation summary at: ${LOG_DIR}/implementation_summary.md"
echo "===================================================="
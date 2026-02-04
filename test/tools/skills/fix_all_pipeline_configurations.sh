#!/bin/bash
#
# Fix All Pipeline Configurations
#
# This script automates the process of fixing pipeline configurations in HuggingFace model test files
# by running both standardization and addition tools in the correct order:
# 1. First, it standardizes existing configurations to use appropriate tasks
# 2. Then, it adds missing configurations to files that don't have them
#
# Usage: ./fix_all_pipeline_configurations.sh [tests_directory]
#

set -e  # Exit on error

# Define directories
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
TESTS_DIR="${1:-$SCRIPT_DIR/fixed_tests}"

echo "================================================"
echo "HuggingFace Model Pipeline Configuration Fixer"
echo "================================================"
echo "Tests directory: $TESTS_DIR"
echo

# Step 1: Standardize existing pipeline configurations
echo "[1/2] Standardizing existing pipeline configurations..."
python "$SCRIPT_DIR/standardize_task_configurations.py" --directory "$TESTS_DIR" --verbose
STANDARDIZE_STATUS=$?

# Step 2: Add missing pipeline configurations
echo "[2/2] Adding missing pipeline configurations..."
python "$SCRIPT_DIR/add_pipeline_configuration.py" --directory "$TESTS_DIR" --verbose
ADD_STATUS=$?

# Print final status
echo
echo "Pipeline configuration fixes complete!"
echo

# Determine overall status
if [ $STANDARDIZE_STATUS -eq 0 ] && [ $ADD_STATUS -eq 0 ]; then
    echo "✅ SUCCESS - All pipeline configurations have been fixed."
    exit 0
else
    echo "⚠️ PARTIAL SUCCESS - Some issues occurred during the fix process."
    exit 1
fi
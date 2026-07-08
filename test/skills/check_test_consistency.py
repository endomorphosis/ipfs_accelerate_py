#!/usr/bin/env python3
"""
Check the consistency of test files.

This script checks that all test files have consistent naming, particularly
focusing on hyphenated model names to ensure they are properly converted to
valid Python identifiers.

Usage:
    python check_test_consistency.py
"""

import os
import sys
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Known hyphenated models
HYPHENATED_MODELS = [
    "gpt-j",
    "gpt-neo",
    "gpt-neox",
    "xlm-roberta"
]

def to_valid_identifier(text):
    """Convert text to a valid Python identifier."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text

def check_test_files(test_dir="fixed_tests"):
    """Check the consistency of test files."""
    # Get all test files
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith("test_hf_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    
    # Check for hyphenated model names in filenames
    hyphenated_filenames = []
    for test_file in test_files:
        file_name = os.path.basename(test_file)
        if "-" in file_name:
            hyphenated_filenames.append(file_name)
    
    if hyphenated_filenames:
        logger.warning("Found test files with hyphens in filenames:")
        for file_name in sorted(hyphenated_filenames):
            logger.warning(f"  - {file_name}")
        logger.warning("These should be renamed to use underscores instead of hyphens.")
    else:
        logger.info("No test files with hyphens in filenames found.")
    
    # Check for hyphenated model names in class names
    for test_file in test_files:
        file_name = os.path.basename(test_file)
        with open(test_file, "r") as f:
            content = f.read()
        
        # Look for class definitions with hyphens
        class_matches = re.findall(r"class\s+Test(\w+)-(\w+)Models", content)
        if class_matches:
            logger.warning(f"Found class with hyphen in {file_name}:")
            for match in class_matches:
                class_name = f"Test{match[0]}-{match[1]}Models"
                logger.warning(f"  - {class_name}")
                
        # Look for variable assignments with hyphens
        var_matches = re.findall(r'(\w+)-(\w+)_MODELS_REGISTRY\s*=', content)
        if var_matches:
            logger.warning(f"Found variable with hyphen in {file_name}:")
            for match in var_matches:
                var_name = f"{match[0]}-{match[1]}_MODELS_REGISTRY"
                logger.warning(f"  - {var_name}")
    
    # Verify each hyphenated model has its properly named test file
    for model in HYPHENATED_MODELS:
        valid_name = to_valid_identifier(model)
        expected_file = f"test_hf_{valid_name}.py"
        expected_path = os.path.join(test_dir, expected_file)
        
        if os.path.exists(expected_path):
            logger.info(f"✅ Test file for {model} exists: {expected_file}")
        else:
            logger.warning(f"❌ Missing test file for {model}: {expected_file}")
    
    return True

def main():
    test_dir = "fixed_tests"
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    
    success = check_test_files(test_dir)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
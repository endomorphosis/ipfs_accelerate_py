#!/usr/bin/env python3
"""
Verify Key Models 

This script verifies that the fixed generators can generate tests for key models
and platforms, and also runs syntax verification on the generated files.
"""

import os
import sys
import subprocess
import logging
import py_compile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Key models to test
KEY_MODELS = [
    "bert",
    "t5",
    "vit",
    "clip",
    "whisper",
    "wav2vec2",
    "clap",
    "llava",
    "detr"
]

# Key platforms to test
PLATFORMS = ["cpu", "cuda,openvino", "webnn,webgpu", "all"]

# Output directory
OUTPUT_DIR = "key_model_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def verify_file(file_path):
    """Verify that a file is syntactically correct Python."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        logger.error(f"Compilation error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error verifying {file_path}: {e}")
        return False

def test_model(model, platform):
    """Test a specific model with a specific platform."""
    # Generate test file
    cmd = f"python fixed_merged_test_generator.py -g {model} -p {platform} -o {OUTPUT_DIR}/"
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True)
        
        # Get the output file path from the result
        file_path = None
        for line in result.stdout.splitlines():
            if "Generated test file:" in line:
                file_path = line.replace("Generated test file:", "").strip()
                break
        
        if file_path:
            # Verify the file
            if verify_file(file_path):
                logger.info(f"Successfully generated and verified test for {model} with {platform}")
                return True
            else:
                logger.error(f"Generated file has syntax errors: {file_path}")
                return False
        else:
            logger.error(f"Could not determine output file for {model} with {platform}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate test for {model} with {platform}: {e}")
        return False

def main():
    """Main function."""
    success_count = 0
    failure_count = 0
    
    for model in KEY_MODELS:
        for platform in PLATFORMS:
            if test_model(model, platform):
                success_count += 1
            else:
                failure_count += 1
    
    logger.info("=" * 50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total model-platform combinations: {success_count + failure_count}")
    logger.info(f"Successful tests: {success_count}")
    logger.info(f"Failed tests: {failure_count}")
    logger.info("=" * 50)
    
    if failure_count == 0:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error(f"{failure_count} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Verify mock detection in HuggingFace test files.

This script runs specified test files with different environment variable configurations
to verify that the mock detection system works correctly.

Usage:
    python verify_mock_detection.py --file FILE_PATH
    python verify_mock_detection.py --models MODEL1 MODEL2 ...
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def verify_mock_detection(file_path):
    """
    Verify mock detection for a specific file.
    
    Args:
        file_path: Path to the file to verify
    
    Returns:
        bool: True if verification passed, False otherwise
    """
    logger.info(f"Verifying mock detection for {file_path}")
    
    # Define the different environment variable configurations to test
    configs = [
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "False", "expected": "REAL INFERENCE"},
        {"MOCK_TORCH": "True", "MOCK_TRANSFORMERS": "False", "expected": "MOCK OBJECTS"},
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "True", "expected": "MOCK OBJECTS"},
        {"MOCK_TORCH": "True", "MOCK_TRANSFORMERS": "True", "expected": "MOCK OBJECTS"}
    ]
    
    success = True
    
    for config in configs:
        # Set up environment for this config
        env = os.environ.copy()
        for key, value in config.items():
            if key != "expected":
                env[key] = value
        
        expected = config["expected"]
        desc = ", ".join([f"{k}={v}" for k, v in config.items() if k != "expected"])
        
        logger.info(f"Testing with {desc}")
        
        try:
            # Run the test file
            result = subprocess.run(
                ["python", file_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if execution was successful
            if result.returncode != 0:
                logger.error(f"Test failed with exit code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                success = False
                continue
            
            # Check for expected output
            if expected in result.stdout:
                logger.info(f"✅ Test passed: Detected '{expected}' as expected")
            else:
                logger.error(f"❌ Test failed: Did not detect '{expected}'")
                logger.error(f"Output: {result.stdout}")
                success = False
        
        except subprocess.TimeoutExpired:
            logger.error(f"Test timed out after 30 seconds")
            success = False
        except Exception as e:
            logger.error(f"Error running test: {e}")
            success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Verify mock detection in HuggingFace test files")
    parser.add_argument("--file", type=str, help="Path to the file to verify")
    parser.add_argument("--models", type=str, nargs="+", help="Model types to verify")
    parser.add_argument("--base-dir", type=str, default="fixed_tests", help="Base directory containing test files")
    
    args = parser.parse_args()
    
    files_to_verify = []
    
    if args.file:
        files_to_verify.append(args.file)
    elif args.models:
        for model in args.models:
            file_path = os.path.join(args.base_dir, f"test_hf_{model}.py")
            if os.path.exists(file_path):
                files_to_verify.append(file_path)
            else:
                logger.error(f"File not found: {file_path}")
    else:
        logger.error("Please specify either --file or --models")
        return 1
    
    if not files_to_verify:
        logger.error("No files to verify")
        return 1
    
    logger.info(f"Verifying {len(files_to_verify)} files")
    
    success_count = 0
    failure_count = 0
    
    for file_path in files_to_verify:
        if verify_mock_detection(file_path):
            success_count += 1
        else:
            failure_count += 1
    
    logger.info(f"Verification summary: {success_count} passed, {failure_count} failed")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
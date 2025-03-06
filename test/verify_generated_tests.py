#\!/usr/bin/env python3
"""
Script to verify the quality of generated test files.
"""

import os
import sys
import importlib.util
import logging
import glob
import ast
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_python_syntax(file_path):
    """Verify that the file has valid Python syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Try to parse the source code
        ast.parse(source)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return False

def verify_imports(file_path):
    """Verify that the file has necessary imports."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Check for essential imports
        essential_imports = [
            "import torch",
            "import numpy",
            "import unittest",
            "import logging"
        ]
        
        for imp in essential_imports:
            if imp not in source:
                logger.warning(f"Missing import in {file_path}: {imp}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking imports in {file_path}: {e}")
        return False

def verify_platform_support(file_path):
    """Verify that the file has tests for various platforms."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Check for platform test methods
        platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
        
        for platform in platforms:
            if f"test_{platform}" not in source:
                logger.warning(f"Missing test method for {platform} in {file_path}")
        
        # At least need CPU test
        if "test_cpu" not in source:
            logger.error(f"Missing essential CPU test in {file_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking platform support in {file_path}: {e}")
        return False

def verify_test_files(directory="./generated_tests"):
    """Verify all test files in the directory."""
    test_files = glob.glob(os.path.join(directory, "test_*.py"))
    
    if not test_files:
        logger.warning(f"No test files found in {directory}")
        return False
    
    logger.info(f"Found {len(test_files)} test files in {directory}")
    
    all_valid = True
    
    for file_path in test_files:
        logger.info(f"Verifying {file_path}...")
        
        # Verify syntax
        if not verify_python_syntax(file_path):
            all_valid = False
            continue
        
        # Verify imports
        if not verify_imports(file_path):
            all_valid = False
            continue
        
        # Verify platform support
        if not verify_platform_support(file_path):
            all_valid = False
            continue
        
        logger.info(f"Verification passed for {file_path}")
    
    return all_valid

def verify_skill_files(directory="./generated_skills"):
    """Verify all skill files in the directory."""
    skill_files = glob.glob(os.path.join(directory, "skill_*.py"))
    
    if not skill_files:
        logger.warning(f"No skill files found in {directory}")
        return False
    
    logger.info(f"Found {len(skill_files)} skill files in {directory}")
    
    all_valid = True
    
    for file_path in skill_files:
        logger.info(f"Verifying {file_path}...")
        
        # Verify syntax
        if not verify_python_syntax(file_path):
            all_valid = False
            continue
        
        # Verify imports
        if not verify_imports(file_path):
            all_valid = False
            continue
        
        logger.info(f"Verification passed for {file_path}")
    
    return all_valid

def main():
    # Verify test files
    test_success = verify_test_files()
    
    # Verify skill files
    skill_success = verify_skill_files()
    
    if test_success and skill_success:
        logger.info("All verifications passed\!")
        return 0
    else:
        logger.error("Verification failed for some files.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#\!/usr/bin/env python3
"""
Validate Generator Improvements

This script validates that the recent improvements to the test and skill generators
are working correctly by generating tests for key models and verifying the results.
"""

import os
import sys
import subprocess
import logging
import tempfile
import shutil
import ast
import re
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Key models to test
KEY_MODELS = [
    "bert", "t5", "vit", "clip", "whisper", "llama"
]

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

def validate_template_support(generator_path, output_dir):
    """Validate that the generator supports templates and produces valid code."""
    results = []
    
    # Create test directory
    os.makedirs(output_dir, exist_ok=True)
    
    for model in KEY_MODELS:
        # Generate test file with template support
        output_file = os.path.join(output_dir, f"test_hf_{model}.py")
        cmd = [
            "python", generator_path,
            "--generate", model,
            "--use-db-templates",
            "--output-dir", output_dir
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check results
        success = process.returncode == 0 and os.path.exists(output_file)
        
        if success:
            # Verify syntax
            syntax_valid = verify_python_syntax(output_file)
            
            # Check for platform tests
            with open(output_file, 'r') as f:
                content = f.read()
                has_platforms = all(f"test_{platform.lower()}" in content for platform in 
                                   ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"])
            
            results.append({
                "model": model,
                "success": success,
                "syntax_valid": syntax_valid,
                "has_platforms": has_platforms,
                "file_path": output_file
            })
        else:
            logger.error(f"Failed to generate test file for {model}")
            logger.error(f"Error: {process.stderr}")
            results.append({
                "model": model,
                "success": False,
                "syntax_valid": False,
                "has_platforms": False,
                "file_path": None
            })
    
    return results

def validate_skill_generator(generator_path, output_dir):
    """Validate that the skill generator works correctly."""
    results = []
    
    # Create test directory
    os.makedirs(output_dir, exist_ok=True)
    
    for model in KEY_MODELS:
        # Generate skill file
        output_file = os.path.join(output_dir, f"skill_hf_{model}.py")
        cmd = [
            "python", generator_path,
            "--model", model,
            "--output-dir", output_dir
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check results
        success = process.returncode == 0 and os.path.exists(output_file)
        
        if success:
            # Verify syntax
            syntax_valid = verify_python_syntax(output_file)
            
            # Check for required elements
            with open(output_file, 'r') as f:
                content = f.read()
                has_get_default_device = "get_default_device" in content
                has_process_method = "def process" in content
            
            results.append({
                "model": model,
                "success": success,
                "syntax_valid": syntax_valid,
                "has_get_default_device": has_get_default_device,
                "has_process_method": has_process_method,
                "file_path": output_file
            })
        else:
            logger.error(f"Failed to generate skill file for {model}")
            logger.error(f"Error: {process.stderr}")
            results.append({
                "model": model,
                "success": False,
                "syntax_valid": False,
                "has_get_default_device": False,
                "has_process_method": False,
                "file_path": None
            })
    
    return results

def print_results(title, results):
    """Print results in a readable format."""
    print(f"\n{title}")
    print("=" * len(title))
    
    success_count = sum(1 for r in results if r["success"])
    syntax_valid_count = sum(1 for r in results if r.get("syntax_valid", False))
    
    print(f"Success: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"Syntax Valid: {syntax_valid_count}/{len(results)} ({syntax_valid_count/len(results)*100:.1f}%)")
    
    print("\nDetails:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        syntax = "✅" if r.get("syntax_valid", False) else "❌"
        print(f"{status} {r['model']}: Generated={r['success']}, Syntax={syntax}")

def main():
    # Create temporary directories
    test_output_dir = tempfile.mkdtemp(prefix="test_generator_validation_")
    skill_output_dir = tempfile.mkdtemp(prefix="skill_generator_validation_")
    
    try:
        # Validate test generator
        test_results = validate_template_support("fixed_merged_test_generator.py", test_output_dir)
        print_results("Test Generator Validation", test_results)
        
        # Validate skill generator
        skill_results = validate_skill_generator("integrated_skillset_generator.py", skill_output_dir)
        print_results("Skill Generator Validation", skill_results)
        
        # Overall success
        test_success = all(r["success"] and r["syntax_valid"] for r in test_results)
        skill_success = all(r["success"] and r["syntax_valid"] for r in skill_results)
        
        if test_success and skill_success:
            print("\nGenerator improvements validation successful! ✅")
            return 0
        else:
            print("\n❌ Generator improvements validation failed.")
            return 1
    
    finally:
        # Clean up temporary directories
        shutil.rmtree(test_output_dir, ignore_errors=True)
        shutil.rmtree(skill_output_dir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())

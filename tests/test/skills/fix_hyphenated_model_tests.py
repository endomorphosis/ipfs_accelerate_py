#!/usr/bin/env python3

"""
Script to fix hyphenated model test generation issues.
This script addresses the specific issue with hyphenated model names like 'qwen2'.
"""

import os
import re
import sys
import time
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_try_block_in_file(file_path):
    """Fix the try block syntax issues in the generated file."""
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Make backup
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        
        # Fix try block without except
        fixed_content = re.sub(
            r'try:\s*\n(\s+)import torch\s*\n\s*HAS_TORCH = True',
            r'try:\n\1import torch\n\1HAS_TORCH = True\nexcept ImportError:\n\1torch = MagicMock()\n\1HAS_TORCH = False\n\1logger.warning("torch not available, using mock")',
            content
        )
        
        # Fix try block for transformers
        fixed_content = re.sub(
            r'try:\s*\n(\s+)import transformers\s*\n\s*HAS_TRANSFORMERS = True',
            r'try:\n\1import transformers\n\1HAS_TRANSFORMERS = True\nexcept ImportError:\n\1transformers = MagicMock()\n\1HAS_TRANSFORMERS = False\n\1logger.warning("transformers not available, using mock")',
            fixed_content
        )
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        # Verify file
        try:
            compile(fixed_content, file_path, 'exec')
            logger.info(f"✅ Successfully fixed {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in fixed file: {e}")
            # Restore backup
            shutil.copy2(backup_path, file_path)
            return False
    
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False

def generate_and_fix_model(model_name, output_dir):
    """Generate a test file for the model and fix any issues."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate test file using test_generator_fixed.py
    logger.info(f"Generating test file for {model_name}...")
    generator_cmd = f"python test_generator_fixed.py --generate {model_name} --output-dir {output_dir}"
    exit_code = os.system(generator_cmd)
    
    if exit_code != 0:
        logger.warning(f"Generator exited with code {exit_code}, attempting to fix file anyway")
    
    # Determine the expected output file path
    # Convert hyphenated names to underscore for file paths
    safe_name = model_name.replace('-', '_')
    output_file = os.path.join(output_dir, f"test_hf_{safe_name}.py")
    
    if not os.path.exists(output_file):
        logger.error(f"Output file {output_file} not found")
        return False
    
    # Fix the generated file
    return fix_try_block_in_file(output_file)

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python fix_hyphenated_model_tests.py MODEL_NAME [OUTPUT_DIR]")
        return 1
    
    model_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "fixed_tests"
    
    success = generate_and_fix_model(model_name, output_dir)
    return 0 if success else 1

if __name__ == "__main__":
    with open(f"hyphenated_fix_{time.strftime('%Y%m%d_%H%M%S')}.log", "w") as log_file:
        # Redirect stdout and stderr to log file
        sys.stdout = log_file
        sys.stderr = log_file
        sys.exit(main())
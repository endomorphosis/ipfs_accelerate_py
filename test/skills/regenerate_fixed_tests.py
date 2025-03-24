#!/usr/bin/env python3
"""
This script regenerates test files using the fixed test generator.
It first runs the fix_generator.py script to fix the generator,
then regenerates tests for specific model types.
"""

import os
import sys
import subprocess
import logging
import datetime
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"regenerate_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def backup_file(file_path):
    """Create a backup of a file with timestamp."""
    if not os.path.exists(file_path):
        logger.warning(f"File not found, cannot backup: {file_path}")
        return
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{file_path}.bak.{timestamp}"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Backed up file to: {backup_path}")

def run_fix_generator():
    """Run the fix_generator.py script."""
    fix_generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_generator.py")
    if not os.path.exists(fix_generator_path):
        logger.error(f"Fix generator script not found: {fix_generator_path}")
        return False
    
    logger.info("Running fix_generator.py to repair the test generator...")
    
    # Backup the original test generator
    generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_generator_fixed.py")
    backup_file(generator_path)
    
    # Run the fix script
    result = subprocess.run([sys.executable, fix_generator_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to fix test generator: {result.stderr}")
        return False
    
    logger.info(f"Fix generator output: {result.stdout}")
    return True

def run_test_generator(model_type, output_dir=None):
    """Run the test generator for a specific model type."""
    generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_generator_fixed.py")
    if not os.path.exists(generator_path):
        logger.error(f"Test generator not found: {generator_path}")
        return False
    
    cmd = [sys.executable, generator_path, "--model-type", model_type]
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    logger.info(f"Running test generator for model type: {model_type}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to generate test for {model_type}: {result.stderr}")
        return False
    
    logger.info(f"Generated test for {model_type}")
    return True

def verify_test_syntax(test_file):
    """Verify the syntax of a generated test file."""
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False
    
    logger.info(f"Verifying syntax of test file: {test_file}")
    
    # Try to compile the file to check for syntax errors
    with open(test_file, 'r') as f:
        content = f.read()
    
    try:
        compile(content, test_file, 'exec')
        logger.info(f"Test file syntax is valid: {test_file}")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {test_file} at line {e.lineno}: {e.msg}")
        if hasattr(e, 'text') and e.text:
            logger.error(f"Line content: {e.text.strip()}")
        return False

def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    return output_dir

def main():
    """Main function to regenerate tests."""
    # Define the output directory for regenerated tests
    output_dir = ensure_output_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixed_tests"))
    
    # 1. Fix the test generator
    if not run_fix_generator():
        logger.error("Failed to fix test generator. Aborting test regeneration.")
        return False
    
    # 2. Define model types to regenerate
    model_types = [
        "bert",            # Encoder-only
        "gpt2",            # Decoder-only
        "t5",              # Encoder-decoder
        "vit",             # Vision
        "clip",            # Vision-text
        "wav2vec2",        # Speech
        "gpt-j",           # Hyphenated model name
        "xlm-roberta",     # Hyphenated model name
        "llava"            # Multimodal
    ]
    
    # 3. Regenerate tests
    successful = 0
    failed = 0
    
    for model_type in model_types:
        logger.info(f"Processing model type: {model_type}")
        
        # Determine the expected output file
        if "-" in model_type:
            # For hyphenated models, replace hyphen with underscore
            test_file_name = f"test_hf_{model_type.replace('-', '_')}.py"
        else:
            test_file_name = f"test_hf_{model_type}.py"
        
        test_file_path = os.path.join(output_dir, test_file_name)
        
        # Backup the existing test file if it exists
        if os.path.exists(test_file_path):
            backup_file(test_file_path)
        
        # Generate the test
        if run_test_generator(model_type, output_dir):
            # Verify the syntax
            if verify_test_syntax(test_file_path):
                successful += 1
                logger.info(f"Successfully regenerated test for {model_type}")
            else:
                failed += 1
                logger.error(f"Test regenerated for {model_type}, but syntax verification failed")
        else:
            failed += 1
            logger.error(f"Failed to regenerate test for {model_type}")
    
    # 4. Report results
    logger.info(f"Test regeneration complete. Success: {successful}, Failed: {failed}")
    
    if failed > 0:
        logger.warning("Some tests failed to regenerate correctly. Review the logs for details.")
        return False
    
    logger.info("All tests regenerated successfully!")
    return True

if __name__ == "__main__":
    main()
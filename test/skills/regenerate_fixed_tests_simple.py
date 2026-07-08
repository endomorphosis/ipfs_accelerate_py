#!/usr/bin/env python3
"""
Simple script to regenerate test files for key model types
using the fixed test generator.
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

def run_test_generator(model_type, output_dir=None):
    """Run the test generator for a specific model type."""
    generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_generator_fixed.py")
    if not os.path.exists(generator_path):
        logger.error(f"Test generator not found: {generator_path}")
        return False
    
    cmd = [sys.executable, generator_path, "--generate", model_type]
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    logger.info(f"Running test generator for model type: {model_type}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to generate test for {model_type}: {result.stderr}")
        logger.error(f"Output: {result.stdout}")
        return False
    
    logger.info(f"Generated test for {model_type}")
    return True

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
    
    # Define model types to regenerate (just one for now as a test)
    model_types = ["bert"]
    
    # Regenerate tests
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
            successful += 1
            logger.info(f"Successfully regenerated test for {model_type}")
        else:
            failed += 1
            logger.error(f"Failed to regenerate test for {model_type}")
    
    # Report results
    logger.info(f"Test regeneration complete. Success: {successful}, Failed: {failed}")
    
    if failed > 0:
        logger.warning("Some tests failed to regenerate. Review the logs for details.")
        return False
    
    logger.info("All tests regenerated successfully!")
    return True

if __name__ == "__main__":
    main()
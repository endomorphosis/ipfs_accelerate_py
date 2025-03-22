#!/usr/bin/env python3
"""
Generate test files for all hyphenated model names.

This script executes the fix_indentation_and_apply_template.py script 
for all known hyphenated models to create properly formatted test files.

Usage:
    python generate_hyphenated_tests.py
"""

import os
import sys
import subprocess
import logging
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

def main():
    # Get directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output directory
    output_dir = os.path.join(script_dir, "fixed_tests")
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the fix script
    fix_script = os.path.join(script_dir, "fix_indentation_and_apply_template.py")
    
    # Ensure the fix script is executable
    os.chmod(fix_script, 0o755)
    
    # Generate test files for all hyphenated models
    for model in HYPHENATED_MODELS:
        logger.info(f"Generating test file for {model}")
        
        # Call the fix script
        cmd = [sys.executable, fix_script, "--model", model, "--output-dir", output_dir]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Check if successful
            output_file = os.path.join(output_dir, f"test_hf_{model.replace('-', '_')}.py")
            if os.path.exists(output_file):
                logger.info(f"✅ Successfully generated {output_file}")
                
                # Verify syntax
                try:
                    subprocess.run(
                        [sys.executable, "-m", "py_compile", output_file], 
                        check=True, capture_output=True
                    )
                    logger.info(f"✅ Syntax verification passed for {output_file}")
                except subprocess.CalledProcessError:
                    logger.error(f"❌ Syntax verification failed for {output_file}")
            else:
                logger.error(f"❌ Failed to generate {output_file}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error running fix script for {model}: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
    
    # Count how many were successfully generated
    generated_files = [f for f in os.listdir(output_dir) 
                       if f.startswith("test_hf_") and f.endswith(".py")]
    
    valid_files = []
    for file in generated_files:
        file_path = os.path.join(output_dir, file)
        try:
            subprocess.run(
                [sys.executable, "-m", "py_compile", file_path], 
                check=True, capture_output=True
            )
            valid_files.append(file)
        except subprocess.CalledProcessError:
            pass
    
    # Print summary
    logger.info(f"\nGenerated {len(generated_files)} test files")
    logger.info(f"Validated {len(valid_files)} test files")
    
    # Print the list of valid files
    logger.info("\nValid test files:")
    for file in sorted(valid_files):
        logger.info(f"  - {file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
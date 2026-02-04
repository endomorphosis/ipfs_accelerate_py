#!/usr/bin/env python3
"""
Regenerate test files for models with hyphenated names.

This script finds all hyphenated model names in the architecture types
and regenerates their test files using the specialized direct string generation approach.
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import the modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the necessary modules
try:
    from simplified_fix_hyphenated import (
        create_hyphenated_test_file, 
        find_hyphenated_models,
        ARCHITECTURE_TYPES
    )
except ImportError:
    logger.error("Could not import simplified_fix_hyphenated.py. Make sure it exists in the current directory.")
    sys.exit(1)

def regenerate_all_hyphenated_tests(output_dir):
    """Regenerate all test files for hyphenated model names."""
    # Find all hyphenated model names
    hyphenated_models = find_hyphenated_models()
    logger.info(f"Found {len(hyphenated_models)} hyphenated model names to regenerate")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Regenerate test files
    success_count = 0
    failure_count = 0
    
    for model_name in hyphenated_models:
        logger.info(f"Regenerating test file for {model_name}")
        success, error = create_hyphenated_test_file(model_name, output_dir)
        
        if success:
            success_count += 1
            logger.info(f"Successfully regenerated test file for {model_name}")
        else:
            failure_count += 1
            logger.error(f"Failed to regenerate test file for {model_name}: {error}")
    
    # Print summary
    logger.info(f"Regeneration complete")
    logger.info(f"Total hyphenated models processed: {len(hyphenated_models)}")
    logger.info(f"Success: {success_count}, Failures: {failure_count}")
    
    return success_count, failure_count

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Regenerate test files for models with hyphenated names")
    
    parser.add_argument("--output-dir", type=str, default="fixed_tests",
                        help="Output directory for regenerated test files")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list the hyphenated models without regenerating files")
    
    args = parser.parse_args()
    
    # Prepare output directory path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = current_dir / output_dir
    
    if args.list_only:
        # Just list the hyphenated models
        hyphenated_models = find_hyphenated_models()
        print("\nHyphenated model names found:")
        for model in hyphenated_models:
            print(f"  - {model}")
        print(f"\nTotal: {len(hyphenated_models)} hyphenated models")
    else:
        # Regenerate the test files
        success_count, failure_count = regenerate_all_hyphenated_tests(output_dir)
        return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
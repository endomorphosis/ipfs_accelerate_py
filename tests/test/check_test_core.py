#!/usr/bin/env python3
# check_test_core.py
# Script to check if test_core.ts compiles

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'check_test_core_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

TARGET_DIR = os.path.abspath("../ipfs_accelerate_js")
TEST_FILE = "src/test_core.ts"

def main():
    """Check if test_core.ts compiles"""
    logger.info(f"Checking TypeScript file: {TEST_FILE}")
    
    # Run TypeScript compiler on the specific file
    logger.info("Running TypeScript compiler on test_core.ts")
    
    result = subprocess.run(
        ["npx", "tsc", TEST_FILE, "--noEmit", "--skipLibCheck"],
        cwd=TARGET_DIR,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Save output to file
    error_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_core_errors.log")
    with open(error_log_path, 'w', encoding='utf-8') as f:
        f.write(result.stdout + "\n\n" + result.stderr)
    
    if result.returncode == 0:
        logger.info("test_core.ts compiled successfully!")
        print("\nSuccess! test_core.ts compiles without errors.")
    else:
        error_count = result.stdout.count("error TS")
        logger.warning(f"TypeScript compilation found {error_count} errors in test_core.ts")
        print(f"\nFound {error_count} TypeScript errors in test_core.ts. Check {error_log_path} for details.")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
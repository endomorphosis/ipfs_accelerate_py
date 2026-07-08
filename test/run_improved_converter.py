#!/usr/bin/env python3
# run_improved_converter.py
# Script to test the improved Python to TypeScript converter

import os
import sys
import logging
from improve_py_to_ts_converter import ConverterImprovements, convert_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('converter_test.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Test the improved converter on a sample file"""
    # Get path to sample file
    sample_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_webgpu_backend.py")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_webgpu_backend_improved.ts")
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample file not found: {sample_file}")
        return
    
    logger.info(f"Converting sample file: {sample_file}")
    
    # Convert file
    success = convert_file(sample_file, output_file)
    
    if success:
        logger.info(f"Conversion successful! Output written to: {output_file}")
        
        # Print success message and file info
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        logger.info(f"Generated TypeScript file has {len(lines)} lines")
        
        # Check for common TypeScript patterns
        import re
        interfaces = len(re.findall(r'interface\s+\w+', content))
        methods = len(re.findall(r'[a-zA-Z0-9_]+\([^)]*\)\s*:\s*[a-zA-Z0-9_<>|]+', content))
        props = len(re.findall(r'[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_<>|]+\s*[;=]', content))
        
        logger.info(f"TypeScript file contains:")
        logger.info(f"- {interfaces} interfaces")
        logger.info(f"- {methods} typed methods")
        logger.info(f"- {props} typed properties")
        
        # Print snippet of the beginning
        print("\nFirst 20 lines of the generated TypeScript file:")
        print("----------------------------------------------")
        print("\n".join(lines[:20]))
        print("...")
    else:
        logger.error("Conversion failed")

if __name__ == "__main__":
    main()
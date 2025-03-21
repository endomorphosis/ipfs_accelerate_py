#!/usr/bin/env python3
"""
Comprehensive test file fixer.

This script fixes all test files with hyphenated model names and syntax issues.
It applies a direct approach to all files in a directory.
"""

import os
import sys
import re
import logging
import argparse
import time
from pathlib import Path
import subprocess

# Set up logging with timestamp
log_file = f"comprehensive_fix_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_test_files(directory):
    """Find all test files in the directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('test_hf_') and filename.endswith('.py'):
                files.append(os.path.join(root, filename))
    return files

def fix_file(file_path, output_dir=None):
    """Fix the file and optionally save to output directory."""
    try:
        # Run the direct fix script
        cmd = [sys.executable, "fix_single_file.py", "--file", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to fix {file_path}: {result.stderr}")
            return False, None
        
        # If output directory is specified, copy the fixed file there
        if output_dir:
            # Get the new file path (it might have been renamed)
            new_file_path = result.stdout.split("Successfully fixed file: ")[1].strip()
            
            # Determine the output file name (using the fixed name without hyphens)
            output_file = os.path.join(output_dir, os.path.basename(new_file_path))
            
            # Copy file to output directory
            with open(new_file_path, 'r') as src, open(output_file, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Copied fixed file to {output_file}")
            return True, output_file
        
        return True, file_path
    
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False, None

def verify_file(file_path):
    """Verify Python syntax in the file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        compile(content, file_path, 'exec')
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix all test files")
    parser.add_argument("--dir", type=str, default="fixed_tests", 
                        help="Directory containing files to fix")
    parser.add_argument("--output-dir", type=str, default="fixed_files_manual", 
                        help="Directory to save fixed files")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify Python syntax after fixing")
    parser.add_argument("--only-hyphenated", action="store_true", 
                        help="Only fix files with hyphenated names")
    parser.add_argument("--specific-file", type=str, 
                        help="Fix a specific file")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fix specific file if requested
    if args.specific_file:
        if os.path.exists(args.specific_file):
            success, output_file = fix_file(args.specific_file, args.output_dir)
            if success:
                logger.info(f"Successfully fixed file: {args.specific_file} -> {output_file}")
                
                if args.verify and output_file:
                    if verify_file(output_file):
                        logger.info(f"Syntax verification passed for {output_file}")
                    else:
                        logger.error(f"Syntax verification failed for {output_file}")
            else:
                logger.error(f"Failed to fix file: {args.specific_file}")
            return 0 if success else 1
        else:
            logger.error(f"File not found: {args.specific_file}")
            return 1
    
    # Find all test files in the directory
    files = find_test_files(args.dir)
    logger.info(f"Found {len(files)} test files in {args.dir}")
    
    # Filter for hyphenated files if requested
    if args.only_hyphenated:
        files = [f for f in files if '-' in os.path.basename(f)]
        logger.info(f"Filtered to {len(files)} hyphenated test files")
    
    # Fix each file
    results = {
        "total": len(files),
        "success": 0,
        "failure": 0,
        "fixed_files": [],
        "failed_files": []
    }
    
    for file_path in files:
        logger.info(f"Processing file: {file_path}")
        
        success, output_file = fix_file(file_path, args.output_dir)
        
        if success:
            results["success"] += 1
            results["fixed_files"].append(output_file)
            
            if args.verify and output_file:
                if verify_file(output_file):
                    logger.info(f"Syntax verification passed for {output_file}")
                else:
                    logger.error(f"Syntax verification failed for {output_file}")
                    # If verification fails, move from success to failure
                    results["success"] -= 1
                    results["failure"] += 1
                    results["fixed_files"].remove(output_file)
                    results["failed_files"].append(file_path)
        else:
            results["failure"] += 1
            results["failed_files"].append(file_path)
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"- Total files: {results['total']}")
    logger.info(f"- Successfully fixed: {results['success']}")
    logger.info(f"- Failed to fix: {results['failure']}")
    
    if results["failed_files"]:
        logger.info("\nFailed files:")
        for file_path in results["failed_files"]:
            logger.info(f"  - {file_path}")
    
    # Write summary to file
    summary_file = f"fix_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Total files: {results['total']}\n")
        f.write(f"Successfully fixed: {results['success']}\n")
        f.write(f"Failed to fix: {results['failure']}\n\n")
        
        if results["fixed_files"]:
            f.write("Fixed files:\n")
            for file_path in results["fixed_files"]:
                f.write(f"  - {file_path}\n")
        
        if results["failed_files"]:
            f.write("\nFailed files:\n")
            for file_path in results["failed_files"]:
                f.write(f"  - {file_path}\n")
    
    logger.info(f"Summary written to {summary_file}")
    return 0 if results["failure"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
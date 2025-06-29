#!/usr/bin/env python3
"""
Execute the comprehensive indentation fix on multiple test files.

This script:
1. Identifies target test files
2. Creates backups
3. Runs the complete_indentation_fix.py script
4. Verifies syntax after fixing
5. Generates a summary report

Usage:
    python run_indentation_fix.py [--pattern PATTERN] [--directory DIR] [--verify] [--force]
"""

import os
import sys
import glob
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_indentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def find_test_files(directory, pattern):
    """Find test files matching the pattern."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return sorted(files)

def fix_file_indentation(file_path, verify=True):
    """Run the indentation fix script on a file."""
    fix_script = "complete_indentation_fix.py"
    
    # Ensure the fix script exists
    if not os.path.exists(fix_script):
        logger.error(f"Fix script not found: {fix_script}")
        return False
    
    # Build command
    cmd = [sys.executable, fix_script, file_path]
    if verify:
        cmd.append("--verify")
    
    # Run the fix script
    logger.info(f"Fixing indentation in {file_path}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully fixed indentation in {file_path}")
            return True
        else:
            logger.error(f"❌ Failed to fix indentation in {file_path}")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error running fix script on {file_path}: {e}")
        return False

def verify_syntax(file_path):
    """Verify Python syntax of a file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {file_path}: Syntax is valid")
            return True
        else:
            logger.error(f"❌ {file_path}: Syntax error")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def run_fixes(directory, pattern, verify=True, force=False):
    """
    Run indentation fixes on multiple files.
    
    Args:
        directory: Directory containing test files
        pattern: File pattern to match
        verify: Whether to verify syntax after fixing
        force: Whether to run even if backup already exists
    
    Returns:
        Tuple of (num_fixed, num_failed, total)
    """
    # Find test files
    files = find_test_files(directory, pattern)
    logger.info(f"Found {len(files)} files matching pattern {pattern}")
    
    fixed = []
    failed = []
    skipped = []
    
    for file_path in files:
        # Check if backup already exists
        backup_path = f"{file_path}.bak"
        if os.path.exists(backup_path) and not force:
            logger.info(f"Skipping {file_path} - backup already exists (use --force to override)")
            skipped.append(file_path)
            continue
        
        # Run fix
        if fix_file_indentation(file_path, verify=verify):
            fixed.append(file_path)
        else:
            failed.append(file_path)
    
    # Print summary
    logger.info("\nIndentation Fix Summary:")
    logger.info(f"- Fixed: {len(fixed)} files")
    logger.info(f"- Failed: {len(failed)} files")
    logger.info(f"- Skipped: {len(skipped)} files")
    logger.info(f"- Total: {len(files)} files")
    
    if failed:
        logger.info("\nFailed files:")
        for f in failed:
            logger.info(f"  - {f}")
    
    return len(fixed), len(failed), len(files)

def main():
    parser = argparse.ArgumentParser(description="Run indentation fixes on multiple test files")
    parser.add_argument("--pattern", type=str, default="test_hf_*.py", 
                        help="File pattern to match (default: test_hf_*.py)")
    parser.add_argument("--directory", type=str, default=".", 
                        help="Directory containing test files (default: current directory)")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify syntax after fixing")
    parser.add_argument("--force", action="store_true", 
                        help="Run even if backup already exists")
    
    args = parser.parse_args()
    
    # Run fixes
    fixed, failed, total = run_fixes(
        directory=args.directory,
        pattern=args.pattern,
        verify=args.verify,
        force=args.force
    )
    
    # Return appropriate exit code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
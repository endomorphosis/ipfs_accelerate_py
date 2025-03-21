#!/usr/bin/env python3
"""
Comprehensive test file fixer script.

This script combines multiple fix scripts to comprehensively fix all issues with test files:
1. Hyphenated model names in variable names, class names, and file names
2. Mock detection issues (imports, class capitalization)
3. Proper environment variable detection
4. Consistent class naming
5. Consistent registry keys

Usage:
    python fix_all_tests.py [--dir DIR] [--file FILE] [--rename-files] [--verify]
"""

import os
import sys
import logging
import argparse
import subprocess
import time
from pathlib import Path

# Set up logging with timestamp
log_file = f"fix_all_tests_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_scripts_available():
    """Ensure all required fix scripts are available."""
    required_scripts = [
        "fix_hyphenated_model_names.py",
        "fix_mock_detection_errors.py",
        "manually_fix_test_file.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.error(f"Missing required scripts: {', '.join(missing_scripts)}")
        return False
    
    return True

def verify_python_file(file_path):
    """Verify that a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Compile to check syntax
        compile(content, file_path, 'exec')
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        # Show the offending line
        if e.lineno and e.text:
            logger.error(f"Line {e.lineno}: {e.text.strip()}")
        return False
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return False

def fix_file(file_path, rename=False, verify=True):
    """Apply all fixes to a file."""
    logger.info(f"Fixing file: {file_path}")
    
    # Create a backup
    backup_path = f"{file_path}.comprehensive.bak"
    try:
        with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup at: {backup_path}")
    except Exception as e:
        logger.error(f"Error creating backup for {file_path}: {e}")
        return False
    
    success = True
    
    # Step 1: Apply hyphenated fixes
    logger.info("Applying hyphenated model name fixes...")
    try:
        result = subprocess.run(
            [sys.executable, "fix_hyphenated_model_names.py", "--file", file_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Hyphenated fixes failed: {result.stderr}")
            success = False
        else:
            logger.info("Hyphenated fixes applied successfully")
    except Exception as e:
        logger.error(f"Error applying hyphenated fixes: {e}")
        success = False
    
    # Step 2: Apply mock detection fixes
    if success:
        logger.info("Applying mock detection fixes...")
        try:
            result = subprocess.run(
                [sys.executable, "fix_mock_detection_errors.py", "--file", file_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Mock detection fixes failed: {result.stderr}")
                success = False
            else:
                logger.info("Mock detection fixes applied successfully")
        except Exception as e:
            logger.error(f"Error applying mock detection fixes: {e}")
            success = False
    
    # Step 3: For stubborn files, try the manual fixer as a last resort
    if not success:
        logger.warning("Previous fixes failed, attempting manual fix...")
        try:
            result = subprocess.run(
                [sys.executable, "manually_fix_test_file.py", "--file", file_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Manual fixes failed: {result.stderr}")
            else:
                logger.info("Manual fixes applied successfully")
                success = True
        except Exception as e:
            logger.error(f"Error applying manual fixes: {e}")
            success = False
    
    # Verify the file has correct syntax
    if success and verify:
        logger.info("Verifying Python syntax...")
        if verify_python_file(file_path):
            logger.info("Syntax verification passed")
        else:
            logger.error("Syntax verification failed")
            success = False
            
            # Restore from backup if verification failed
            logger.info("Restoring from backup due to syntax errors")
            try:
                with open(backup_path, 'r') as src, open(file_path, 'w') as dst:
                    dst.write(src.read())
                logger.info("Restored from backup")
            except Exception as e:
                logger.error(f"Error restoring from backup: {e}")
    
    # Rename file if requested and successful
    if success and rename and '-' in os.path.basename(file_path):
        new_name = file_path.replace('-', '_')
        try:
            os.rename(file_path, new_name)
            logger.info(f"Renamed file to: {new_name}")
        except Exception as e:
            logger.error(f"Error renaming file: {e}")
    
    return success

def fix_directory(directory, rename=False, verify=True):
    """Apply fixes to all Python files in a directory."""
    logger.info(f"Fixing all Python files in directory: {directory}")
    
    # Find all Python files
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and "test_hf_" in file:
                python_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(python_files)} test files to fix")
    
    # Apply fixes to each file
    results = {
        "total": len(python_files),
        "success": 0,
        "failure": 0,
        "failures": []
    }
    
    for file_path in python_files:
        if fix_file(file_path, rename=rename, verify=verify):
            results["success"] += 1
        else:
            results["failure"] += 1
            results["failures"].append(file_path)
    
    # Print summary
    logger.info(f"\nSummary:")
    logger.info(f"- Successfully fixed: {results['success']} files")
    logger.info(f"- Failed to fix: {results['failure']} files")
    
    if results["failures"]:
        logger.info("\nFailed files:")
        for file_path in results["failures"]:
            logger.info(f"  - {file_path}")
    
    return results["failure"] == 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix all issues in test files")
    parser.add_argument("--dir", type=str, help="Directory containing files to fix")
    parser.add_argument("--file", type=str, help="Single file to fix")
    parser.add_argument("--rename-files", action="store_true", help="Rename files to replace hyphens with underscores")
    parser.add_argument("--verify", action="store_true", help="Verify Python syntax after fixes")
    
    args = parser.parse_args()
    
    # Ensure required scripts are available
    if not ensure_scripts_available():
        return 1
    
    # Fix a single file or directory
    if args.file:
        success = fix_file(args.file, rename=args.rename_files, verify=args.verify)
        if success:
            logger.info(f"Successfully fixed file: {args.file}")
        else:
            logger.error(f"Failed to fix file: {args.file}")
        return 0 if success else 1
    
    elif args.dir:
        success = fix_directory(args.dir, rename=args.rename_files, verify=args.verify)
        if success:
            logger.info(f"Successfully fixed all files in directory: {args.dir}")
        else:
            logger.warning(f"Some files in directory could not be fixed: {args.dir}")
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
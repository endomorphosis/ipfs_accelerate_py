#!/usr/bin/env python3
"""
Cleanup script to remove temporary files after successful integration.
This script should only be run after verifying that the integration is working correctly.
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Files to remove after successful integration
TEMP_FILES = [
    "test_generator_fixed.py",
    "regenerate_tests.py",
    "fixed_tests",
    "integrate_generator_fixes.py",
    "test_integration.py",
    "FIXED_GENERATOR_README.md",
    "INTEGRATION_PLAN.md",
    "TESTING_FIXES_SUMMARY.md",
    "fix_file_indentation.py",
    "fix_test_indentation.py",
    "regenerate_tests_with_fixes.py",
    "fix_indentation_*.log",
    "regenerate_tests_*.log"
]

def create_backup(path, backup_dir="backups"):
    """Create a backup of a file or directory before removing."""
    if not os.path.exists(path):
        return None
        
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create backup filename/dirname
    backup_name = os.path.basename(path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{backup_name}.bak.{timestamp}")
    
    # Create backup
    try:
        if os.path.isdir(path):
            shutil.copytree(path, backup_path)
        else:
            shutil.copy2(path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup of {path}: {e}")
        return None

def cleanup_files(backup_dir="backups"):
    """Remove temporary files after successful integration."""
    logger.info("Cleaning up temporary files...")
    
    backups = []
    removed = []
    failed = []
    
    # Process glob patterns first
    expanded_files = []
    for pattern in TEMP_FILES:
        if "*" in pattern:
            # Handle globs
            for matched_file in Path().glob(pattern):
                expanded_files.append(str(matched_file))
        else:
            expanded_files.append(pattern)
    
    # Remove files/directories
    for file_path in expanded_files:
        if not os.path.exists(file_path):
            logger.info(f"Skipped (not found): {file_path}")
            continue
            
        # Create backup
        backup_path = create_backup(file_path, backup_dir)
        if backup_path:
            backups.append((file_path, backup_path))
        
        # Remove file/directory
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
                logger.info(f"Removed directory: {file_path}")
                removed.append(file_path)
            else:
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
                removed.append(file_path)
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
            failed.append(file_path)
    
    # Print summary
    logger.info("\nCleanup Summary:")
    logger.info(f"- Files processed: {len(expanded_files)}")
    logger.info(f"- Files removed: {len(removed)}")
    logger.info(f"- Backups created: {len(backups)}")
    logger.info(f"- Failed removals: {len(failed)}")
    
    if backups:
        logger.info("\nBackups created:")
        for original, backup in backups:
            logger.info(f"- {original} -> {backup}")
        
        # Create restore script
        restore_script = "restore_files.sh"
        with open(restore_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Script to restore backups if needed\n\n")
            for original, backup in backups:
                if os.path.isdir(backup):
                    f.write(f"# Restore directory\n")
                    f.write(f"mkdir -p $(dirname {original})\n")
                    f.write(f"cp -r {backup}/* {original}/\n\n")
                else:
                    f.write(f"# Restore file\n")
                    f.write(f"mkdir -p $(dirname {original})\n")
                    f.write(f"cp {backup} {original}\n\n")
            
        os.chmod(restore_script, 0o755)
        logger.info(f"Created restore script: {restore_script}")
    
    logger.info("\nCleanup completed successfully.")
    logger.info("\nRemaining tasks:")
    logger.info("1. Update project documentation to reflect the architecture-aware approach")
    logger.info("2. Consider creating a test suite for the generator itself to prevent regressions")
    logger.info("3. Update the test automation documentation with the new approach")
    logger.info("4. Implement CI/CD integration for syntax validation as specified in INTEGRATION_PLAN.md")


def main():
    parser = argparse.ArgumentParser(description="Clean up temporary files after successful integration")
    parser.add_argument("--force", action="store_true", help="Force cleanup without confirmation")
    parser.add_argument("--backup-dir", type=str, default="backups", 
                      help="Directory for backups (default: ./backups)")
    parser.add_argument("--dry-run", action="store_true", 
                      help="Show what would be removed without actually removing")
    args = parser.parse_args()
    
    # Backup directory
    backup_dir = os.path.abspath(args.backup_dir)
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE: No files will be removed")
        logger.info("Files that would be processed:")
        
        for pattern in TEMP_FILES:
            if "*" in pattern:
                # Handle globs
                for matched_file in Path().glob(pattern):
                    logger.info(f"- {matched_file}")
            else:
                if os.path.exists(pattern):
                    logger.info(f"- {pattern}")
                else:
                    logger.info(f"- {pattern} (not found)")
        return
    
    # Confirmation if not forced
    if not args.force:
        confirm = input("This will remove all temporary files. Are you sure? (y/N): ")
        if confirm.lower() != 'y':
            logger.info("Cleanup aborted.")
            return
    
    cleanup_files(backup_dir=backup_dir)


if __name__ == "__main__":
    main()
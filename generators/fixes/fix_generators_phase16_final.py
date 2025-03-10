#!/usr/bin/env python3
"""
Fix Generators Phase 16 - Final Fix Script

This script copies the clean versions of the generators to the main
generator files, ensuring they work correctly without syntax errors.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File mapping: clean version -> original version
FILE_MAPPING = {
    "fixed_merged_test_generator_clean.py": "fixed_merged_test_generator.py",
    "merged_test_generator_clean.py": "merged_test_generator.py", 
    "integrated_skillset_generator_clean.py": "integrated_skillset_generator.py"
}

# Ensure backup directory exists
backup_dir = Path("backups")
backup_dir.mkdir(exist_ok=True)

def create_backup(file_path):
    """Create a backup of a file."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File does not exist, cannot back up: {file_path}")
        return False
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f"{file_path.name}.bak_{timestamp}"
    
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def fix_generator(clean_version, original_version):
    """Fix a generator by copying the clean version to the original."""
    clean_path = Path(clean_version)
    original_path = Path(original_version)
    
    if not clean_path.exists():
        logger.error(f"Clean version does not exist: {clean_path}")
        return False
    
    # Backup original if it exists
    if original_path.exists():
        if not create_backup(original_path):
            logger.error(f"Failed to create backup for {original_path}, skipping fix")
            return False
    
    # Copy clean version to original
    try:
        shutil.copy2(clean_path, original_path)
        os.chmod(original_path, 0o755)  # Make executable
        logger.info(f"Fixed generator: {original_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to fix generator: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting Phase 16 generator fixes...")
    
    success = True
    for clean_version, original_version in FILE_MAPPING.items():
        logger.info(f"Fixing {original_version} with clean version {clean_version}...")
        if not fix_generator(clean_version, original_version):
            success = False
    
    if success:
        logger.info("All generators fixed successfully!")
        
        # Provide instructions for using the fixed generators
        print("\nYou can now use the fixed generators:")
        print("  python fixed_merged_test_generator.py -g bert -p cpu,cuda,webgpu -o test_outputs/")
        print("  python merged_test_generator.py -g vit-base -p all -o test_outputs/")
        print("  python integrated_skillset_generator.py -m clip -p all -o test_outputs/")
        print("\nYou can also verify all generators with:")
        print("  python test_all_generators.py")
        
        return 0
    else:
        logger.error("Some generators could not be fixed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
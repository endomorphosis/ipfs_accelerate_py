#!/usr/bin/env python3
"""
Update fix_hyphenated_models.py to use the fixed templates.

This script modifies the fix_hyphenated_models.py script to use the templates from the fixed_templates directory.
"""

import os
import re
import shutil
import logging
from pathlib import Path
import argparse
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TEMPLATES_DIR = CURRENT_DIR / "fixed_templates"
FIX_HYPHENATED_MODELS_PATH = CURRENT_DIR / "fix_hyphenated_models.py"
BACKUP_PATH = CURRENT_DIR / "fix_hyphenated_models.py.bak"

def update_script():
    """Update the fix_hyphenated_models.py script to use the fixed templates."""
    # Backup original script
    if os.path.exists(FIX_HYPHENATED_MODELS_PATH):
        shutil.copy(FIX_HYPHENATED_MODELS_PATH, BACKUP_PATH)
        logger.info(f"Backed up original script to {BACKUP_PATH}")
    
    # Read the original script
    with open(FIX_HYPHENATED_MODELS_PATH, 'r') as f:
        content = f.read()
    
    # Update template directory constant
    content = re.sub(
        r'TEMPLATES_DIR = CURRENT_DIR / "templates"',
        f'TEMPLATES_DIR = CURRENT_DIR / "fixed_templates"',
        content
    )
    
    # Check if any templates exist in the fixed_templates directory
    if not os.path.exists(FIXED_TEMPLATES_DIR) or not os.listdir(FIXED_TEMPLATES_DIR):
        logger.error(f"No templates found in {FIXED_TEMPLATES_DIR}")
        return False
    
    # Write the updated script
    with open(FIX_HYPHENATED_MODELS_PATH, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated {FIX_HYPHENATED_MODELS_PATH} to use templates from {FIXED_TEMPLATES_DIR}")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update fix_hyphenated_models.py to use fixed templates")
    parser.add_argument("--restore", action="store_true", help="Restore original script from backup")
    
    args = parser.parse_args()
    
    if args.restore:
        # Restore from backup
        if os.path.exists(BACKUP_PATH):
            shutil.copy(BACKUP_PATH, FIX_HYPHENATED_MODELS_PATH)
            logger.info(f"Restored original script from {BACKUP_PATH}")
            return 0
        else:
            logger.error(f"Backup file not found: {BACKUP_PATH}")
            return 1
    
    # Update the script
    success = update_script()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
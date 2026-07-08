#!/usr/bin/env python3
"""
Fix syntax errors in template files for HuggingFace test generation.

This script directly edits the template files to correct syntax issues
that prevent the templates from being parsed correctly.
"""

import os
import sys
import re
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = CURRENT_DIR / "templates"

def fix_template_file(file_path):
    """Fix syntax errors in a template file."""
    logger.info(f"Processing template file: {file_path}")
    
    try:
        # Read the template file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create a backup
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Fix unterminated string literals in print statements with trailing parentheses
        content = re.sub(r'print\(f?"(.*?)"\)', r'print(f"\1")', content)
        content = re.sub(r'print\(f?\'(.*?)\'\)', r'print(f"\1")', content)
        
        # Fix multiline string issues
        content = re.sub(r'print\(\"\n', r'print(f"\n', content)
        content = re.sub(r'print\(\"\r\n', r'print(f"\r\n', content)
        
        # Fix triple quote issues
        content = re.sub(r'\"\"\"\"', '"""', content)
        
        # Fix escaped characters in string literals
        content = re.sub(r'\\([^tnrb\'\"\\])', r'\\\\\1', content)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify the syntax
        try:
            compile(content, file_path, 'exec')
            logger.info(f"✅ Fixed syntax in {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax error remains in {file_path}: {e}")
            error_line = content.split('\n')[e.lineno - 1] if e.lineno <= len(content.split('\n')) else "Line not available"
            logger.error(f"   Line {e.lineno}: {error_line}")
            
            # Restore from backup
            with open(backup_path, 'r') as f:
                original = f.read()
            with open(file_path, 'w') as f:
                f.write(original)
            logger.info(f"Restored {file_path} from backup")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def fix_all_templates():
    """Fix all template files in the templates directory."""
    logger.info(f"Fixing all templates in {TEMPLATES_DIR}")
    
    # Get all template files
    template_files = list(TEMPLATES_DIR.glob("*_template.py"))
    
    if not template_files:
        logger.error(f"No template files found in {TEMPLATES_DIR}")
        return False
    
    logger.info(f"Found {len(template_files)} template files")
    
    # Apply fixes to all templates
    success_count = 0
    failure_count = 0
    
    for template_file in template_files:
        if fix_template_file(template_file):
            success_count += 1
        else:
            failure_count += 1
    
    logger.info(f"Fixed {success_count} of {len(template_files)} template files")
    
    return failure_count == 0

def main():
    """Main entry point."""
    if not TEMPLATES_DIR.exists():
        logger.error(f"Templates directory not found: {TEMPLATES_DIR}")
        return 1
    
    if fix_all_templates():
        logger.info("Successfully fixed all template files")
        return 0
    else:
        logger.error("Failed to fix some template files")
        return 1

if __name__ == "__main__":
    sys.exit(main())
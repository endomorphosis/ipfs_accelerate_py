#!/usr/bin/env python3
"""
Direct fix for common indentation issues in templates and test files.

This script focuses on fixing the critical indentation issues at the top of template files,
including try/except blocks around imports which were found to be problematic.

Usage:
    python direct_fix.py <template_file> [--apply]
"""

import os
import sys
import re
import argparse
import logging
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_try_except_imports(content):
    """
    Fix try/except blocks around imports at the top of the file.
    
    Args:
        content: The file content as a string
        
    Returns:
        Fixed content with proper try/except indentation
    """
    # Let's directly replace the problematic hardware detection section
    import_section = """# Import hardware detection capabilities if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback"""
    
    # Find the hardware detection section
    if "# Import hardware detection capabilities if available" in content:
        # Use regex to replace the entire section
        pattern = re.compile(r'# Import hardware detection capabilities if available.*?# We\'ll detect hardware manually as fallback', re.DOTALL)
        fixed_content = pattern.sub(import_section, content)
        return fixed_content
    
    return content

def fix_template_file(file_path, apply=False):
    """
    Fix common indentation issues in a template file.
    
    Args:
        file_path: Path to the template file
        apply: Whether to apply changes (True) or just show them (False)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the template file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Fix try/except blocks around imports
        fixed_content = fix_try_except_imports(content)
        
        # Check if any changes were made
        if content == fixed_content:
            logger.info(f"No changes needed for {file_path}")
            return True
            
        # Create a backup if applying changes
        if apply:
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
            
            # Write the fixed content
            with open(file_path, 'w') as f:
                f.write(fixed_content)
                
            logger.info(f"Applied fixes to {file_path}")
            
            # Verify the syntax
            try:
                compile(fixed_content, file_path, 'exec')
                logger.info(f"✅ {file_path}: Syntax is valid")
            except SyntaxError as e:
                logger.error(f"❌ {file_path}: Syntax error after fixing: {e}")
                logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
                return False
        else:
            # Just show the differences
            logger.info(f"Recommended fixes for {file_path}:")
            
            # Show diff in a simple format
            old_lines = content.split('\n')
            new_lines = fixed_content.split('\n')
            
            for i, (old, new) in enumerate(zip(old_lines, new_lines)):
                if old != new:
                    logger.info(f"Line {i+1}:")
                    logger.info(f"  - {old}")
                    logger.info(f"  + {new}")
            
            # Show any added or removed lines
            if len(old_lines) < len(new_lines):
                for i in range(len(old_lines), len(new_lines)):
                    logger.info(f"Line {i+1} (added):")
                    logger.info(f"  + {new_lines[i]}")
            elif len(old_lines) > len(new_lines):
                for i in range(len(new_lines), len(old_lines)):
                    logger.info(f"Line {i+1} (removed):")
                    logger.info(f"  - {old_lines[i]}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error fixing template file {file_path}: {e}")
        return False
        
def main():
    parser = argparse.ArgumentParser(description="Fix common indentation issues in templates")
    parser.add_argument("file_path", help="Path to the template file to fix")
    parser.add_argument("--apply", action="store_true", help="Apply the fixes (default is just to show them)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        logger.error(f"File not found: {args.file_path}")
        return 1
        
    success = fix_template_file(args.file_path, args.apply)
    
    return 0 if success else 1
    
if __name__ == "__main__":
    sys.exit(main())
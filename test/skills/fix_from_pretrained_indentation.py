#!/usr/bin/env python3
"""
Fix indentation in the CUDA blocks of test_from_pretrained method in the test_generator_fixed.py file.
This addresses the primary issue with the test generator.

Usage:
    python fix_from_pretrained_indentation.py
"""

import os
import sys
import re
import logging
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to the test generator
TEST_GENERATOR_PATH = "/home/barberb/ipfs_accelerate_py/test/skills/test_generator_fixed.py"
BACKUP_PATH = f"{TEST_GENERATOR_PATH}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def fix_cuda_indentation(content):
    """
    Fix indentation issues in the CUDA testing block of the template.
    These issues affect the test_from_pretrained method.
    """
    # Pattern for CUDA block with indentation issues
    pattern = r'(if device == "cuda":\s+try:.*?with torch\.no_grad\(\):.*?_ = model\(\*\*inputs\).*?except Exception:.*?)(\s+pass\s+)(\s+# Run multiple)'
    
    def reindent_block(match):
        """Fix indentation for the matched CUDA block."""
        cuda_block = match.group(1)
        pass_stmt = match.group(2)
        next_section = match.group(3)
        
        # Calculate proper indentation level
        lines = cuda_block.split('\n')
        try_line_idx = -1
        for i, line in enumerate(lines):
            if "try:" in line:
                try_line_idx = i
                break
        
        if try_line_idx >= 0:
            try_line = lines[try_line_idx]
            indent_level = len(try_line) - len(try_line.lstrip())
            indentation = " " * indent_level
            
            # Fix indentation and rebuild block
            return f"{cuda_block}{pass_stmt}{indentation}{next_section.strip()}"
    
    try:
        # Apply the fix with regex substitution
        fixed_content = re.sub(pattern, reindent_block, content, flags=re.DOTALL)
        if fixed_content == content:
            logger.info("No CUDA block indentation issues found or pattern didn't match")
        else:
            logger.info("Successfully fixed CUDA block indentation")
        return fixed_content
    except Exception as e:
        logger.error(f"Error fixing CUDA indentation: {e}")
        return content

def fix_registry_duplication(content):
    """
    Fix registry name duplication for hyphenated models.
    This prevents issues like GPT_GPT_GPT_GPT_J_MODELS_REGISTRY.
    """
    # Function to replace duplicated registry patterns
    def replace_duplicate_registry(match):
        model_name = match.group(1)
        model_name_upper = model_name.upper()
        # Replace any duplicate segments like GPT_GPT_GPT with just GPT
        cleaned_name = re.sub(r'([A-Z]+)(?:_+\1)+', r'\1', model_name_upper)
        return f"{cleaned_name}_MODELS_REGISTRY"
    
    # Pattern to find duplicated registry names like GPT_GPT_GPT_GPT_J_MODELS_REGISTRY
    pattern = r'([A-Za-z0-9_]+)(?:_+[A-Za-z0-9_]+)*_MODELS_REGISTRY'
    
    try:
        # Apply the fix with regex substitution
        fixed_content = re.sub(pattern, replace_duplicate_registry, content)
        if fixed_content == content:
            logger.info("No registry duplication issues found or pattern didn't match")
        else:
            logger.info("Successfully fixed registry duplication")
        return fixed_content
    except Exception as e:
        logger.error(f"Error fixing registry duplication: {e}")
        return content

def enhance_valid_identifier(content):
    """
    Find and enhance the to_valid_identifier function to properly handle hyphenated model names.
    """
    # Look for the existing function
    pattern = r'def to_valid_identifier\([^)]*\):[^}]*?return.*?$'
    
    # New improved implementation that handles hyphenated models better
    replacement = """def to_valid_identifier(name):
    # Replace hyphens with underscores
    valid_name = name.replace("-", "_")
    
    # Ensure the name doesn't start with a number
    if valid_name and valid_name[0].isdigit():
        valid_name = f"m{valid_name}"
    
    # Replace any invalid characters with underscores
    valid_name = re.sub(r'[^a-zA-Z0-9_]', '_', valid_name)
    
    # Deduplicate consecutive underscores
    valid_name = re.sub(r'_+', '_', valid_name)
    
    return valid_name"""
    
    try:
        # Replace the function with improved version
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            fixed_content = content.replace(match.group(0), replacement)
            logger.info("Successfully enhanced to_valid_identifier function")
            return fixed_content
        else:
            logger.warning("Could not find to_valid_identifier function")
            return content
    except Exception as e:
        logger.error(f"Error enhancing to_valid_identifier function: {e}")
        return content

def main():
    """Apply the fixes to the test generator file."""
    try:
        # Read the original file
        with open(TEST_GENERATOR_PATH, 'r') as f:
            content = f.read()
        
        # Create a backup
        shutil.copy2(TEST_GENERATOR_PATH, BACKUP_PATH)
        logger.info(f"Created backup at {BACKUP_PATH}")
        
        # Apply the fixes
        fixed_content = content
        fixed_content = fix_cuda_indentation(fixed_content)
        fixed_content = fix_registry_duplication(fixed_content)
        fixed_content = enhance_valid_identifier(fixed_content)
        
        # Write back the fixed content
        with open(TEST_GENERATOR_PATH, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Successfully updated {TEST_GENERATOR_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error fixing file: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"✅ Successfully fixed test generator at {TEST_GENERATOR_PATH}")
        print(f"  Original file backed up to {BACKUP_PATH}")
    else:
        print(f"❌ Failed to fix test generator. See log for details.")
    
    sys.exit(0 if success else 1)
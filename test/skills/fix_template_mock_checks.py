#!/usr/bin/env python3
"""
Fix mock check blocks in templates.

This script directly adds the missing mock checks to import blocks in template files.

Usage:
    python fix_template_mock_checks.py [--template TEMPLATE_FILE]
"""

import os
import sys
import re
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def fix_tokenizers_import(content):
    """
    Fix tokenizers import block to include mock check.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with fixed import
    """
    # If the file already has a proper mock check, return unchanged
    if "if MOCK_TOKENIZERS:" in content:
        return content
    
    # Check if the file has a tokenizers import without mock check
    tokenizers_import_pattern = r"try:\s+import tokenizers\s+HAS_TOKENIZERS = True\s+except ImportError:"
    match = re.search(tokenizers_import_pattern, content, re.DOTALL)
    
    if match:
        old_import = match.group(0)
        new_import = """try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:"""
        
        # Replace the import
        content = content.replace(old_import, new_import)
        logger.info("Added mock check to tokenizers import")
    
    return content

def fix_sentencepiece_import(content):
    """
    Fix sentencepiece import block to include mock check.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with fixed import
    """
    # If the file already has a proper mock check, return unchanged
    if "if MOCK_SENTENCEPIECE:" in content:
        return content
    
    # Check if the file has a sentencepiece import without mock check
    sentencepiece_import_pattern = r"try:\s+import sentencepiece\s+HAS_SENTENCEPIECE = True\s+except ImportError:"
    match = re.search(sentencepiece_import_pattern, content, re.DOTALL)
    
    if match:
        old_import = match.group(0)
        new_import = """try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:"""
        
        # Replace the import
        content = content.replace(old_import, new_import)
        logger.info("Added mock check to sentencepiece import")
    
    return content

def fix_template(template_path):
    """
    Fix mock checks in a template file.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if the file exists
        if not os.path.exists(template_path):
            logger.error(f"File not found: {template_path}")
            return False
        
        # Read the file
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = f"{template_path}.mock_checks.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply fixes
        updated_content = fix_tokenizers_import(content)
        updated_content = fix_sentencepiece_import(updated_content)
        
        # If no changes made, return
        if content == updated_content:
            logger.info(f"No changes needed for {template_path}")
            return True
        
        # Write the updated content
        with open(template_path, 'w') as f:
            f.write(updated_content)
        logger.info(f"Updated {template_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error fixing template {template_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix mock checks in template files")
    parser.add_argument("--template", type=str, help="Specific template to fix")
    parser.add_argument("--templates-dir", type=str, default="templates", help="Directory containing templates")
    
    args = parser.parse_args()
    
    if args.template:
        # Fix a specific template
        if not os.path.exists(args.template):
            template_path = os.path.join(args.templates_dir, args.template)
            if not os.path.exists(template_path):
                logger.error(f"Template not found: {args.template}")
                return 1
        else:
            template_path = args.template
        
        success = fix_template(template_path)
        
        if success:
            print(f"{GREEN}✅ Successfully fixed {os.path.basename(template_path)}{RESET}")
        else:
            print(f"{RED}❌ Failed to fix {os.path.basename(template_path)}{RESET}")
        
        return 0 if success else 1
    else:
        # Fix all templates in the directory
        if not os.path.exists(args.templates_dir):
            logger.error(f"Templates directory not found: {args.templates_dir}")
            return 1
        
        template_files = []
        for file in os.listdir(args.templates_dir):
            if file.endswith("_template.py"):
                template_files.append(os.path.join(args.templates_dir, file))
        
        logger.info(f"Found {len(template_files)} template files to process")
        
        success_count = 0
        fail_count = 0
        
        for template_path in template_files:
            print(f"{BLUE}Processing {os.path.basename(template_path)}...{RESET}")
            if fix_template(template_path):
                success_count += 1
                print(f"{GREEN}✅ Successfully fixed {os.path.basename(template_path)}{RESET}")
            else:
                fail_count += 1
                print(f"{RED}❌ Failed to fix {os.path.basename(template_path)}{RESET}")
        
        print(f"\n{BLUE}=== SUMMARY ==={RESET}")
        print(f"Total templates: {len(template_files)}")
        print(f"{GREEN}✅ Successfully fixed: {success_count}{RESET}")
        print(f"{RED}❌ Failed to fix: {fail_count}{RESET}")
        
        return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
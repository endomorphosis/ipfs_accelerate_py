#!/usr/bin/env python3
"""
Fix a single test file with hyphenated model name and syntax issues.

This script takes a direct approach to fixing a specific file.
"""

import os
import sys
import re
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def replace_registry_keys(content, model_name):
    """Replace registry keys with proper format."""
    model_upper = model_name.replace('-', '_').upper()
    
    # Replace registry declaration
    pattern = r'([A-Z_\-]+)_MODELS_REGISTRY'
    match = re.search(pattern, content)
    if match:
        old_registry = match.group(0)
        new_registry = f"{model_upper}_MODELS_REGISTRY"
        content = content.replace(old_registry, new_registry)
        logger.info(f"Fixed registry key: {old_registry} -> {new_registry}")
    return content

def replace_file_name_references(content, model_name):
    """Replace references to file names."""
    model_valid = model_name.replace('-', '_').lower()
    
    # Replace file name references in save_results
    file_pattern = r'hf_([a-z0-9_\-]+)_'
    matches = re.finditer(file_pattern, content)
    for match in matches:
        old_ref = match.group(0)
        new_ref = f"hf_{model_valid}_"
        content = content.replace(old_ref, new_ref)
        logger.info(f"Fixed filename reference: {old_ref} -> {new_ref}")
    return content

def fix_triple_quotes(content):
    """Fix triple quotes and other string issues."""
    # Fix extra quotes ("""")
    content = content.replace('""""', '"""')
    
    # Check for unclosed triple quotes
    triple_quotes_count = content.count('"""')
    if triple_quotes_count % 2 != 0:
        logger.info(f"Odd number of triple quotes found: {triple_quotes_count}")
        # Try to find the problem location
        lines = content.split('\n')
        found_docstring = False
        line_num = 0
        for i, line in enumerate(lines):
            if '"""' in line:
                if found_docstring:
                    found_docstring = False
                else:
                    found_docstring = True
                    line_num = i
            
            # If we found an open docstring and there are extra quotes, fix them
            if found_docstring and '""""' in line:
                lines[i] = line.replace('""""', '"""')
                logger.info(f"Fixed extra quotes on line {i+1}")
        
        content = '\n'.join(lines)
    
    return content

def fix_file(file_path):
    """Fix all issues in a file and save it."""
    try:
        # Extract model type from filename
        filename = os.path.basename(file_path)
        model_type = filename.replace('test_hf_', '').replace('.py', '')
        logger.info(f"Processing file: {file_path} for model: {model_type}")
        
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = f"{file_path}.direct.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at: {backup_path}")
        
        # Apply fixes
        content = replace_registry_keys(content, model_type)
        content = replace_file_name_references(content, model_type)
        content = fix_triple_quotes(content)
        
        # Replace any remaining hyphens in model references
        model_valid = model_type.replace('-', '_')
        content = content.replace(f'"{model_type}"', f'"{model_valid}"')
        content = content.replace(f" {model_type} ", f" {model_valid} ")
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        logger.info(f"Written fixed content to {file_path}")
        
        # Verify syntax
        try:
            compile(content, file_path, 'exec')
            logger.info(f"✅ Syntax is valid after fixes")
            
            # Check if file name has hyphen and rename if needed
            if '-' in file_path:
                new_path = file_path.replace('-', '_')
                os.rename(file_path, new_path)
                logger.info(f"Renamed file to: {new_path}")
                return new_path, True
            
            return file_path, True
        
        except SyntaxError as e:
            logger.error(f"❌ Syntax error after fixes: {e}")
            # Restore from backup
            with open(backup_path, 'r') as f:
                content = f.read()
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Restored from backup due to syntax error")
            return file_path, False
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return file_path, False

def main():
    parser = argparse.ArgumentParser(description="Fix a single test file")
    parser.add_argument("--file", type=str, required=True, help="Path to file to fix")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return 1
    
    file_path, success = fix_file(args.file)
    
    if success:
        print(f"Successfully fixed file: {file_path}")
        return 0
    else:
        print(f"Failed to fix file: {args.file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
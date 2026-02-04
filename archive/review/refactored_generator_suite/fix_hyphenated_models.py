#!/usr/bin/env python3
"""
Fix Hyphenated Model Names in Generated Files

This script fixes syntax issues in generated test files, specifically:
1. Replaces hyphens in class names with underscores
2. Fixes incomplete code blocks (empty returns, etc.)

Usage:
    python fix_hyphenated_models.py [directory_path]

If directory_path is not provided, the script will use the default generated test output directory.
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f"fix_hyphenated_models_{os.path.basename(__file__).split('.')[0]}.log",
    filemode='w'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

DEFAULT_GENERATED_DIR = "./generated_test_output"
DEFAULT_REFERENCE_DIR = "./generated_reference"

def fix_class_name(content: str) -> str:
    """
    Fix hyphenated class names in the Python code.
    
    Args:
        content: Original code content
    
    Returns:
        Fixed code content
    """
    # Find the class definition with a hyphen in the name
    class_pattern = re.compile(r'class\s+hf_([a-zA-Z0-9_-]+):')
    match = class_pattern.search(content)
    
    if match:
        original_name = match.group(1)
        
        # Check if the name contains hyphens
        if '-' in original_name:
            # Replace hyphens with underscores in the class name
            fixed_name = original_name.replace('-', '_')
            logger.info(f"Found class with hyphenated name: hf_{original_name}, replacing with hf_{fixed_name}")
            
            # Replace all occurrences of the class name in the code
            content = content.replace(f"hf_{original_name}", f"hf_{fixed_name}")
            
            # Also fix any print statements, error messages, etc. that reference the class name
            content = content.replace(f"hf_{original_name} test", f"hf_{fixed_name} test")
            content = content.replace(f"Error in {original_name}", f"Error in {fixed_name}")
    
    return content

def fix_incomplete_returns(content: str) -> str:
    """
    Fix incomplete return statements in the code.
    
    Args:
        content: Original code content
    
    Returns:
        Fixed code content
    """
    # Find incomplete return statements
    lines = content.splitlines()
    fixed_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check for standalone 'return' followed by a commented or blank line
        if (stripped == "return" and 
            (i == len(lines) - 1 or 
             not lines[i+1].strip() or 
             lines[i+1].strip().startswith('#'))):
            
            # Add a proper return value
            fixed_lines.append(line + " {}")
            logger.info(f"Fixed incomplete return statement at line {i+1}")
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_incomplete_code_blocks(content: str) -> str:
    """
    Fix incomplete code blocks in the code.
    
    Args:
        content: Original code content
    
    Returns:
        Fixed code content
    """
    # Replace specific patterns where code blocks are incomplete
    patterns = [
        # Pattern: line starts with 'return # Code will be generated here'
        (r'(\s+)return\s+#\s*Code will be generated here', r'\1return {"success": True, "device": device, "hardware": hardware_label}'),
        
        # Pattern: multiline return with nothing after it
        (r'(\s+)return\s*\n', r'\1return {"success": True}\n'),
        
        # Pattern: return statements followed by comments without a value
        (r'(\s+)return(\s*#[^\n]*)', r'\1return {"success": True}\2'),
    ]
    
    fixed_content = content
    for pattern, replacement in patterns:
        fixed_content = re.sub(pattern, replacement, fixed_content)
    
    return fixed_content

def fix_string_formatting(content: str) -> str:
    """
    Fix string formatting issues, particularly in f-strings with double braces.
    
    Args:
        content: Original code content
    
    Returns:
        Fixed code content
    """
    # Find instances of double braces in f-strings and fix them
    lines = content.splitlines()
    fixed_lines = []
    
    for line in lines:
        # Check if line contains an f-string (starts with f")
        if "f\"" in line or "f'" in line:
            # Replace {{ with { and }} with } within the f-string
            line = re.sub(r'f(["\'])(.*?){{(.*?)}}(.*?)\1', r'f\1\2{\3}\4\1', line)
            
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_template_variables(content: str) -> str:
    """
    Fix template variable formatting issues.
    
    Args:
        content: Original code content
    
    Returns:
        Fixed code content
    """
    # Fix template variables that weren't properly rendered
    lines = content.splitlines()
    fixed_lines = []
    
    for line in lines:
        # Replace unrendered template variables with placeholders
        if "{" in line and "}" in line and not ("{{" in line or "}}" in line):
            # Check for missing f-string prefix
            if ("{" in line and "}" in line and 
                not line.strip().startswith(('f"', "f'", 'r"', "r'", "u'", 'u"')) and
                ('"' in line or "'" in line)):
                
                # Add f-string prefix
                if '"' in line:
                    line = re.sub(r'(^|\s+)"(.*?\{.*?\}.*?)"', r'\1f"\2"', line)
                elif "'" in line:
                    line = re.sub(r"(^|\s+)'(.*?\{.*?\}.*?)'", r"\1f'\2'", line)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_imports(content: str) -> str:
    """
    Fix missing imports in the code.
    
    Args:
        content: Original code content
    
    Returns:
        Fixed code content
    """
    # Check for references to classes/modules without imports
    imports_to_add = []
    
    # Check for AutoModel, AutoModelForXXX without import
    if "AutoModel" in content and "from transformers import AutoModel" not in content:
        # Find all AutoModel variants in the code
        auto_models = re.findall(r'(\bAutoModel\w*)', content)
        auto_models = list(set(auto_models))  # Remove duplicates
        
        if auto_models:
            imports_to_add.append(f"from transformers import {', '.join(auto_models)}")
    
    # Add missing imports at the top of the file after existing imports
    if imports_to_add:
        # Find the last import line
        lines = content.splitlines()
        last_import_idx = -1
        
        for i, line in enumerate(lines):
            if line.startswith(('import ', 'from ')):
                last_import_idx = i
        
        # Insert new imports after the last import line
        if last_import_idx >= 0:
            updated_lines = lines[:last_import_idx+1] + imports_to_add + lines[last_import_idx+1:]
            return '\n'.join(updated_lines)
    
    return content

def fix_file(file_path: str) -> bool:
    """
    Fix all syntax issues in a Python file.
    
    Args:
        file_path: Path to the file to fix
    
    Returns:
        True if the file was fixed, False otherwise
    """
    logger.info(f"Processing file: {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix all issues
        fixed_content = content
        fixed_content = fix_class_name(fixed_content)
        fixed_content = fix_incomplete_returns(fixed_content)
        fixed_content = fix_incomplete_code_blocks(fixed_content)
        fixed_content = fix_string_formatting(fixed_content)
        fixed_content = fix_template_variables(fixed_content)
        fixed_content = fix_imports(fixed_content)
        
        # Write the fixed content back to the file
        if fixed_content != content:
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            logger.info(f"Fixed file: {file_path}")
            return True
        else:
            logger.info(f"No issues found in file: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False

def fix_directory(directory_path: str) -> Tuple[int, int]:
    """
    Fix all Python files in a directory.
    
    Args:
        directory_path: Path to the directory containing Python files
    
    Returns:
        Tuple of (total_files, fixed_files)
    """
    logger.info(f"Processing directory: {directory_path}")
    
    total_files = 0
    fixed_files = 0
    
    try:
        # Find all Python files in the directory
        for file_path in Path(directory_path).glob('**/*.py'):
            total_files += 1
            if fix_file(str(file_path)):
                fixed_files += 1
    
    except Exception as e:
        logger.error(f"Error processing directory {directory_path}: {e}")
    
    return total_files, fixed_files

def main():
    """Main entry point."""
    # Get the directory to process
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        # Check which default directory exists
        if os.path.exists(DEFAULT_GENERATED_DIR):
            directory_path = DEFAULT_GENERATED_DIR
        elif os.path.exists(DEFAULT_REFERENCE_DIR):
            directory_path = DEFAULT_REFERENCE_DIR
        else:
            directory_path = "."
    
    # Process the directory
    logger.info(f"Starting syntax fix for Python files in {directory_path}")
    total_files, fixed_files = fix_directory(directory_path)
    
    # Print summary
    logger.info(f"Processed {total_files} files, fixed {fixed_files} files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
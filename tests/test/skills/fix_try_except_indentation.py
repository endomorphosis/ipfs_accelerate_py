#!/usr/bin/env python3
"""
Specialized fixer for try/except block indentation issues in generated test files.

This script focuses specifically on fixing indentation in try/except blocks:
1. Looks for a common pattern where import statements immediately follow try: with incorrect indentation
2. Fixes indentation after try: statements that have no proper indentation
3. Serves as a specialized complement to the complete_indentation_fix.py

Usage:
    python fix_try_except_indentation.py <file_path> [--verify]
"""

import sys
import re
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"try_except_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def fix_import_indentation_in_try_blocks(content):
    """Fix indentation for import statements inside try blocks."""
    # This pattern catches import statements that immediately follow try: with no indentation
    pattern = re.compile(r'try:\s*\n(from|import)([^\n]*)', re.MULTILINE)
    
    def fix_match(match):
        import_type = match.group(1)
        import_content = match.group(2)
        return f"try:\n    {import_type}{import_content}"
    
    # Apply the fix
    return pattern.sub(fix_match, content)

def fix_missing_indentation_after_try(content):
    """Fix cases where there is no indentation after try statements."""
    # Fix try blocks that have statements with no indentation
    lines = content.splitlines()
    fixed_lines = []
    in_try_block = False
    try_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # If this is a try statement, remember we're in a try block
        if stripped == "try:":
            fixed_lines.append(line)
            in_try_block = True
            try_indent = len(line) - len(line.lstrip())
            continue
            
        # If we're in a try block and the line isn't indented properly
        if in_try_block and stripped and not line.startswith(" " * (try_indent + 4)):
            # Exclude except/finally/else which end the try block
            if not any(stripped.startswith(end) for end in ["except", "finally", "else"]):
                # Add proper indentation
                fixed_lines.append(" " * (try_indent + 4) + stripped)
                continue
            else:
                # We've reached the end of the try block
                in_try_block = False
        
        # If we see except/finally/else, we're no longer in the try part
        if stripped.startswith(("except", "finally", "else:")):
            in_try_block = False
        
        # Add the line unchanged
        fixed_lines.append(line)
    
    return "\n".join(fixed_lines)

def fix_mock_class_indentation(content):
    """Fix indentation issues in mock class definitions."""
    # Fix issues with MockTokenizer class
    # First fix the class definition line with method definition
    pattern = re.compile(r'class\s+MockTokenizer:def\s+__init__')
    fixed_content = pattern.sub(r'class MockTokenizer:\n        def __init__', content)
    
    # Fix other method definitions in the class
    pattern = re.compile(r'def\s+encode\(self,([^\n]+)\)def\s+decode')
    fixed_content = pattern.sub(r'def encode(self,\1)\n        def decode', fixed_content)
    
    # Fix mock tokenizer static method
    pattern = re.compile(r'@staticmethod\s*\ndef\s+from_file')
    fixed_content = pattern.sub(r'@staticmethod\n        def from_file', fixed_content)
    
    # Fix any similar issues in other mock classes
    pattern = re.compile(r'def\s+get_piece_size\(([^\n]+)\)def\s+')
    fixed_content = pattern.sub(r'def get_piece_size(\1)\n        def ', fixed_content)
    
    return fixed_content

def fix_try_except_blocks(file_path, backup=True):
    """
    Fix indentation issues in try/except blocks.
    
    Args:
        file_path: Path to the file to fix
        backup: Whether to create a backup of the original file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if backup:
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Created backup at {backup_path}")
        
        # Apply fixes for try/except blocks
        fixed_content = fix_import_indentation_in_try_blocks(content)
        fixed_content = fix_missing_indentation_after_try(fixed_content)
        
        # Fix indentation in mock class definitions
        fixed_content = fix_mock_class_indentation(fixed_content)
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Fixed try/except indentation in {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing try/except indentation in {file_path}: {e}")
        return False

def verify_python_syntax(file_path):
    """
    Verify that the Python file has valid syntax.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, file_path, 'exec')
        logger.info(f"✅ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ {file_path}: Syntax error: {e}")
        logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
        return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix try/except indentation issues in HuggingFace test files")
    parser.add_argument("file_path", type=str, help="Path to the test file to fix")
    parser.add_argument("--verify", action="store_true", help="Verify Python syntax after fixing")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup of the original file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        logger.error(f"File not found: {args.file_path}")
        return 1
    
    # Fix indentation
    success = fix_try_except_blocks(args.file_path, backup=not args.no_backup)
    
    if success and args.verify:
        # Verify syntax
        syntax_valid = verify_python_syntax(args.file_path)
        if not syntax_valid:
            return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
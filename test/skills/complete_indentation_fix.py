#!/usr/bin/env python3
"""
Fix indentation issues in Python files.

This script detects and fixes indentation issues in Python files, especially those
that were generated or have mixed tabs and spaces.

Usage:
    python complete_indentation_fix.py FILE_PATH [--verify] [--backup]
"""

import sys
import os
import re
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"indentation_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def detect_indentation_style(content):
    """
    Detect the indentation style used in the file.
    
    Args:
        content: String content of the file
        
    Returns:
        Tuple of (char, size) where char is ' ' or '\t' and size is the indent size
    """
    # Count leading whitespace in lines
    space_counts = {}
    tab_counts = 0
    
    for line in content.splitlines():
        if line.strip() and line.startswith(' '):
            # Count spaces at the start
            spaces = len(line) - len(line.lstrip(' '))
            if spaces > 0:
                space_counts[spaces] = space_counts.get(spaces, 0) + 1
        elif line.strip() and line.startswith('\t'):
            # Count tabs
            tab_counts += 1
    
    # Determine dominant indent size for spaces
    dominant_space_size = 4  # Default to 4 spaces
    max_count = 0
    
    for size, count in space_counts.items():
        if count > max_count:
            max_count = count
            dominant_space_size = size
    
    # Determine if tabs or spaces are dominant
    if tab_counts > sum(space_counts.values()):
        return ('\t', 1)
    else:
        return (' ', dominant_space_size)

def fix_indentation(content, indent_char=' ', indent_size=4):
    """
    Fix indentation issues in the content.
    
    Args:
        content: String content of the file
        indent_char: Character to use for indentation (' ' or '\t')
        indent_size: Size of each indent level
        
    Returns:
        Fixed content
    """
    lines = content.splitlines()
    fixed_lines = []
    
    # Define regex pattern to find indentation
    indent_pattern = re.compile(r'^(\s*)')
    
    # Process each line
    for line in lines:
        if not line.strip():
            # Empty line, just add it
            fixed_lines.append(line)
            continue
        
        # Find the current indentation
        match = indent_pattern.match(line)
        if match:
            indent = match.group(1)
            stripped_line = line.lstrip()
            
            # Calculate the indent level
            if '\t' in indent:
                # Convert tabs to spaces for calculation
                spaces_equivalent = indent.replace('\t', ' ' * 4)
                indent_level = len(spaces_equivalent) // indent_size
            else:
                indent_level = len(indent) // indent_size if indent_size > 0 else 0
            
            # Create new indentation
            new_indent = indent_char * indent_level * indent_size if indent_char == ' ' else indent_char * indent_level
            
            # Reconstruct the line
            fixed_lines.append(new_indent + stripped_line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_try_except_indentation(content):
    """
    Fix common indentation issues with try/except blocks.
    
    Args:
        content: String content of the file
        
    Returns:
        Fixed content
    """
    # Find try/except blocks with indentation issues
    try_pattern = re.compile(r'(\s*)try:(?:\s*\n)(\s*)')
    except_pattern = re.compile(r'(\s*)except(\s+\w+)?:(?:\s*\n)(\s*)')
    
    # Fix try blocks
    matches = list(try_pattern.finditer(content))
    for match in reversed(matches):  # Process in reverse to maintain indices
        indent = match.group(1)
        inner_indent = match.group(2)
        
        # If inner block is not indented more than try line
        if len(inner_indent) <= len(indent):
            new_inner_indent = indent + '    '
            content = content[:match.start(2)] + new_inner_indent + content[match.end(2):]
    
    # Fix except blocks
    matches = list(except_pattern.finditer(content))
    for match in reversed(matches):  # Process in reverse to maintain indices
        indent = match.group(1)
        exception = match.group(2) or ''
        inner_indent = match.group(3)
        
        # If inner block is not indented more than except line
        if len(inner_indent) <= len(indent):
            new_inner_indent = indent + '    '
            content = content[:match.start(3)] + new_inner_indent + content[match.end(3):]
    
    return content

def fix_file(file_path, verify=False, create_backup=True):
    """
    Fix indentation in a file.
    
    Args:
        file_path: Path to the file to fix
        verify: If True, verify the fixed file with Python's compile
        create_backup: If True, create a backup of the original file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup if requested
        if create_backup:
            backup_path = f"{file_path}.indentation.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Created backup at {backup_path}")
        
        # Detect indentation style
        indent_char, indent_size = detect_indentation_style(content)
        logger.info(f"Detected indentation style: {repr(indent_char)} with size {indent_size}")
        
        # Fix indentation
        fixed_content = fix_indentation(content, indent_char, indent_size)
        
        # Fix try/except blocks
        fixed_content = fix_try_except_indentation(fixed_content)
        
        # Verify if requested
        if verify:
            try:
                compile(fixed_content, file_path, 'exec')
                logger.info(f"Verification successful: No syntax errors in the fixed content")
            except SyntaxError as e:
                logger.error(f"Verification failed: Syntax error in the fixed content: {e}")
                return False
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Successfully fixed indentation in {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing indentation in {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix indentation issues in Python files")
    parser.add_argument("file_path", help="Path to the file to fix")
    parser.add_argument("--verify", action="store_true", help="Verify the fixed file with Python's compile")
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original file")
    
    args = parser.parse_args()
    
    success = fix_file(args.file_path, verify=args.verify, create_backup=not args.no_backup)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
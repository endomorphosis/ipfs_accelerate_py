#!/usr/bin/env python3
"""
Quick Fix for Indentation in Generated Files

This script fixes the indentation in the generated test files.
It's a temporary solution until the improved renderer is fully working.
"""

import os
import sys
import re
import glob
from pathlib import Path

def fix_file_indentation(file_path):
    """Fix indentation in a generated file."""
    print(f"Fixing indentation in {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix indentation
    fixed_content = fix_indentation(content)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed indentation in {file_path}")
    
    return True

def fix_indentation(content):
    """Fix indentation in a string."""
    lines = content.split('\n')
    fixed_lines = []
    stack = []  # Stack to track nested blocks
    
    # First, reset all indentation
    clean_lines = []
    for line in lines:
        clean_lines.append(line.lstrip())
    
    # Now, reindent properly
    current_indent = 0
    for i, line in enumerate(clean_lines):
        # Skip empty lines
        if not line:
            fixed_lines.append('')
            continue
        
        # Handling outdent for these tokens at the beginning of the line
        if line.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ', 'elif', ')', '}', ']')):
            if current_indent > 0:
                current_indent -= 1
        
        # Add the current line with proper indentation
        fixed_lines.append(' ' * (4 * current_indent) + line)
        
        # Check for indentation triggers
        # Functions, classes, if/else blocks, etc. that end with a colon
        if line.endswith(':') and not line.startswith(('import ', 'from ')):
            current_indent += 1
            stack.append(':')
        # Opening brackets
        elif line.endswith('{') or line.endswith('[') or line.endswith('('):
            current_indent += 1
            stack.append(line[-1])
    
    # Join the lines back together
    return '\n'.join(fixed_lines)

def process_directory(directory):
    """Process all Python files in a directory."""
    # Find all Python files
    files = list(Path(directory).glob('**/*.py'))
    
    # Fix indentation in each file
    for file_path in files:
        fix_file_indentation(str(file_path))
    
    print(f"Processed {len(files)} files in {directory}")

def main():
    """Main function."""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python quick_fix_indentation.py <directory>")
        return 1
    
    # Get the directory from arguments
    directory = sys.argv[1]
    
    # Validate the directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return 1
    
    # Process the directory
    process_directory(directory)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
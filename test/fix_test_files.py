#!/usr/bin/env python3
"""
Fix Hugging Face Test Files

This script fixes the indentation and structure of Hugging Face test files
to ensure consistency and proper hardware integration.

It focuses on:
1. Fixing indentation issues
2. Placing methods inside the correct class
3. Organizing methods in a consistent order
4. Ensuring proper implementation of hardware methods

Usage:
python fix_test_files.py --file test_hf_bert.py
python fix_test_files.py --directory key_models_hardware_fixes
"""

import os
import re
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Current directory
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def fix_file_structure(file_path: str) -> bool:
    """
    Fix the structure of a test file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        True if changes were made, False otherwise
    """
    print(f"Processing {file_path}...")
    
    # Read the file
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False
    
    # Find the class definition
    class_pattern = r'^class\s+(\w+):'
    class_match = re.search(class_pattern, content, re.MULTILINE)
    if not class_match:
        print(f"No class definition found in {file_path}")
        return False
    
    class_name = class_match.group(1)
    class_start = class_match.start()
    
    # Find methods that should be inside the class
    method_patterns = [
        r'^def\s+init_(\w+)\(',
        r'^def\s+test_with_(\w+)\(',
        r'^def\s+create_(\w+)_handler\('
    ]
    
    methods_to_move = []
    
    for pattern in method_patterns:
        for match in re.finditer(pattern, content, re.MULTILINE):
            method_start = match.start()
            
            # Find the end of the method (next def or class or EOF)
            next_def = re.search(r'^(def|class)\s', content[method_start+1:], re.MULTILINE)
            if next_def:
                method_end = method_start + 1 + next_def.start()
            else:
                method_end = len(content)
            
            # Extract the method code
            method_code = content[method_start:method_end]
            
            # Check if this method is already inside the class
            if method_start > class_start:
                # Check if the indentation is correct (4 spaces)
                lines = method_code.split('\n')
                if not lines[0].startswith('    '):
                    # Method is not correctly indented
                    methods_to_move.append({
                        'start': method_start,
                        'end': method_end,
                        'code': method_code,
                        'indented_code': '    ' + method_code.replace('\n', '\n    ')
                    })
            else:
                # Method is outside the class and needs to be moved inside
                methods_to_move.append({
                    'start': method_start,
                    'end': method_end,
                    'code': method_code,
                    'indented_code': '    ' + method_code.replace('\n', '\n    ')
                })
    
    # If no methods need to be moved, we're done
    if not methods_to_move:
        print(f"No structure issues found in {file_path}")
        return False
    
    # Sort methods by their position in the file (reverse order to avoid changing offsets)
    methods_to_move.sort(key=lambda m: m['start'], reverse=True)
    
    # Remove the methods from their original positions
    new_content = content
    for method in methods_to_move:
        new_content = new_content[:method['start']] + new_content[method['end']:]
    
    # Find a good insertion point within the class
    # Look for the end of __init__ method
    init_pattern = r'def\s+__init__\s*\(.*?\):.*?(?=\n\s*def|\n\s*$|\Z)'
    init_match = re.search(init_pattern, new_content[class_start:], re.DOTALL)
    
    if init_match:
        # Insert after __init__
        insertion_point = class_start + init_match.end()
    else:
        # Insert right after class definition if no __init__
        # Find the end of the class line
        class_line_end = new_content.find('\n', class_start) + 1
        insertion_point = class_line_end
    
    # Insert the methods in a logical order
    # 1. init_* methods
    # 2. create_*_handler methods
    # 3. test_with_* methods
    
    # Sort methods by type and name
    methods_by_type = {
        'init': [],
        'create': [],
        'test': []
    }
    
    for method in methods_to_move:
        if 'def init_' in method['code']:
            methods_by_type['init'].append(method)
        elif 'def create_' in method['code']:
            methods_by_type['create'].append(method)
        elif 'def test_with_' in method['code']:
            methods_by_type['test'].append(method)
    
    # Sort each type by name
    for type_key in methods_by_type:
        methods_by_type[type_key].sort(key=lambda m: m['code'])
    
    # Insert methods in order
    methods_text = "\n\n"
    for type_key in ['init', 'create', 'test']:
        if methods_by_type[type_key]:
            methods_text += "\n".join(m['indented_code'] for m in methods_by_type[type_key])
            methods_text += "\n\n"
    
    # Insert the methods
    fixed_content = new_content[:insertion_point] + methods_text + new_content[insertion_point:]
    
    # Save the fixed file
    try:
        # Create a backup of the original file
        backup_path = file_path + '.bak'
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        print(f"Fixed {len(methods_to_move)} methods in {file_path}")
        print(f"Backup saved to {backup_path}")
        return True
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        return False

def process_directory(directory: str) -> int:
    """
    Process all test files in a directory.
    
    Args:
        directory: Directory to process
        
    Returns:
        Number of files fixed
    """
    # Find all test files
    pattern = os.path.join(directory, "test_hf_*.py")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No test files found in {directory}")
        return 0
    
    print(f"Found {len(files)} test files in {directory}")
    fixed_count = 0
    
    for file_path in files:
        if fix_file_structure(file_path):
            fixed_count += 1
    
    return fixed_count

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix Hugging Face test files structure")
    
    # Source options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a specific test file to fix")
    group.add_argument("--directory", type=str, help="Path to a directory containing test files to fix")
    
    args = parser.parse_args()
    
    if args.file:
        # Fix a single file
        if fix_file_structure(args.file):
            print(f"Successfully fixed file: {args.file}")
            return 0
        else:
            print(f"No changes made to file: {args.file}")
            return 1
    
    if args.directory:
        # Process a directory
        fixed_count = process_directory(args.directory)
        print(f"Fixed {fixed_count} files in directory: {args.directory}")
        return 0 if fixed_count > 0 else 1
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
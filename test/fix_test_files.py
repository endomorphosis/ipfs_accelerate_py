#!/usr/bin/env python3
"""
Script to fix common issues in generated HuggingFace test files.
"""

import os
import sys
import glob
import re

def fix_unterminated_string(file_path):
    """Fix unterminated string literals in the file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix unterminated string in print statement
    if 'print("\nWould you like to update' in content:
        print(f"Fixing unterminated string in {file_path}")
        content = content.replace('print("\nWould you like to update', 'print("Would you like to update')
    
    # Fix missing class_name variable
    if '    print(f"Warning: {class_name} module not found' in content:
        # Extract the class name from the import statement or class definition
        module_name = os.path.basename(file_path).replace("test_", "").replace(".py", "")
        print(f"Fixing missing class_name in {file_path} (using {module_name})")
        content = content.replace('    print(f"Warning: {class_name} module not found',
                               f'    print(f"Warning: {module_name} module not found')
    
    # Fix tensor contains string issue
    if '"implementation_type": "REAL" if "implementation_type" not in output else output["implementation_type"]' in content:
        print(f"Fixing tensor contains string issue in {file_path}")
        content = content.replace(
            '"implementation_type": "REAL" if "implementation_type" not in output else output["implementation_type"]',
            '"implementation_type": "REAL" if not isinstance(output, dict) or "implementation_type" not in output else output["implementation_type"]'
        )
        content = content.replace(
            '"implementation_type": "MOCK" if "implementation_type" not in output else output["implementation_type"]',
            '"implementation_type": "MOCK" if not isinstance(output, dict) or "implementation_type" not in output else output["implementation_type"]'
        )
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True

def main():
    # Get all test files
    test_files = glob.glob('skills/test_hf_*.py')
    print(f"Found {len(test_files)} test files")
    
    fixed_count = 0
    
    for file_path in test_files:
        try:
            if fix_unterminated_string(file_path):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
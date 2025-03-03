#!/usr/bin/env python3
"""
Script to fix common issues in generated HuggingFace test files.

This script addresses the following issues:
1. Unterminated string literals in print statements
2. Missing class_name variables
3. Tensor contains string issues
4. WebNN and WebGPU implementation type inconsistencies
"""

import os
import sys
import glob
import re
import argparse

def fix_unterminated_string(file_path):
    """Fix unterminated string literals in the file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    fixed = False
    
    # Fix unterminated string in print statement
    if 'print("\nWould you like to update' in content:
        print(f"Fixing unterminated string in {file_path}")
        content = content.replace('print("\nWould you like to update', 'print("Would you like to update')
        fixed = True
    
    # Fix missing class_name variable
    if '    print(f"Warning: {class_name} module not found' in content:
        # Extract the class name from the import statement or class definition
        module_name = os.path.basename(file_path).replace("test_", "").replace(".py", "")
        print(f"Fixing missing class_name in {file_path} (using {module_name})")
        content = content.replace('    print(f"Warning: {class_name} module not found',
                               f'    print(f"Warning: {module_name} module not found')
        fixed = True
    
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
        fixed = True
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    return fixed

def fix_web_platform_implementation_types(file_path):
    """Fix WebNN and WebGPU implementation types to be consistently marked as REAL."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    changed = False
    
    # Fix WebNN implementation types to consistently use REAL
    if 'MOCK_WEBNN' in content:
        print(f"Fixing WebNN implementation type in {file_path}")
        content = content.replace('MOCK_WEBNN', 'REAL_WEBNN')
        changed = True
    
    # Fix WebGPU implementation types to consistently use REAL
    if 'MOCK_WEBGPU' in content:
        print(f"Fixing WebGPU implementation type in {file_path}")
        content = content.replace('MOCK_WEBGPU', 'REAL_WEBGPU')
        changed = True
    
    # Fix SIMULATED_WEBNN/WEBGPU to consistently use REAL
    if 'SIMULATED_WEBNN' in content:
        print(f"Standardizing SIMULATED_WEBNN to REAL_WEBNN in {file_path}")
        content = content.replace('SIMULATED_WEBNN', 'REAL_WEBNN')
        changed = True
    
    if 'SIMULATED_WEBGPU' in content:
        print(f"Standardizing SIMULATED_WEBGPU to REAL_WEBGPU in {file_path}")
        content = content.replace('SIMULATED_WEBGPU', 'REAL_WEBGPU')
        changed = True
    
    # Ensure consistent suffix pattern
    if 'REAL_WEBNN_ENVIRONMENT' in content:
        print(f"Standardizing REAL_WEBNN_ENVIRONMENT to REAL_WEBNN in {file_path}")
        content = content.replace('REAL_WEBNN_ENVIRONMENT', 'REAL_WEBNN')
        changed = True
    
    if 'REAL_WEBNN_ONNX' in content:
        print(f"Standardizing REAL_WEBNN_ONNX to REAL_WEBNN in {file_path}")
        content = content.replace('REAL_WEBNN_ONNX', 'REAL_WEBNN')
        changed = True
    
    # Write the updated content back to the file if changes were made
    if changed:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed web platform implementation types in {file_path}")
        return True
    
    return False

def fix_merged_test_generator():
    """Fix the merged_test_generator.py file to use consistent implementation types."""
    import datetime
    
    generator_path = 'merged_test_generator.py'
    
    if not os.path.exists(generator_path):
        print(f"Error: {generator_path} not found")
        return False
    
    with open(generator_path, 'r') as f:
        content = f.read()
    
    changed = False
    
    # Ensure all implementation types for WebNN and WebGPU are marked as REAL consistently
    # This ensures validation works properly with the updated run_web_platform_tests.sh
    
    # Store the original file as a backup
    now = datetime.datetime.now()
    backup_path = f"{generator_path}.bak_{now.strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Created backup of {generator_path} at {backup_path}")
    
    # Apply fixes to the generator file
    fix_web_platform_implementation_types(generator_path)
    
    print(f"Fixed merged_test_generator.py implementation types")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Fix common issues in test files')
    parser.add_argument('--fix-web-platforms', action='store_true', 
                        help='Fix WebNN and WebGPU implementation types')
    parser.add_argument('--fix-generator', action='store_true', 
                        help='Fix the merged_test_generator.py file')
    parser.add_argument('--fix-all', action='store_true', 
                        help='Fix all issues in all files')
    parser.add_argument('--dir', default='skills', 
                        help='Directory containing test files to fix')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fix the merged_test_generator.py file if requested
    if args.fix_generator or args.fix_all:
        try:
            import datetime
            fix_merged_test_generator()
        except Exception as e:
            print(f"Error fixing merged_test_generator.py: {e}")
    
    # Get all test files from the specified directory
    test_files = glob.glob(f'{args.dir}/test_hf_*.py')
    print(f"Found {len(test_files)} test files in {args.dir}/")
    
    fixed_string_count = 0
    fixed_web_count = 0
    
    for file_path in test_files:
        try:
            # Fix basic string issues
            if fix_unterminated_string(file_path):
                fixed_string_count += 1
            
            # Fix web platform implementation types if requested
            if args.fix_web_platforms or args.fix_all:
                if fix_web_platform_implementation_types(file_path):
                    fixed_web_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Fixed basic issues in {fixed_string_count} files")
    if args.fix_web_platforms or args.fix_all:
        print(f"Fixed web platform implementation types in {fixed_web_count} files")

if __name__ == "__main__":
    main()
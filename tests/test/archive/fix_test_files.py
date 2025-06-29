#!/usr/bin/env python3
"""
Fix Hugging Face Test Files

This script fixes the indentation and structure of Hugging Face test files
to ensure consistency and proper hardware integration.

It focuses on:
1. Fixing indentation issues for test generator files and test implementations
2. Placing methods inside the correct class
3. Organizing methods in a consistent order
4. Ensuring proper implementation of hardware methods
5. Fixing string termination, table examples, and model modality detection functions
6. Ensuring KEY_MODEL_HARDWARE_MAP is defined properly

Usage:
python fix_test_files.py --file test_hf_bert.py
python fix_test_files.py --directory key_models_hardware_fixes
python fix_test_files.py --generators  # Fix test generator files
"""

import os
import re
import sys
import glob
import argparse
import datetime
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

# Generator file fixing functions
def backup_generator_file(file_path):
    """Create a backup of the file before modifying it."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".py.bak_{timestamp}")
    
    with open(file_path, 'r', encoding='utf-8') as source_file:
        with open(backup_path, 'w', encoding='utf-8') as backup_file:
            backup_file.write(source_file.read())
    
    print(f"Created backup of {file_path} at {backup_path}")
    return backup_path

def fix_table_example_indentation(content):
    """Fix the table example indentation issue."""
    # Pattern to find the table example with improper indentation
    pattern = r"(if\s+primary_task\s+==\s+\"table-question-answering\"[^\n]*\n)(\s*)table_example\s+=\s+'''self\.test_table"
    replacement = r"\1\2    table_example = '''self.test_table"
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    return fixed_content

def fix_function_indentation(content):
    """Fix indentation issues in functions."""
    # Pattern for functions that should be indented
    pattern = r"(\s*)(def\s+[a-zA-Z0-9_]+\([^)]*\):)(\s*)([\r\n]+)([^\s])"
    replacement = r"\1\2\3\4\1    \5"
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    return fixed_content

def fix_class_indentation(content):
    """Fix indentation issues in class methods."""
    # Pattern for class methods that should be indented
    pattern = r"(\s*)(class\s+[a-zA-Z0-9_]+(?:\([^)]*\))?:)(\s*)([\r\n]+)([^\s])"
    replacement = r"\1\2\3\4\1    \5"
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    return fixed_content

def fix_string_termination(content):
    """Fix any unterminated string literals."""
    # Pattern for triple-quoted strings that might be unterminated
    # This is a simplified approach; for robust parsing you'd need a proper parser
    pattern = r"('''[^']*?)(?:(?!''')[^'])*$"
    replacement = r"\1'''"
    
    # Apply the fix (note: this might not catch all cases)
    fixed_content = re.sub(pattern, replacement, content)
    
    return fixed_content

def fix_hardware_map_placement(content):
    """Fix the KEY_MODEL_HARDWARE_MAP definition placement."""
    # If KEY_MODEL_HARDWARE_MAP is referenced before defined, this fixes that
    if "KEY_MODEL_HARDWARE_MAP" in content:
        hardware_map_def = re.search(r"KEY_MODEL_HARDWARE_MAP\s*=\s*\{[^}]*\}", content, re.DOTALL)
        if hardware_map_def:
            # Extract the full definition
            map_definition = hardware_map_def.group(0)
            
            # Remove it from its current location
            content = content.replace(map_definition, "")
            
            # Find a good place to put it (after imports but before first function)
            first_function = re.search(r"def\s+[a-zA-Z0-9_]+\(", content)
            if first_function:
                insertion_point = first_function.start()
                # Add the definition before the first function
                content = content[:insertion_point] + "\n# Define hardware map\n" + map_definition + "\n\n" + content[insertion_point:]
    
    return content

def fix_detect_modality_function(content):
    """Fix specific issues with the detect_model_modality function."""
    # Pattern for incomplete detect_model_modality function
    pattern = r"(def\s+detect_model_modality\([^)]*\):.*?\n)(\s*# Normalize the model type.*?\n)(\s*model_name = model_type\.lower\(\)\.replace\('-', '_'\)\.replace\('\.', '_'\).*?\n)(\s*# Check direct matches in defined modality types.*?\n)(\s*for modality, types in MODALITY_TYPES\.items\(\):.*?\n)(\s*if model_name in types or any\(t in model_name for t in types\):)"
    
    replacement = r"\1\2\3\4\5\6\n        return modality\n        \n    # Check specialized models\n    for model_key, task in SPECIALIZED_MODELS.items():\n        if model_name == model_key.lower().replace('-', '_').replace('.', '_'):\n            # Determine modality based on task\n            if 'audio' in task or 'speech' in task:\n                return 'audio'\n            elif 'image' in task or 'vision' in task or 'depth' in task:\n                return 'vision'\n            elif 'text-to-text' in task or 'text-generation' in task:\n                return 'text'\n            elif 'multimodal' in task or ('image' in task and 'text' in task):\n                return 'multimodal'\n    \n    # Default to text if no match found\n    return 'text'"
    
    # Apply the fix (this is a partial fix, a complete solution would need proper parsing)
    fixed_content = re.sub(pattern, replacement, content)
    
    return fixed_content

def fix_duplicate_hardware_detection(content):
    """Remove duplicate hardware detection blocks."""
    # Find the first complete hardware detection block
    first_block = re.search(r"# Hardware Detection.*?HW_CAPABILITIES = check_hardware\(\)", content, re.DOTALL)
    if first_block:
        first_block_text = first_block.group(0)
        
        # Find and remove subsequent hardware detection blocks
        duplicate_blocks = re.finditer(r"# Hardware Detection.*?HW_CAPABILITIES = check_hardware\(\)", content[first_block.end():], re.DOTALL)
        for duplicate in duplicate_blocks:
            duplicate_text = duplicate.group(0)
            content = content.replace(duplicate_text, "")
    
    return content

def fix_duplicate_optimizations(content):
    """Remove duplicate web platform optimization definitions."""
    # Find the first complete optimization block
    first_block = re.search(r"# Web Platform Optimizations.*?def apply_web_platform_optimizations\(.*?def detect_browser_for_optimizations\(.*?return browser_info", content, re.DOTALL)
    if first_block:
        first_block_text = first_block.group(0)
        
        # Find and remove subsequent optimization blocks
        duplicate_blocks = re.finditer(r"# Web Platform Optimizations.*?def apply_web_platform_optimizations\(.*?def detect_browser_for_optimizations\(.*?return browser_info", content[first_block.end():], re.DOTALL)
        for duplicate in duplicate_blocks:
            duplicate_text = duplicate.group(0)
            content = content.replace(duplicate_text, "")
    
    return content

def process_generator_file(file_path):
    """Process a single generator file to fix the identified issues."""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Create backup
    backup_path = backup_generator_file(file_path)
    
    try:
        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes
        fixed_content = content
        fixed_content = fix_duplicate_hardware_detection(fixed_content)
        fixed_content = fix_duplicate_optimizations(fixed_content)
        fixed_content = fix_table_example_indentation(fixed_content)
        fixed_content = fix_function_indentation(fixed_content)
        fixed_content = fix_class_indentation(fixed_content)
        fixed_content = fix_string_termination(fixed_content)
        fixed_content = fix_hardware_map_placement(fixed_content)
        fixed_content = fix_detect_modality_function(fixed_content)
        
        # Write fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Successfully fixed issues in {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Restore from backup
        with open(backup_path, 'r', encoding='utf-8') as backup_file:
            with open(file_path, 'w', encoding='utf-8') as source_file:
                source_file.write(backup_file.read())
        
        print(f"Restored {file_path} from backup")
        return False

def process_generator_files():
    """Process all test generator files."""
    generator_files = [
        CURRENT_DIR / "fixed_merged_test_generator.py",
        CURRENT_DIR / "merged_test_generator.py",
        CURRENT_DIR / "integrated_skillset_generator.py"
    ]
    
    success_count = 0
    
    for file_path in generator_files:
        if process_generator_file(file_path):
            success_count += 1
    
    print(f"\nSummary: Successfully fixed {success_count} of {len(generator_files)} generator files")
    
    return success_count == len(generator_files)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix Hugging Face test files structure")
    
    # Source options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a specific test file to fix")
    group.add_argument("--directory", type=str, help="Path to a directory containing test files to fix")
    group.add_argument("--generators", action="store_true", help="Fix test generator files")
    
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
    
    if args.generators:
        # Fix test generator files
        if process_generator_files():
            print("Successfully fixed all generator files")
            return 0
        else:
            print("Some generator files could not be fixed")
            return 1
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
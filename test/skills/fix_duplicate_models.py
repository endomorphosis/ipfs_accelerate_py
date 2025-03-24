#!/usr/bin/env python3

"""
Fix duplicate models in the test_generator_fixed.py file.

This script:
1. Creates a backup of the original file
2. Identifies duplicate model entries in MODEL_REGISTRY
3. Keeps only one instance of each model
4. Removes syntax errors
5. Writes the fixed content back to the file
"""

import os
import sys
import re
import time
from pathlib import Path
import shutil

def fix_test_generator(file_path):
    """Fix duplicate model entries in the test_generator_fixed.py file."""
    print(f"Fixing {file_path}...")
    
    # Create backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract the MODEL_REGISTRY section
    start_index = -1
    for i, line in enumerate(lines):
        if "MODEL_REGISTRY = {" in line:
            start_index = i
            break
    
    if start_index == -1:
        print("Error: MODEL_REGISTRY not found")
        return False
    
    # Look for duplicate entry start points (starts of model entries)
    model_entries = {}
    duplicate_ranges = []
    current_index = start_index + 1
    
    while current_index < len(lines):
        line = lines[current_index]
        
        # Check for model entry start
        match = re.match(r'\s*"([\w-]+)":\s*{', line)
        if match:
            model_name = match.group(1)
            
            if model_name in model_entries:
                # Found a duplicate - mark the range
                duplicate_start = current_index
                duplicate_range = [duplicate_start]
                
                # Find where this duplicate entry ends
                brace_count = 1  # Starting with the opening brace
                dup_index = duplicate_start + 1
                
                while dup_index < len(lines) and brace_count > 0:
                    dup_line = lines[dup_index]
                    brace_count += dup_line.count('{') - dup_line.count('}')
                    dup_index += 1
                
                duplicate_range.append(dup_index - 1)
                duplicate_ranges.append(duplicate_range)
                print(f"Found duplicate model '{model_name}' at lines {duplicate_start+1}-{dup_index}")
                
                # Skip to the end of this duplicate
                current_index = dup_index
            else:
                # First time seeing this model, add to dict
                model_entries[model_name] = current_index
                current_index += 1
        else:
            # Check if we've reached the end of the registry
            if line.strip() == "}":
                break
            current_index += 1
    
    if not duplicate_ranges:
        print("No duplicates found. Check if there's another issue.")
        return False
    
    # Create new lines with duplicates removed
    filtered_lines = lines.copy()
    
    # Remove the duplicates (in reverse order to maintain line numbers)
    for start, end in sorted(duplicate_ranges, reverse=True):
        del filtered_lines[start:end+1]
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"✅ Removed {len(duplicate_ranges)} duplicate model entries")
    
    # Verify file is valid Python
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Use compile to check syntax
        compile(content, file_path, 'exec')
        print("✅ Fixed file is syntactically valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error still exists at line {e.lineno}: {e.msg}")
        print(f"Near: {e.text}")
        return False

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "test_generator_fixed.py"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return 1
    
    success = fix_test_generator(file_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
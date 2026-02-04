#!/usr/bin/env python3

"""
Fix indentation issues in the test_generator_fixed.py file.

This script:
1. Identifies and corrects the MODEL_REGISTRY dictionary duplications
2. Validates the syntax of the fixed file
3. Saves a backup of the original file
4. Writes the fixed content back to the original file
"""

import os
import sys
import ast
import re
import time
from pathlib import Path
import shutil

def fix_model_registry_duplication(file_path):
    """Fix the duplication in the MODEL_REGISTRY dictionary."""
    print(f"Analyzing {file_path}...")
    
    # Create backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find MODEL_REGISTRY start and end
    registry_start = content.find("MODEL_REGISTRY = {")
    if registry_start == -1:
        print("Error: Could not find MODEL_REGISTRY in file")
        return False
    
    # Find the correct end of the dictionary by matching braces
    registry_end = -1
    brace_count = 0
    in_string = False
    string_char = None
    
    # Scan for the end of the registry
    i = registry_start + len("MODEL_REGISTRY = {")
    while i < len(content):
        char = content[i]
        
        # Handle strings
        if char in ['"', "'"]:
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char and content[i-1] != '\\':
                in_string = False
        
        # Only count braces when not in a string
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                if brace_count == 0:
                    registry_end = i
                    break
                brace_count -= 1
        
        i += 1
    
    if registry_end == -1:
        print("Error: Could not find the end of MODEL_REGISTRY")
        return False
    
    # Extract the registry section
    registry_text = content[registry_start:registry_end+1]
    
    # Identify duplicated models
    model_entries = {}
    duplicates = []
    
    # Check for duplicates by looking for model names
    model_pattern = re.compile(r'["\']([\w-]+)["\']:\s*{')
    for match in model_pattern.finditer(registry_text):
        model_name = match.group(1)
        if model_name in model_entries:
            duplicates.append(model_name)
        else:
            model_entries[model_name] = match.start()
    
    print(f"Found {len(model_entries)} model entries, {len(duplicates)} duplicates")
    
    if duplicates:
        print(f"Duplicate models found: {', '.join(duplicates)}")
    
    # Fix the registry by removing duplicates
    # First let's try parsing it as a valid dictionary by adding a missing closing brace
    fixed_registry = registry_text
    
    # If we have duplicates, we need to restructure the dictionary
    if duplicates:
        # Get the registry section with the malformed ending to create a fixed version
        # Find where the correct entries end (before duplicates start)
        for duplicate in duplicates:
            duplicate_pos = registry_text.find(f'"{duplicate}":')
            if duplicate_pos > 0:
                # Look for the previous valid closing brace
                prev_brace = registry_text.rfind('},', 0, duplicate_pos)
                if prev_brace > 0:
                    # Check if this brace is followed by another model entry
                    next_entry = registry_text[prev_brace+2:].lstrip()
                    if not next_entry.startswith('"'):
                        # This is where the valid part ends and malformed part begins
                        fixed_registry = registry_text[:prev_brace+1] + "}"
                        break
    
    # Try to parse the fixed registry to see if it's valid
    try:
        # Add dummy variable name for parsing
        dummy_code = "dummy = " + fixed_registry
        ast.parse(dummy_code)
        print("✅ Fixed registry is syntactically valid")
    except SyntaxError as e:
        print(f"❌ Syntax error in fixed registry: {e}")
        return False
    
    # Replace the original registry with the fixed version
    fixed_content = content[:registry_start] + fixed_registry + content[registry_end+1:]
    
    # Verify the entire file
    try:
        ast.parse(fixed_content)
        print("✅ Fixed file is syntactically valid")
    except SyntaxError as e:
        print(f"❌ Syntax error in fixed file at line {e.lineno}: {e.msg}")
        print(f"Near: {e.text}")
        return False
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {file_path}")
    return True

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "test_generator_fixed.py"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return 1
    
    success = fix_model_registry_duplication(file_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
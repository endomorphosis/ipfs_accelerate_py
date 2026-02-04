#!/usr/bin/env python3
"""
Phase 11: Comprehensive fix for ALL remaining 223 relative import issues
"""

import os
import re
import ast
from pathlib import Path

test_base = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test")

def get_absolute_import_path(file_path, relative_import_level, module_name):
    """Convert relative import to absolute import"""
    file_path = Path(file_path)
    parts = list(file_path.relative_to(test_base).parts[:-1])  # Remove filename
    
    # Go up 'level' directories
    for _ in range(relative_import_level):
        if parts:
            parts.pop()
    
    # Construct absolute path
    absolute_parts = ["test"] + parts
    if module_name and module_name != '.':
        absolute_parts.append(module_name)
    
    return ".".join(absolute_parts)

def fix_imports_in_file(filepath):
    """Fix all relative imports in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            # Match: from .module import X
            # Match: from ..module import X
            # Match: from ...module import X
            match = re.match(r'^(\s*)from\s+(\.+)([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)?(?:\s+import\s+(.+))$', line)
            
            if match:
                indent, dots, module, imports = match.groups()
                level = len(dots)
                module = module or ''
                
                # Calculate absolute import
                try:
                    abs_path = get_absolute_import_path(filepath, level - 1, module)
                    new_line = f"{indent}from {abs_path} import {imports}"
                    new_lines.append(new_line)
                    continue
                except Exception as e:
                    # If we can't calculate, keep original
                    pass
            
            new_lines.append(line)
        
        new_content = '\n'.join(new_lines)
        
        if new_content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

# Process all Python files in test directory
fixed_count = 0
total_count = 0

for root, dirs, files in os.walk(test_base):
    for file in files:
        if file.endswith('.py'):
            filepath = Path(root) / file
            total_count += 1
            if fix_imports_in_file(filepath):
                fixed_count += 1
                print(f"Fixed: {filepath.relative_to(test_base)}")

print(f"\n{'='*80}")
print(f"Processed {total_count} files, fixed {fixed_count} files")
print(f"{'='*80}")

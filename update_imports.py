#!/usr/bin/env python3
"""
Update imports after test directory refactoring.
Fixes relative and absolute imports in moved files.
"""

import os
import re
from pathlib import Path
import ast

def find_python_files(directory):
    """Find all Python files in directory and subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files

def get_imports(file_path):
    """Extract all import statements from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        imports = []
        # Find relative imports
        for match in re.finditer(r'^from\s+\.\s+import\s+(.+)$', content, re.MULTILINE):
            imports.append(('relative_from_current', match.group(0), match.group(1)))
        
        # Find parent relative imports
        for match in re.finditer(r'^from\s+\.\.(\.\w+)*\s+import\s+(.+)$', content, re.MULTILINE):
            imports.append(('relative_from_parent', match.group(0), match.group(2)))
        
        # Find absolute imports from test
        for match in re.finditer(r'^from\s+test\.(\w+)\s+import\s+(.+)$', content, re.MULTILINE):
            imports.append(('absolute_test', match.group(0), match.group(1), match.group(2)))
        
        # Find direct test imports
        for match in re.finditer(r'^import\s+test\.(\w+)$', content, re.MULTILINE):
            imports.append(('import_test', match.group(0), match.group(1)))
        
        return imports
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def update_imports(file_path, old_location, new_location):
    """Update imports in a file based on its move."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Calculate relative path depth
        old_depth = len(Path(old_location).relative_to('test').parents) - 1
        new_depth = len(Path(new_location).relative_to('test').parents) - 1
        
        # Update relative imports if depth changed
        if old_depth != new_depth:
            depth_change = new_depth - old_depth
            if depth_change > 0:
                # Need more ".." in relative imports
                content = re.sub(
                    r'from\s+\.',
                    f'from {"." * (depth_change + 1)}',
                    content
                )
            elif depth_change < 0:
                # Need fewer ".." in relative imports
                content = re.sub(
                    r'from\s+\.{' + str(abs(depth_change) + 1) + r'}',
                    'from .',
                    content
                )
        
        # Update absolute imports from test
        # Example: from test.utils import X -> from test.scripts.utilities.utils import X
        # This is complex and depends on where files moved to
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def create_import_mapping():
    """Create a mapping of old imports to new imports."""
    # This would map test.filename -> test.category.filename
    mapping = {}
    
    # Read the refactoring plan
    plan_file = Path('/tmp/refactoring_plan.txt')
    if plan_file.exists():
        with open(plan_file, 'r') as f:
            for line in f:
                if '->' in line:
                    parts = line.strip().split(' -> ')
                    if len(parts) == 2:
                        old_path = parts[0].replace('/', '.').replace('.py', '')
                        new_path = parts[1].replace('/', '.').replace('.py', '')
                        mapping[old_path] = new_path
    
    return mapping

def update_all_imports(test_dir):
    """Update imports in all Python files."""
    print("=" * 80)
    print("UPDATING IMPORTS")
    print("=" * 80)
    
    # Get mapping of old to new paths
    mapping = create_import_mapping()
    print(f"\nLoaded {len(mapping)} file mappings")
    
    # Find all Python files
    python_files = find_python_files(test_dir)
    print(f"Found {len(python_files)} Python files to check\n")
    
    updated_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Update each mapped import
            for old_import, new_import in mapping.items():
                # Handle various import forms
                old_module = old_import.split('.')[-1]
                new_module_path = new_import
                
                # from test.filename import X -> from test.category.filename import X
                pattern1 = f'from {old_import} import'
                replacement1 = f'from {new_import} import'
                content = content.replace(pattern1, replacement1)
                
                # import test.filename -> import test.category.filename
                pattern2 = f'import {old_import}'
                replacement2 = f'import {new_import}'
                content = content.replace(pattern2, replacement2)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_files.append(file_path)
                print(f"✓ Updated: {file_path}")
        
        except Exception as e:
            print(f"✗ Error with {file_path}: {e}")
    
    print("\n" + "=" * 80)
    print("IMPORT UPDATE COMPLETE")
    print("=" * 80)
    print(f"Updated {len(updated_files)} files")
    
    return updated_files

def main():
    """Main import update logic."""
    test_dir = Path('test')
    
    if not test_dir.exists():
        print(f"Error: {test_dir} does not exist")
        return
    
    updated_files = update_all_imports(test_dir)
    
    print("\nNext steps:")
    print("1. Review updated files")
    print("2. Run tests to verify imports work")
    print("3. Fix any remaining import issues manually")

if __name__ == '__main__':
    main()

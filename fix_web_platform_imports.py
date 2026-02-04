#!/usr/bin/env python3
"""
Fix test.web_platform.* imports to test.tests.web.web_platform.*
"""
import os
import re
import sys

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: from test.web_platform.X import Y
        content = re.sub(
            r'from test\.web_platform\.([a-zA-Z0-9_]+) import',
            r'from test.tests.web.web_platform.\1 import',
            content
        )
        
        # Pattern 2: from test.web_platform import X
        content = re.sub(
            r'from test\.web_platform import',
            r'from test.tests.web.web_platform import',
            content
        )
        
        # Pattern 3: import test.web_platform.X
        content = re.sub(
            r'import test\.web_platform\.([a-zA-Z0-9_]+)',
            r'import test.tests.web.web_platform.\1',
            content
        )
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    base_path = '/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py'
    test_dir = os.path.join(base_path, 'test')
    
    print("=" * 80)
    print("Fixing test.web_platform.* imports")
    print("=" * 80)
    
    fixed_count = 0
    total_files = 0
    
    for root, dirs, files in os.walk(test_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                total_files += 1
                if fix_imports_in_file(filepath):
                    fixed_count += 1
                    rel_path = os.path.relpath(filepath, base_path)
                    print(f"Fixed: {rel_path}")
    
    print(f"\n{'=' * 80}")
    print(f"Summary:")
    print(f"  Total Python files: {total_files}")
    print(f"  Files modified: {fixed_count}")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

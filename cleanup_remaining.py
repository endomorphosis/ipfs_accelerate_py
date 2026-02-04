#!/usr/bin/env python3
"""Cleanup remaining test/ directories."""

import subprocess
import shutil
from pathlib import Path

def safe_remove(path):
    """Remove directory from git and filesystem."""
    try:
        subprocess.run(['git', 'rm', '-rf', str(path)], 
                      capture_output=True, check=False)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        return True
    except Exception as e:
        print(f"Error removing {path}: {e}")
        return False

def main():
    test_dir = Path('test')
    
    # Remaining directories to handle
    to_delete = [
        'output',  # Empty output directory
        'temp_docs',  # Temporary docs
        'template_integration',  # Empty
        'template_system',  # Empty
        'web_platform_test_output',  # Output directory
    ]
    
    print("Cleaning up remaining empty/temporary directories...")
    for dirname in to_delete:
        path = test_dir / dirname
        if path.exists():
            if safe_remove(path):
                print(f"  [DEL] {path}")
            else:
                print(f"  [ERR] Failed to remove {path}")
    
    # test/common should stay but not nest
    # It's already in the right place
    print("\nKeeping test/common/ as shared utilities")
    
    print("\nRemaining directories in test/:")
    remaining = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    for d in remaining:
        py_count = len(list(d.rglob('*.py')))
        print(f"  {d.name:30s} ({py_count} .py files)")
    
    print(f"\nTotal: {len(remaining)} directories remain in test/")

if __name__ == '__main__':
    main()

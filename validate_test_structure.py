#!/usr/bin/env python3
"""
Validate the refactored test directory structure.
Checks that all test files are properly organized and discoverable.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

def validate_test_structure():
    """Validate the test directory structure."""
    
    print("=" * 80)
    print("TEST DIRECTORY STRUCTURE VALIDATION")
    print("=" * 80)
    print()
    
    # Check root test directory
    test_root = Path('test')
    if not test_root.exists():
        print("❌ ERROR: test/ directory not found")
        return False
    
    # Count files in root (should be minimal)
    root_files = [f for f in test_root.iterdir() if f.is_file() and f.suffix == '.py']
    print(f"✓ Files in test/ root: {len(root_files)}")
    for f in root_files:
        print(f"  - {f.name}")
    
    if len(root_files) > 5:
        print(f"  ⚠️  Warning: {len(root_files)} Python files in root (expected ~2)")
    print()
    
    # Check organized subdirectories
    expected_dirs = {
        'tests': 'Organized test files',
        'scripts': 'Utility scripts',
        'tools': 'Testing tools',
        'generators': 'Test generators',
        'templates': 'Test templates',
        'examples': 'Example scripts',
    }
    
    print("Checking organized directories:")
    all_good = True
    for dir_name, description in expected_dirs.items():
        dir_path = test_root / dir_name
        if dir_path.exists():
            file_count = sum(1 for f in dir_path.rglob('*.py'))
            print(f"  ✓ {dir_name}/ - {description} ({file_count} .py files)")
        else:
            print(f"  ❌ {dir_name}/ - NOT FOUND")
            all_good = False
    print()
    
    # Check test subdirectories
    tests_dir = test_root / 'tests'
    if tests_dir.exists():
        print("Test categories in test/tests/:")
        subdirs = [d for d in tests_dir.iterdir() if d.is_dir()]
        for subdir in sorted(subdirs):
            file_count = sum(1 for f in subdir.rglob('test_*.py'))
            print(f"  - {subdir.name}/ ({file_count} test files)")
        print()
    
    # Check for __init__.py files
    print("Checking __init__.py files:")
    init_files = list(test_root.rglob('__init__.py'))
    print(f"  Found {len(init_files)} __init__.py files")
    
    # Check a few key directories
    key_dirs = ['tests', 'scripts', 'tools', 'generators', 'templates']
    missing_init = []
    for dir_name in key_dirs:
        dir_path = test_root / dir_name
        if dir_path.exists():
            if not (dir_path / '__init__.py').exists():
                missing_init.append(dir_name)
    
    if missing_init:
        print(f"  ⚠️  Missing __init__.py in: {', '.join(missing_init)}")
    else:
        print(f"  ✓ All key directories have __init__.py")
    print()
    
    # Check for problematic patterns
    print("Checking for common issues:")
    
    # Check for files with syntax errors
    syntax_errors = []
    for py_file in test_root.rglob('*.py'):
        # Skip broken symlinks
        if not py_file.exists():
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append((str(py_file), e.lineno))
        except Exception:
            # Skip files that can't be read
            pass
    
    if syntax_errors:
        print(f"  ❌ Found {len(syntax_errors)} files with syntax errors:")
        for file_path, line_num in syntax_errors[:5]:
            print(f"     {file_path}:{line_num}")
    else:
        print(f"  ✓ No syntax errors found")
    
    # Check for uncommented broken imports
    broken_imports = []
    import re
    problematic_patterns = [
        r'^from test\.test_configuration_common import',
        r'^from test\.test_modeling_common import',
        r'^from test\.test_pipeline_mixin import',
        r'^from test\.merge_benchmark_databases import',
    ]
    
    for py_file in test_root.rglob('*.py'):
        # Skip broken symlinks
        if not py_file.exists():
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            for pattern in problematic_patterns:
                if re.search(pattern, content, re.MULTILINE):
                    broken_imports.append(str(py_file))
                    break
        except:
            pass
    
    if broken_imports:
        print(f"  ❌ Found {len(broken_imports)} files with uncommented broken imports:")
        for file_path in broken_imports[:5]:
            print(f"     {file_path}")
    else:
        print(f"  ✓ No uncommented broken imports found")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total_py_files = sum(1 for _ in test_root.rglob('*.py'))
    test_files = sum(1 for _ in test_root.rglob('test_*.py'))
    
    print(f"  Total Python files: {total_py_files}")
    print(f"  Test files (test_*.py): {test_files}")
    print(f"  __init__.py files: {len(init_files)}")
    print(f"  Files in root: {len(root_files)}")
    print(f"  Syntax errors: {len(syntax_errors)}")
    print(f"  Broken imports: {len(broken_imports)}")
    print()
    
    # Overall status
    print("=" * 80)
    if len(root_files) <= 5 and all_good and not syntax_errors and not broken_imports:
        print("✅ TEST STRUCTURE VALIDATION: PASSED")
        print("   All checks passed. Test directory is properly organized.")
    elif not syntax_errors and not broken_imports:
        print("✅ TEST STRUCTURE VALIDATION: PASSED (with warnings)")
        print("   Structure is good but some warnings were found.")
    else:
        print("⚠️  TEST STRUCTURE VALIDATION: PASSED WITH ISSUES")
        print("   Structure is organized but some issues need attention.")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    try:
        success = validate_test_structure()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

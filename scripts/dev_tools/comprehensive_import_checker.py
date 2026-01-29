#!/usr/bin/env python3
"""
Python Import Checker for ipfs_accelerate_py

Validates import statements in Python files.
"""

import os
import sys
import argparse
import ast
from pathlib import Path
from typing import List, Set
import importlib.util


def check_imports(file_path: Path) -> tuple[bool, List[str]]:
    """Check if all imports in a file are valid."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(file_path))
        
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check if module exists
                    spec = importlib.util.find_spec(alias.name.split('.')[0])
                    if spec is None:
                        errors.append(f"Cannot import '{alias.name}'")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    spec = importlib.util.find_spec(module_name)
                    if spec is None:
                        errors.append(f"Cannot import from '{node.module}'")
        
        return len(errors) == 0, errors
        
    except SyntaxError as e:
        return False, [f"SyntaxError: {e.msg}"]
    except Exception as e:
        return False, [f"Error: {str(e)}"]


def find_python_files(directory: Path, exclude_patterns: List[str]) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []
    
    for py_file in directory.glob('**/*.py'):
        if py_file.is_file():
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(py_file):
                    should_exclude = True
                    break
            
            if not should_exclude:
                python_files.append(py_file)
    
    return python_files


def main():
    parser = argparse.ArgumentParser(description="Check Python imports")
    parser.add_argument('--directory', required=True, help='Directory to scan')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--exclude', nargs='*',
                       default=['__pycache__', '.venv', 'venv', '.git', 'build', 'dist'],
                       help='Patterns to exclude')
    
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    print(f"Checking imports in: {directory}")
    print(f"Excluding: {', '.join(args.exclude)}")
    print()
    
    python_files = find_python_files(directory, args.exclude)
    
    if not python_files:
        print("No Python files found")
        return 0
    
    print(f"Found {len(python_files)} Python files to check\n")
    
    successful = 0
    failed = 0
    
    for py_file in sorted(python_files):
        success, errors = check_imports(py_file)
        
        if success:
            successful += 1
            if args.verbose:
                print(f"✓ {py_file.relative_to(directory)}")
        else:
            failed += 1
            print(f"✗ {py_file.relative_to(directory)}")
            for error in errors:
                print(f"  {error}")
    
    print("\n" + "="*60)
    print("IMPORT CHECK SUMMARY")
    print("="*60)
    print(f"Total files checked: {len(python_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if len(python_files) > 0:
        print(f"Success rate: {successful/len(python_files)*100:.1f}%")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Python File Compilation Checker for ipfs_accelerate_py

Checks Python files for syntax errors and compilation issues.
"""

import os
import sys
import argparse
import py_compile
import ast
from pathlib import Path
from typing import List, Tuple
import time


def compile_file(file_path: Path) -> Tuple[bool, str]:
    """Attempt to compile a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse with AST to catch syntax errors
        ast.parse(source_code, filename=str(file_path))
        
        # Try to compile
        py_compile.compile(file_path, doraise=True)
        
        return True, "OK"
        
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def find_python_files(directory: Path, exclude_patterns: List[str]) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []
    
    for py_file in directory.glob('**/*.py'):
        if py_file.is_file():
            # Check if file should be excluded
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(py_file):
                    should_exclude = True
                    break
            
            if not should_exclude:
                python_files.append(py_file)
    
    return python_files


def main():
    parser = argparse.ArgumentParser(description="Check Python file compilation")
    parser.add_argument('path', help='Directory to scan for Python files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--exclude', nargs='*', 
                       default=['__pycache__', '.venv', 'venv', '.git', 'build', 'dist'],
                       help='Patterns to exclude')
    
    args = parser.parse_args()
    
    directory = Path(args.path).resolve()
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    print(f"Checking Python files in: {directory}")
    print(f"Excluding: {', '.join(args.exclude)}")
    print()
    
    # Find files
    python_files = find_python_files(directory, args.exclude)
    
    if not python_files:
        print("No Python files found")
        return 0
    
    print(f"Found {len(python_files)} Python files to check\n")
    
    # Check each file
    successful = 0
    failed = 0
    failed_files = []
    
    for py_file in sorted(python_files):
        success, message = compile_file(py_file)
        
        if success:
            successful += 1
            if args.verbose:
                print(f"✓ {py_file.relative_to(directory)}")
        else:
            failed += 1
            failed_files.append((py_file, message))
            print(f"✗ {py_file.relative_to(directory)}")
            print(f"  {message}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPILATION CHECK SUMMARY")
    print("="*60)
    print(f"Total files checked: {len(python_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if len(python_files) > 0:
        print(f"Success rate: {successful/len(python_files)*100:.1f}%")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Python Code Quality Checker for ipfs_accelerate_py

Checks Python code quality using multiple metrics.
"""

import os
import sys
import argparse
import ast
from pathlib import Path
from typing import List, Dict
import subprocess


def check_file_quality(file_path: Path) -> Dict[str, any]:
    """Check quality of a single Python file."""
    results = {
        'file': str(file_path),
        'size': file_path.stat().st_size,
        'lines': 0,
        'functions': 0,
        'classes': 0,
        'complexity_issues': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            results['lines'] = len(content.splitlines())
        
        tree = ast.parse(content, filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                results['functions'] += 1
                # Check function length
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        results['complexity_issues'].append(
                            f"Function '{node.name}' is {func_lines} lines (consider refactoring if >50)"
                        )
            
            elif isinstance(node, ast.ClassDef):
                results['classes'] += 1
        
        # Check file length
        if results['lines'] > 500:
            results['complexity_issues'].append(
                f"File has {results['lines']} lines (consider splitting if >500)"
            )
        
        results['success'] = True
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results


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
    parser = argparse.ArgumentParser(description="Check Python code quality")
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
    
    print(f"Checking code quality in: {directory}")
    print()
    
    python_files = find_python_files(directory, args.exclude)
    
    if not python_files:
        print("No Python files found")
        return 0
    
    print(f"Analyzing {len(python_files)} Python files\n")
    
    total_lines = 0
    total_functions = 0
    total_classes = 0
    files_with_issues = 0
    
    for py_file in sorted(python_files):
        result = check_file_quality(py_file)
        
        if result['success']:
            total_lines += result['lines']
            total_functions += result['functions']
            total_classes += result['classes']
            
            if result['complexity_issues']:
                files_with_issues += 1
                print(f"⚠ {py_file.relative_to(directory)}")
                for issue in result['complexity_issues']:
                    print(f"  {issue}")
            elif args.verbose:
                print(f"✓ {py_file.relative_to(directory)}")
        else:
            print(f"✗ {py_file.relative_to(directory)}: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    print("CODE QUALITY SUMMARY")
    print("="*60)
    print(f"Total files analyzed: {len(python_files)}")
    print(f"Total lines of code: {total_lines}")
    print(f"Total functions: {total_functions}")
    print(f"Total classes: {total_classes}")
    print(f"Files with complexity issues: {files_with_issues}")
    
    if len(python_files) > 0:
        print(f"Average lines per file: {total_lines/len(python_files):.1f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

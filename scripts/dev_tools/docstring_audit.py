#!/usr/bin/env python3
"""
Docstring Auditor for ipfs_accelerate_py

Audits Python files for documentation completeness.
"""

import os
import sys
import argparse
import ast
import json
from pathlib import Path
from typing import List, Dict, Any


def audit_file_docstrings(file_path: Path) -> Dict[str, Any]:
    """Audit docstrings in a single Python file."""
    results = {
        'file': str(file_path),
        'module_docstring': False,
        'classes': 0,
        'classes_with_docstrings': 0,
        'functions': 0,
        'functions_with_docstrings': 0,
        'missing_docstrings': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        # Check module docstring
        results['module_docstring'] = ast.get_docstring(tree) is not None
        if not results['module_docstring']:
            results['missing_docstrings'].append('module')
        
        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                results['classes'] += 1
                if ast.get_docstring(node):
                    results['classes_with_docstrings'] += 1
                else:
                    results['missing_docstrings'].append(f"class {node.name}")
            
            elif isinstance(node, ast.FunctionDef):
                # Skip private functions (starting with _)
                if not node.name.startswith('_'):
                    results['functions'] += 1
                    if ast.get_docstring(node):
                        results['functions_with_docstrings'] += 1
                    else:
                        results['missing_docstrings'].append(f"function {node.name}")
        
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
    parser = argparse.ArgumentParser(description="Audit Python docstrings")
    parser.add_argument('--directory', required=True, help='Directory to scan')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--exclude', nargs='*',
                       default=['__pycache__', '.venv', 'venv', '.git', 'build', 'dist', 'test'],
                       help='Patterns to exclude')
    
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    print(f"Auditing docstrings in: {directory}")
    print()
    
    python_files = find_python_files(directory, args.exclude)
    
    if not python_files:
        print("No Python files found")
        return 0
    
    print(f"Analyzing {len(python_files)} Python files\n")
    
    all_results = []
    total_classes = 0
    total_classes_documented = 0
    total_functions = 0
    total_functions_documented = 0
    files_without_module_docstring = 0
    
    for py_file in sorted(python_files):
        result = audit_file_docstrings(py_file)
        all_results.append(result)
        
        if result['success']:
            total_classes += result['classes']
            total_classes_documented += result['classes_with_docstrings']
            total_functions += result['functions']
            total_functions_documented += result['functions_with_docstrings']
            
            if not result['module_docstring']:
                files_without_module_docstring += 1
            
            if result['missing_docstrings']:
                print(f"⚠ {py_file.relative_to(directory)}")
                print(f"  Missing docstrings: {len(result['missing_docstrings'])}")
                if args.verbose:
                    for missing in result['missing_docstrings'][:5]:  # Show first 5
                        print(f"    - {missing}")
            elif args.verbose:
                print(f"✓ {py_file.relative_to(directory)}")
    
    print("\n" + "="*60)
    print("DOCSTRING AUDIT SUMMARY")
    print("="*60)
    print(f"Total files analyzed: {len(python_files)}")
    print(f"Files with module docstrings: {len(python_files) - files_without_module_docstring}/{len(python_files)}")
    print(f"Classes with docstrings: {total_classes_documented}/{total_classes}")
    print(f"Functions with docstrings: {total_functions_documented}/{total_functions}")
    
    if total_classes > 0:
        print(f"Class documentation rate: {total_classes_documented/total_classes*100:.1f}%")
    if total_functions > 0:
        print(f"Function documentation rate: {total_functions_documented/total_functions*100:.1f}%")
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'summary': {
                    'total_files': len(python_files),
                    'files_with_module_docstrings': len(python_files) - files_without_module_docstring,
                    'total_classes': total_classes,
                    'classes_documented': total_classes_documented,
                    'total_functions': total_functions,
                    'functions_documented': total_functions_documented
                },
                'files': all_results
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

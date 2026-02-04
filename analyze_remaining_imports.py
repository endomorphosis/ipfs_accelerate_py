#!/usr/bin/env python3
"""
Analyze remaining relative import issues in detail.
Categorize and prepare for Phase 10 fixes.
"""

import os
import ast
import re
from collections import defaultdict
from pathlib import Path

def analyze_remaining_imports():
    """Analyze the remaining 478 import issues in detail."""
    
    test_dir = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test")
    
    # Categories for analysis
    internal_refs = []  # from .module within same package
    deep_nested = []     # from ...  (triple dot or more)
    conditional = []     # imports in try/except
    other = []          # other patterns
    
    # Patterns to look for
    relative_patterns = {
        'single_dot': re.compile(r'from\s+\.(\w+)'),
        'double_dot': re.compile(r'from\s+\.\.(\w+)'),
        'triple_dot': re.compile(r'from\s+\.\.\.(\w+)'),
        'deeper': re.compile(r'from\s+\.{4,}'),
    }
    
    file_count = 0
    error_count = 0
    
    for py_file in test_dir.rglob("*.py"):
        if py_file.name == "__pycache__":
            continue
            
        file_count += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Try to parse with AST
            try:
                tree = ast.parse(content, filename=str(py_file))
                
                # Analyze imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.level > 0:
                            # Relative import
                            rel_path = str(py_file.relative_to(test_dir))
                            import_info = {
                                'file': rel_path,
                                'level': node.level,
                                'module': node.module,
                                'line': node.lineno,
                                'names': [alias.name for alias in node.names]
                            }
                            
                            if node.level >= 3:
                                deep_nested.append(import_info)
                            elif node.level == 2:
                                # Check if it's internal or needs fixing
                                if 'skillset' in rel_path or 'plugins' in rel_path:
                                    internal_refs.append(import_info)
                                else:
                                    other.append(import_info)
                            else:  # level == 1
                                internal_refs.append(import_info)
                                
            except SyntaxError as e:
                error_count += 1
                # Can't parse - skip
                continue
                
        except Exception as e:
            error_count += 1
            continue
    
    # Print results
    print("="*80)
    print("REMAINING IMPORT ANALYSIS")
    print("="*80)
    print(f"\nTotal Python files scanned: {file_count}")
    print(f"Files with parse errors: {error_count}")
    print()
    
    print(f"Internal references (level 1): {len(internal_refs)}")
    print(f"Deep nested (level 3+): {len(deep_nested)}")
    print(f"Other patterns: {len(other)}")
    print(f"TOTAL: {len(internal_refs) + len(deep_nested) + len(other)}")
    print()
    
    # Show samples of each category
    if internal_refs:
        print("\n" + "="*80)
        print("INTERNAL REFERENCES (first 10):")
        print("="*80)
        for item in internal_refs[:10]:
            print(f"  {item['file']}:{item['line']}")
            print(f"    from {'.' * item['level']}{item['module']} import {', '.join(item['names'])}")
    
    if deep_nested:
        print("\n" + "="*80)
        print("DEEP NESTED IMPORTS (first 10):")
        print("="*80)
        for item in deep_nested[:10]:
            print(f"  {item['file']}:{item['line']}")
            print(f"    from {'.' * item['level']}{item['module']} import {', '.join(item['names'])}")
    
    if other:
        print("\n" + "="*80)
        print("OTHER PATTERNS (first 10):")
        print("="*80)
        for item in other[:10]:
            print(f"  {item['file']}:{item['line']}")
            print(f"    from {'.' * item['level']}{item['module']} import {', '.join(item['names'])}")
    
    # Group by directory for better understanding
    print("\n" + "="*80)
    print("ISSUES BY DIRECTORY:")
    print("="*80)
    
    dir_counts = defaultdict(int)
    for item in internal_refs + deep_nested + other:
        dir_path = os.path.dirname(item['file'])
        dir_counts[dir_path] += 1
    
    for dir_path, count in sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {count:3d}  {dir_path}")
    
    return {
        'internal_refs': internal_refs,
        'deep_nested': deep_nested,
        'other': other,
        'total': len(internal_refs) + len(deep_nested) + len(other)
    }

if __name__ == "__main__":
    results = analyze_remaining_imports()
    print(f"\n{'='*80}")
    print(f"Analysis complete. Total remaining issues: {results['total']}")
    print(f"{'='*80}")

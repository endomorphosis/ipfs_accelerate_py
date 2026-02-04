#!/usr/bin/env python3
"""Analyze remaining subdirectories in test/ to determine what to do with them."""

import os
from pathlib import Path
from collections import defaultdict

def count_files(directory):
    """Count Python files in a directory recursively."""
    py_files = list(Path(directory).rglob('*.py'))
    return len(py_files)

def analyze_directory(dir_path):
    """Analyze a directory and suggest what to do with it."""
    name = dir_path.name
    name_lower = name.lower()
    
    # Count files
    py_count = count_files(dir_path)
    all_count = sum(1 for _ in dir_path.rglob('*') if _.is_file())
    
    # Analysis rules
    if name_lower in ['venv', 'venvs', 'test_venv', '__pycache__']:
        return 'DELETE', 'Virtual environment or cache'
    
    if 'legacy' in name_lower or 'old' in name_lower or 'backup' in name_lower:
        return 'ARCHIVE', 'Legacy or backup directory'
    
    if name_lower in ['improved', 'improvements', 'fixes', 'refactored_test_suite', 
                      'refactored_generator_suite', 'refactored_benchmark_suite']:
        return 'REVIEW', 'Refactored/improved version - check if supersedes original'
    
    if name_lower.startswith('temp') or 'output' in name_lower:
        return 'DELETE', 'Temporary or output directory'
    
    if 'doc' in name_lower or 'docs' in name_lower:
        return 'MOVE', f'Documentation - move to docs/ ({py_count} py, {all_count} total)'
    
    if name in ['tests', 'scripts', 'tools', 'generators', 'templates', 'examples', 'data']:
        return 'KEEP', 'Already organized'
    
    # Check if it's actual test content
    if py_count > 0:
        return 'EVALUATE', f'Has {py_count} Python files, {all_count} total files'
    
    if all_count == 0:
        return 'DELETE', 'Empty directory'
    
    return 'EVALUATE', f'{all_count} files - needs manual review'

def main():
    test_dir = Path('test')
    
    # Get all subdirectories
    subdirs = [d for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    subdirs = sorted(subdirs, key=lambda x: x.name)
    
    print(f"Found {len(subdirs)} subdirectories in test/")
    print()
    
    # Categorize
    actions = defaultdict(list)
    for subdir in subdirs:
        action, reason = analyze_directory(subdir)
        actions[action].append((subdir.name, reason))
    
    # Print results
    print("=" * 80)
    print("DIRECTORY ANALYSIS RESULTS")
    print("=" * 80)
    
    for action in ['KEEP', 'EVALUATE', 'MOVE', 'ARCHIVE', 'DELETE', 'REVIEW']:
        if action not in actions:
            continue
        
        dirs = actions[action]
        print(f"\n{action} ({len(dirs)} directories)")
        print("-" * 80)
        for name, reason in sorted(dirs)[:20]:  # Show first 20
            print(f"  {name:45s} - {reason}")
        if len(dirs) > 20:
            print(f"  ... and {len(dirs) - 20} more")
    
    print("\n" + "=" * 80)
    print(f"\nSummary:")
    for action, dirs in sorted(actions.items()):
        print(f"  {action:10s}: {len(dirs):3d} directories")
    
    # Write detailed report
    with open('/tmp/test_subdir_analysis.txt', 'w') as f:
        f.write("DETAILED TEST SUBDIRECTORY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for action in ['KEEP', 'EVALUATE', 'MOVE', 'ARCHIVE', 'DELETE', 'REVIEW']:
            if action not in actions:
                continue
            
            dirs = actions[action]
            f.write(f"\n{action} ({len(dirs)} directories)\n")
            f.write("-" * 80 + "\n")
            for name, reason in sorted(dirs):
                f.write(f"test/{name}\n  → {reason}\n\n")
    
    print(f"\nDetailed report written to: /tmp/test_subdir_analysis.txt")

if __name__ == '__main__':
    main()

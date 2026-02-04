#!/usr/bin/env python3
"""Categorize remaining non-Python, non-MD files in test/ directory."""

from pathlib import Path
from collections import defaultdict

def categorize_file(filename):
    """Categorize a file based on its name and extension."""
    name_lower = filename.lower()
    
    # HTML/CSS/JSX demos and examples
    if any(ext in filename for ext in ['.html', '.css', '.jsx']):
        if 'demo' in name_lower:
            return 'examples/demos'
        elif 'example' in name_lower:
            return 'examples'
        else:
            return 'examples'
    
    # JavaScript config files
    if filename.endswith('.js') and ('config' in name_lower or 'setup' in name_lower or 'rollup' in name_lower):
        return 'config'
    
    # Requirements files
    if filename.startswith('requirements'):
        return 'config'
    
    # Text files - analysis
    if filename.endswith('.txt'):
        if 'summary' in name_lower or 'error' in name_lower:
            return 'reports'
        elif 'files' in name_lower:
            return 'reports'
        else:
            return 'config'
    
    # Makefile
    if 'makefile' in name_lower:
        return 'config'
    
    # Updated markdown files
    if filename.endswith('.updated'):
        return 'temporary'
    
    return 'other'

def main():
    test_dir = Path('test')
    
    # Find all non-Python, non-directory files in test/ root
    all_files = [f for f in test_dir.iterdir() if f.is_file() and not f.name.endswith('.py')]
    
    print(f"Found {len(all_files)} non-Python files in test/ root")
    print()
    
    # Categorize
    categorized = defaultdict(list)
    for f in sorted(all_files):
        if f.name == '__init__.py' or f.name == 'conftest.py':
            continue
        category = categorize_file(f.name)
        categorized[category].append(f.name)
    
    # Print categorization
    print("File Categorization:")
    print("=" * 80)
    
    for category in sorted(categorized.keys()):
        files = categorized[category]
        print(f"\n{category.upper()} ({len(files)} files)")
        print("-" * 80)
        for f in sorted(files):
            print(f"  test/{f}")
    
    print("\n" + "=" * 80)
    print(f"\nRecommended moves:")
    print("-" * 80)
    print("examples/demos/ : HTML/CSS/JSX demo files")
    print("examples/       : Example files")
    print("config/         : Requirements and config files (or keep in root)")
    print("reports/        : Analysis/summary text files (or move to docs/reports/)")
    print("temporary/      : Delete or review .updated files")

if __name__ == '__main__':
    main()

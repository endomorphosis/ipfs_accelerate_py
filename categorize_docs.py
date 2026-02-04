#!/usr/bin/env python3
"""Categorize markdown documentation files from test/ directory."""

import os
import re
from pathlib import Path
from collections import defaultdict

def categorize_doc(filename):
    """Categorize a documentation file based on its name."""
    name_lower = filename.lower()
    
    # Category patterns
    categories = {
        'testing': [
            'test', 'benchmark', 'validation', 'pytest', 'playwright',
            'coverage', 'integration', 'unit'
        ],
        'api': [
            'api', 'endpoint', 'backend', 'interface', 'duckdb'
        ],
        'implementation': [
            'implementation', 'conversion', 'migration', 'refactor',
            'standardization', 'typescript'
        ],
        'guides': [
            'guide', 'tutorial', 'how', 'usage', 'setup', 'getting',
            'readme'
        ],
        'reports': [
            'report', 'summary', 'status', 'completion', 'final',
            'analysis'
        ],
        'web': [
            'webgpu', 'webnn', 'browser', 'web', 'shader', 'gpu'
        ],
        'hardware': [
            'hardware', 'gpu', 'npu', 'apple', 'silicon', 'amd',
            'nvidia', 'metal', 'cuda', 'rocm'
        ],
        'mobile': [
            'mobile', 'ios', 'android', 'battery', 'thermal'
        ],
        'monitoring': [
            'monitoring', 'dashboard', 'visualization', 'metrics',
            'logging'
        ],
        'models': [
            'model', 'huggingface', 'hf_', 'transformer', 'template'
        ],
        'ipfs': [
            'ipfs', 'storage', 'distributed', 'p2p'
        ],
        'mcp': [
            'mcp', 'copilot', 'copilot_'
        ]
    }
    
    # Check each category
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    
    return 'other'

def main():
    test_dir = Path('test')
    
    # Find all markdown files in test/ root
    md_files = sorted([f for f in test_dir.glob('*.md')])
    
    print(f"Found {len(md_files)} markdown files in test/ root")
    print()
    
    # Categorize files
    categorized = defaultdict(list)
    for md_file in md_files:
        category = categorize_doc(md_file.name)
        categorized[category].append(md_file.name)
    
    # Print categorization
    print("Documentation Categorization:")
    print("=" * 80)
    
    for category in sorted(categorized.keys()):
        files = categorized[category]
        print(f"\n{category.upper()} ({len(files)} files)")
        print("-" * 80)
        for f in sorted(files)[:10]:  # Show first 10
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(md_files)} files across {len(categorized)} categories")
    
    # Write detailed categorization to file
    output_file = Path('/tmp/doc_categorization.txt')
    with open(output_file, 'w') as f:
        f.write("DOCUMENTATION FILE CATEGORIZATION\n")
        f.write("=" * 80 + "\n\n")
        
        for category in sorted(categorized.keys()):
            files = categorized[category]
            f.write(f"\n{category.upper()} ({len(files)} files)\n")
            f.write("-" * 80 + "\n")
            for file in sorted(files):
                f.write(f"test/{file} -> docs/{category}/{file}\n")
    
    print(f"\nDetailed categorization written to: {output_file}")

if __name__ == '__main__':
    main()

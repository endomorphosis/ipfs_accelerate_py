#!/usr/bin/env python3
"""Move documentation files from test/ to docs/ with proper categorization."""

import os
import subprocess
from pathlib import Path
from collections import defaultdict

def categorize_doc(filename):
    """Categorize a documentation file based on its name."""
    name_lower = filename.lower()
    
    categories = {
        'testing': ['test', 'benchmark', 'validation', 'pytest', 'playwright', 'coverage', 'integration', 'unit'],
        'api': ['api', 'endpoint', 'backend', 'interface', 'duckdb'],
        'implementation': ['implementation', 'conversion', 'migration', 'refactor', 'standardization', 'typescript'],
        'guides': ['guide', 'tutorial', 'how', 'usage', 'setup', 'getting', 'readme'],
        'reports': ['report', 'summary', 'status', 'completion', 'final', 'analysis'],
        'web': ['webgpu', 'webnn', 'browser', 'web', 'shader', 'gpu'],
        'hardware': ['hardware', 'gpu', 'npu', 'apple', 'silicon', 'amd', 'nvidia', 'metal', 'cuda', 'rocm'],
        'mobile': ['mobile', 'ios', 'android', 'battery', 'thermal'],
        'monitoring': ['monitoring', 'dashboard', 'visualization', 'metrics', 'logging'],
        'models': ['model', 'huggingface', 'hf_', 'transformer', 'template'],
        'ipfs': ['ipfs', 'storage', 'distributed', 'p2p'],
        'mcp': ['mcp', 'copilot', 'copilot_']
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    
    return 'other'

def main():
    test_dir = Path('test')
    docs_dir = Path('docs')
    
    # Find all markdown files in test/ root
    md_files = sorted([f for f in test_dir.glob('*.md')])
    
    print(f"Found {len(md_files)} markdown files to move")
    print()
    
    # Categorize and move files
    categorized = defaultdict(list)
    moves_made = 0
    
    for md_file in md_files:
        category = categorize_doc(md_file.name)
        categorized[category].append(md_file.name)
        
        # Create target directory
        target_dir = docs_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py if it doesn't exist (not needed for docs but for consistency)
        # Actually, we don't need __init__.py for markdown directories
        
        source = md_file
        target = target_dir / md_file.name
        
        # Use git mv to preserve history
        try:
            result = subprocess.run(
                ['git', 'mv', str(source), str(target)],
                capture_output=True,
                text=True,
                check=True
            )
            moves_made += 1
            if moves_made <= 10 or moves_made % 50 == 0:
                print(f"  [{moves_made:3d}] {source} -> {target}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERR] Failed to move {source}: {e.stderr.strip()}")
    
    print()
    print("=" * 80)
    print(f"Successfully moved {moves_made}/{len(md_files)} documentation files")
    print()
    
    # Print summary by category
    print("Files moved by category:")
    for category in sorted(categorized.keys()):
        count = len(categorized[category])
        print(f"  {category:20s}: {count:3d} files")
    
    print()
    print("Documentation files are now organized in docs/ subdirectories!")

if __name__ == '__main__':
    main()

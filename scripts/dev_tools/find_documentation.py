#!/usr/bin/env python3
"""
Documentation Finder for ipfs_accelerate_py

Finds and catalogs documentation files (TODO.md, CHANGELOG.md, README.md, etc.).
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def find_documentation_files(directory: Path, doc_types: List[str], exclude_patterns: List[str]) -> List[Dict]:
    """Find documentation files in directory."""
    doc_files = []
    
    for doc_type in doc_types:
        for doc_file in directory.glob(f'**/{doc_type}'):
            if doc_file.is_file():
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern in str(doc_file):
                        should_exclude = True
                        break
                
                if not should_exclude:
                    stat = doc_file.stat()
                    doc_files.append({
                        'path': str(doc_file.relative_to(directory)),
                        'type': doc_type,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'lines': count_lines(doc_file)
                    })
    
    return doc_files


def count_lines(file_path: Path) -> int:
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Find documentation files")
    parser.add_argument('--directory', required=True, help='Directory to scan')
    parser.add_argument('--format', choices=['json', 'text'], default='text', 
                       help='Output format')
    parser.add_argument('--exclude', nargs='*',
                       default=['__pycache__', '.venv', 'venv', '.git', 'build', 'dist', 'node_modules'],
                       help='Patterns to exclude')
    
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Find different types of documentation
    doc_types = ['README.md', 'TODO.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 
                 'LICENSE', 'MANIFEST.in', '*.rst']
    
    print(f"Finding documentation files in: {directory}")
    print()
    
    doc_files = find_documentation_files(directory, doc_types, args.exclude)
    
    if not doc_files:
        print("No documentation files found")
        return 0
    
    # Sort by type then path
    doc_files.sort(key=lambda x: (x['type'], x['path']))
    
    if args.format == 'json':
        # Output as JSON
        result = {
            'directory': str(directory),
            'scan_time': datetime.now().isoformat(),
            'total_files': len(doc_files),
            'files': doc_files
        }
        print(json.dumps(result, indent=2))
    else:
        # Output as text
        print(f"Found {len(doc_files)} documentation files:\n")
        
        current_type = None
        for doc in doc_files:
            if doc['type'] != current_type:
                current_type = doc['type']
                print(f"\n{current_type}:")
                print("-" * 60)
            
            print(f"  {doc['path']}")
            print(f"    Size: {doc['size']} bytes, Lines: {doc['lines']}, Modified: {doc['modified']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

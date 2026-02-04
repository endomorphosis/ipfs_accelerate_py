#\!/usr/bin/env python3
"""
Migration Helper Script

This script helps identify files that need to be migrated to the new directory structure.
"""

import os
import sys
import re
import argparse
from pathlib import Path

def find_generator_files(directory):
    """Find generator-related files in the directory."""
    patterns = []],,
    r'.*generator.*\.py$',
    r'.*template.*\.py$',
    r'.*skill.*\.py$',
    r'run_.*test.*\.py$',
    r'create_.*\.py$',
    ]
    
    generator_files = []],,]
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                for pattern in patterns:
                    if re.match(pattern, file) and not re.match(r'.*database.*\.py$', file):
                        generator_files.append(os.path.join(root, file))
                    break
    
                return generator_files

def find_db_files(directory):
    """Find database-related files in the directory."""
    patterns = []],,
    r'.*db_.*\.py$',
    r'.*database.*\.py$',
    r'.*duckdb.*\.py$',
    r'view_benchmark.*\.py$',
    r'.*cleanup.*reports.*\.py$',
    r'.*simulation.*\.py$',
    ]
    
    db_files = []],,]
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                for pattern in patterns:
                    if re.match(pattern, file):
                        db_files.append(os.path.join(root, file))
                    break
    
                return db_files

def check_migrated_files(source_files, target_dir):
    """Check which files have been migrated."""
    migrated = []],,]
    not_migrated = []],,]
    
    for src_file in source_files:
        filename = os.path.basename(src_file)
        # Check if file exists anywhere in the target directory
        found = False:
        for root, _, files in os.walk(target_dir):
            if filename in files:
                migrated.append((src_file, os.path.join(root, filename)))
                found = True
            break
        
        if not found:
            not_migrated.append(src_file)
    
            return migrated, not_migrated

def main():
    parser = argparse.ArgumentParser(description="Helper script for code migration")
    parser.add_argument("--generators", action="store_true", help="Check generator files")
    parser.add_argument("--db", action="store_true", help="Check database files")
    parser.add_argument("--all", action="store_true", help="Check all files")
    parser.add_argument("--source", default=".", help="Source directory to check")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    
    if args.generators or args.all:
        generator_files = find_generator_files(source_dir)
        generators_dir = Path("../generators")
        migrated, not_migrated = check_migrated_files(generator_files, generators_dir)
        
        print(f"\n==== GENERATOR FILES ====")
        print(f"Found {len(generator_files)} generator-related files")
        print(f"- {len(migrated)} files migrated")
        print(f"- {len(not_migrated)} files not migrated\n")
        
        if not_migrated:
            print("Files that need migration to generators/:")
            for file in not_migrated:
                print(f"\1{os.path.basename(file)}\3")
    
    if args.db or args.all:
        db_files = find_db_files(source_dir)
        db_dir = Path("../duckdb_api")
        migrated, not_migrated = check_migrated_files(db_files, db_dir)
        
        print(f"\n==== DATABASE FILES ====")
        print(f"Found {len(db_files)} database-related files")
        print(f"- {len(migrated)} files migrated")
        print(f"- {len(not_migrated)} files not migrated\n")
        
        if not_migrated:
            print("Files that need migration to duckdb_api/:")
            for file in not_migrated:
                print(f"\1{os.path.basename(file)}\3")

if __name__ == "__main__":
    main()
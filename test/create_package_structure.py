#!/usr/bin/env python3
"""
Creates the necessary directory structure for the new package organization.

This script creates the generators/ and duckdb_api/ directories and their 
subdirectories according to the reorganization plan in CLAUDE.md.
It also creates __init__.py files in each directory to ensure proper
Python package structure.

Usage:
    python create_package_structure.py [--dry-run]
"""

import os
import sys
import argparse
from pathlib import Path

# Directory structure definition
PACKAGE_STRUCTURE = {
    "generators": {
        "benchmark_generators": {},
        "models": {},
        "runners": {
            "web": {}
        },
        "skill_generators": {},
        "template_generators": {},
        "templates": {},
        "test_generators": {},
        "utils": {},
        "hardware": {},
        "validators": {},
        "web": {},
    },
    "duckdb_api": {
        "core": {},
        "migration": {},
        "schema": {
            "creation": {}
        },
        "utils": {},
        "visualization": {},
        "distributed_testing": {},
        "web": {},
        "ci": {},
        "examples": {},
    }
}

def create_directory_structure(base_dir, structure, dry_run=False):
    """
    Recursively creates directory structure and __init__.py files.
    
    Args:
        base_dir: Base directory path
        structure: Dictionary representing the directory structure
        dry_run: If True, only print actions without creating directories
    
    Returns:
        Tuple of (created_dirs, created_inits)
    """
    created_dirs = 0
    created_inits = 0
    
    for dirname, subdirs in structure.items():
        dirpath = os.path.join(base_dir, dirname)
        
        # Create directory if it doesn't exist
        if not os.path.exists(dirpath):
            if not dry_run:
                try:
                    os.makedirs(dirpath, exist_ok=True)
                    print(f"✅ Created directory: {dirpath}")
                    created_dirs += 1
                except Exception as e:
                    print(f"❌ Error creating directory {dirpath}: {e}")
                    continue
            else:
                print(f"🔍 [DRY RUN] Would create directory: {dirpath}")
                created_dirs += 1
        else:
            print(f"👉 Directory already exists: {dirpath}")
        
        # Create __init__.py if it doesn't exist
        init_path = os.path.join(dirpath, "__init__.py")
        if not os.path.exists(init_path):
            if not dry_run:
                try:
                    with open(init_path, "w") as f:
                        package_name = dirname.replace("-", "_")
                        f.write(f'"""\n{package_name} module for IPFS Accelerate.\n"""\n')
                        print(f"✅ Created __init__.py: {init_path}")
                        created_inits += 1
                except Exception as e:
                    print(f"❌ Error creating __init__.py in {dirpath}: {e}")
            else:
                print(f"🔍 [DRY RUN] Would create __init__.py: {init_path}")
                created_inits += 1
        else:
            print(f"👉 __init__.py already exists: {init_path}")
        
        # Recursively process subdirectories
        if subdirs:
            subdirs_created, inits_created = create_directory_structure(dirpath, subdirs, dry_run)
            created_dirs += subdirs_created
            created_inits += inits_created
    
    return created_dirs, created_inits

def main():
    parser = argparse.ArgumentParser(description='Create package directory structure')
    parser.add_argument('--dry-run', action='store_true', help='Show actions without creating directories')
    
    args = parser.parse_args()
    
    # Get project root directory (parent of test/)
    project_root = Path(__file__).parent.parent
    
    print(f"Creating package structure in {project_root}")
    print("Directory structure follows the plan in CLAUDE.md")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE: No directories will be created\n")
    
    total_dirs_created, total_inits_created = create_directory_structure(
        project_root, PACKAGE_STRUCTURE, args.dry_run
    )
    
    print("\n=== SUMMARY ===")
    if args.dry_run:
        print(f"Would create {total_dirs_created} directories and {total_inits_created} __init__.py files")
        print("Run without --dry-run to actually create the structure")
    else:
        print(f"Created {total_dirs_created} directories and {total_inits_created} __init__.py files")
        print("\nNext steps:")
        print("1. Run fix_syntax_errors.py to fix syntax issues in generator files")
        print("2. Run update_imports.py to update import statements")
        print("3. Move files to the new structure based on their types")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
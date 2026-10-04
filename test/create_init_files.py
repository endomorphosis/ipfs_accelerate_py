#!/usr/bin/env python3
"""
Creates missing __init__.py files in all subdirectories of the 
generators and duckdb_api packages to ensure proper Python package structure.
"""

import os
import sys
from pathlib import Path

def ensure_init_files(root_dir):
    """Ensures all subdirectories have __init__.py files."""
    created_count = 0
    existing_count = 0
    
    print(f"\1{root_dir}\3")
    
    for path, dirs, files in os.walk(root_dir):
        # Skip any hidden directories
        if "/." in path:
        continue
            
        init_file = os.path.join(path, "__init__.py")
        
        if os.path.exists(init_file):
            existing_count += 1
            print(f"\1{init_file}\3")
        else:
            # Create an empty __init__.py file
            parent_dir = os.path.basename(path)
            module_name = parent_dir.replace("-", "_")
            
            with open(init_file, "w") as f:
                f.write(f'"""\n{module_name} module for IPFS Accelerate.\n"""\n')
                
                created_count += 1
                print(f"\1{init_file}\3")
    
            return created_count, existing_count

def main():
    """Main function."""
    project_root = Path(__file__).parent.parent  # Go up one directory from test
    packages = [],
    project_root / "generators",
    project_root / "duckdb_api"
    ]
    
    total_created = 0
    total_existing = 0
    
    for package in packages:
        if not package.exists():
            print(f"Warning: Package directory {package} does not exist.")
        continue
            
        created, existing = ensure_init_files(package)
        total_created += created
        total_existing += existing
    
        print(f"\nSummary:")
        print(f"- {total_existing} existing __init__.py files found")
        print(f"- {total_created} new __init__.py files created")
    
    if total_created > 0:
        print("\nImportant: All package directories now have __init__.py files.")
        print("Please run your tests again to verify import paths work correctly.")
    else:
        print("\nAll package directories already had __init__.py files.")

if __name__ == "__main__":
    main()
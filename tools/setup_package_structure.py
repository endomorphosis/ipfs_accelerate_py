#!/usr/bin/env python
"""
Script to set up Python package structure for the reorganized project.
This script creates necessary __init__.py files in the new directory structure.
"""

import os
from pathlib import Path

def create_init_files(project_root=None):
    """Create __init__.py files in all directories and subdirectories."""
    if project_root is None:
        project_root = Path.cwd().parent  # Go up one level from current directory
    
    # Directories that should be treated as packages
    package_dirs = [
        project_root / "duckdb_api",
        project_root / "generators",
        project_root / "fixed_web_platform",
        project_root / "predictive_performance"
    ]
    
    init_files_created = []
    
    for package_dir in package_dirs:
        if not package_dir.exists():
            print(f"Directory not found: {package_dir}")
            continue
        
        print(f"Creating __init__.py files in {package_dir}...")
        
        # Create __init__.py in the main package directory
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write(f'"""\n{package_dir.name} package for IPFS Accelerate Python Framework.\n"""\n')
            init_files_created.append(str(init_file))
        
        # Create __init__.py in all subdirectories
        for subdir in package_dir.glob("**"):
            if subdir.is_dir():
                init_file = subdir / "__init__.py"
                if not init_file.exists():
                    with open(init_file, 'w') as f:
                        # Create a package docstring
                        package_name = f"{package_dir.name}.{subdir.relative_to(package_dir)}"
                        package_name = package_name.replace('/', '.')
                        f.write(f'"""\n{package_name} module for IPFS Accelerate Python Framework.\n"""\n')
                    init_files_created.append(str(init_file))
    
    # Print summary
    print("\nSummary:")
    print(f"Created {len(init_files_created)} __init__.py files")
    
    return init_files_created

if __name__ == "__main__":
    create_init_files()
#!/usr/bin/env python
"""
Environment Setup Script for IPFS Accelerate Python Framework

This script helps set up a development environment for the IPFS Accelerate Python
Framework by checking dependencies and ensuring the correct packages are installed.

Usage:
    python setup_environment.py [--full | --minimal | --dev]
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if the Python version is adequate."""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️ WARNING: Python 3.8 or higher is recommended. You are using Python "
              f"{python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    
    print(f"✅ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is adequate")
    return True

def check_dependencies(requirement_sets):
    """Check if dependencies are installed."""
    missing_packages = {}
    
    for set_name, required_packages in requirement_sets.items():
        missing_packages[set_name] = []
        
        for package in required_packages:
            # Strip version info
            package_name = package.split('>=')[0].split('==')[0].strip()
            
            # Skip comments
            if package_name.startswith('#'):
                continue
            
            # Check if package is importable
            is_installed = importlib.util.find_spec(package_name) is not None
            
            if not is_installed:
                missing_packages[set_name].append(package)
    
    return missing_packages

def install_requirements(requirements_file, upgrade=False):
    """Install requirements from a requirements file."""
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements_file]
    
    if upgrade:
        cmd.insert(4, '--upgrade')
    
    try:
        print(f"Installing requirements from {requirements_file}...")
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_package_structure():
    """Set up the package structure with __init__.py files."""
    project_root = Path.cwd()
    
    # Create directories if they don't exist
    directories = [
        project_root / "generators",
        project_root / "generators" / "models",
        project_root / "generators" / "test_runners",
        project_root / "duckdb_api",
        project_root / "duckdb_api" / "analysis",
        project_root / "duckdb_api" / "core",
        project_root / "fixed_web_platform",
        project_root / "predictive_performance"
    ]
    
    for directory in directories:
        if not directory.exists():
            print(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py if it doesn't exist
        init_file = directory / "__init__.py"
        if not init_file.exists():
            print(f"Creating {init_file}")
            with open(init_file, 'w') as f:
                package_name = str(directory.relative_to(project_root)).replace('/', '.')
                f.write(f'"""\n{package_name} package for IPFS Accelerate Python Framework.\n"""\n')

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up the IPFS Accelerate Python Framework environment")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', help='Install all dependencies')
    group.add_argument('--minimal', action='store_true', help='Install only essential dependencies')
    group.add_argument('--dev', action='store_true', help='Install development dependencies')
    parser.add_argument('--upgrade', action='store_true', help='Upgrade existing packages')
    parser.add_argument('--init', action='store_true', help='Create package structure with __init__.py files')
    args = parser.parse_args()
    
    # Check Python version
    check_python_version()
    
    # Set up requirements
    requirements_file = 'requirements.txt'
    
    # Define dependency groups
    requirement_sets = {
        "core": [
            "torch", "transformers", "numpy", "scipy",
            "ipfs_kit_py", "libp2p_kit_py", "ipfs_model_manager_py"
        ],
        "database": [
            "duckdb", "pandas", "pyarrow", "fastapi", "uvicorn"
        ],
        "visualization": [
            "matplotlib", "plotly", "seaborn"
        ],
        "web": [
            "selenium", "websockets", "jinja2"
        ],
        "testing": [
            "pytest", "pytest-cov"
        ]
    }
    
    # Check missing dependencies
    missing_packages = check_dependencies(requirement_sets)
    
    # Report missing dependencies by category
    print("\nDependency Status:")
    has_missing = False
    
    for set_name, packages in missing_packages.items():
        status = "✅ All installed" if not packages else f"❌ Missing {len(packages)} packages"
        print(f"{set_name.capitalize()}: {status}")
        
        if packages:
            has_missing = True
            for package in packages:
                print(f"  - {package}")
    
    # Install dependencies if requested or if there are missing packages
    if args.full or args.minimal or args.dev or (has_missing and input("\nInstall missing packages? [y/N]: ").lower() == 'y'):
        install_requirements(requirements_file, upgrade=args.upgrade)
    
    # Set up package structure if requested
    if args.init:
        setup_package_structure()
    
    # Print summary
    print("\nEnvironment setup complete!")
    print("For more information on the IPFS Accelerate Python Framework, see the README.md file.")

if __name__ == "__main__":
    main()
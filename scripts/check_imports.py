#!/usr/bin/env python
"""
Import checker script to identify files that need to be updated after migration.
This script attempts to import the modules in the new directory structure and 
reports any errors that occur.
"""

import os
import sys
from pathlib import Path
import importlib.util
import inspect

def check_import(module_path):
    """
    Attempt to import a module and report any errors.
    
    Args:
        module_path: Path to the Python module to import
    
    Returns:
        (bool, str): Success status and error message if any
    """
    try:
        # Convert file path to module name
        module_name = str(module_path).replace('/', '.').replace('.py', '')
        
        # Try to import the module directly
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False, f"Could not find module specification for {module_path}"
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return True, "Successfully imported module"
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        return False, f"{error_type}: {error_msg}"

def analyze_migrated_files():
    """
    Analyze all migrated files and check their imports.
    """
    project_root = Path.cwd().parent  # Go up one level from current directory
    
    # List of directories to check
    dirs_to_check = [
        project_root / "duckdb_api",
        project_root / "generators",
        project_root / "fixed_web_platform",
        project_root / "predictive_performance"
    ]
    
    results = {
        "success": [],
        "failure": []
    }
    
    for directory in dirs_to_check:
        if not directory.exists():
            print(f"Directory not found: {directory}")
            continue
        
        print(f"Checking files in {directory}...")
        
        # Find all Python files in the directory
        for py_file in directory.glob("**/*.py"):
            if py_file.name.startswith("__"):
                continue  # Skip __init__.py etc.
                
            success, error_msg = check_import(py_file)
            
            if success:
                results["success"].append(str(py_file))
            else:
                results["failure"].append({
                    "file": str(py_file),
                    "error": error_msg
                })
    
    # Print results
    print("\n" + "="*50)
    print(f"Import Check Results")
    print("-"*50)
    print(f"Successfully imported: {len(results['success'])}")
    print(f"Failed imports: {len(results['failure'])}")
    
    if results["failure"]:
        print("\nFiles with import issues:")
        for item in results["failure"]:
            print(f"\n{item['file']}:")
            print(f"  Error: {item['error']}")
    
    return results

if __name__ == "__main__":
    print("Checking imports for migrated files...")
    analyze_migrated_files()
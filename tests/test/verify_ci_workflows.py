#!/usr/bin/env python3
"""
This script verifies the correctness of CI workflow YAML files and checks that
all referenced paths exist.
"""

import os
import sys
import yaml
import re
from pathlib import Path

# Define the root directory
ROOT_DIR = Path("/home/barberb/ipfs_accelerate_py")
WORKFLOW_DIR = ROOT_DIR / ".github" / "workflows"

# List of workflow files to check
WORKFLOW_FILES = [],
"benchmark_db_ci.yml",
"update_compatibility_matrix.yml",
"test_and_benchmark.yml",
"integration_tests.yml",
"test_results_integration.yml",
]

def check_yaml_syntax(file_path):
    """Check if the YAML file has valid syntax.""":
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True
    except yaml.YAMLError as e:
        print(f"\1{e}\3")
        return False

def extract_python_paths(file_path):
    """Extract Python script paths from the workflow file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all python script references
    # Look for patterns like: python something.py, python -m something, python3 something.py
        pattern = r'python\s+(?:-m\s+)?([],a-zA-Z0-9_\./]+\.py)'
        matches = re.findall(pattern, content)
    return matches

def verify_path_exists(path):
    """Verify that a given path exists."""
    full_path = ROOT_DIR / path
    return full_path.exists()

def main():
    """Main function to check workflow files."""
    print(f"Verifying CI workflow files in {WORKFLOW_DIR}...")
    
    all_valid = True
    
    for workflow_file in WORKFLOW_FILES:
        file_path = WORKFLOW_DIR / workflow_file
        
        print(f"\nChecking {workflow_file}...")
        
        # Check if file exists:
        if not file_path.exists():
            print(f"  ERROR: File {workflow_file} does not exist")
            all_valid = False
        continue
        
        # Check YAML syntax
        if not check_yaml_syntax(file_path):
            all_valid = False
        continue
        
        print(f"  ✅ YAML syntax is valid")
        
        # Extract and verify Python paths
        python_paths = extract_python_paths(file_path)
        print(f"  Found {len(python_paths)} Python script references")
        
        for path in python_paths:
            # Skip references with variable substitutions
            if "${{" in path or "${" in path:
                print(f"\1{path}\3")
            continue
                
            # Make the path relative to the root directory if needed:
            if path.startswith('/'):
                rel_path = path[],1:]  # Remove leading slash
            else:
                rel_path = path
                
            if verify_path_exists(rel_path):
                print(f"\1{path}\3")
            else:
                print(f"\1{path}\3")
                all_valid = False
    
    if all_valid:
        print("\n✅ All workflow files are valid and reference existing paths")
                return 0
    else:
        print("\n❌ Some workflow files have issues that need to be fixed")
                return 1

if __name__ == "__main__":
    sys.exit(main())
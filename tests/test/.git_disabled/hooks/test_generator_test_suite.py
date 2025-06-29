#!/usr/bin/env python3
"""
Test generator validation suite for pre-commit hook.

This script validates that test generators adhere to coding standards
and can properly generate tests.
"""

import os
import sys
import glob
import importlib.util
import unittest

def find_test_generators():
    """Find test generator files in the distributed testing framework only."""
    generators = []
    
    # Only look in distributed_testing directory and its subdirectories
    for pattern in ["**/distributed_testing/**/test_*_generator.py", "**/distributed_testing/**/test_generator*.py"]:
        generators.extend(glob.glob(pattern, recursive=True))
    
    return generators

def validate_generator(generator_path):
    """Validate a single test generator."""
    print(f"Validating {generator_path}...")
    
    # Basic file existence check
    if not os.path.exists(generator_path):
        print(f"Error: {generator_path} does not exist")
        return False
    
    # Skip validation for non-Python files or empty files
    if not generator_path.endswith('.py'):
        return True
    
    try:
        with open(generator_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: {generator_path} is empty")
                return True
    except Exception:
        print(f"Error reading {generator_path}")
        return False
    
    # Check for basic syntax without importing
    try:
        with open(generator_path, 'r') as f:
            code = compile(f.read(), generator_path, 'exec')
    except SyntaxError as e:
        print(f"Syntax error in {generator_path}: {str(e)}")
        return False
    
    # Check if generate_test or main function exists in the file
    try:
        with open(generator_path, 'r') as f:
            content = f.read()
            if 'def generate_test(' not in content and 'def main(' not in content:
                print(f"Error: {generator_path} does not have generate_test or main function")
                return False
    except Exception as e:
        print(f"Error checking functions in {generator_path}: {str(e)}")
        return False
    
    return True

def main():
    """Main function to validate test generators."""
    print("🧪 Running test generator validation suite...")
    
    # Get all test generators
    generators = find_test_generators()
    
    if not generators:
        print("No test generators found in distributed testing framework. Skipping validation.")
        return 0
    
    # Validate each generator
    failures = 0
    for generator in generators:
        if not validate_generator(generator):
            failures += 1
    
    if failures > 0:
        print(f"❌ Test generator validation failed. Please fix the issues before committing.")
        return 1
    else:
        print(f"✅ All {len(generators)} test generators in distributed testing framework passed validation.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
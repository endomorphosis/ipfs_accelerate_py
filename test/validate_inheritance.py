#!/usr/bin/env python3
"""
Validation tool to check if test files properly inherit from ModelTest.

This script validates that test files in the refactored test suite inherit from ModelTest
and implement required methods.
"""

import os
import sys
import ast
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required methods for ModelTest subclasses
REQUIRED_METHODS = ['test_model_loading', 'load_model', 'verify_model_output', 'detect_preferred_device']

def is_modeltest_subclass(file_path):
    """Check if the file contains a class that inherits from ModelTest."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
            
        tree = ast.parse(content)
        
        # Track imports to find ModelTest
        imports = []
        model_test_import = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'model_test' in node.module:
                    for name in node.names:
                        if name.name == 'ModelTest':
                            model_test_import = f"{node.module}.{name.name}"
                            imports.append(model_test_import)
                            
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if 'model_test' in name.name:
                        imports.append(name.name)
                        if name.name.endswith('ModelTest'):
                            model_test_import = name.name
        
        if not model_test_import:
            return False, "ModelTest not imported", None
        
        # Find classes that inherit from ModelTest
        valid_classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'ModelTest':
                        valid_classes.append(node.name)
                    elif isinstance(base, ast.Attribute) and base.attr == 'ModelTest':
                        valid_classes.append(node.name)
        
        if not valid_classes:
            return False, "No class inherits from ModelTest", None
            
        # Check for required methods
        class_methods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in valid_classes:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_methods.add(item.name)
        
        missing_methods = []
        for method in REQUIRED_METHODS:
            if method not in class_methods:
                missing_methods.append(method)
        
        if missing_methods:
            return False, f"Missing required methods: {', '.join(missing_methods)}", valid_classes[0]
            
        return True, f"Valid ModelTest subclass: {valid_classes[0]}", valid_classes[0]
        
    except Exception as e:
        return False, f"Error processing file: {str(e)}", None

def find_test_files(root_dir):
    """Find all Python test files in the directory tree."""
    test_files = []
    for path in Path(root_dir).rglob("*.py"):
        if path.name.startswith("test_") and path.is_file():
            test_files.append(str(path))
    return test_files

def validate_directory(directory):
    """Validate all test files in the directory."""
    test_files = find_test_files(directory)
    results = {"valid": [], "invalid": [], "total": len(test_files)}
    
    for file_path in test_files:
        valid, message, class_name = is_modeltest_subclass(file_path)
        relative_path = os.path.relpath(file_path, directory)
        
        if valid:
            results["valid"].append((relative_path, class_name, message))
        else:
            results["invalid"].append((relative_path, message))
    
    return results

def print_results(results):
    """Print validation results in a readable format."""
    print(f"\nValidation Results for {results['total']} test files:")
    print(f"Valid: {len(results['valid'])} files")
    print(f"Invalid: {len(results['invalid'])} files")
    print("\n--- Valid Files ---")
    for path, class_name, message in results["valid"]:
        print(f"✅ {path} - {class_name}")
    
    print("\n--- Invalid Files ---")
    for path, message in results["invalid"]:
        print(f"❌ {path} - {message}")

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        # Handle both file and directory paths
        if os.path.isfile(path):
            # Process a single file
            logger.info(f"Validating test file: {path}")
            valid, message, class_name = is_modeltest_subclass(path)
            file_name = os.path.basename(path)
            
            if valid:
                print(f"✅ {file_name} - {class_name}: {message}")
                return 0
            else:
                print(f"❌ {file_name} - {message}")
                return 1
        elif os.path.isdir(path):
            # Process a directory
            directory = path
        else:
            logger.error(f"Path not found: {path}")
            sys.exit(1)
    else:
        directory = os.path.join(os.path.dirname(__file__), "refactored_test_suite")
    
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    logger.info(f"Validating test files in {directory}")
    results = validate_directory(directory)
    print_results(results)
    
    # Calculate compliance percentage
    compliance = (len(results["valid"]) / results["total"]) * 100 if results["total"] > 0 else 0
    print(f"\nCompliance: {compliance:.1f}% ({len(results['valid'])}/{results['total']} files)")
    
    # Return non-zero exit code if any invalid files
    return 1 if results["invalid"] else 0

if __name__ == "__main__":
    sys.exit(main())
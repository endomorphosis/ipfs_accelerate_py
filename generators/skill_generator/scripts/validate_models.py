#!/usr/bin/env python3
"""
Validate Generated Models

This script validates the syntax and structure of generated test files.
"""

import os
import sys
import ast
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# Add parent directory to path to allow imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from syntax.validator import SyntaxValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_syntax(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate the syntax of a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Parse with ast to check for syntax errors
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
    except Exception as e:
        return False, f"Error validating syntax: {str(e)}"

def validate_imports(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate the imports in a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary file to check imports
            temp_file = os.path.join(temp_dir, "check_imports.py")
            
            with open(file_path, "r") as f:
                content = f.read()
            
            # Extract only the import statements
            tree = ast.parse(content)
            import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            import_content = "\n".join([ast.unparse(node) for node in import_nodes])
            import_content += "\n\nprint('Imports valid')"
            
            with open(temp_file, "w") as f:
                f.write(import_content)
            
            # Run the temporary file with Python
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, f"Import error: {result.stderr.strip()}"
            
            return True, None
    except Exception as e:
        return False, f"Error validating imports: {str(e)}"

def validate_class_structure(file_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate the class structure of a test file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (is_valid, error_message, class_info)
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find all class definitions
        class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not class_nodes:
            return False, "No classes found in the file", {}
        
        # Check for test class
        test_classes = [node for node in class_nodes if node.name.endswith("Test")]
        
        if not test_classes:
            return False, "No test classes found in the file", {}
        
        test_class = test_classes[0]
        
        # Extract methods
        method_names = [node.name for node in ast.walk(test_class) if isinstance(node, ast.FunctionDef)]
        
        # Check for required methods
        required_methods = ["__init__", "run", "test_pipeline", "test_model", "save_results"]
        missing_methods = [method for method in required_methods if method not in method_names]
        
        if missing_methods:
            return False, f"Missing required methods: {', '.join(missing_methods)}", {}
        
        # Extract class info
        class_info = {
            "name": test_class.name,
            "methods": method_names,
            "has_required_methods": len(missing_methods) == 0
        }
        
        return True, None, class_info
    except Exception as e:
        return False, f"Error validating class structure: {str(e)}", {}

def validate_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a test file.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        Dictionary with validation results
    """
    filename = os.path.basename(file_path)
    logger.info(f"Validating {filename}...")
    
    results = {
        "file": filename,
        "path": file_path,
        "valid": True,
        "errors": [],
        "warnings": [],
        "class_info": {}
    }
    
    # Validate syntax
    syntax_valid, syntax_error = validate_syntax(file_path)
    if not syntax_valid:
        results["valid"] = False
        results["errors"].append(f"Syntax: {syntax_error}")
    
    # Validate imports
    imports_valid, imports_error = validate_imports(file_path)
    if not imports_valid:
        results["warnings"].append(f"Imports: {imports_error}")
    
    # Validate class structure
    structure_valid, structure_error, class_info = validate_class_structure(file_path)
    results["class_info"] = class_info
    
    if not structure_valid:
        results["valid"] = False
        results["errors"].append(f"Structure: {structure_error}")
    
    # Use our SyntaxValidator
    validator = SyntaxValidator()
    validator_result = validator.validate_file(file_path)
    
    if not validator_result["valid"]:
        results["warnings"].extend([f"Validator: {error}" for error in validator_result["errors"]])
    
    return results

def validate_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Validate all test files in a directory.
    
    Args:
        directory: Directory containing test files
        
    Returns:
        List of validation results
    """
    results = []
    
    try:
        for file in os.listdir(directory):
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(directory, file)
                result = validate_file(file_path)
                results.append(result)
    except Exception as e:
        logger.error(f"Error validating directory: {str(e)}")
    
    return results

def print_results(results: List[Dict[str, Any]]):
    """
    Print validation results.
    
    Args:
        results: List of validation results
    """
    # Count valid and invalid files
    valid_count = sum(1 for result in results if result["valid"])
    invalid_count = len(results) - valid_count
    
    print("\n\033[1mValidation Results\033[0m")
    print(f"Total files: {len(results)}")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {invalid_count}")
    print()
    
    # Print details for each file
    for result in results:
        filename = result["file"]
        status = "\033[32m✓\033[0m" if result["valid"] else "\033[31m✗\033[0m"
        
        print(f"{status} {filename}")
        
        if result["errors"]:
            print("  \033[31mErrors:\033[0m")
            for error in result["errors"]:
                print(f"    - {error}")
        
        if result["warnings"]:
            print("  \033[33mWarnings:\033[0m")
            for warning in result["warnings"]:
                print(f"    - {warning}")
        
        if not result["errors"] and not result["warnings"]:
            print("  No issues found")
        
        print()

def generate_report(results: List[Dict[str, Any]], output_file: str):
    """
    Generate a validation report.
    
    Args:
        results: List of validation results
        output_file: Path to output file
    """
    with open(output_file, "w") as f:
        f.write("# Model Test Validation Report\n\n")
        
        # Summary statistics
        valid_count = sum(1 for result in results if result["valid"])
        invalid_count = len(results) - valid_count
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Total files:** {len(results)}\n")
        f.write(f"- **Valid files:** {valid_count}\n")
        f.write(f"- **Invalid files:** {invalid_count}\n\n")
        
        # Valid files
        f.write(f"## Valid Files\n\n")
        valid_files = [result for result in results if result["valid"]]
        
        if valid_files:
            for result in valid_files:
                f.write(f"- {result['file']}\n")
        else:
            f.write("No valid files found.\n")
        
        f.write("\n")
        
        # Invalid files
        f.write(f"## Invalid Files\n\n")
        invalid_files = [result for result in results if not result["valid"]]
        
        if invalid_files:
            for result in invalid_files:
                f.write(f"### {result['file']}\n\n")
                
                if result["errors"]:
                    f.write("**Errors:**\n\n")
                    for error in result["errors"]:
                        f.write(f"- {error}\n")
                    f.write("\n")
                
                if result["warnings"]:
                    f.write("**Warnings:**\n\n")
                    for warning in result["warnings"]:
                        f.write(f"- {warning}\n")
                    f.write("\n")
        else:
            f.write("No invalid files found.\n")
    
    logger.info(f"Report generated at {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Validate model test files")
    parser.add_argument("--dir", help="Directory containing test files")
    parser.add_argument("--file", help="Specific test file to validate")
    parser.add_argument("--report", help="Generate a validation report")
    
    args = parser.parse_args()
    
    if not args.dir and not args.file:
        parser.error("Either --dir or --file must be specified")
    
    results = []
    
    if args.file:
        # Validate a specific file
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1
        
        result = validate_file(args.file)
        results.append(result)
    
    if args.dir:
        # Validate all files in a directory
        if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
            logger.error(f"Directory not found: {args.dir}")
            return 1
        
        results = validate_directory(args.dir)
    
    # Print results
    print_results(results)
    
    # Generate report if requested
    if args.report:
        generate_report(results, args.report)
    
    # Return non-zero exit code if any files are invalid
    return 0 if all(result["valid"] for result in results) else 1

if __name__ == "__main__":
    sys.exit(main())
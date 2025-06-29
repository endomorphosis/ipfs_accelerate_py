#!/usr/bin/env python3
"""
Validate the syntax and structure of generated test files.

This script verifies that test files generated from templates are valid.
It checks syntax, inheritance, required methods, and other structural elements.
"""

import os
import sys
import ast
import argparse
import logging
import json
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
log_filename = f"validate_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Required methods that should be in all test files
REQUIRED_METHODS = [
    "setUp",
    "test_model_loading"
]

# Required base class for all refactored test files
REQUIRED_BASE_CLASS = "ModelTest"

class TestFileValidator:
    """Validator for generated test files."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the validator."""
        self.verbose = verbose
        
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a single test file.
        
        Args:
            file_path: Path to the test file.
            
        Returns:
            Dictionary with validation results.
        """
        result = {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "valid_syntax": False,
            "valid_inheritance": False,
            "valid_methods": False,
            "missing_methods": [],
            "has_model_id": False,
            "model_id": None,
            "errors": [],
            "warnings": [],
            "test_class": None,
            "overall_valid": False
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            result["errors"].append(f"File does not exist: {file_path}")
            return result
        
        # Read file content
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
        except Exception as e:
            result["errors"].append(f"Error reading file: {e}")
            return result
        
        # Parse AST to check syntax
        try:
            tree = ast.parse(file_content)
            result["valid_syntax"] = True
        except SyntaxError as e:
            result["errors"].append(f"Syntax error on line {e.lineno}: {e.msg}")
            if self.verbose:
                logger.error(f"Syntax error in {file_path} on line {e.lineno}: {e.msg}")
            return result
        
        # Find all class definitions
        class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Find test class
        test_classes = [cls for cls in class_defs if cls.name.startswith('Test')]
        
        if not test_classes:
            result["errors"].append("No test class found (class starting with 'Test')")
            return result
        
        # Use the first test class
        test_class = test_classes[0]
        result["test_class"] = test_class.name
        
        # Check inheritance
        base_classes = []
        for base in test_class.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(base.attr)
        
        if REQUIRED_BASE_CLASS in base_classes:
            result["valid_inheritance"] = True
        else:
            result["errors"].append(
                f"Test class {test_class.name} does not inherit from {REQUIRED_BASE_CLASS}. "
                f"Base classes: {', '.join(base_classes)}"
            )
        
        # Get all method names
        method_names = []
        for node in ast.walk(test_class):
            if isinstance(node, ast.FunctionDef):
                method_names.append(node.name)
        
        # Check required methods
        missing_methods = []
        for required_method in REQUIRED_METHODS:
            if required_method not in method_names:
                missing_methods.append(required_method)
        
        if missing_methods:
            result["missing_methods"] = missing_methods
            result["errors"].append(f"Missing required methods: {', '.join(missing_methods)}")
        else:
            result["valid_methods"] = True
        
        # Find model_id assignment in setUp
        for node in ast.walk(test_class):
            if isinstance(node, ast.FunctionDef) and node.name == 'setUp':
                for sub_node in ast.walk(node):
                    if (isinstance(sub_node, ast.Assign) and 
                        len(sub_node.targets) == 1 and 
                        isinstance(sub_node.targets[0], ast.Attribute)):
                        
                        attr = sub_node.targets[0]
                        if (isinstance(attr.value, ast.Name) and 
                            attr.value.id == 'self' and 
                            attr.attr == 'model_id'):
                            
                            # Found self.model_id assignment
                            result["has_model_id"] = True
                            
                            # Try to extract the model ID value
                            if isinstance(sub_node.value, ast.Constant):
                                result["model_id"] = sub_node.value.value
                            elif isinstance(sub_node.value, ast.Str):  # For Python < 3.8
                                result["model_id"] = sub_node.value.s
        
        if not result["has_model_id"]:
            result["warnings"].append("self.model_id assignment not found in setUp method")
        
        # Check for test_ methods (other than the required ones)
        test_methods = [m for m in method_names if m.startswith('test_') 
                       and m not in REQUIRED_METHODS]
        
        if len(test_methods) == 0:
            result["warnings"].append("No additional test methods found beyond required ones")
        
        # Determine overall validity
        result["overall_valid"] = (
            result["valid_syntax"] and 
            result["valid_inheritance"] and 
            result["valid_methods"]
        )
        
        return result
    
    def validate_directory(self, directory: str, 
                          recursive: bool = True,
                          pattern: str = "test_*.py") -> Dict[str, Any]:
        """
        Validate all test files in a directory.
        
        Args:
            directory: Directory to scan for test files.
            recursive: Whether to scan subdirectories.
            pattern: File name pattern to match.
            
        Returns:
            Dictionary with validation results.
        """
        results = {
            "directory": directory,
            "pattern": pattern,
            "timestamp": datetime.now().isoformat(),
            "files": {},
            "summary": {
                "total_files": 0,
                "valid_files": 0,
                "invalid_files": 0,
                "files_with_warnings": 0
            }
        }
        
        # Get all test files
        if recursive:
            file_paths = list(Path(directory).rglob(pattern))
        else:
            file_paths = list(Path(directory).glob(pattern))
        
        # Sort file paths for consistent output
        file_paths.sort()
        
        # Initialize counters
        total_files = 0
        valid_files = 0
        invalid_files = 0
        files_with_warnings = 0
        
        # Validate each file
        for file_path in file_paths:
            total_files += 1
            file_result = self.validate_file(str(file_path))
            
            # Update counters
            if file_result["overall_valid"]:
                valid_files += 1
            else:
                invalid_files += 1
            
            if file_result["warnings"]:
                files_with_warnings += 1
            
            # Add to results
            relative_path = os.path.relpath(file_path, directory)
            results["files"][relative_path] = file_result
            
            # Log result
            if file_result["overall_valid"]:
                if file_result["warnings"]:
                    logger.info(f"✓ Valid with warnings: {relative_path}")
                    if self.verbose:
                        for warning in file_result["warnings"]:
                            logger.info(f"  ⚠ {warning}")
                else:
                    logger.info(f"✓ Valid: {relative_path}")
            else:
                logger.warning(f"✗ Invalid: {relative_path}")
                if self.verbose:
                    for error in file_result["errors"]:
                        logger.error(f"  ✗ {error}")
        
        # Update summary
        results["summary"]["total_files"] = total_files
        results["summary"]["valid_files"] = valid_files
        results["summary"]["invalid_files"] = invalid_files
        results["summary"]["files_with_warnings"] = files_with_warnings
        
        # Calculate percentages if there are any files
        if total_files > 0:
            results["summary"]["valid_percentage"] = (valid_files / total_files) * 100
        else:
            results["summary"]["valid_percentage"] = 0
        
        # Log summary
        logger.info(f"Validation complete: {valid_files}/{total_files} valid files "
                   f"({results['summary']['valid_percentage']:.1f}%)")
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Generate a human-readable validation report.
        
        Args:
            results: Validation results from validate_directory.
            output_file: Path to write the report to.
        """
        with open(output_file, 'w') as f:
            f.write("# Test File Validation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            f.write("## Summary\n\n")
            f.write(f"- Directory: `{results['directory']}`\n")
            f.write(f"- Pattern: `{results['pattern']}`\n")
            f.write(f"- Total files: {results['summary']['total_files']}\n")
            f.write(f"- Valid files: {results['summary']['valid_files']} ")
            if results['summary']['total_files'] > 0:
                f.write(f"({results['summary']['valid_percentage']:.1f}%)\n")
            else:
                f.write("(0%)\n")
            f.write(f"- Invalid files: {results['summary']['invalid_files']}\n")
            f.write(f"- Files with warnings: {results['summary']['files_with_warnings']}\n\n")
            
            # Write invalid files
            if results['summary']['invalid_files'] > 0:
                f.write("## Invalid Files\n\n")
                for file_path, file_result in results["files"].items():
                    if not file_result["overall_valid"]:
                        f.write(f"### {file_path}\n\n")
                        f.write(f"- Test class: `{file_result['test_class']}`\n")
                        f.write(f"- Model ID: `{file_result['model_id']}`\n")
                        f.write("\n**Errors:**\n\n")
                        for error in file_result["errors"]:
                            f.write(f"- {error}\n")
                        f.write("\n")
                f.write("\n")
            
            # Write files with warnings
            if results['summary']['files_with_warnings'] > 0:
                f.write("## Files with Warnings\n\n")
                for file_path, file_result in results["files"].items():
                    if file_result["warnings"]:
                        f.write(f"### {file_path}\n\n")
                        f.write(f"- Test class: `{file_result['test_class']}`\n")
                        f.write(f"- Model ID: `{file_result['model_id']}`\n")
                        f.write("\n**Warnings:**\n\n")
                        for warning in file_result["warnings"]:
                            f.write(f"- {warning}\n")
                        f.write("\n")
                f.write("\n")
            
            # Write valid files
            if results['summary']['valid_files'] > 0:
                f.write("## Valid Files\n\n")
                
                # Group by model architecture
                architectures = {}
                for file_path, file_result in results["files"].items():
                    if file_result["overall_valid"]:
                        # Try to determine architecture from file path
                        parts = file_path.split(os.sep)
                        if len(parts) >= 2 and parts[-2] == "models":
                            arch = "other"
                        elif len(parts) >= 3 and parts[-3] == "models":
                            arch = parts[-2]
                        else:
                            arch = "other"
                        
                        if arch not in architectures:
                            architectures[arch] = []
                        
                        architectures[arch].append((file_path, file_result))
                
                # Write each architecture group
                for arch, files in sorted(architectures.items()):
                    f.write(f"### {arch.capitalize()} Models\n\n")
                    for file_path, file_result in sorted(files, key=lambda x: x[0]):
                        f.write(f"- `{file_path}`: {file_result['test_class']}")
                        if file_result["model_id"]:
                            f.write(f" (Model: {file_result['model_id']})")
                        f.write("\n")
                    f.write("\n")
            
        logger.info(f"Validation report written to {output_file}")

def save_json_results(results: Dict[str, Any], output_file: str) -> None:
    """Save validation results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Validation results saved to {output_file}")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Validate test files")
    
    # Target specification
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--file", type=str, help="Validate a single test file")
    target_group.add_argument("--directory", type=str, help="Validate all test files in a directory")
    
    # Directory options
    directory_group = parser.add_argument_group("Directory Options")
    directory_group.add_argument("--pattern", type=str, default="test_*.py",
                               help="File name pattern to match (default: test_*.py)")
    directory_group.add_argument("--no-recursive", action="store_true",
                               help="Don't recursively scan subdirectories")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--json", type=str, help="Save results to JSON file")
    output_group.add_argument("--report", type=str, help="Generate human-readable report")
    output_group.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Create validator
    validator = TestFileValidator(verbose=args.verbose)
    
    # Validate target
    if args.file:
        # Validate single file
        result = validator.validate_file(args.file)
        
        # Print result
        if result["overall_valid"]:
            print(f"✓ File is valid: {args.file}")
            if result["warnings"]:
                print("Warnings:")
                for warning in result["warnings"]:
                    print(f"  ⚠ {warning}")
        else:
            print(f"✗ File is invalid: {args.file}")
            print("Errors:")
            for error in result["errors"]:
                print(f"  ✗ {error}")
        
        # Save JSON results if requested
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.json}")
        
        # Return appropriate exit code
        return 0 if result["overall_valid"] else 1
    
    elif args.directory:
        # Validate directory
        recursive = not args.no_recursive
        results = validator.validate_directory(
            args.directory,
            recursive=recursive,
            pattern=args.pattern
        )
        
        # Save JSON results if requested
        if args.json:
            save_json_results(results, args.json)
        
        # Generate report if requested
        if args.report:
            validator.generate_report(results, args.report)
        
        # Print summary
        print("\nValidation Summary:")
        print(f"Total files: {results['summary']['total_files']}")
        print(f"Valid files: {results['summary']['valid_files']} ", end="")
        if results['summary']['total_files'] > 0:
            print(f"({results['summary']['valid_percentage']:.1f}%)")
        else:
            print("(0%)")
        print(f"Invalid files: {results['summary']['invalid_files']}")
        print(f"Files with warnings: {results['summary']['files_with_warnings']}")
        
        # Return appropriate exit code
        return 0 if results['summary']['invalid_files'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
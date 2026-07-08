#!/usr/bin/env python3
"""
Script to verify the generated model test files for syntax and structure correctness.
"""

import os
import sys
import time
import json
import logging
import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common test patterns to check for
TEST_PATTERNS = [
    "import unittest",
    "class Test",
    "def test_",
    "self.assert",
    "mock.patch",
    "torch.device",
    "cuda.is_available"
]

def verify_file_syntax(file_path: str) -> Tuple[bool, str]:
    """
    Verify that a Python file has valid syntax.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        compile(source, file_path, 'exec')
        return True, "Valid syntax"
    except Exception as e:
        return False, f"Syntax error: {type(e).__name__}: {str(e)}"

def verify_file_imports(file_path: str) -> Tuple[bool, str]:
    """
    Verify that a Python file has valid imports.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (imports_valid, error_message)
    """
    # Get the directory of the file to add to path
    file_dir = os.path.dirname(os.path.abspath(file_path))
    
    try:
        # Create a new module spec
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if spec is None or spec.loader is None:
            return False, f"Could not create module spec for {file_path}"
        
        # Create the module
        module = importlib.util.module_from_spec(spec)
        
        # Add file directory to path temporarily
        sys.path.insert(0, file_dir)
        
        # Execute the module
        try:
            spec.loader.exec_module(module)
            return True, "Valid imports"
        except ImportError as e:
            return False, f"Import error: {str(e)}"
        finally:
            # Remove file directory from path
            if file_dir in sys.path:
                sys.path.remove(file_dir)
    except Exception as e:
        return False, f"Failed to load module: {type(e).__name__}: {str(e)}"

def verify_test_patterns(file_path: str) -> Tuple[bool, Set[str], str]:
    """
    Verify that a test file contains expected test patterns.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        Tuple of (has_patterns, found_patterns, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        found_patterns = set()
        for pattern in TEST_PATTERNS:
            if pattern in content:
                found_patterns.add(pattern)
        
        if len(found_patterns) >= 3:  # Require at least 3 patterns
            return True, found_patterns, "Found expected test patterns"
        else:
            missing = set(TEST_PATTERNS) - found_patterns
            return False, found_patterns, f"Missing expected test patterns: {', '.join(missing)}"
    except Exception as e:
        return False, set(), f"Error checking test patterns: {str(e)}"

def verify_model_test(file_path: str) -> Dict[str, Any]:
    """
    Verify a model test file for syntax, imports, and patterns.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        Dictionary with verification results
    """
    result = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "verification_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "syntax_valid": False,
        "imports_valid": False,
        "patterns_valid": False,
        "found_patterns": [],
        "errors": [],
        "overall_valid": False
    }
    
    # Step 1: Verify syntax
    syntax_valid, syntax_msg = verify_file_syntax(file_path)
    result["syntax_valid"] = syntax_valid
    if not syntax_valid:
        result["errors"].append(syntax_msg)
    
    # Step 2: Verify imports (only if syntax is valid)
    if syntax_valid:
        imports_valid, imports_msg = verify_file_imports(file_path)
        result["imports_valid"] = imports_valid
        if not imports_valid:
            result["errors"].append(imports_msg)
    
    # Step 3: Verify test patterns
    patterns_valid, found_patterns, patterns_msg = verify_test_patterns(file_path)
    result["patterns_valid"] = patterns_valid
    result["found_patterns"] = list(found_patterns)
    if not patterns_valid:
        result["errors"].append(patterns_msg)
    
    # Overall validity
    result["overall_valid"] = syntax_valid and patterns_valid  # Don't require import validity
    
    return result

def verify_directory(directory: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify all Python test files in a directory.
    
    Args:
        directory: Directory containing test files
        output_file: Optional file to write verification results
        
    Returns:
        Dictionary with verification summary
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return {"error": f"Directory not found: {directory}"}
    
    # Find all Python files in the directory
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file.startswith('test_'):
                python_files.append(os.path.join(root, file))
    
    if not python_files:
        logger.warning(f"No Python test files found in {directory}")
        return {"error": f"No Python test files found in {directory}"}
    
    # Verify each file
    results = []
    valid_count = 0
    
    for file_path in python_files:
        logger.info(f"Verifying {file_path}...")
        result = verify_model_test(file_path)
        results.append(result)
        
        if result["overall_valid"]:
            valid_count += 1
            logger.info(f"✅ Valid: {file_path}")
        else:
            errors = ', '.join(result["errors"])
            logger.warning(f"❌ Invalid: {file_path} - {errors}")
    
    # Generate summary
    summary = {
        "directory": directory,
        "total_files": len(python_files),
        "valid_files": valid_count,
        "invalid_files": len(python_files) - valid_count,
        "verification_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    
    # Write to output file if specified
    if output_file:
        logger.info(f"Writing verification results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Generate markdown report
    report_file = output_file.replace('.json', '.md') if output_file else os.path.join(directory, "verification_report.md")
    generate_verification_report(summary, report_file)
    
    return summary

def generate_verification_report(summary: Dict[str, Any], output_file: str) -> None:
    """
    Generate a markdown report from verification results.
    
    Args:
        summary: Verification summary
        output_file: Path to output markdown file
    """
    with open(output_file, 'w') as f:
        f.write(f"# Model Test Verification Report\n\n")
        f.write(f"**Date:** {summary['verification_time']}\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Directory:** {summary['directory']}\n")
        f.write(f"- **Total files:** {summary['total_files']}\n")
        f.write(f"- **Valid files:** {summary['valid_files']} ({round(summary['valid_files'] / summary['total_files'] * 100, 1)}%)\n")
        f.write(f"- **Invalid files:** {summary['invalid_files']} ({round(summary['invalid_files'] / summary['total_files'] * 100, 1)}%)\n\n")
        
        # List valid files
        valid_files = [r for r in summary['results'] if r['overall_valid']]
        if valid_files:
            f.write(f"## Valid Files ({len(valid_files)})\n\n")
            for result in valid_files:
                f.write(f"- ✅ `{result['file_name']}`\n")
            f.write("\n")
        
        # List invalid files with errors
        invalid_files = [r for r in summary['results'] if not r['overall_valid']]
        if invalid_files:
            f.write(f"## Invalid Files ({len(invalid_files)})\n\n")
            for result in invalid_files:
                f.write(f"### ❌ `{result['file_name']}`\n\n")
                f.write(f"- **Syntax valid:** {'✅' if result['syntax_valid'] else '❌'}\n")
                f.write(f"- **Imports valid:** {'✅' if result['imports_valid'] else '❌'}\n")
                f.write(f"- **Patterns valid:** {'✅' if result['patterns_valid'] else '❌'}\n")
                
                if result['found_patterns']:
                    f.write(f"- **Found patterns:** {', '.join(result['found_patterns'])}\n")
                
                if result['errors']:
                    f.write(f"- **Errors:**\n")
                    for error in result['errors']:
                        f.write(f"  - {error}\n")
                f.write("\n")
    
    logger.info(f"Verification report written to {output_file}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Verify generated model test files")
    parser.add_argument("--directory", required=True, help="Directory containing test files")
    parser.add_argument("--output", help="Output file for verification results")
    args = parser.parse_args()
    
    # Determine output file if not specified
    output_file = args.output
    if not output_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"verification_results_{timestamp}.json"
    
    # Verify the directory
    summary = verify_directory(args.directory, output_file)
    
    # Print summary
    print("\nVerification Summary:")
    print(f"- Directory: {summary['directory']}")
    print(f"- Total files: {summary['total_files']}")
    print(f"- Valid files: {summary['valid_files']} ({round(summary['valid_files'] / summary['total_files'] * 100, 1)}%)")
    print(f"- Invalid files: {summary['invalid_files']} ({round(summary['invalid_files'] / summary['total_files'] * 100, 1)}%)")
    print(f"- Report: {output_file.replace('.json', '.md')}")
    
    return 0 if summary["invalid_files"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
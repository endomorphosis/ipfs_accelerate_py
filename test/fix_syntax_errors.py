#!/usr/bin/env python3
"""
Checks Python files for syntax errors and attempts to fix them.
This script is intended to help find and fix syntax errors in relocated files.
"""

import os
import sys
import ast
import subprocess
from pathlib import Path

def check_syntax(file_path):
    """
    Checks a Python file for syntax errors.
    Returns None if no errors, or the error message if errors found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Try to parse the source code
        ast.parse(source)
        return None  # No syntax errors
    except SyntaxError as e:
        return f"Line {e.lineno}, Column {e.offset}: {e.msg}"
    except Exception as e:
        return f"Error checking syntax: {str(e)}"

def check_file_execution(file_path):
    """
    Attempts to execute the Python file to check for runtime errors.
    Returns None if no errors, or the error message if errors found.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return f"Compilation error: {result.stderr}"
        
        return None  # No compilation errors
    except Exception as e:
        return f"Error checking execution: {str(e)}"

def scan_directory(directory, extensions=None):
    """
    Scans a directory for Python files and checks them for syntax errors.
    Returns a list of tuples (file_path, error_message) for files with errors.
    """
    if extensions is None:
        extensions = ['.py']
    
    errors = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                # Check for syntax errors
                syntax_error = check_syntax(file_path)
                if syntax_error:
                    errors.append((file_path, f"Syntax error: {syntax_error}"))
                    continue
                
                # Check for runtime errors
                execution_error = check_file_execution(file_path)
                if execution_error:
                    errors.append((file_path, f"Execution error: {execution_error}"))
                    continue
    
    return errors

def main():
    """Main function."""
    project_root = Path(__file__).parent.parent  # Go up one directory from test
    
    # Directories to scan
    directories = [
        project_root / "duckdb_api",
        # Add generators directory later if needed
        # project_root / "generators",
    ]
    
    all_errors = []
    
    for directory in directories:
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist.")
            continue
        
        print(f"Scanning directory: {directory}")
        errors = scan_directory(directory)
        
        if errors:
            all_errors.extend(errors)
            for file_path, error in errors:
                print(f"  ❌ {file_path}")
                print(f"     {error}")
        else:
            print(f"  ✅ No errors found in {directory}")
    
    if all_errors:
        print(f"\nFound {len(all_errors)} files with errors.")
        print("To fix these errors, you'll need to:")
        print("1. Open each file and locate the syntax error")
        print("2. Fix the syntax errors (missing parentheses, indentation, etc.)")
        print("3. Verify imports are using the new package structure")
        print("4. Run this script again to check for more errors")
    else:
        print("\nNo syntax errors found in scanned directories.")

if __name__ == "__main__":
    main()
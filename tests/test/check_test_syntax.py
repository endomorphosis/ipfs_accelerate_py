#!/usr/bin/env python3
"""
Script to check syntax of all test files.
This file scans all test files in the specified directory and checks for Python syntax errors.
"""

import os
import sys
import ast
import argparse
import concurrent.futures
from typing import List, Dict, Optional, Tuple

class TestSyntaxChecker:
    """Checks syntax of Python test files."""
    
    def __init__(self, test_dir: str = "skills/fixed_tests"):
        """Initialize the syntax checker.
        
        Args:
            test_dir: Directory containing test files
        """
        self.test_dir = test_dir
        self.results = {"success": [], "failure": []}
        
    def find_test_files(self) -> List[str]:
        """Find all Python test files in the specified directory.
        
        Returns:
            List of paths to test files
        """
        test_files = []
        for root, _, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith('.py') and file.startswith('test_'):
                    test_files.append(os.path.join(root, file))
        return test_files
    
    def check_syntax(self, file_path: str) -> Tuple[str, bool, Optional[str]]:
        """Check syntax of a single Python file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple of (file_path, success, error_message)
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Try to parse the file as Python code
            ast.parse(content)
            return file_path, True, None
            
        except SyntaxError as e:
            return file_path, False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return file_path, False, str(e)
    
    def check_all_files(self, parallel: bool = True) -> Dict[str, List[str]]:
        """Check syntax of all test files.
        
        Args:
            parallel: If True, use parallel processing
            
        Returns:
            Dict with 'success' and 'failure' lists of file paths
        """
        test_files = self.find_test_files()
        print(f"Found {len(test_files)} test files to check")
        
        if parallel:
            # Use parallel processing for faster checks
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(self.check_syntax, test_files))
        else:
            # Sequential processing
            results = [self.check_syntax(file) for file in test_files]
        
        # Process results
        for file_path, success, error_message in results:
            if success:
                self.results["success"].append(file_path)
            else:
                self.results["failure"].append((file_path, error_message))
        
        return self.results
    
    def print_report(self) -> None:
        """Print a report of the syntax check results."""
        success_count = len(self.results["success"])
        failure_count = len(self.results["failure"])
        total = success_count + failure_count
        
        print("\n=== SYNTAX CHECK REPORT ===")
        print(f"Total files checked: {total}")
        print(f"Success: {success_count} ({success_count/total*100:.1f}%)")
        print(f"Failures: {failure_count} ({failure_count/total*100:.1f}%)")
        
        if failure_count > 0:
            print("\nFiles with syntax errors:")
            for file_path, error_message in self.results["failure"]:
                print(f"  - {os.path.basename(file_path)}: {error_message}")
        
        print("\nSyntax check completed.")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Check syntax of Python test files")
    parser.add_argument("--dir", type=str, default="skills/fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--sequential", action="store_true",
                        help="Run checks sequentially (not in parallel)")
    args = parser.parse_args()
    
    checker = TestSyntaxChecker(test_dir=args.dir)
    checker.check_all_files(parallel=not args.sequential)
    checker.print_report()
    
    # Return non-zero exit code if there are failures
    if len(checker.results["failure"]) > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Script to fix common syntax errors in Python test files.
This script addresses issues like:
- Unmatched parentheses
- Indentation errors 
- Missing code blocks after try statements
- And other syntax problems
"""

import os
import sys
import re
import ast
import argparse
from typing import List, Dict, Tuple, Optional, Any

class SyntaxFixer:
    """Fixes common syntax errors in Python test files."""
    
    def __init__(self, files_to_fix: List[str] = None, test_dir: str = "skills"):
        """Initialize the fixer.
        
        Args:
            files_to_fix: List of specific files to fix
            test_dir: Directory containing test files if no specific files are provided
        """
        self.files_to_fix = files_to_fix or []
        self.test_dir = test_dir
        self.results = {"fixed": [], "skipped": [], "errors": []}
        
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
    
    def check_syntax(self, file_path: str) -> Tuple[bool, str]:
        """Check if file has syntax errors.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple of (has_syntax_errors, error_message)
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Try to parse the file as Python code
            ast.parse(content)
            return False, "No syntax errors"
            
        except SyntaxError as e:
            return True, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return True, str(e)
    
    def fix_unmatched_parentheses(self, content: str) -> str:
        """Fix unmatched parentheses in content.
        
        Args:
            content: File content to fix
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        fixed_lines = []
        
        # Stack to track opening brackets
        stack = []
        line_indices = []  # Track which line each opening bracket is on
        
        # First pass - identify unmatched parentheses
        for i, line in enumerate(lines):
            j = 0
            while j < len(line):
                if line[j] in '({[':
                    stack.append(line[j])
                    line_indices.append(i)
                elif line[j] in ')}]':
                    if stack:
                        # Check if brackets match
                        if (line[j] == ')' and stack[-1] == '(') or \
                           (line[j] == '}' and stack[-1] == '{') or \
                           (line[j] == ']' and stack[-1] == '['):
                            stack.pop()
                            line_indices.pop()
                        else:
                            # Mismatched brackets - remove this closing bracket
                            line = line[:j] + line[j+1:]
                            j -= 1
                    else:
                        # Extra closing bracket - remove it
                        line = line[:j] + line[j+1:]
                        j -= 1
                j += 1
            
            fixed_lines.append(line)
        
        # Second pass - add missing closing brackets at the end of lines
        for _ in range(len(stack)):
            if stack and line_indices:
                bracket = stack.pop()
                line_idx = line_indices.pop()
                
                # Add corresponding closing bracket
                if bracket == '(':
                    fixed_lines[line_idx] += ')'
                elif bracket == '{':
                    fixed_lines[line_idx] += '}'
                elif bracket == '[':
                    fixed_lines[line_idx] += ']'
        
        return '\n'.join(fixed_lines)
    
    def fix_try_indentation_errors(self, content: str) -> str:
        """Fix indentation errors after try statements.
        
        Args:
            content: File content to fix
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            fixed_lines.append(line)
            
            # Check if this line has a try statement
            if re.search(r'^\s*try\s*:', line):
                # Look for the next line
                j = i + 1
                
                # Skip empty lines
                while j < len(lines) and not lines[j].strip():
                    fixed_lines.append(lines[j])
                    j += 1
                    i += 1
                
                if j < len(lines):
                    next_line = lines[j]
                    current_indent = len(line) - len(line.lstrip())
                    next_indent = len(next_line) - len(next_line.lstrip())
                    expected_indent = current_indent + 4  # 4 spaces per level
                    
                    # Check if indentation is incorrect (not properly indented)
                    if next_indent <= current_indent:
                        # Fix the indentation of the next line
                        spaces = ' ' * expected_indent
                        fixed_lines.append(spaces + next_line.lstrip())
                        i += 1
                    
            i += 1
            
        return '\n'.join(fixed_lines)
    
    def fix_indentation_levels(self, content: str) -> str:
        """Fix general indentation levels throughout the file.
        
        Args:
            content: File content to fix
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        fixed_lines = []
        
        # Track indentation stack (blocks that increase indentation)
        indent_stack = []
        current_indent = 0
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
                
            # Get current line indentation
            leading_spaces = len(line) - len(line.lstrip())
            
            # Check if this line should decrease indentation
            if leading_spaces < current_indent:
                # This line decreases indentation - pop from stack until we match
                while indent_stack and leading_spaces < current_indent:
                    indent_stack.pop()
                    current_indent -= 4
            
            # Check if line ends with colon (should increase indentation for next line)
            if line.rstrip().endswith(':'):
                indent_stack.append(current_indent)
                current_indent += 4
            
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def fix_try_except_blocks(self, content: str) -> str:
        """Specifically fix misaligned try/except blocks.
        
        Args:
            content: File content to fix
            
        Returns:
            Fixed content
        """
        # Replace patterns like:
        # try:
        # logger.info
        # 
        # With:
        # try:
        #     logger.info
        pattern = r'(^\s*try\s*:\s*$\n)(\s*)(\S)'
        
        def fix_indent(match):
            try_line = match.group(1)
            existing_indent = match.group(2)
            next_content = match.group(3)
            
            # Calculate proper indentation (4 spaces more than the try statement)
            try_indent = len(try_line) - len(try_line.lstrip())
            proper_indent = ' ' * (try_indent + 4)
            
            return f"{try_line}{proper_indent}{next_content}"
        
        # Apply the fix using regex
        fixed_content = re.sub(pattern, fix_indent, content, flags=re.MULTILINE)
        return fixed_content
    
    def fix_file(self, file_path: str) -> Tuple[bool, str]:
        """Fix syntax errors in a single file.
        
        Args:
            file_path: Path to the file to fix
            
        Returns:
            Tuple of (success, message)
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Check if file has syntax errors
            has_errors, error_msg = self.check_syntax(file_path)
            if not has_errors:
                return False, "No syntax errors to fix"
            
            # Apply fixers in sequence
            original_content = content
            
            # Fix unmatched parentheses
            content = self.fix_unmatched_parentheses(content)
            
            # Look for specific patterns in the error message
            if "expected an indented block" in error_msg and "try" in error_msg:
                # This is a try statement indentation error
                content = self.fix_try_except_blocks(content)
                content = self.fix_try_indentation_errors(content)
            
            # If no changes were made, try more comprehensive fixes
            if content == original_content:
                content = self.fix_try_except_blocks(content)
                content = self.fix_try_indentation_errors(content)
                content = self.fix_indentation_levels(content)
            
            # If no changes were made, skip
            if content == original_content:
                return False, "No changes needed"
            
            # Check if fixes resolved the syntax errors
            try:
                ast.parse(content)
                
                # Write fixed content
                with open(file_path, 'w') as file:
                    file.write(content)
                
                return True, f"Fixed syntax errors: {error_msg}"
                
            except SyntaxError as e:
                # Failed to fix all errors
                return False, f"Could not fix all syntax errors: Line {e.lineno}: {e.msg}"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def fix_specific_try_statement(self, file_path: str) -> Tuple[bool, str]:
        """Directly target and fix indentation errors in try statements.
        
        Args:
            file_path: Path to the file to fix
            
        Returns:
            Tuple of (success, message)
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Find the problematic line
            problematic_line = -1
            problematic_try = -1
            
            for i, line in enumerate(lines):
                if line.strip() == 'try:':
                    if i+1 < len(lines) and not lines[i+1].strip().startswith(' '):
                        problematic_try = i
                        problematic_line = i+1
                        break
            
            if problematic_line >= 0:
                # Add proper indentation to the line after try:
                try_indent = len(lines[problematic_try]) - len(lines[problematic_try].lstrip())
                leading_spaces = ' ' * (try_indent + 4)  # Try block with 4 more spaces
                
                # Add proper indentation
                lines[problematic_line] = leading_spaces + lines[problematic_line].lstrip()
                
                # Write the fixed content
                with open(file_path, 'w') as file:
                    file.writelines(lines)
                
                return True, f"Fixed try-statement indentation at line {problematic_line+1}"
            
            return False, "No problematic try statements found"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def process_file(self, file_path: str) -> Tuple[bool, str]:
        """Process a single file, trying multiple fix strategies.
        
        Args:
            file_path: Path to the file to fix
            
        Returns:
            Tuple of (success, message)
        """
        # First try the general fix
        success, message = self.fix_file(file_path)
        if success:
            return success, message
        
        # If that didn't work, try the specific try statement fix
        success, message = self.fix_specific_try_statement(file_path)
        if success:
            return success, message
        
        # If all else fails, try a direct text-based approach
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Try to identify specific pattern: try:[\n]unindented_line
            pattern = r'(try\s*:)\s*\n\s*(\S)'
            
            def add_indent(match):
                return f"{match.group(1)}\n    {match.group(2)}"
            
            fixed_content = re.sub(pattern, add_indent, content)
            
            if fixed_content != content:
                # Write the fixed content
                with open(file_path, 'w') as file:
                    file.write(fixed_content)
                
                # Verify the fix worked
                try:
                    ast.parse(fixed_content)
                    return True, "Fixed direct try-except pattern"
                except SyntaxError:
                    # Didn't work, restore original
                    with open(file_path, 'w') as file:
                        file.write(content)
            
            return False, "Could not fix the file using any method"
            
        except Exception as e:
            return False, f"Error in direct fix: {str(e)}"
    
    def fix_all_files(self) -> Dict:
        """Fix syntax errors in all files.
        
        Returns:
            Dict with results
        """
        files_to_process = self.files_to_fix or self.find_test_files()
        print(f"Found {len(files_to_process)} files to check")
        
        for file_path in files_to_process:
            success, message = self.process_file(file_path)
            
            if success:
                self.results["fixed"].append((file_path, message))
            elif "Error" in message:
                self.results["errors"].append((file_path, message))
            else:
                self.results["skipped"].append((file_path, message))
        
        return self.results
    
    def print_report(self) -> None:
        """Print a report of the fix results."""
        fixed_count = len(self.results["fixed"])
        skipped_count = len(self.results["skipped"])
        error_count = len(self.results["errors"])
        total = fixed_count + skipped_count + error_count
        
        print("\n=== SYNTAX FIX REPORT ===")
        print(f"Total files processed: {total}")
        print(f"Fixed: {fixed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        
        if fixed_count > 0:
            print("\nFiles fixed:")
            for file_path, message in self.results["fixed"]:
                print(f"  - {os.path.basename(file_path)}: {message}")
        
        if error_count > 0:
            print("\nFiles with errors:")
            for file_path, message in self.results["errors"]:
                print(f"  - {os.path.basename(file_path)}: {message}")
        
        print("\nFix completed.")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Fix syntax errors in Python test files")
    parser.add_argument("--dir", type=str, default="skills",
                       help="Directory containing test files")
    parser.add_argument("--files", nargs='+', help="Specific files to fix")
    parser.add_argument("--manual", action="store_true", help="Manually fix try-except blocks")
    args = parser.parse_args()
    
    fixer = SyntaxFixer(files_to_fix=args.files, test_dir=args.dir)
    fixer.fix_all_files()
    fixer.print_report()
    
    # Return non-zero exit code if there are errors
    if len(fixer.results["errors"]) > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
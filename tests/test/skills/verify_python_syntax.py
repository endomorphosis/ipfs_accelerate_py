#!/usr/bin/env python3

"""
Verify Python syntax for all generated test files.

This script:
1. Scans all test files in a directory
2. Checks each file for valid Python syntax
3. Attempts to fix common syntax errors
4. Reports on success/failure

Usage:
    python verify_python_syntax.py [--directory DIRECTORY] [--fix] [--verbose]
"""

import os
import sys
import re
import argparse
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_syntax(file_path):
    """
    Check if a file has valid Python syntax.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        tuple: (is_valid, error_message, line_number, offset)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify syntax
        compile(content, file_path, 'exec')
        return True, None, None, None
    except SyntaxError as e:
        return False, str(e), e.lineno, e.offset
    except Exception as e:
        return False, str(e), None, None

def fix_common_syntax_errors(file_path):
    """
    Fix common syntax errors in a file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if fixes were applied, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Common fixes
        fixed_content = content
        
        # Fix 1: Extra quotes ("""")
        fixed_content = fixed_content.replace('""""', '"""')
        
        # Fix 2: Unclosed triple quotes - find and fix
        triple_quotes_count = fixed_content.count('"""')
        if triple_quotes_count % 2 != 0:
            logger.info(f"Odd number of triple quotes found in {file_path}, fixing...")
            # Try to find the problem location
            lines = fixed_content.split('\n')
            found_docstring = False
            line_num = 0
            
            for i, line in enumerate(lines):
                if '"""' in line:
                    if found_docstring:
                        found_docstring = False
                    else:
                        found_docstring = True
                        line_num = i
                
                # If we found an open docstring and there are extra quotes, fix them
                if found_docstring and '""""' in line:
                    lines[i] = line.replace('""""', '"""')
                    logger.info(f"Fixed extra quotes on line {i+1}")
            
            fixed_content = '\n'.join(lines)
        
        # Fix 3: Fix colons without spaces (if:x vs. if: x)
        fixed_content = re.sub(r'(if|elif|else|for|while|try|except|finally|with|def|class):([^\s])', r'\1: \2', fixed_content)
        
        # Fix 4: Fix indentation issues (detect inconsistent indentation patterns)
        lines = fixed_content.split('\n')
        indent_pattern = re.compile(r'^(\s+)')
        indents = []
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                match = indent_pattern.match(line)
                if match:
                    indent = match.group(1)
                    if indent not in indents:
                        indents.append(indent)
        
        # If we have mixed tabs and spaces, convert tabs to spaces
        if any('\t' in indent for indent in indents) and any(' ' in indent for indent in indents):
            logger.info(f"Mixed tabs and spaces found in {file_path}, fixing...")
            fixed_content = fixed_content.replace('\t', '    ')
        
        # Fix 5: Remove stray backslashes (common in formatting errors)
        fixed_content = re.sub(r'\\(?!\n|\r|t|"|\')', '', fixed_content)
        
        # Fix 6: Fix syntax error in try: statements (try:_x vs try: x)
        fixed_content = re.sub(r'try:_', 'try: ', fixed_content)
        fixed_content = re.sub(r'try::_', 'try: ', fixed_content)
        fixed_content = re.sub(r'try::([^\s])', r'try: \1', fixed_content)
        
        # Fix 7: Fix except: statements (except::x vs except: x)
        fixed_content = re.sub(r'except::([^\s])', r'except: \1', fixed_content)
        
        # Fix 8: Fix lists with double commas ([1,,2] vs [1,2])
        fixed_content = re.sub(r'\[([^]]*),\s*,([^]]*)\]', r'[\1,\2]', fixed_content)
        
        # Fix 9: Fix missing closing parentheses in function calls
        # This is more complex, so we'll use a simple heuristic
        for i in range(fixed_content.count('(') - fixed_content.count(')')):
            if fixed_content.endswith('\n'):
                fixed_content = fixed_content[:-1] + ')\n'
            else:
                fixed_content += ')'
        
        # Check if any changes were made
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            logger.info(f"Applied syntax fixes to {file_path}")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error fixing syntax in {file_path}: {e}")
        return False

def verify_files_in_directory(directory, fix=False, verbose=False):
    """
    Verify all Python files in a directory.
    
    Args:
        directory: Directory containing Python files
        fix: Whether to attempt to fix syntax errors
        verbose: Whether to print detailed information
        
    Returns:
        tuple: (passed_files, failed_files)
    """
    directory_path = Path(directory)
    
    # Find all Python files
    python_files = list(directory_path.glob("**/*.py"))
    
    if verbose:
        logger.info(f"Found {len(python_files)} Python files in {directory}")
    
    passed_files = []
    failed_files = []
    
    for file_path in python_files:
        relative_path = file_path.relative_to(directory_path) if directory_path in file_path.parents else file_path
        
        is_valid, error_message, line_number, offset = check_syntax(file_path)
        
        if is_valid:
            passed_files.append(str(relative_path))
            if verbose:
                logger.info(f"✅ {relative_path}: Syntax is valid")
        else:
            if fix:
                logger.info(f"❌ {relative_path}: Syntax error - attempting to fix")
                fix_common_syntax_errors(file_path)
                
                # Check again after fixing
                is_valid, error_message, line_number, offset = check_syntax(file_path)
                
                if is_valid:
                    passed_files.append(str(relative_path))
                    logger.info(f"✅ {relative_path}: Fixed successfully")
                else:
                    failed_files.append((str(relative_path), error_message, line_number, offset))
                    logger.error(f"❌ {relative_path}: Failed to fix - {error_message} (line {line_number}, offset {offset})")
            else:
                failed_files.append((str(relative_path), error_message, line_number, offset))
                logger.error(f"❌ {relative_path}: Syntax error - {error_message} (line {line_number}, offset {offset})")
    
    return passed_files, failed_files

def main():
    """Main function to verify Python syntax."""
    parser = argparse.ArgumentParser(description="Verify Python syntax for files in a directory")
    parser.add_argument("--directory", type=str, default="fixed_tests", help="Directory containing Python files")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix syntax errors")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    # Verify the directory exists
    if not os.path.isdir(args.directory):
        logger.error(f"Directory not found: {args.directory}")
        return 1
    
    # Verify files
    passed_files, failed_files = verify_files_in_directory(args.directory, fix=args.fix, verbose=args.verbose)
    
    # Print summary
    logger.info("\nSYNTAX VERIFICATION SUMMARY")
    logger.info(f"Passed: {len(passed_files)} files")
    logger.info(f"Failed: {len(failed_files)} files")
    
    if failed_files:
        logger.info("\nFailed files:")
        for file_path, error_message, line_number, offset in failed_files:
            logger.info(f"- {file_path}: {error_message} (line {line_number}, offset {offset})")
    
    return 0 if not failed_files else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3

import os
import sys
import re
import glob
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_unterminated_triple_quotes(content):
    """Fix unterminated triple quotes in a file."""
    # Check if the number of triple quotes is uneven
    triple_quotes_count = content.count('"""')
    single_triple_quotes_count = content.count("'''")
    
    # No issue if counts are even
    if triple_quotes_count % 2 == 0 and single_triple_quotes_count % 2 == 0:
        return content, 0
    
    lines = content.split('\n')
    changes = 0
    
    # Fix double triple quotes
    if triple_quotes_count % 2 != 0:
        logger.info(f"Found uneven number of triple quotes: {triple_quotes_count}")
        
        # Track open/close states
        open_positions = []
        
        for i, line in enumerate(lines):
            # Count occurrences in this line
            line_triple_quotes = line.count('"""')
            
            # Skip if even number in this line (open and close on same line)
            if line_triple_quotes > 0 and line_triple_quotes % 2 == 0:
                continue
                
            # Handle lines with odd number of triple quotes
            for j in range(line_triple_quotes):
                pos = line.find('"""', 0 if j == 0 else line.find('"""') + 3)
                if len(open_positions) == 0:
                    # Opening quote
                    open_positions.append((i, pos, len(line[:pos].rstrip())))
                else:
                    # Closing quote
                    open_positions.pop()
        
        # If we still have unclosed quotes, add closing quotes
        if open_positions:
            for line_num, pos, indent in open_positions:
                # Find a good place to add the closing quote
                # Try to find the next non-docstring line
                next_code_line = None
                for j in range(line_num + 1, len(lines)):
                    if (lines[j].strip() and 
                        not lines[j].strip().startswith('#') and 
                        not '"""' in lines[j]):
                        next_code_line = j
                        break
                
                if next_code_line is not None:
                    # Add closing quotes before the next code line
                    indent_str = ' ' * indent
                    lines.insert(next_code_line, f"{indent_str}\"\"\"")
                    logger.info(f"Added closing triple quotes before line {next_code_line + 1}")
                    changes += 1
                else:
                    # Add to the end of the file
                    indent_str = ' ' * indent
                    lines.append(f"{indent_str}\"\"\"")
                    logger.info(f"Added closing triple quotes at end of file")
                    changes += 1
    
    # Similar process for single triple quotes
    if single_triple_quotes_count % 2 != 0:
        logger.info(f"Found uneven number of single triple quotes: {single_triple_quotes_count}")
        
        # Track open/close states
        open_positions = []
        
        for i, line in enumerate(lines):
            # Count occurrences in this line
            line_triple_quotes = line.count("'''")
            
            # Skip if even number in this line (open and close on same line)
            if line_triple_quotes > 0 and line_triple_quotes % 2 == 0:
                continue
                
            # Handle lines with odd number of triple quotes
            for j in range(line_triple_quotes):
                pos = line.find("'''", 0 if j == 0 else line.find("'''") + 3)
                if len(open_positions) == 0:
                    # Opening quote
                    open_positions.append((i, pos, len(line[:pos].rstrip())))
                else:
                    # Closing quote
                    open_positions.pop()
        
        # If we still have unclosed quotes, add closing quotes
        if open_positions:
            for line_num, pos, indent in open_positions:
                # Find a good place to add the closing quote
                # Try to find the next non-docstring line
                next_code_line = None
                for j in range(line_num + 1, len(lines)):
                    if (lines[j].strip() and 
                        not lines[j].strip().startswith('#') and 
                        not "'''" in lines[j]):
                        next_code_line = j
                        break
                
                if next_code_line is not None:
                    # Add closing quotes before the next code line
                    indent_str = ' ' * indent
                    lines.insert(next_code_line, f"{indent_str}'''")
                    logger.info(f"Added closing single triple quotes before line {next_code_line + 1}")
                    changes += 1
                else:
                    # Add to the end of the file
                    indent_str = ' ' * indent
                    lines.append(f"{indent_str}'''")
                    logger.info(f"Added closing single triple quotes at end of file")
                    changes += 1
    
    return '\n'.join(lines), changes

def fix_try_blocks(content):
    """Fix try/except blocks with missing indentation."""
    lines = content.split('\n')
    changes = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Look for try blocks with missing indentation
        if line.strip() == 'try:':
            indent = len(line) - len(line.lstrip())
            
            # Look ahead to next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
                
            # If we found a non-empty line
            if j < len(lines):
                next_line = lines[j]
                next_indent = len(next_line) - len(next_line.lstrip())
                
                # Check if next line needs indentation
                if not next_line.strip().startswith(('except', 'finally')) and next_indent <= indent:
                    # Fix indentation
                    lines[j] = ' ' * (indent + 4) + next_line.lstrip()
                    logger.info(f"Fixed missing indentation after try: on line {i+1}")
                    changes += 1
        
        # Look for except blocks with missing indentation
        elif line.strip().startswith('except ') and line.strip().endswith(':'):
            indent = len(line) - len(line.lstrip())
            
            # Look ahead to next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
                
            # If we found a non-empty line
            if j < len(lines):
                next_line = lines[j]
                next_indent = len(next_line) - len(next_line.lstrip())
                
                # Check if next line needs indentation
                if not next_line.strip().startswith(('except', 'finally', 'try:')) and next_indent <= indent:
                    # Fix indentation
                    lines[j] = ' ' * (indent + 4) + next_line.lstrip()
                    logger.info(f"Fixed missing indentation after except: on line {i+1}")
                    changes += 1
                    
        i += 1
    
    return '\n'.join(lines), changes

def fix_syntax_file(file_path):
    """Fix syntax issues in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        original_content = content
        total_changes = 0
        
        # Fix unterminated triple quotes
        content, changes = fix_unterminated_triple_quotes(content)
        total_changes += changes
        
        # Fix try/except blocks
        content, changes = fix_try_blocks(content)
        total_changes += changes
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Fixed {total_changes} issues in {file_path}")
            
            # Validate syntax
            try:
                compile(content, file_path, 'exec')
                logger.info(f"✅ {file_path}: Syntax is valid after fixes")
                return True
            except SyntaxError as e:
                logger.error(f"❌ {file_path}: Syntax errors remain: {e}")
                return False
        else:
            logger.info(f"No issues found in {file_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix syntax issues in Python files")
    parser.add_argument("--file", type=str, help="Process a single file")
    parser.add_argument("--directory", type=str, help="Process all Python files in directory")
    parser.add_argument("--pattern", type=str, default="*.py", help="File pattern to match (default: *.py)")
    
    args = parser.parse_args()
    
    if not (args.file or args.directory):
        parser.error("Either --file or --directory must be specified")
    
    success_count = 0
    fail_count = 0
    
    if args.file:
        if fix_syntax_file(args.file):
            success_count += 1
        else:
            fail_count += 1
    
    if args.directory:
        files = glob.glob(os.path.join(args.directory, args.pattern))
        logger.info(f"Found {len(files)} Python files in {args.directory}")
        
        for file_path in files:
            if fix_syntax_file(file_path):
                success_count += 1
            else:
                fail_count += 1
    
    logger.info(f"Processed {success_count + fail_count} files: {success_count} fixed successfully, {fail_count} with errors")
    
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
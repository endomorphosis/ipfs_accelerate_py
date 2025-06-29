#!/usr/bin/env python3

"""
Fix syntax and indentation issues in template files for test generation.

This script:
1. Reads template files from the templates directory
2. Fixes indentation issues, especially in try/except blocks
3. Fixes syntax errors like unclosed quotes
4. Ensures proper spacing between methods
5. Handles special cases for nested blocks
6. Writes the fixed templates back to the files
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_syntax_errors(content):
    """Fix common syntax errors like unterminated string literals."""
    # Fix extra quotes ("""")
    content = content.replace('""""', '"""')
    
    # Check for unclosed triple quotes
    triple_quotes_count = content.count('"""')
    if triple_quotes_count % 2 != 0:
        logger.info(f"Odd number of triple quotes found: {triple_quotes_count}, fixing...")
        # Try to find the problem location
        lines = content.split('\n')
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
        
        content = '\n'.join(lines)
    
    return content

def fix_try_except_blocks(content):
    """Fix try/except block indentation specifically."""
    lines = content.split('\n')
    fixed_lines = []
    in_try_block = False
    in_except_block = False
    try_indent = ""
    except_indent = ""
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            continue
        
        # Handle try blocks
        if stripped == "try:":
            in_try_block = True
            try_indent = line[:line.find("try:")]
            fixed_lines.append(line)
            continue
            
        # Handle except blocks
        if stripped.startswith("except") and stripped.endswith(":"):
            in_try_block = False
            in_except_block = True
            except_indent = line[:line.find("except")]
            fixed_lines.append(line)
            continue
            
        # Handle content inside try block
        if in_try_block:
            # Add proper indentation (4 spaces)
            indent = try_indent + "    "
            if not line.startswith(indent) and line.strip():
                fixed_line = indent + stripped
                fixed_lines.append(fixed_line)
                logger.debug(f"Fixed try block indentation at line {i+1}")
            else:
                fixed_lines.append(line)
            continue
                
        # Handle content inside except block
        if in_except_block:
            # Add proper indentation (4 spaces)
            indent = except_indent + "    "
            if not line.startswith(indent) and line.strip():
                # If line starts a new block, this is the end of the except block
                if stripped.endswith(":") and not stripped.startswith(("if", "for", "while", "try", "with")):
                    in_except_block = False
                    fixed_lines.append(line)
                else:
                    fixed_line = indent + stripped
                    fixed_lines.append(fixed_line)
                    logger.debug(f"Fixed except block indentation at line {i+1}")
            else:
                fixed_lines.append(line)
                # If line is properly indented and starts a new block, we're out of the except block
                if line.startswith(except_indent) and not line.startswith(indent) and line.strip():
                    in_except_block = False
            continue
            
        # Regular line
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_blank_lines(content):
    """Fix excessive blank lines."""
    # Replace multiple blank lines with just two
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Ensure one blank line between methods
    content = re.sub(r'(\n\s+def [^\n]+:\s*\n)(\s+def )', r'\1\n\2', content)
    return content

def fix_method_boundaries(content):
    """Fix method boundaries to ensure proper spacing and indentation."""
    # Fix common issues with return statements followed by method declarations
    content = re.sub(r'(\s+)return ([^"\n]+)\n(\s+)def ', r'\1return \2\n\n\3def ', content)
    content = re.sub(r'(\s+)return ([^"\n]+)\s+def ', r'\1return \2\n\n    def ', content)
    
    # First add proper spacing between methods for various return patterns
    content = content.replace("        return results\n    def ", "        return results\n\n    def ")
    content = content.replace("        return None\n    def ", "        return None\n\n    def ")
    content = content.replace("        return path\n    def ", "        return path\n\n    def ")
    content = content.replace("        return True\n    def ", "        return True\n\n    def ")
    content = content.replace("        return False\n    def ", "        return False\n\n    def ")
    
    # Make sure __init__ has correct spacing after it
    content = content.replace("        self.performance_stats = {}\n    def ", "        self.performance_stats = {}\n\n    def ")
    content = content.replace("        self.examples = []\n    def ", "        self.examples = []\n\n    def ")
        
    # Place all method declarations at the right indentation level
    content = re.sub(r'(\s+)def (\w+)\(', r'    def \2(', content)
    
    # Fix any other top-level methods (save_results, main, etc.)
    content = re.sub(r'^(\s*)def ([^(]+)\(', r'def \2(', content, flags=re.MULTILINE)
    
    # Ensure proper line breaks after class declarations
    content = re.sub(r'class (\w+):(.*?)\n(\s+)def', r'class \1:\2\n\n\3def', content, flags=re.DOTALL)
    
    return content

def fix_template(template_path):
    """Fix syntax and indentation issues in a template file."""
    try:
        logger.info(f"Processing template file: {template_path}")
        
        # Read the file
        with open(template_path, 'r') as f:
            content = f.read()
            
        # Store original content for comparison
        original_content = content
        
        # Apply fixes
        content = fix_syntax_errors(content)
        content = fix_try_except_blocks(content)
        content = fix_blank_lines(content)
        content = fix_method_boundaries(content)
        
        # Check if content was changed
        if content != original_content:
            logger.info(f"Fixed issues in {template_path}")
            
            # Create backup of original file
            backup_path = f"{template_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(original_content)
            logger.info(f"Created backup at {backup_path}")
            
            # Write fixed content
            with open(template_path, 'w') as f:
                f.write(content)
                
            # Validate syntax
            try:
                compile(content, template_path, 'exec')
                logger.info(f"✅ Syntax is valid for {template_path}")
                return True
            except SyntaxError as e:
                logger.error(f"❌ Syntax error in fixed file: {e}")
                if hasattr(e, 'lineno') and e.lineno is not None:
                    lines = content.split('\n')
                    line_no = e.lineno - 1  # 0-based index
                    if 0 <= line_no < len(lines):
                        logger.error(f"Problematic line {e.lineno}: {lines[line_no].rstrip()}")
                return False
        else:
            logger.info(f"No issues found in {template_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error processing {template_path}: {e}")
        return False

def fix_all_templates(templates_dir):
    """Fix all template files in the specified directory."""
    templates_dir = Path(templates_dir)
    
    if not templates_dir.exists() or not templates_dir.is_dir():
        logger.error(f"Templates directory not found: {templates_dir}")
        return False
        
    logger.info(f"Processing templates in: {templates_dir}")
    
    # Find all Python files in the templates directory
    template_files = list(templates_dir.glob("*.py"))
    
    if not template_files:
        logger.warning(f"No template files found in {templates_dir}")
        return False
        
    logger.info(f"Found {len(template_files)} template files")
    
    success_count = 0
    for template_file in template_files:
        if fix_template(template_file):
            success_count += 1
            
    logger.info(f"Successfully fixed {success_count} of {len(template_files)} template files")
    return success_count == len(template_files)

def check_indentation_issues(content):
    """Check for common indentation issues and report them."""
    issues = []
    
    # Check for try blocks without proper indentation
    try_blocks = re.finditer(r'(\s+)try:\s*\n\s*([^\s])', content)
    for match in try_blocks:
        line_num = content[:match.start()].count('\n') + 1
        issues.append(f"Line {line_num}: try block without proper indentation")
    
    # Check for except blocks without proper indentation
    except_blocks = re.finditer(r'(\s+)except.*?:\s*\n\s*([^\s])', content)
    for match in except_blocks:
        line_num = content[:match.start()].count('\n') + 1
        issues.append(f"Line {line_num}: except block without proper indentation")
    
    # Check for class method indentation issues
    methods = re.finditer(r'(\s+)def\s+(\w+)\(', content)
    for match in methods:
        indent = match.group(1)
        method_name = match.group(2)
        if method_name != '__init__' and not indent.startswith('    '):
            line_num = content[:match.start()].count('\n') + 1
            issues.append(f"Line {line_num}: method {method_name} with improper indentation")
    
    return issues

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Fix syntax and indentation issues in template files")
    parser.add_argument("--templates-dir", type=str, default="templates", help="Directory containing template files")
    parser.add_argument("--file", type=str, help="Fix a specific template file")
    parser.add_argument("--dry-run", action="store_true", help="Check issues without modifying files")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Handle single file mode
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1
            
        if args.dry_run:
            # Just check issues
            with open(args.file, 'r') as f:
                content = f.read()
            issues = check_indentation_issues(content)
            if issues:
                print(f"Found {len(issues)} issues in {args.file}:")
                for issue in issues:
                    print(f"- {issue}")
            else:
                print(f"No issues found in {args.file}")
            return 0
        else:
            # Fix the file
            success = fix_template(args.file)
            return 0 if success else 1
    
    # Handle directory mode
    if args.dry_run:
        # Check issues in all templates
        all_issues = {}
        templates_dir = Path(args.templates_dir)
        
        if not templates_dir.exists() or not templates_dir.is_dir():
            logger.error(f"Templates directory not found: {templates_dir}")
            return 1
            
        template_files = list(templates_dir.glob("*.py"))
        
        for template_file in template_files:
            with open(template_file, 'r') as f:
                content = f.read()
            issues = check_indentation_issues(content)
            if issues:
                all_issues[template_file.name] = issues
                
        if all_issues:
            print(f"Found issues in {len(all_issues)} template files:")
            for file_name, issues in all_issues.items():
                print(f"\n{file_name} ({len(issues)} issues):")
                for issue in issues:
                    print(f"- {issue}")
        else:
            print("No issues found in template files")
        return 0
    else:
        # Fix all templates
        success = fix_all_templates(args.templates_dir)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
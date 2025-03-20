#!/usr/bin/env python3
"""
Complete indentation fixer for HuggingFace test files.

This script combines all the best approaches from our previous work:
1. The apply_indentation helper function from test_generator_fixed.py
2. Method boundary fixing from fix_method_boundaries()
3. Comprehensive indentation normalizing from fix_file_indentation.py
4. Code validation with Python's compile() function

Usage:
    python complete_indentation_fix.py <file_path> [--verify]
"""

import sys
import re
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"indentation_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def apply_indentation(code, base_indent=0):
    """
    Apply consistent indentation to code blocks.
    
    Args:
        code: The code string to indent
        base_indent: The base indentation level (number of spaces)
        
    Returns:
        Properly indented code string
    """
    # Split the code into lines
    lines = code.strip().split('\n')
    
    # Determine the minimum indentation of non-empty lines
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Skip empty lines
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)
    
    # If no indentation found, set to 0
    if min_indent == float('inf'):
        min_indent = 0
    
    # Remove the minimum indentation from all lines and add the base indentation
    indented_lines = []
    indent_spaces = ' ' * base_indent
    
    for line in lines:
        if line.strip():  # If not an empty line
            # Remove original indentation and add new base indentation
            indented_line = indent_spaces + line[min_indent:]
            indented_lines.append(indented_line)
        else:
            # For empty lines, just add base indentation
            indented_lines.append(indent_spaces)
    
    # Join the lines back into a single string
    return '\n'.join(indented_lines)

def fix_method_boundaries(content):
    """Fix method boundaries to ensure proper spacing and indentation."""
    # First add proper spacing between methods
    content = content.replace("        return results\n    def ", "        return results\n\n    def ")
    
    # Make sure __init__ has correct spacing after it
    content = content.replace("        self.performance_stats = {}\n    def ", "        self.performance_stats = {}\n\n    def ")
        
    # Place all method declarations at the right indentation level
    content = re.sub(r'(\s+)def test_pipeline\(', r'    def test_pipeline(', content)
    content = re.sub(r'(\s+)def test_from_pretrained\(', r'    def test_from_pretrained(', content)
    content = re.sub(r'(\s+)def run_tests\(', r'    def run_tests(', content)
    
    # Fix any other methods (save_results, main, etc.)
    content = re.sub(r'^(\s*)def ([^(]+)\(', r'def \2(', content, flags=re.MULTILINE)
    
    return content

def extract_method(content, method_name):
    """
    Extract a method from the class content
    
    Args:
        content: The class content
        method_name: The name of the method to extract
        
    Returns:
        The extracted method text
    """
    # Find the method definition
    pattern = re.compile(rf'(\s+)def {method_name}\([^)]*\):(.*?)(?=\s+def|\Z)', re.DOTALL)
    match = pattern.search(content)
    
    if match:
        return match.group(0)
    return None

def fix_method_content(method_text, method_name):
    """
    Fix the indentation of a method's content.
    
    Args:
        method_text: The method text to fix
        method_name: The name of the method
        
    Returns:
        The fixed method text with proper indentation
    """
    # Normalize method indentation first
    lines = method_text.split('\n')
    method_lines = []
    
    # First line should be the method definition with exactly 4 spaces
    if lines and lines[0].strip().startswith(f"def {method_name}"):
        method_lines.append(f"    def {method_name}" + lines[0].strip()[4 + len(method_name):])
    else:
        # If we can't find the method definition, return unchanged
        return method_text
    
    # Process the remaining lines with proper indentation for method body
    i = 1
    in_docstring = False
    
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            method_lines.append("")
            i += 1
            continue
        
        # Handle docstrings
        if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) > 3:
            # Single line docstring
            method_lines.append("        " + stripped)
            i += 1
            continue
        elif stripped.startswith('"""'):
            # Start of multi-line docstring
            method_lines.append("        " + stripped)
            in_docstring = True
            i += 1
            continue
        elif stripped.endswith('"""') and in_docstring:
            # End of multi-line docstring
            method_lines.append("        " + stripped)
            in_docstring = False
            i += 1
            continue
        elif in_docstring:
            # Inside multi-line docstring
            method_lines.append("        " + stripped)
            i += 1
            continue
        
        # Handle method body with 8 spaces base indentation
        if stripped.startswith("if ") or stripped.startswith("for ") or stripped.startswith("while ") or \
           stripped.startswith("try:") or stripped.startswith("except ") or stripped.startswith("else:") or \
           stripped.startswith("elif ") or stripped.startswith("with ") or stripped.startswith("class "):
            # Control flow statements at 8 spaces
            method_lines.append("        " + stripped)
        elif stripped.startswith("return "):
            # Return statements at 8 spaces
            method_lines.append("        " + stripped)
        elif stripped.startswith(("self.", "results[", "logger.")):
            # Method level variable access at 8 spaces
            method_lines.append("        " + stripped)
        elif stripped.startswith("#"):
            # Comments at same level as surrounding code
            method_lines.append("        " + stripped)
        elif stripped in ["pass", "continue", "break"]:
            # Simple statements at 8 spaces
            method_lines.append("        " + stripped)
        elif "=" in stripped and not stripped.startswith(" "):
            # Variable assignments at 8 spaces
            method_lines.append("        " + stripped)
        elif stripped.startswith(("(", "[", "{")) or stripped.endswith((")", "]", "}")):
            # Collection literals or continuations at 8 spaces
            method_lines.append("        " + stripped)
        else:
            # Most nested blocks at 12 spaces
            # This is a heuristic that can be improved
            method_lines.append("            " + stripped)
        
        i += 1
    
    return "\n".join(method_lines)

def fix_dependency_checks(content):
    """Fix indentation in dependency check blocks."""
    # Fix dependency checks indentation to 8 spaces inside methods
    content = re.sub(r'(\s+)if not HAS_(\w+):', r'        if not HAS_\2:', content)
    
    # Fix returns in dependency checks
    content = re.sub(r'(\s+)return results', r'        return results', content)
    
    return content

def fix_imports(content):
    """Fix import section indentation."""
    # Make all top-level imports properly unindented
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            # Top-level imports should have no indentation
            fixed_lines.append(stripped)
        elif stripped.startswith(('try:', 'except ')):
            # Try/except blocks around imports should have no indentation
            fixed_lines.append(stripped)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_mock_definitions(content):
    """Fix mock class definitions indentation."""
    # Fix indentation of mock classes
    mock_classes = re.findall(r'(\s+)class Mock(\w+):', content)
    for indent, class_name in mock_classes:
        # Replace with proper indentation (4 spaces for class inside a conditional block)
        content = content.replace(f"{indent}class Mock{class_name}:", f"    class Mock{class_name}:")
    
    return content

def fix_try_except_blocks(content):
    """Fix try/except block indentation."""
    # Find all try blocks and properly indent their content
    try_pattern = re.compile(r'(\s+)try:(.*?)(\s+)except', re.DOTALL)
    
    def fix_try_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        except_indent = match.group(3)
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{indent}try:{fixed_block}\n{except_indent}except"
    
    content = try_pattern.sub(fix_try_block, content)
    
    # Fix except blocks with similar approach
    except_pattern = re.compile(r'(\s+)except.*?:(.*?)(?=\s+(?:try:|except|else:|finally:|def|$))', re.DOTALL)
    
    def fix_except_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{indent}except Exception:{fixed_block}"
    
    content = except_pattern.sub(fix_except_block, content)
    
    return content

def fix_if_blocks(content):
    """Fix if/else block indentation."""
    # Find all if blocks and properly indent their content
    if_pattern = re.compile(r'(\s+)if\s+.*?:(.*?)(?=\s+(?:elif|else:|try:|except|def|$))', re.DOTALL)
    
    def fix_if_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{indent}if{fixed_block}"
    
    content = if_pattern.sub(fix_if_block, content)
    
    # Fix else blocks with similar approach
    else_pattern = re.compile(r'(\s+)else:(.*?)(?=\s+(?:try:|except|def|if|$))', re.DOTALL)
    
    def fix_else_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{indent}else:{fixed_block}"
    
    content = else_pattern.sub(fix_else_block, content)
    
    return content

def fix_class_method_indentation(file_path, backup=True):
    """
    Fix indentation issues in the class methods of the generated test file.
    
    Args:
        file_path: Path to the test file
        backup: Whether to create a backup before modification
    
    Returns:
        bool: True if successful, False if an error occurred
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if backup:
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Created backup at {backup_path}")
        
        # Fix top-level imports and class definitions
        content = fix_imports(content)
        
        # Find the class definition(s)
        class_matches = re.finditer(r'class\s+(\w+):', content)
        
        for class_match in class_matches:
            class_name = class_match.group(1)
            class_start_pos = class_match.start()
            
            # Find all methods of this class 
            # This looks for methods until another class definition or EOF
            class_content_pattern = re.compile(r'class\s+' + class_name + r':(.*?)(?=class\s+\w+:|$)', re.DOTALL)
            class_content_match = class_content_pattern.search(content, class_start_pos)
            
            if not class_content_match:
                logger.warning(f"Could not extract content for class {class_name}")
                continue
            
            class_content = class_content_match.group(1)
            
            # Fix method indentation for common methods
            for method_name in ['__init__', 'test_pipeline', 'test_from_pretrained', 'run_tests']:
                method_text = extract_method(class_content, method_name)
                if method_text:
                    fixed_method = fix_method_content(method_text, method_name)
                    class_content = class_content.replace(method_text, fixed_method)
            
            # Fix dependency checks in methods
            class_content = fix_dependency_checks(class_content)
            
            # Fix mock class definitions inside the class
            class_content = fix_mock_definitions(class_content)
            
            # Fix try/except blocks
            class_content = fix_try_except_blocks(class_content)
            
            # Fix if/else blocks
            class_content = fix_if_blocks(class_content)
            
            # Fix spacing between methods
            class_content = fix_method_boundaries(class_content)
            
            # Replace the original class content with fixed content
            content = content[:class_start_pos] + "class " + class_name + ":" + class_content + content[class_content_match.end():]
        
        # Fix indentation of utility functions and main function
        for func_match in re.finditer(r'def\s+(\w+)\s*\(', content):
            func_name = func_match.group(1)
            if func_name not in ['__init__', 'test_pipeline', 'test_from_pretrained', 'run_tests']:
                func_pattern = re.compile(r'def\s+' + func_name + r'\s*\(.*?\):(.*?)(?=def\s+\w+\s*\(|$)', re.DOTALL)
                func_match = func_pattern.search(content, func_match.start())
                if func_match:
                    func_text = func_match.group(0)
                    fixed_func = apply_indentation(func_text, 0)  # Top-level functions have 0 indentation
                    content = content.replace(func_text, fixed_func)
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Fixed indentation in {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing indentation in {file_path}: {e}")
        return False

def verify_python_syntax(file_path):
    """
    Verify that the Python file has valid syntax.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, file_path, 'exec')
        logger.info(f"✅ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ {file_path}: Syntax error: {e}")
        logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
        return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix indentation issues in HuggingFace test files")
    parser.add_argument("file_path", type=str, help="Path to the test file to fix")
    parser.add_argument("--verify", action="store_true", help="Verify Python syntax after fixing")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup of the original file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        logger.error(f"File not found: {args.file_path}")
        return 1
    
    # Fix indentation
    success = fix_class_method_indentation(args.file_path, backup=not args.no_backup)
    
    if success and args.verify:
        # Verify syntax
        syntax_valid = verify_python_syntax(args.file_path)
        if not syntax_valid:
            return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
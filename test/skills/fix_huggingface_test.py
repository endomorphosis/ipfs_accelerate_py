#!/usr/bin/env python3
"""
Specialized fixer for HuggingFace test files.

This script specifically targets common indentation patterns in HuggingFace test files:
1. Mock class definitions in conditional blocks
2. Multiline try/except blocks
3. Method declarations with self parameter
4. Nested indentation in if/else blocks
5. Hardware detection sections

Usage:
    python fix_huggingface_test.py <file_path> [--verify]
"""

import sys
import os
import re
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_file_content(content):
    """Initial cleanup of file content."""
    # Replace tabs with spaces
    content = content.replace('\t', '    ')
    
    # Ensure consistent line endings
    content = content.replace('\r\n', '\n')
    
    # Fix any trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix excessive blank lines (more than 2)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content

def fix_indentation_patterns(content):
    """Apply pattern-based indentation fixes."""
    
    # 1. Fix class definitions (should be at column 0)
    content = re.sub(r'^\s+class\s+(\w+):', r'class \1:', content, flags=re.MULTILINE)
    
    # 2. Fix method definitions with self (should have 4 spaces)
    content = re.sub(r'^\s*def\s+(\w+)\(self', r'    def \1(self', content, flags=re.MULTILINE)
    
    # 3. Fix regular function definitions (should be at column 0)
    content = re.sub(r'^\s+def\s+(\w+)\((?!self)', r'def \1(', content, flags=re.MULTILINE)
    
    # 4. Fix import statements (should be at column 0)
    content = re.sub(r'^\s+import\s+', r'import ', content, flags=re.MULTILINE)
    content = re.sub(r'^\s+from\s+', r'from ', content, flags=re.MULTILINE)
    
    # 5. Fix docstrings in methods (should have 8 spaces)
    content = re.sub(r'(    def \w+\(.*?\):.*?)(\s+)"""', r'\1\n        """', content, flags=re.DOTALL)
    
    # 6. Fix if statements in methods (should have 8 spaces)
    content = re.sub(r'(    def \w+\(.*?\):.*?)(\s+)if\s+', r'\1\n        if ', content, flags=re.DOTALL)
    
    # 7. Fix return statements in methods (should have 8 spaces)
    content = re.sub(r'(\s*)return\s+', r'        return ', content)
    
    # 8. Fix mock class definitions in conditionals (should have 4 spaces inside if block)
    mock_pattern = re.compile(r'(\s+)if not HAS_(\w+):(.*?)(?=\s+if not HAS_|\s+def|\Z)', re.DOTALL)
    
    def fix_mock_block(match):
        indent = match.group(1)
        has_check = match.group(2)
        block_content = match.group(3)
        
        # Properly indent the block content
        lines = block_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                fixed_lines.append('')
                continue
                
            stripped = line.strip()
            if stripped.startswith('class Mock'):
                fixed_lines.append(f"{indent}    class Mock{stripped[11:]}")
            elif 'return' in stripped or 'logger.' in stripped:
                fixed_lines.append(f"{indent}        {stripped}")
            elif '=' in stripped and not stripped.startswith((' ', '(')):
                fixed_lines.append(f"{indent}        {stripped}")
            elif stripped.startswith(('def ', '@')):
                fixed_lines.append(f"{indent}        {stripped}")
            else:
                fixed_lines.append(f"{indent}            {stripped}")
                
        return f"{indent}if not HAS_{has_check}:" + '\n'.join(fixed_lines)
    
    content = mock_pattern.sub(fix_mock_block, content)
    
    # 9. Fix try/except blocks (ensure proper indentation)
    try_pattern = re.compile(r'(\s+)try:(.*?)(\s+)except', re.DOTALL)
    
    def fix_try_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        except_indent = match.group(3)
        
        # Properly indent the block content
        lines = block_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                fixed_lines.append('')
                continue
                
            # Add 4 more spaces to the indent
            fixed_lines.append(f"{indent}    {line.strip()}")
                
        return f"{indent}try:" + '\n'.join(fixed_lines) + f"\n{except_indent}except"
    
    content = try_pattern.sub(fix_try_block, content)
    
    # 10. Fix except blocks with similar approach
    except_pattern = re.compile(r'(\s+)except.*?:(.*?)(?=\s+(?:try:|except|else:|finally:|def|$))', re.DOTALL)
    
    def fix_except_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        
        # Properly indent the block content
        lines = block_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                fixed_lines.append('')
                continue
                
            # Add 4 more spaces to the indent
            fixed_lines.append(f"{indent}    {line.strip()}")
                
        return f"{indent}except Exception:" + '\n'.join(fixed_lines)
    
    content = except_pattern.sub(fix_except_block, content)
    
    # 11. Fix continuation of method content after docstring
    method_docstring_pattern = re.compile(r'(    def \w+\(.*?\):.*?""".*?""")(.*?)(?=\s+def|\Z)', re.DOTALL)
    
    def fix_method_after_docstring(match):
        method_start = match.group(1)
        method_content = match.group(2)
        
        # Properly indent the method content after docstring
        lines = method_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                fixed_lines.append('')
                continue
                
            # Basic indent for method body is 8 spaces
            fixed_lines.append(f"        {line.strip()}")
                
        return method_start + '\n' + '\n'.join(fixed_lines)
    
    content = method_docstring_pattern.sub(fix_method_after_docstring, content)
    
    # 12. Fix if blocks inside methods
    if_in_method_pattern = re.compile(r'(\s+)if\s+(.*?):(.*?)(?=\s+(?:elif|else:|except|def|\S|\Z))', re.DOTALL)
    
    def fix_if_in_method(match):
        indent = match.group(1)
        condition = match.group(2)
        block_content = match.group(3)
        
        # Properly indent the if block content
        lines = block_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                fixed_lines.append('')
                continue
                
            # Add 4 more spaces to the indent
            fixed_lines.append(f"{indent}    {line.strip()}")
                
        return f"{indent}if {condition}:" + '\n'.join(fixed_lines)
    
    # Apply the fix multiple times to catch nested if blocks
    for _ in range(3):
        content = if_in_method_pattern.sub(fix_if_in_method, content)
    
    # 13. Fix hardware detection sections
    hw_detection_pattern = re.compile(r'(# Hardware detection.*?HW_CAPABILITIES = check_hardware\(\))', re.DOTALL)
    
    def fix_hw_detection(match):
        hw_section = match.group(1)
        
        # Split into lines and fix indentation
        lines = hw_section.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                fixed_lines.append('')
                continue
                
            stripped = line.strip()
            if stripped.startswith('def check_hardware'):
                fixed_lines.append(f"def check_hardware{stripped[16:]}")
            elif stripped.startswith('#'):
                fixed_lines.append(stripped)
            elif stripped.startswith('"""'):
                fixed_lines.append(f"    {stripped}")
            elif stripped.startswith('capabilities'):
                fixed_lines.append(f"    {stripped}")
            elif 'return capabilities' in stripped:
                fixed_lines.append(f"    {stripped}")
            elif 'if ' in stripped or 'try:' in stripped or 'except ' in stripped:
                fixed_lines.append(f"    {stripped}")
            else:
                fixed_lines.append(f"        {stripped}")
                
        return '\n'.join(fixed_lines)
    
    content = hw_detection_pattern.sub(fix_hw_detection, content)
    
    return content

def fix_method_spacing(content):
    """Ensure proper spacing between methods."""
    # Add blank lines between methods
    content = re.sub(r'(\s+)def\s+(\w+)\(self.*?\):(.*?)(\s+)def\s+', 
                     r'\1def \2(self\3)\n\n\4def ', content, flags=re.DOTALL)
    
    # Fix spacing after the last class method before class end
    content = re.sub(r'(return [^}]*})(\s+)def\s+', r'\1\n\n\2def ', content)
    
    return content

def fix_dictionary_indentation(content):
    """Fix indentation of dictionaries."""
    # Find dictionary definitions
    dict_pattern = re.compile(r'(\s+)([^=\s]+)\s*=\s*\{(.*?)\}', re.DOTALL)
    
    def fix_dict(match):
        indent = match.group(1)
        var_name = match.group(2)
        dict_content = match.group(3)
        
        # Skip if it's a simple one-line dict
        if '\n' not in dict_content:
            return f"{indent}{var_name} = {{{dict_content}}}"
        
        # Split the content into lines
        lines = dict_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                continue
                
            stripped = line.strip()
            # Add 4 more spaces to the indent
            fixed_lines.append(f"{indent}    {stripped}")
                
        # Join with proper closing brace indent
        return f"{indent}{var_name} = {{\n" + '\n'.join(fixed_lines) + f"\n{indent}}}"
    
    content = dict_pattern.sub(fix_dict, content)
    
    return content

def verify_syntax(content, file_path):
    """
    Verify that the Python content has valid syntax.
    
    Args:
        content: Python code content
        file_path: Path to report in error messages
        
    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        # Try to compile the code
        compile(content, file_path, 'exec')
        logger.info(f"✅ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ {file_path}: Syntax error: {e}")
        logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
        context_start = max(0, e.lineno - 3)
        context_end = e.lineno + 3
        
        # Show context
        logger.error("Context:")
        lines = content.split('\n')
        for i in range(context_start, min(context_end, len(lines))):
            prefix = ">" if i + 1 == e.lineno else " "
            logger.error(f"{prefix} {i+1}: {lines[i]}")
        
        return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def fix_huggingface_test(file_path, verify=True, backup=True):
    """
    Fix indentation issues in a HuggingFace test file.
    
    Args:
        file_path: Path to the file to fix
        verify: Whether to verify syntax after fixing
        backup: Whether to create a backup before modifying
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the original content
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(original_content)
            logger.info(f"Created backup at {backup_path}")
        
        # Clean the content first
        content = clean_file_content(original_content)
        
        # Apply pattern-based fixes
        content = fix_indentation_patterns(content)
        
        # Fix method spacing
        content = fix_method_spacing(content)
        
        # Fix dictionary indentation
        content = fix_dictionary_indentation(content)
        
        # Verify if requested
        if verify:
            if not verify_syntax(content, file_path):
                # If verification fails, try to keep original content
                # but still apply basic fixes
                logger.warning("Fixing failed, attempting minimal fixes...")
                content = clean_file_content(original_content)
                content = re.sub(r'^\s+class\s+(\w+):', r'class \1:', content, flags=re.MULTILINE)
                content = re.sub(r'^\s*def\s+(\w+)\(self', r'    def \1(self', content, flags=re.MULTILINE)
                content = re.sub(r'^\s+def\s+(\w+)\((?!self)', r'def \1(', content, flags=re.MULTILINE)
                
                if not verify_syntax(content, file_path):
                    logger.error("All fixing attempts failed, reverting to original")
                    with open(file_path, 'w') as f:
                        f.write(original_content)
                    return False
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Successfully fixed {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix indentation in HuggingFace test files")
    parser.add_argument("file_path", help="Path to the file to fix")
    parser.add_argument("--verify", action="store_true", help="Verify syntax after fixing")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a backup")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        logger.error(f"File not found: {args.file_path}")
        return 1
    
    # Fix file
    if fix_huggingface_test(args.file_path, verify=args.verify, backup=not args.no_backup):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
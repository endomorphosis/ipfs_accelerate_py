#!/usr/bin/env python3
"""
Apply comprehensive indentation fixes to a single file.

This script combines multiple approaches to fix indentation:
1. Pattern-based replacements for common issues
2. Method-level extraction and fixing
3. Python syntax verification

Usage:
    python fix_single_file.py <file_path> [--verify]
"""

import sys
import os
import re
import logging
import argparse
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_indentation(code, base_indent=0):
    """Apply consistent indentation to code blocks."""
    lines = code.strip().split('\n')
    
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Skip empty lines
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)
    
    if min_indent == float('inf'):
        min_indent = 0
    
    indented_lines = []
    indent_spaces = ' ' * base_indent
    
    for line in lines:
        if line.strip():  # If not an empty line
            indented_line = indent_spaces + line[min_indent:]
            indented_lines.append(indented_line)
        else:
            indented_lines.append(indent_spaces)
    
    return '\n'.join(indented_lines)

def fix_common_patterns(content):
    """Fix common indentation issues using pattern-based replacements."""
    # 1. Fix method boundaries
    content = re.sub(r'return results\s+def', 'return results\n\n    def', content)
    content = re.sub(r'self\.performance_stats = \{\}\s+def', 
                     'self.performance_stats = {}\n\n    def', content)
    
    # 2. Fix method declarations
    content = re.sub(r'^(\s*)def test_(\w+)', r'    def test_\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*)def run_tests', r'    def run_tests', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*)def __init__', r'    def __init__', content, flags=re.MULTILINE)
    
    # 3. Fix class declarations
    content = re.sub(r'^(\s*)class Test(\w+)', r'class Test\2', content, flags=re.MULTILINE)
    
    # 4. Fix top-level function declarations
    content = re.sub(r'^(\s+)def (?!test_|run_|__init__)(\w+)', r'def \2', content, flags=re.MULTILINE)
    
    # 5. Fix dependency check indentation
    content = re.sub(r'(\s+)if not HAS_(\w+):', r'        if not HAS_\2:', content)
    
    # 6. Fix import statements
    content = re.sub(r'^(\s+)import (\w+)', r'import \2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)from ([\w\.]+) import', r'from \2 import', content, flags=re.MULTILINE)
    
    # 7. Fix try-except blocks
    content = re.sub(r'try:\s+(\w+)', r'try:\n            \1', content)
    content = re.sub(r'except Exception:\s+(\w+)', r'except Exception:\n            \1', content)
    
    # 8. Fix if-else blocks
    content = re.sub(r'(\s+)if (.+):\s+(\w+)', r'\1if \2:\n\1    \3', content)
    content = re.sub(r'(\s+)else:\s+(\w+)', r'\1else:\n\1    \2', content)
    
    # 9. Fix common content indentation
    string_fixes = [
        ('            else:\n            results', '            else:\n                results'),
        ('            elif', '            elif'),
        ('    if device', '        if device'),
        ('    for _ in range', '        for _ in range'),
        ('    try:', '        try:'),
        ('    logger.', '        logger.'),
        ('        try:\n        with', '        try:\n            with'),
        ('        except Exception:\n        pass', '        except Exception:\n            pass'),
    ]
    
    for old, new in string_fixes:
        content = content.replace(old, new)
    
    return content

def normalize_imports(content):
    """Ensure import statements are properly indented."""
    import_section = re.search(r'(^import.*?)(?=^class|\Z)', content, re.MULTILINE | re.DOTALL)
    
    if import_section:
        imports = import_section.group(0)
        # Split imports into lines
        import_lines = imports.split('\n')
        # Fix each line
        fixed_imports = []
        for line in import_lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                fixed_imports.append(stripped)
            elif stripped.startswith(('try:', 'except ')):
                fixed_imports.append(stripped)
            elif stripped.startswith(('#', 'logging.', 'sys.')):
                fixed_imports.append(stripped)
            else:
                fixed_imports.append(line)
        
        # Join back together
        fixed_import_section = '\n'.join(fixed_imports)
        # Replace in content
        content = content.replace(imports, fixed_import_section)
    
    return content

def normalize_class_methods(content, backup=True):
    """Normalize indentation in class methods."""
    # Extract classes and their methods
    class_pattern = re.compile(r'class\s+(\w+):(.*?)(?=^class|\Z)', re.MULTILINE | re.DOTALL)
    classes = class_pattern.findall(content)
    
    for class_name, class_content in classes:
        # Extract methods from class
        method_pattern = re.compile(r'(\s+)def\s+(\w+)\s*\(self(?:,.*?)?\):(.*?)(?=\s+def|\Z)', re.DOTALL)
        methods = method_pattern.findall(class_content)
        
        # Fix each method's indentation
        for indent, method_name, method_body in methods:
            original_method = f"{indent}def {method_name}(self{', ' if method_body.strip() else ''}){method_body}"
            
            # Fix method indentation
            fixed_body = []
            for line in method_body.split('\n'):
                stripped = line.strip()
                if not stripped:
                    fixed_body.append('')
                    continue
                
                # Basic rules for indentation
                if stripped.startswith('"""'):
                    # Docstring: 8 spaces
                    fixed_body.append(f"        {stripped}")
                elif stripped.startswith(('if ', 'for ', 'while ', 'try:', 'except ', 'else:', 'elif ')):
                    # Control flow: 8 spaces
                    fixed_body.append(f"        {stripped}")
                elif stripped.startswith(('return ', 'self.', 'logger.')):
                    # Method statements: 8 spaces
                    fixed_body.append(f"        {stripped}")
                else:
                    # Most other content: 12 spaces
                    fixed_body.append(f"            {stripped}")
            
            # Create fixed method
            fixed_method = f"    def {method_name}(self{', ' if method_body.strip() else ''})\n" + '\n'.join(fixed_body)
            
            # Replace in content
            content = content.replace(original_method, fixed_method)
    
    return content

def apply_whitespace_cleanup(content):
    """Clean up whitespace issues in the content."""
    # Fix multiple blank lines
    content = re.sub(r'\n{3,}', r'\n\n', content)
    
    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix indentation around closing braces
    content = re.sub(r'(\s+)(\}|\]|\))', r'\2', content)
    
    return content

def verify_python_syntax(content, file_path):
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
        return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def fix_indentation_in_file(file_path, verify=True, backup=True):
    """
    Apply comprehensive indentation fixes to a file.
    
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
        
        # Apply fixes in sequence
        content = original_content
        
        # First normalize the imports
        content = normalize_imports(content)
        
        # Apply pattern-based fixes
        content = fix_common_patterns(content)
        
        # Normalize class methods
        content = normalize_class_methods(content)
        
        # Final cleanup
        content = apply_whitespace_cleanup(content)
        
        # Verify if requested
        if verify:
            syntax_valid = verify_python_syntax(content, file_path)
            if not syntax_valid:
                logger.warning(f"Fixed content still has syntax errors, using incremental approach")
                
                # Try incremental approach if verification fails
                content = original_content
                
                # Fix imports
                content = normalize_imports(content)
                if verify and verify_python_syntax(content, file_path):
                    logger.info("Imports fixed successfully")
                
                # Fix common patterns
                content = fix_common_patterns(content)
                if verify and verify_python_syntax(content, file_path):
                    logger.info("Common patterns fixed successfully")
                
                # Fix class methods
                content = normalize_class_methods(content)
                if verify and verify_python_syntax(content, file_path):
                    logger.info("Class methods fixed successfully")
                
                # Clean up whitespace
                content = apply_whitespace_cleanup(content)
                
                # Final verification
                syntax_valid = verify_python_syntax(content, file_path)
                if not syntax_valid:
                    logger.error(f"Failed to fix all syntax errors, using minimal approach")
                    
                    # Try minimal approach - just fix indentation
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Fix each line individually
                    fixed_lines = []
                    indent_level = 0
                    
                    for line in lines:
                        stripped = line.strip()
                        
                        # Skip empty lines
                        if not stripped:
                            fixed_lines.append('')
                            continue
                        
                        # Adjust indent level based on line content
                        if stripped.startswith(('class ', 'import ', 'from ')):
                            indent_level = 0
                        elif stripped.startswith('def '):
                            if 'self' in stripped:
                                indent_level = 4
                            else:
                                indent_level = 0
                        
                        # Add indentation
                        fixed_lines.append(' ' * indent_level + stripped)
                    
                    # Join fixed lines
                    content = '\n'.join(fixed_lines)
                    
                    # Final verification
                    syntax_valid = verify_python_syntax(content, file_path)
                    if not syntax_valid:
                        # If all approaches fail, just keep the original
                        logger.error(f"All fix approaches failed, reverting to original content")
                        with open(file_path, 'w') as f:
                            f.write(original_content)
                        return False
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Fixed indentation in {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error fixing indentation in {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix indentation in a Python file")
    parser.add_argument("file_path", help="Path to the file to fix")
    parser.add_argument("--verify", action="store_true", help="Verify syntax after fixing")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a backup")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        logger.error(f"File not found: {args.file_path}")
        return 1
    
    # Fix indentation
    success = fix_indentation_in_file(
        args.file_path,
        verify=args.verify,
        backup=not args.no_backup
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Comprehensive indentation fixer for generated test files.
This script combines all the indentation fixing approaches.
"""

import os
import sys
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_indentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    # Add proper spacing between methods
    content = content.replace("        return results\n    def ", "        return results\n\n    def ")
    
    # Make sure __init__ has correct spacing after it
    content = content.replace("        self.performance_stats = {}\n    def ", "        self.performance_stats = {}\n\n    def ")
        
    # Place all method declarations at the right indentation level
    content = re.sub(r'(\s+)def test_pipeline\(', r'    def test_pipeline(', content)
    content = re.sub(r'(\s+)def test_from_pretrained\(', r'    def test_from_pretrained(', content)
    content = re.sub(r'(\s+)def run_tests\(', r'    def run_tests(', content)
    
    return content

def extract_method(content, method_name):
    """Extract a method's content from a class."""
    pattern = None
    
    if method_name == "test_pipeline":
        pattern = r'(\s+def test_pipeline\(self,.*?(?=\s+def test_from_pretrained|\s+def run_tests|$))'
    elif method_name == "test_from_pretrained":
        pattern = r'(\s+def test_from_pretrained\(self,.*?(?=\s+def run_tests|$))'
    elif method_name == "run_tests":
        pattern = r'(\s+def run_tests\(self,.*?(?=\s+def save_results|$))'
    
    if not pattern:
        logger.error(f"Unknown method name: {method_name}")
        return None
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(0)
    else:
        logger.warning(f"Could not find method {method_name}")
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
    lines = method_text.split('\n')
    fixed_lines = []
    
    # First line should be properly indented with 4 spaces for class methods
    first_line = lines[0]
    if not first_line.strip().startswith('def'):
        return method_text  # Not a method definition, return as is
    
    # Ensure the method definition has exactly 4 spaces
    fixed_first_line = f"    def {method_name}(self,"
    for part in first_line.split('def')[1].split('(self,')[1:]:
        fixed_first_line += f"(self,{part}"
    fixed_lines.append(fixed_first_line)
    
    # Process the rest of the method body with 8 spaces (4 for class + 4 for method content)
    for line in lines[1:]:
        if not line.strip():
            # Empty line
            fixed_lines.append(line)
            continue
            
        content = line.strip()
        # Determine the correct indentation based on content
        if content.startswith('"""') or content.endswith('"""'):
            # Docstring line - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('if ') or content.startswith('for ') or content.startswith('try:') or content.startswith('else:') or content.startswith('elif '):
            # Control flow - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('return '):
            # Return statement - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('results[') or content.startswith('self.'):
            # Assignment to results or self - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('logger.'):
            # Logging - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('# '):
            # Comments - 8 spaces for top-level method comments
            fixed_lines.append(f"        {content}")
        elif content.startswith('}') or content.startswith(']'):
            # Closing brackets - 8 spaces for method level collections
            fixed_lines.append(f"        {content}")
        else:
            # Default deeper indentation for nested blocks - 12 spaces
            # This is a heuristic that works well for this specific code structure
            fixed_lines.append(f"            {content}")
    
    return '\n'.join(fixed_lines)

def fix_dependency_checks(content):
    """Fix dependency check blocks which often have indentation issues."""
    # Normalize dependency check blocks to have 8 spaces of indentation
    pattern = r'(\s+)if not HAS_(\w+):'
    content = re.sub(pattern, r'        if not HAS_\2:', content)
    
    # Also fix imports with incorrect indentation
    mock_imports = [
        (r'    torch = MagicMock\(\)\n        HAS_TORCH = False', r'    torch = MagicMock()\n    HAS_TORCH = False'),
        (r'    transformers = MagicMock\(\)\n        HAS_TRANSFORMERS = False', r'    transformers = MagicMock()\n    HAS_TRANSFORMERS = False'),
        (r'    tokenizers = MagicMock\(\)\n        HAS_TOKENIZERS = False', r'    tokenizers = MagicMock()\n    HAS_TOKENIZERS = False'),
        (r'    sentencepiece = MagicMock\(\)\n        HAS_SENTENCEPIECE = False', r'    sentencepiece = MagicMock()\n    HAS_SENTENCEPIECE = False'),
        (r'        logger.warning\("', r'    logger.warning("')
    ]
    
    for pattern, replacement in mock_imports:
        content = re.sub(pattern, replacement, content)
    
    return content

def fix_string_patterns(content):
    """Fix common string patterns that have indentation issues."""
    fixes = [
        # Method declarations with proper spacing
        ('return results    def test_from_pretrained', 'return results\n\n    def test_from_pretrained'),
        ('return results    def run_tests', 'return results\n\n    def run_tests'),
        
        # Error handling indentation
        ('            else:\n            results', '            else:\n                results'),
        ('            elif', '            elif'),
        
        # Block indentation
        ('    if device', '        if device'),
        ('    for _ in range', '        for _ in range'),
        ('    try:', '        try:'),
        ('    logger.', '        logger.'),
        
        # Nested blocks
        ('        try:\n        with', '        try:\n            with'),
        ('        except Exception:\n        pass', '        except Exception:\n            pass'),
        
        # Tokenizer setup - common issue
        ('            # First load the tokenizer', '        # First load the tokenizer'),
        ('            tokenizer =', '        tokenizer ='),
        ('            if tokenizer.pad_token', '        if tokenizer.pad_token'),
        ('            tokenizer.pad_token', '        tokenizer.pad_token'),
        ('            logger.info', '        logger.info'),
        ('            # Create pipeline', '        # Create pipeline'),
        ('            pipeline_kwargs', '        pipeline_kwargs'),
    ]
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    return content

def fix_spacing(content):
    """Fix spacing around blocks and method boundaries."""
    # Add newlines between methods
    content = re.sub(r'(\s+def \w+\(self,.+\):.*?return [^}]+})\s+(\s+def)', r'\1\n\n\2', content, flags=re.DOTALL)
    
    # Add proper spacing after class definition before first method
    content = re.sub(r'(class \w+:.*?performance_stats = \{\})\s+(\s+def)', r'\1\n\n\2', content, flags=re.DOTALL)
    
    # Add proper spacing between functions outside classes
    content = re.sub(r'(def \w+\(.+\):.*?return [^}]+})\s+(\s*def)', r'\1\n\n\n\2', content, flags=re.DOTALL)
    
    return content

def fix_class_method_indentation(file_path, backup=True):
    """
    Fix indentation issues in the class methods of the generated test file.
    
    This is a comprehensive approach that:
    1. Extracts and properly indents each method
    2. Fixes common indentation patterns
    3. Normalizes indentation throughout the file
    
    Args:
        file_path: Path to the test file to fix
        backup: If True, create a backup of the file before modifying
    """
    if backup:
        backup_path = f"{file_path}.bak"
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # First, apply general fixes
    content = fix_dependency_checks(content)
    content = fix_string_patterns(content)
    content = fix_method_boundaries(content)
    
    # Extract and fix each method
    methods_to_fix = ["test_pipeline", "test_from_pretrained", "run_tests"]
    for method_name in methods_to_fix:
        original_method = extract_method(content, method_name)
        if original_method:
            fixed_method = fix_method_content(original_method, method_name)
            content = content.replace(original_method, fixed_method)
    
    # Apply final spacing fixes
    content = fix_spacing(content)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"✅ Fixed indentation in {file_path}")
    return file_path

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Fix indentation issues in generated test files")
    parser.add_argument("file_path", help="Path to the test file to fix")
    parser.add_argument("--output", "-o", help="Output path (defaults to overwriting the input file)")
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original file")
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    file_path = args.file_path
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    # Run fix
    fixed_path = fix_class_method_indentation(file_path, backup=not args.no_backup)
    
    # Save to new file if specified
    if args.output:
        with open(fixed_path, 'r') as src:
            with open(args.output, 'w') as dst:
                dst.write(src.read())
        logger.info(f"Saved fixed file to {args.output}")
    
    logger.info(f"Successfully fixed indentation in {file_path}")
    
    # Check if file is a test generator or if it's a test file
    if "generator" in file_path:
        logger.info(f"Fixed file appears to be a generator - remember to regenerate test files")
    elif "test_hf_" in file_path:
        logger.info(f"Fixed file is a test file - try running it to verify proper execution")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
Fix syntax and indentation issues in the encoder_only_template.py file.

This script specifically targets common issues in the encoder_only_template.py
file to make it syntactically valid.
"""

import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_try_except_blocks(content):
    """Fix try/except block indentation."""
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

def fix_if_conditions(content):
    """Fix if block indentation."""
    # Find all 'if' statements
    pattern = r'(if\s+.*?:)\s*\n([^\s].*?\n)'
    
    def repl(match):
        if_statement = match.group(1)
        next_line = match.group(2)
        # Add proper indentation to the next line
        return f"{if_statement}\n    {next_line}"
    
    fixed_content = re.sub(pattern, repl, content)
    return fixed_content

def fix_class_definitions(content):
    """Fix class definitions and methods."""
    # Fix class method indentation
    pattern = r'class\s+(\w+):\s*\n([^\n]+?\n)'
    
    def repl(match):
        class_def = match.group(1)
        next_line = match.group(2)
        # If next line isn't indented and not another class
        if not next_line.startswith('    ') and not next_line.startswith('class '):
            return f"class {class_def}:\n    {next_line.lstrip()}"
        return match.group(0)
    
    fixed_content = re.sub(pattern, repl, content)
    
    # Fix docstrings
    pattern = r'(class\s+\w+:)\s*\n\s*"""(.*?)"""\s*\n([^\s])'
    fixed_content = re.sub(pattern, r'\1\n    """\2"""\n    \3', fixed_content, flags=re.DOTALL)
    
    # Fix method bodies
    pattern = r'(def\s+\w+\([^)]*\):)\s*\n([^\s])'
    fixed_content = re.sub(pattern, r'\1\n    \2', fixed_content)
    
    # Fix class method spacing
    fixed_content = fixed_content.replace("\ndef ", "\n\ndef ")
    fixed_content = fixed_content.replace("    def ", "\n    def ")
    
    return fixed_content

def ensure_proper_spacing(content):
    """Ensure proper spacing between blocks."""
    # Add blank line between methods
    content = re.sub(r'(\n\s+def \w+\([^)]*\):.*?return [^"]+?\n)(\s+def)', 
                   r'\1\n\2', content, flags=re.DOTALL)
    
    # Add blank line after docstring
    content = re.sub(r'(""".+?""")(\n[^\s])', r'\1\n\2', content, flags=re.DOTALL)
    
    # Fix registry definition spacing
    content = re.sub(r'(}\s*)\n([a-zA-Z])', r'\1\n\n\2', content)
    
    return content

def fix_encoder_template(file_path):
    """Fix issues in the encoder_only_template.py file."""
    try:
        logger.info(f"Processing template file: {file_path}")
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Store original content for comparison
        original_content = content
        
        # Manual replacement of problematic sections
        # Fix CUDA detection section at line ~130
        cuda_detection_pattern = r'if HAS_TORCH:(.*?)HAS_CUDA = torch.cuda.is_available'
        replacement = r'if HAS_TORCH:\n    HAS_CUDA = torch.cuda.is_available'
        content = re.sub(cuda_detection_pattern, replacement, content, flags=re.DOTALL)
        
        # Apply fixes
        content = fix_try_except_blocks(content)
        content = fix_if_conditions(content)
        content = fix_class_definitions(content)
        content = ensure_proper_spacing(content)
        
        # Check if content was changed
        if content != original_content:
            logger.info(f"Fixed issues in {file_path}")
            
            # Create backup of original file
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(original_content)
            logger.info(f"Created backup at {backup_path}")
            
            # Write fixed content
            with open(file_path, 'w') as f:
                f.write(content)
                
            # Validate syntax
            try:
                compile(content, file_path, 'exec')
                logger.info(f"✅ Syntax is valid for {file_path}")
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
            logger.info(f"No issues found in {file_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix the encoder template."""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "templates/encoder_only_template.py"
    
    success = fix_encoder_template(file_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
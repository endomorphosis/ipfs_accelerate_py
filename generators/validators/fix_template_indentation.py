#!/usr/bin/env python3
"""
Fix Template Indentation

This script automatically fixes indentation issues in template strings.
It can process specific files or entire directories.
"""

import os
import re
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_template_strings(content: str) -> List[Tuple[str, int, int]]:
    """
    Extract triple-quoted template strings from content.
    
    Args:
        content: File content
        
    Returns:
        List of tuples (template_string, start_pos, end_pos)
    """
    # Find all triple-quoted strings
    templates = []
    
    # Use regex to find templates in one go
    triple_quoted_pattern = r'(?P<quote>\'\'\'|""")(?P<content>.*?)(?P=quote)'
    matches = re.finditer(triple_quoted_pattern, content, re.DOTALL)
    
    for match in matches:
        start_pos = match.start()
        end_pos = match.end()
        template_str = match.group(0)
        
        # Check if this is likely a template 
        # Templates typically have placeholders or Python code references
        if (
            "{{" in template_str or 
            "}}" in template_str or 
            "self." in template_str or
            "import" in template_str or
            "logger." in template_str or
            # Model input pattern
            "inputs = " in template_str
        ):
            templates.append((template_str, start_pos, end_pos))
    
    return templates

def fix_indentation(template: str) -> str:
    """
    Fix indentation in a template string.
    
    Args:
        template: Template string
        
    Returns:
        Fixed template string
    """
    # Extract the template content (without the triple quotes)
    if template.startswith("'''"):
        content = template[3:-3]
        quote_type = "'''"
    elif template.startswith('"""'):
        content = template[3:-3]
        quote_type = '"""'
    else:
        return template
    
    # Split into lines
    lines = content.split("\n")
    if not lines:
        return template
    
    # Check if this is a simple template without significant indentation structure
    # If it's just a few lines without complex structure, don't modify it
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) < 3:
        return template
    
    # Analyze code structure
    is_python_code = any(line.strip().startswith(('def ', 'class ', 'import ', 'from ')) for line in lines)
    
    # Find the indentation baseline
    # The first line is typically blank or has special indentation
    # So we'll look at subsequent lines
    base_indent = None
    min_indent = None
    indent_levels = []
    
    # First pass: Find baseline and all indentation levels
    for i in range(len(lines)):
        if lines[i].strip():  # Non-empty line
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            
            # Track all indentation levels
            indent_levels.append(current_indent)
            
            # Track minimum indentation
            if min_indent is None or current_indent < min_indent:
                min_indent = current_indent
                
            # For baseline, skip first line and use the most common non-zero indentation
            if i > 0 and (base_indent is None or current_indent < base_indent):
                base_indent = current_indent
    
    if base_indent is None:
        # No indentation detected
        return template
    
    # Count occurrences of each indentation level
    from collections import Counter
    indentation_counts = Counter(indent_levels)
    
    # Check if indentation is already consistent
    # For Python code, we expect multiples of 4 spaces
    if is_python_code and all(level % 4 == 0 for level in indentation_counts.keys()):
        return template
        
    # For general templates, we'll normalize to either 2-space or 4-space indentation
    # based on what's most common in the template
    indent_unit = 4  # Default to 4-space indentation
    
    # Detect if 2-space indentation might be deliberate
    # This happens if there are lots of indentation levels that are 2, 6, 10, etc.
    if sum(1 for level in indentation_counts.keys() if level % 2 == 0 and level % 4 != 0) > 2:
        indent_unit = 2
    
    # Adjust all lines to use consistent indentation
    fixed_lines = []
    
    # First line is usually the empty line after the opening triple quote
    # So we'll keep it as is
    first_line_empty = not lines[0].strip() if lines else False
    
    for i, line in enumerate(lines):
        if i == 0 and first_line_empty:  # First empty line special case
            fixed_lines.append(line)
            continue
            
        if line.strip():  # Non-empty line
            # Measure current indentation
            current_indent = len(line) - len(line.lstrip())
            
            # Normalize indentation to be a multiple of indent_unit spaces from the base
            if current_indent >= base_indent:
                relative_indent = current_indent - base_indent
                # Round to the nearest multiple of indent_unit
                normalized_indent = base_indent + (relative_indent // indent_unit) * indent_unit
                # Apply the normalized indentation
                fixed_line = " " * normalized_indent + line.lstrip()
                fixed_lines.append(fixed_line)
            else:
                # Line already has less indentation than base - keep it as is
                fixed_lines.append(line)
        else:
            # Empty line - keep as is
            fixed_lines.append(line)
    
    # Reassemble the template
    fixed_content = "\n".join(fixed_lines)
    fixed_template = quote_type + fixed_content + quote_type
    
    return fixed_template

def identify_template_variables(file_path: str) -> Dict[str, List[str]]:
    """
    Identify template variables in a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary mapping variable names to their assignments
    """
    variables = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Look for variable assignments
        # Pattern: VARIABLE_NAME = '''...''' or VARIABLE_NAME = """..."""
        pattern = r'([A-Z_]+)\s*=\s*(\'\'\'|""").*?(\'\'\'|""")'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            var_name = match.group(1)
            
            # Extract the template value
            start_pos = match.start(2)
            end_quote = match.group(3)
            end_pos = content.find(end_quote, start_pos + 3) + 3
            
            if end_pos > start_pos:
                template_value = content[start_pos:end_pos]
                variables[var_name] = template_value
        
        return variables
    except Exception as e:
        logger.error(f"Error identifying template variables in {file_path}: {str(e)}")
        return {}

def fix_file(file_path: str, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Fix indentation issues in templates within a file.
    
    Args:
        file_path: Path to the file
        dry_run: If True, only report changes without modifying file
        
    Returns:
        Tuple of (success, number of templates fixed)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Look for template dictionary structures first
        template_dicts = []
        dict_pattern = r'([A-Z_]+)\s*=\s*\{([^}]*)\}'
        dict_matches = re.finditer(dict_pattern, content, re.DOTALL)
        
        for match in dict_matches:
            dict_name = match.group(1)
            dict_content = match.group(2)
            
            # Check if this looks like a template dictionary
            if '"""' in dict_content or "'''" in dict_content:
                template_dicts.append(dict_name)
                logger.info(f"Found template dictionary: {dict_name}")
        
        # Extract template strings
        templates = extract_template_strings(content)
        if not templates:
            logger.info(f"No template strings found in {file_path}")
            return True, 0
            
        logger.info(f"Found {len(templates)} candidate template strings in {file_path}")
        
        # Fix each template
        fixed_count = 0
        templates_sorted = sorted(templates, key=lambda x: x[1], reverse=True)  # Sort by start position, descending
        
        fixed_content = content
        for template_str, start_pos, end_pos in templates_sorted:
            fixed_template = fix_indentation(template_str)
            
            if fixed_template != template_str:
                if dry_run:
                    # Just report what would be changed
                    template_snippet = template_str[:50] + "..." if len(template_str) > 50 else template_str
                    logger.info(f"Would fix template at position {start_pos}-{end_pos}: {template_snippet}")
                else:
                    # Replace the template in the content
                    fixed_content = fixed_content[:start_pos] + fixed_template + fixed_content[end_pos:]
                
                fixed_count += 1
        
        if fixed_count > 0 and not dry_run:
            # Write the updated content
            with open(file_path, 'w') as f:
                f.write(fixed_content)
                
            logger.info(f"Fixed indentation in {fixed_count} templates in {file_path}")
        elif fixed_count > 0 and dry_run:
            logger.info(f"Would fix indentation in {fixed_count} templates in {file_path} (dry run)")
        else:
            logger.info(f"No templates needed fixing in {file_path}")
            
        return True, fixed_count
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {str(e)}")
        return False, 0
        
def fix_directory(directory: str, pattern: str = "*.py", dry_run: bool = False) -> Tuple[bool, int, int]:
    """
    Fix indentation issues in templates within files in a directory.
    
    Args:
        directory: Path to the directory
        pattern: Glob pattern for files
        dry_run: If True, only report changes without modifying files
        
    Returns:
        Tuple of (success, number of files processed, number of templates fixed)
    """
    import glob
    
    try:
        # Find matching files
        file_pattern = os.path.join(directory, pattern)
        files = glob.glob(file_pattern, recursive=True)
        
        if not files:
            logger.info(f"No files matching {file_pattern} found")
            return True, 0, 0
            
        logger.info(f"Found {len(files)} files matching {file_pattern}")
        
        # Process each file
        success_count = 0
        template_count = 0
        
        for file_path in files:
            success, fixed_count = fix_file(file_path, dry_run)
            if success:
                success_count += 1
                template_count += fixed_count
        
        if dry_run:
            logger.info(f"Would fix {template_count} templates in {success_count}/{len(files)} files (dry run)")
        else:
            logger.info(f"Successfully processed {success_count}/{len(files)} files, fixed {template_count} templates")
        
        return True, success_count, template_count
    except Exception as e:
        logger.error(f"Error fixing directory {directory}: {str(e)}")
        return False, 0, 0

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Fix Template Indentation")
    parser.add_argument("--file", type=str, help="Path to a specific file to fix")
    parser.add_argument("--directory", type=str, help="Path to a directory to fix")
    parser.add_argument("--pattern", type=str, default="**/*.py", help="Glob pattern for files when using --directory")
    parser.add_argument("--identify-templates", action="store_true", help="Only identify template variables without fixing")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files, just report what would be changed")
    
    args = parser.parse_args()
    
    dry_run = args.dry_run
    
    if args.file:
        if args.identify_templates:
            # Just identify templates
            variables = identify_template_variables(args.file)
            if variables:
                logger.info(f"Found {len(variables)} template variables in {args.file}:")
                for var_name in sorted(variables.keys()):
                    logger.info(f"  - {var_name}")
            else:
                logger.info(f"No template variables found in {args.file}")
        else:
            # Fix the file
            success, count = fix_file(args.file, dry_run)
            if success:
                if dry_run:
                    logger.info(f"Would fix {count} templates in {args.file} (dry run)")
                else:
                    logger.info(f"Successfully fixed {count} templates in {args.file}")
            else:
                logger.error(f"Failed to fix {args.file}")
    elif args.directory:
        # Fix files in a directory
        success, file_count, template_count = fix_directory(args.directory, args.pattern, dry_run)
        if success:
            if dry_run:
                logger.info(f"Would fix {template_count} templates in {file_count} files in {args.directory} (dry run)")
            else:
                logger.info(f"Successfully processed {file_count} files, fixed {template_count} templates in {args.directory}")
        else:
            logger.error(f"Failed to process directory {args.directory}")
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
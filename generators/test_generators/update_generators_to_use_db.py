#!/usr/bin/env python3
"""
Generator updater to use database templates.

This script patches existing test generators to use the new template database system
instead of static files. This completes the migration of the generator system to use
DuckDB-backed templates.
"""

import os
import sys
import re
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path to import template_database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates.template_database import TemplateDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target generator files to update
TARGET_GENERATORS = [
    'merged_test_generator.py',
    'qualified_test_generator.py',
    'fixed_merged_test_generator_clean.py',
    'simple_test_generator.py',
    'fixed_merged_test_generator.py',
    'integrated_skillset_generator.py'
]

def update_static_to_db_templates(file_path: str) -> Tuple[bool, str]:
    """Update a generator file to use database templates.
    
    This function searches for static template loading patterns
    and replaces them with calls to the template database.
    
    Args:
        file_path: Path to the generator file
        
    Returns:
        (success, message) tuple
    """
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Template loading from file or string constant
        file_pattern = r"(?:template_file\s*=\s*['\"](.*)['\"]|template\s*=\s*['\"{3}](.*?)['\"{3}])"
        
        # Pattern 2: Template directory references
        dir_pattern = r"(?:template_dir|TEMPLATE_DIR)\s*=\s*['\"](.*)['\"]"
        
        # Pattern 3: Add import for template database if needed
        if "from templates.template_database import TemplateDatabase" not in content:
            # Find appropriate import section
            import_section = re.search(r"import.*?\n\n", content, re.DOTALL)
            if import_section:
                # Insert after existing imports
                content = content.replace(
                    import_section.group(0),
                    import_section.group(0) + "from templates.template_database import TemplateDatabase\n"
                )
            else:
                # Add to beginning after any module docstring
                match = re.search(r"^((?:['\"]{3}.*?['\"]{3}\s*\n)?)", content, re.DOTALL)
                if match:
                    content = content.replace(
                        match.group(0),
                        match.group(0) + "from templates.template_database import TemplateDatabase\n\n"
                    )
                else:
                    content = "from templates.template_database import TemplateDatabase\n\n" + content
        
        # Pattern 4: Initialize template database
        if "self.template_db = TemplateDatabase()" not in content and "template_db = TemplateDatabase()" not in content:
            # Find the class or function that should contain the database initialization
            class_match = re.search(r"class\s+\w+.*?:\s*\n(?:\s+.*?\n)*?\s+def\s+__init__\s*\(self", content, re.DOTALL)
            if class_match:
                # Add to __init__ method
                init_block = re.search(r"def\s+__init__\s*\(self.*?\).*?:(?:\n\s+.*?)+?\n", content[class_match.start():], re.DOTALL)
                if init_block:
                    init_end = class_match.start() + init_block.end()
                    content = content[:init_end] + "        self.template_db = TemplateDatabase()\n" + content[init_end:]
            else:
                # Find main function or entry point
                main_match = re.search(r"def\s+main\s*\(", content)
                if main_match:
                    # Add after function definition
                    func_header = re.search(r"def\s+main.*?:\s*\n", content[main_match.start():])
                    if func_header:
                        main_end = main_match.start() + func_header.end()
                        content = content[:main_end] + "    template_db = TemplateDatabase()\n" + content[main_end:]
        
        # Pattern 5: Replace template loading from files with database
        # Find places where templates are loaded from files
        load_template_pattern = r"(?:def\s+load_template|with\s+open\([^)]*['\"]([^'\"]+\.py)['\"][^)]*\)\s+as\s+f:\s*\n\s+template\s*=\s*f\.read\(\))"
        for match in re.finditer(load_template_pattern, content, re.DOTALL):
            # Get the surrounding function or block
            block_start = match.start()
            block_end = content.find("\n\n", match.end())
            if block_end == -1:
                block_end = len(content)
            
            template_block = content[block_start:block_end]
            
            # Replace with template database code
            if "def load_template" in template_block:
                # Replace entire function with database version
                new_func = """
    def load_template(self, model_type, template_type='test', hardware_platform=None):
        """Load a template from the database."""
        template = self.template_db.get_template(model_type, template_type, hardware_platform)
        if template is None:
            logger.warning(f"Template not found for {model_type}/{template_type}/{hardware_platform}, using default")
            # Fall back to default template
            template = self.template_db.get_template('default', template_type)
        return template
"""
                content = content.replace(template_block, new_func)
            elif "with open" in template_block:
                # Extract model type and template type from filename if possible
                file_match = re.search(r"['\"]([^'\"]+)\.py['\"]", template_block)
                if file_match:
                    template_file = file_match.group(1)
                    parts = template_file.split('_')
                    if len(parts) >= 2:
                        model_type = parts[0]
                        template_type = parts[1]
                        
                        # Replace file loading with database call
                        replacement = f"""
        # Load template from database
        template = template_db.get_template('{model_type}', '{template_type}')
        if template is None:
            logger.warning(f"Template not found for {model_type}/{template_type}, using default")
            template = template_db.get_template('default', '{template_type}')
"""
                        content = content.replace(template_block, replacement)
        
        # Pattern 6: Add use_db_templates flag to argument parser if it exists
        parser_pattern = r"parser\s*=\s*argparse\.ArgumentParser\("
        if re.search(parser_pattern, content) and "--use-db-templates" not in content:
            # Find the argument groups
            group_pattern = r"(?:parser\.add_argument_group|parser\.add_argument)\(.*?\)"
            last_group = None
            for match in re.finditer(group_pattern, content, re.DOTALL):
                last_group = match
            
            if last_group:
                # Add the argument after the last group
                db_arg = """
    parser.add_argument(
        "--use-db-templates", action="store_true",
        help="Use database templates instead of static files"
    )
"""
                insert_pos = last_group.end()
                content = content[:insert_pos] + db_arg + content[insert_pos:]
        
        # Pattern 7: Update the generate_test_file function or similar to use database templates when flag is set
        generate_pattern = r"def\s+generate_(?:test|skill)_file\s*\("
        for match in re.finditer(generate_pattern, content, re.DOTALL):
            func_start = match.start()
            func_name_end = content.find("(", func_start)
            func_body_start = content.find(":", func_start)
            func_body_end = find_function_end(content, func_body_start)
            
            func_body = content[func_body_start:func_body_end]
            
            # Check if it already has db template logic
            if "template_db" in func_body:
                continue
            
            # Add db template logic with feature flag
            indent = get_indent(content, func_body_start)
            
            # Find where template loading happens
            template_load = re.search(r"template\s*=", func_body)
            if template_load:
                template_load_start = func_body_start + template_load.start()
                template_load_line_end = content.find("\n", template_load_start)
                
                # Get argument names
                args_str = content[func_name_end+1:func_body_start].strip()
                args = [a.strip().split('=')[0].strip() for a in args_str.split(',')]
                
                # Check if model_type/name is available
                model_arg = None
                for arg in args:
                    if arg in ['model_type', 'model_name', 'model']:
                        model_arg = arg
                        break
                
                if model_arg:
                    # Insert db template code before template loading
                    db_code = f"""
{indent}    # Use database templates if requested
{indent}    if getattr(args, 'use_db_templates', False) or os.environ.get('USE_DB_TEMPLATES', '').lower() in ('1', 'true', 'yes'):
{indent}        logger.info("Using database templates")
{indent}        template = self.template_db.get_template({model_arg}, 'test')
{indent}        if template is None:
{indent}            logger.warning(f"Template not found for {{model_arg}}, using default")
{indent}            template = self.template_db.get_template('default', 'test')
{indent}    else:
{indent}"""
                    
                    # Find the beginning of the line
                    line_start = content.rfind("\n", 0, template_load_start) + 1
                    
                    # Insert db code and indent the original template loading
                    indented_original = add_indent(content[line_start:template_load_line_end], 4)
                    content = content[:line_start] + db_code + indented_original + content[template_load_line_end:]
        
        # Save changes only if content has been modified
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True, f"Updated {file_path} to use database templates"
        
        return False, f"No changes needed for {file_path}"
    
    except Exception as e:
        return False, f"Error updating {file_path}: {e}"

def find_function_end(content: str, func_body_start: int) -> int:
    """Find the end of a function body.
    
    Args:
        content: File content
        func_body_start: Starting position of function body
        
    Returns:
        Ending position of function body
    """
    # Find indent level of function
    func_line_start = content.rfind("\n", 0, func_body_start) + 1
    indent = len(content[func_line_start:func_body_start].replace(content[func_line_start:func_body_start].lstrip(), ''))
    
    # Scan forward to find end of function (first line with same or less indent)
    lines = content[func_body_start:].split("\n")
    line_count = 0
    in_docstring = False
    
    for i, line in enumerate(lines):
        # Skip first line (function header)
        if i == 0:
            line_count += 1
            continue
        
        # Handle docstrings
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.endswith('"""') or stripped.endswith("'''"):
                # Single line docstring
                line_count += 1
                continue
            
            # Toggle docstring mode
            in_docstring = not in_docstring
            line_count += 1
            continue
        
        if in_docstring:
            # Still in docstring
            line_count += 1
            continue
        
        # Skip empty lines or comments
        if not stripped or stripped.startswith('#'):
            line_count += 1
            continue
        
        # Check indent level
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= indent and i > 0:
            # Found end of function
            break
        
        line_count += 1
    
    # Calculate end position
    end_pos = func_body_start + sum(len(lines[i]) + 1 for i in range(line_count))
    return min(end_pos, len(content))

def get_indent(content: str, pos: int) -> str:
    """Get the indentation at a position in the content.
    
    Args:
        content: File content
        pos: Position to check
        
    Returns:
        Indentation string
    """
    line_start = content.rfind("\n", 0, pos) + 1
    return content[line_start:pos].replace(content[line_start:pos].lstrip(), '')

def add_indent(text: str, spaces: int) -> str:
    """Add indentation to text.
    
    Args:
        text: Text to indent
        spaces: Number of spaces to add
        
    Returns:
        Indented text
    """
    indent = ' ' * spaces
    lines = text.split("\n")
    return "\n".join(indent + line for line in lines)

def update_all_generators(base_dir: str, generators: List[str] = None) -> Dict[str, Any]:
    """Update all generator files to use database templates.
    
    Args:
        base_dir: Base directory containing generators
        generators: List of generator files to update (default: TARGET_GENERATORS)
        
    Returns:
        Dictionary with results
    """
    generators = generators or TARGET_GENERATORS
    results = {
        'total': 0,
        'updated': 0,
        'skipped': 0,
        'failed': 0,
        'details': []
    }
    
    # Find generator files in the directory tree
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in generators:
                results['total'] += 1
                file_path = os.path.join(root, file)
                
                # Update the generator
                success, message = update_static_to_db_templates(file_path)
                
                if success:
                    logger.info(message)
                    results['updated'] += 1
                    results['details'].append({
                        'file': file,
                        'path': file_path,
                        'status': 'updated',
                        'message': message
                    })
                else:
                    if "No changes needed" in message:
                        logger.info(message)
                        results['skipped'] += 1
                        results['details'].append({
                            'file': file,
                            'path': file_path,
                            'status': 'skipped',
                            'message': message
                        })
                    else:
                        logger.error(message)
                        results['failed'] += 1
                        results['details'].append({
                            'file': file,
                            'path': file_path,
                            'status': 'failed',
                            'message': message
                        })
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update generator files to use database templates"
    )
    parser.add_argument(
        "--dir", type=str, default="/home/barberb/ipfs_accelerate_py/generators",
        help="Base directory containing generators"
    )
    parser.add_argument(
        "--generators", type=str,
        help="Comma-separated list of generator files to update"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Get list of generators to update
    generators = TARGET_GENERATORS
    if args.generators:
        generators = args.generators.split(',')
    
    # Update all generator files
    logger.info(f"Updating generators in {args.dir}")
    results = update_all_generators(args.dir, generators)
    
    logger.info(f"Total generators: {results['total']}")
    logger.info(f"Updated generators: {results['updated']}")
    logger.info(f"Skipped generators: {results['skipped']}")
    logger.info(f"Failed generators: {results['failed']}")
    
    return 0 if results['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
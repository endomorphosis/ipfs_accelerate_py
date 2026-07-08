#!/usr/bin/env python3
"""
Enable DB templates in test generators.

This script adds the necessary code to test generators to support the --use-db-templates flag
and environment variable, completing the migration to database templates.
"""

import os
import re
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target generators
DEFAULT_GENERATORS = [
    'simple_test_generator.py',
    'merged_test_generator.py',
    'fixed_merged_test_generator.py',
    'fixed_merged_test_generator_clean.py',
    'qualified_test_generator.py'
]

def update_generator(file_path):
    """
    Update a generator file to support the --use-db-templates flag.
    
    Args:
        file_path: Path to the generator file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # 1. Add DB templates flag to argument parser if needed
        if '--use-db-templates' not in content:
            parser_match = re.search(r'parser\s*=\s*argparse\.ArgumentParser\(', content)
            if parser_match:
                # Find where arguments are added
                arg_matches = list(re.finditer(r'parser\.add_argument\(', content))
                if arg_matches:
                    # Find the last argument
                    last_arg = arg_matches[-1]
                    # Find the end of the add_argument call
                    end_pos = content.find('\n', last_arg.start())
                    end_pos = content.find('\n', end_pos + 1)
                    
                    # Add our flag after the last argument
                    db_arg = """
    parser.add_argument(
        "--use-db-templates", action="store_true",
        help="Use database templates instead of static files"
    )"""
                    
                    content = content[:end_pos+1] + db_arg + content[end_pos+1:]
                    logger.info(f"Added --use-db-templates flag to {file_path}")
        
        # 2. Add code to check for environment variable if not present
        if 'USE_DB_TEMPLATES' not in content:
            # Find a good place to add the check - inside main function or generate function
            main_match = re.search(r'def\s+main\s*\(', content)
            if main_match:
                # Find the start of the main function body
                body_start = content.find(':', main_match.start())
                body_start = content.find('\n', body_start) + 1
                
                # Add env var check
                env_check = """
    # Check for database templates flag from environment variable
    use_db_templates = args.use_db_templates if hasattr(args, 'use_db_templates') else False
    if not use_db_templates and os.environ.get('USE_DB_TEMPLATES', '').lower() in ('1', 'true', 'yes'):
        logger.info("Using database templates from environment variable")
        use_db_templates = True
        
"""
                
                content = content[:body_start] + env_check + content[body_start:]
                logger.info(f"Added environment variable check to {file_path}")
        
        # 3. Pass the flag to the generate function
        if '--use-db-templates' in content and 'use_db_templates=' not in content:
            # Find the call to generate function
            generate_match = re.search(r'generate\w*\(\s*(?:model|model_name|model_type)', content)
            if generate_match:
                # Find the end of the function call
                end_pos = content.find(')', generate_match.start())
                
                # Add the flag
                content = content[:end_pos] + ", use_db_templates=use_db_templates" + content[end_pos:]
                logger.info(f"Added use_db_templates flag to generate call in {file_path}")
        
        # 4. Update the generate function to handle the flag
        if 'use_db_templates=' in content and 'use_db_templates=False' not in content:
            # Find the generate function definition
            generate_func = re.search(r'def\s+generate\w*\(', content)
            if generate_func:
                # Find the end of the parameter list
                params_end = content.find(')', generate_func.start())
                
                # Add the parameter if not present
                if 'use_db_templates' not in content[generate_func.start():params_end]:
                    content = content[:params_end] + ", use_db_templates=False" + content[params_end:]
                    logger.info(f"Added use_db_templates parameter to generate function in {file_path}")
                
                # Find the function body
                body_start = content.find(':', generate_func.start())
                body_end = find_function_end(content, body_start)
                
                # Check if the function already has code for db templates
                if 'if use_db_templates' not in content[body_start:body_end]:
                    # Find where template loading happens
                    template_loading = re.search(r'template\s*=', content[body_start:body_end])
                    if template_loading:
                        # Position in the full content
                        template_pos = body_start + template_loading.start()
                        line_start = content.rfind('\n', 0, template_pos) + 1
                        
                        # Get the indentation
                        indent = ' ' * (len(content[line_start:template_pos]) - len(content[line_start:template_pos].lstrip()))
                        
                        # Add the conditional code
                        db_code = f"""
{indent}# Use database templates if requested
{indent}if use_db_templates:
{indent}    logger.info(f"Using database templates for {{model_type}}")
{indent}    # This is a placeholder - in a real implementation, 
{indent}    # you would load from a template database here
{indent}    # Example:
{indent}    # from templates.template_database import TemplateDatabase
{indent}    # db = TemplateDatabase()
{indent}    # template = db.get_template(model_type, 'test')
{indent}else:
{indent}"""
                        
                        # Indent the original template loading
                        line_end = content.find('\n', template_pos)
                        original_line = content[line_start:line_end]
                        indented_line = indent + '    ' + original_line.lstrip()
                        
                        # Replace the line
                        content = content[:line_start] + db_code + '\n' + indented_line + content[line_end:]
                        logger.info(f"Added template database support to generate function in {file_path}")
        
        # Save changes if content was modified
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Updated {file_path}")
            return True
        
        logger.info(f"No changes needed for {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")
        return False

def find_function_end(content, start_pos):
    """
    Find the end of a function definition.
    
    Args:
        content: File content
        start_pos: Starting position of the function
        
    Returns:
        Position of function end
    """
    # Find the line with function definition
    line_start = content.rfind('\n', 0, start_pos) + 1
    
    # Get the indentation level
    indent = 0
    for char in content[line_start:start_pos]:
        if char == ' ':
            indent += 1
        else:
            break
    
    # Find the end of the function
    pos = start_pos
    while pos < len(content):
        next_line = content.find('\n', pos + 1)
        if next_line == -1:
            return len(content)
        
        # Check indentation of next non-empty line
        line_pos = next_line + 1
        while line_pos < len(content) and content[line_pos:line_pos+1].isspace():
            line_pos = content.find('\n', line_pos) + 1
            if line_pos <= 0:
                return len(content)
        
        # If we find a line with same or less indentation, we've reached the end
        line_indent = 0
        for i in range(line_pos, min(line_pos + indent + 1, len(content))):
            if content[i] == ' ':
                line_indent += 1
            else:
                break
        
        if line_indent <= indent and not content[line_pos:].strip().startswith('#'):
            return line_pos
        
        pos = next_line
    
    return len(content)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enable DB templates in test generators"
    )
    parser.add_argument(
        '--dir', type=str, default='.',
        help='Directory to search for generators'
    )
    parser.add_argument(
        '--generators', type=str,
        help='Comma-separated list of generators to update (default: all)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
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
    generators = DEFAULT_GENERATORS
    if args.generators:
        generators = args.generators.split(',')
    
    logger.info(f"Searching for generators in {args.dir}")
    
    # Find and update generators
    found_count = 0
    updated_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file in generators:
                found_count += 1
                file_path = os.path.join(root, file)
                logger.info(f"Found generator: {file_path}")
                
                # Update the generator
                if update_generator(file_path):
                    updated_count += 1
                else:
                    failed_count += 1
    
    # Print summary
    print(f"\nFound {found_count} generators")
    print(f"Updated {updated_count} generators")
    print(f"Failed to update {failed_count} generators")
    
    if found_count == 0:
        logger.warning(f"No generators found in {args.dir}")
        print(f"\nNo generators found in {args.dir}")
        print(f"Looking for: {', '.join(generators)}")
        return 1
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
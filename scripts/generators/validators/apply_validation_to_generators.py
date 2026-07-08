#!/usr/bin/env python3
"""
Apply Validation to Generators

This script adds template validation to existing generator files by adding import
statements and validation logic.
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMPORT_PATTERN = '''
# Import template validator
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.generators.validators.template_validator_integration import (
        validate_template_for_generator,
        validate_template_file_for_generator
    )
    HAS_VALIDATOR = True
    logger.info("Template validator loaded successfully")
except ImportError:
    HAS_VALIDATOR = False
    logger.warning("Template validator not found. Templates will not be validated.")
    
    # Define minimal validation function
    def validate_template_for_generator(template_content, generator_type, **kwargs):
        return True, []
        
    def validate_template_file_for_generator(file_path, generator_type, **kwargs):
        return True, []
'''

VALIDATION_ARGS_PATTERN = '''
    # Validation options
    parser.add_argument("--validate", action="store_true", 
                     help="Validate templates before generation (default if validator available)")
    parser.add_argument("--skip-validation", action="store_true",
                     help="Skip template validation even if validator is available")
    parser.add_argument("--strict-validation", action="store_true",
                     help="Fail on validation errors")
'''

VALIDATION_LOGIC_PATTERN = '''
        # Validate the generated template content
        should_validate = HAS_VALIDATOR and (getattr(self.args, "validate", True) and not getattr(self.args, "skip_validation", False))
        
        if should_validate:
            logger.info(f"Validating template for {model_name}...")
            is_valid, validation_errors = validate_template_for_generator(
                content, 
                "{generator_type}",
                validate_hardware=True,
                check_resource_pool=True
            )
            
            if not is_valid:
                logger.warning(f"Generated template has validation errors:")
                for error in validation_errors:
                    logger.warning(f"  - {error}")
                
                if getattr(self.args, "strict_validation", False):
                    raise ValueError(f"Template validation failed for {model_name}")
                else:
                    logger.warning("Continuing despite validation errors (use --strict-validation to fail on errors)")
            else:
                logger.info(f"Template validation passed for {model_name}")
        elif getattr(self.args, "validate", False) and not HAS_VALIDATOR:
            logger.warning("Template validation requested but validator not available. Skipping validation.")
'''

def apply_imports(content: str) -> str:
    """
    Add validator imports to the file content.
    
    Args:
        content: Original file content
        
    Returns:
        Updated file content with imports
    """
    # Check if imports already exist
    if "from scripts.generators.validators.template_validator_integration import" in content:
        logger.info("Imports already exist, skipping")
        return content
        
    # Find the logging import
    logging_match = re.search(r'import logging.*?\n', content, re.DOTALL)
    if logging_match:
        insert_position = logging_match.end()
        return content[:insert_position] + IMPORT_PATTERN + content[insert_position:]
    
    # If no logging import, find last import
    import_matches = list(re.finditer(r'^import .*?$|^from .*? import', content, re.MULTILINE))
    if import_matches:
        last_import = import_matches[-1]
        insert_position = last_import.end()
        while insert_position < len(content) and content[insert_position] != '\n':
            insert_position += 1
        if insert_position < len(content):
            insert_position += 1  # Move past the newline
        return content[:insert_position] + IMPORT_PATTERN + content[insert_position:]
    
    # If no imports, add after shebang or at beginning
    shebang_match = re.search(r'^#!/usr/bin/env python.*?\n', content)
    if shebang_match:
        insert_position = shebang_match.end()
        return content[:insert_position] + IMPORT_PATTERN + content[insert_position:]
    
    # Add at beginning as last resort
    return IMPORT_PATTERN + content
    
def apply_args(content: str) -> str:
    """
    Add validation arguments to the argument parser.
    
    Args:
        content: Original file content
        
    Returns:
        Updated file content with validation arguments
    """
    # Check if args already exist
    if "--validate" in content and "--skip-validation" in content:
        logger.info("Validation arguments already exist, skipping")
        return content
        
    # Find the argument parser definition
    arg_parser_match = re.search(r'parser\s*=\s*argparse\.ArgumentParser\(.*?\)', content, re.DOTALL)
    if not arg_parser_match:
        logger.warning("Could not find argument parser, skipping arguments")
        return content
        
    # Find the end of argument definitions
    # Look for the parse_args() call
    parse_args_match = re.search(r'args\s*=\s*parser\.parse_args\(\)', content, re.DOTALL)
    if parse_args_match:
        # Insert before parse_args()
        insert_position = parse_args_match.start()
        # Find the last occurrence of parser.add_argument before parse_args
        parser_args = list(re.finditer(r'parser\.add_argument\(.*?\)', content[:insert_position], re.DOTALL))
        if parser_args:
            last_arg = parser_args[-1]
            insert_position = last_arg.end()
            # Find the next line after the last argument
            while insert_position < len(content) and content[insert_position] != '\n':
                insert_position += 1
            if insert_position < len(content):
                insert_position += 1  # Move past the newline
            return content[:insert_position] + VALIDATION_ARGS_PATTERN + content[insert_position:]
    
    logger.warning("Could not find parse_args() call, skipping arguments")
    return content
    
def apply_validation_logic(content: str, generator_type: str) -> str:
    """
    Add validation logic to the file content.
    
    Args:
        content: Original file content
        generator_type: Type of generator
        
    Returns:
        Updated file content with validation logic
    """
    # Check if validation logic already exists
    if "validate_template_for_generator" in content and "should_validate" in content:
        logger.info("Validation logic already exists, skipping")
        return content
        
    # Find where to insert validation logic
    # Look for places where a template is generated and written to a file
    patterns = [
        r'generate_test_file\(.*?\)',
        r'content\s*=\s*[^=]*template\s*\..*?',
        r'with\s+open\(.*?\)\s+as\s+f:\s*\n\s*f\.write\(.*?\)'
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, content, re.DOTALL))
        for match in matches:
            # Find where to insert validation logic
            context_start = max(0, match.start() - 500)  # Look at surrounding context
            context_end = min(len(content), match.end() + 500)
            context = content[context_start:context_end]
            
            # Look for the pattern "content = " in the context
            content_match = re.search(r'content\s*=\s*.*?', context, re.DOTALL)
            if content_match:
                # Find the end of the content assignment statement
                content_end = content_match.end()
                # Find the next line after content assignment
                while content_end < len(context) and context[content_end] != '\n':
                    content_end += 1
                if content_end < len(context):
                    content_end += 1  # Move past the newline
                
                # Insert the validation logic
                validation_logic = VALIDATION_LOGIC_PATTERN.format(generator_type=generator_type)
                full_content = context[:content_end] + validation_logic + context[content_end:]
                
                # Replace the context in the original content
                return content[:context_start] + full_content + content[context_end:]
    
    logger.warning("Could not find appropriate place to insert validation logic")
    return content
    
def apply_validation(file_path: str, generator_type: str) -> bool:
    """
    Apply validation to a generator file.
    
    Args:
        file_path: Path to the generator file
        generator_type: Type of generator
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Adding validation to {file_path}")
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply transformations
        content = apply_imports(content)
        content = apply_args(content)
        content = apply_validation_logic(content, generator_type)
        
        # Write the file back
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Successfully added validation to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error adding validation to {file_path}: {str(e)}")
        return False
        
def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Apply Template Validation to Generators")
    parser.add_argument("--file", type=str, help="Path to the generator file")
    parser.add_argument("--generator-type", type=str, default="generic", help="Type of generator")
    parser.add_argument("--directory", type=str, help="Directory containing generator files")
    parser.add_argument("--pattern", type=str, default="*_generator.py", help="Pattern for generator files")
    
    args = parser.parse_args()
    
    if args.file:
        # Apply to a single file
        apply_validation(args.file, args.generator_type)
    elif args.directory:
        # Apply to all matching files in directory
        import glob
        
        pattern = os.path.join(args.directory, args.pattern)
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"No files matching {pattern} found")
            return 1
            
        logger.info(f"Found {len(files)} files matching {pattern}")
        
        success_count = 0
        for file_path in files:
            # Derive generator type from filename
            filename = os.path.basename(file_path)
            generator_type = os.path.splitext(filename)[0]
            
            if apply_validation(file_path, generator_type):
                success_count += 1
                
        logger.info(f"Successfully added validation to {success_count}/{len(files)} files")
    else:
        parser.print_help()
        
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script to verify the syntax of template files.

This script checks all template files in the templates directory to ensure
they have valid Python syntax.
"""

import os
import sys
import ast
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_template_syntax(template_path):
    """
    Check if a template file has valid Python syntax.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Replace placeholders with dummy values to make it valid Python
        template_content = template_content.replace("{model_type}", "bert")
        template_content = template_content.replace("{model_type_upper}", "BERT")
        template_content = template_content.replace("{model_class_name}", "Bert")
        template_content = template_content.replace("{processor_class_name}", "BertProcessor")
        template_content = template_content.replace("{model_class_name_short}", "Bert")
        template_content = template_content.replace("{default_model_id}", "bert-base-uncased")
        template_content = template_content.replace("{sampling_rate}", "16000")
        template_content = template_content.replace("{device_init_code}", "# CPU initialization")
        
        # Try to parse the template as Python code
        ast.parse(template_content)
        return True, "Template has valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error in {template_path.name}: {str(e)}"
    except Exception as e:
        return False, f"Error checking template {template_path.name}: {str(e)}"

def main():
    """Main function to check all templates."""
    parser = argparse.ArgumentParser(description="Check template syntax")
    parser.add_argument("--template-dir", type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
                        help="Directory containing template files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    template_dir = Path(args.template_dir)
    if not template_dir.exists() or not template_dir.is_dir():
        logger.error(f"Template directory {template_dir} does not exist or is not a directory")
        return 1
    
    # Find all template files
    template_files = list(template_dir.glob("*_template.py"))
    
    if not template_files:
        logger.warning(f"No template files found in {template_dir}")
        return 0
    
    logger.info(f"Found {len(template_files)} template files")
    
    # Check each template
    all_valid = True
    for template_path in template_files:
        is_valid, message = check_template_syntax(template_path)
        if is_valid:
            status = "✅ VALID"
            if args.verbose:
                logger.info(f"{status}: {template_path.name}")
        else:
            status = "❌ INVALID"
            logger.error(f"{status}: {template_path.name} - {message}")
            all_valid = False
    
    if all_valid:
        logger.info("All templates have valid syntax")
        return 0
    else:
        logger.error("Some templates have syntax errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
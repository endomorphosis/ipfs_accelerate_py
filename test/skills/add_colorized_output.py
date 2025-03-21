#!/usr/bin/env python3
"""
Add colorized output to the mock detection indicators in template files.

This script enhances the visibility of mock detection indicators
with ANSI color codes for better readability in terminals.

Usage:
    python add_colorized_output.py [--check-only] [--template TEMPLATE_FILE]
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
        logging.FileHandler(f"colorize_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def check_colorized_output(content):
    """
    Check if colorized output is implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        bool: True if colorized output is implemented, False otherwise
    """
    # Check for key color indicator patterns
    has_color_definitions = "GREEN =" in content and "BLUE =" in content and "RESET =" in content
    has_green_rocket = 'f"{GREEN}🚀' in content
    has_blue_diamond = 'f"{BLUE}🔷' in content
    
    return has_color_definitions and has_green_rocket and has_blue_diamond

def add_colorized_output(content):
    """
    Add colorized output to the content.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with colorized output
    """
    if check_colorized_output(content):
        return content
    
    # Add color definitions after the imports
    if "import datetime" in content:
        datetime_import_index = content.find("import datetime")
        next_line_index = content.find("\n", datetime_import_index)
        if next_line_index != -1:
            color_definitions = """
# ANSI color codes for terminal output
GREEN = "\\033[32m"
BLUE = "\\033[34m"
RESET = "\\033[0m"
"""
            content = content[:next_line_index + 1] + color_definitions + content[next_line_index + 1:]
    
    # Update the real inference visual indicator
    content = content.replace(
        'print(f"🚀 Using REAL INFERENCE with actual models")',
        'print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")'
    )
    
    # Update the mock objects visual indicator
    content = content.replace(
        'print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")',
        'print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")'
    )
    
    return content

def process_template(template_path, check_only=False, create_backup=True):
    """
    Process a template file to ensure it has colorized output implemented.
    
    Args:
        template_path: Path to the template file
        check_only: If True, only check for colorized output without modifying
        create_backup: If True, create a backup before modifying
        
    Returns:
        bool: True if template has or now has colorized output, False otherwise
    """
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check if colorized output is implemented
        has_colorized_output = check_colorized_output(content)
        
        if has_colorized_output:
            logger.info(f"✅ {template_path}: Colorized output is already implemented")
            return True
        else:
            logger.warning(f"❌ {template_path}: Colorized output is missing")
            
            if check_only:
                return False
            
            # Create backup if requested
            if create_backup:
                backup_path = f"{template_path}.bak"
                with open(backup_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created backup at {backup_path}")
            
            # Add colorized output
            updated_content = add_colorized_output(content)
            
            # Write updated content
            with open(template_path, 'w') as f:
                f.write(updated_content)
            
            # Verify update
            with open(template_path, 'r') as f:
                new_content = f.read()
            
            has_colorized_output = check_colorized_output(new_content)
            if has_colorized_output:
                logger.info(f"✅ {template_path}: Successfully added colorized output")
                return True
            else:
                logger.error(f"❌ {template_path}: Failed to add colorized output")
                return False
            
    except Exception as e:
        logger.error(f"Error processing template {template_path}: {e}")
        return False

def process_all_templates(templates_dir="templates", check_only=False):
    """
    Process all template files in the given directory.
    
    Args:
        templates_dir: Directory containing template files
        check_only: If True, only check for colorized output without modifying
        
    Returns:
        Tuple of (success_count, failure_count, total_count)
    """
    success_count = 0
    failure_count = 0
    
    try:
        templates_path = os.path.join(os.path.dirname(__file__), templates_dir)
        template_files = []
        
        for file in os.listdir(templates_path):
            if file.endswith("_template.py"):
                template_files.append(os.path.join(templates_path, file))
        
        logger.info(f"Found {len(template_files)} template files to process")
        
        for template_path in template_files:
            if process_template(template_path, check_only):
                success_count += 1
            else:
                failure_count += 1
        
        return success_count, failure_count, len(template_files)
    
    except Exception as e:
        logger.error(f"Error processing templates: {e}")
        return success_count, failure_count, 0

def main():
    parser = argparse.ArgumentParser(description="Add colorized output to template files")
    parser.add_argument("--check-only", action="store_true", help="Only check for colorized output without modifying")
    parser.add_argument("--template", type=str, help="Process a specific template file")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    
    args = parser.parse_args()
    
    create_backup = not args.no_backup
    
    if args.template:
        template_path = args.template
        if not os.path.exists(template_path):
            template_path = os.path.join(os.path.dirname(__file__), "templates", args.template)
        
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return 1
        
        success = process_template(template_path, args.check_only, create_backup)
        
        if args.check_only:
            print(f"\nTemplate check: {'✅ Has colorized output' if success else '❌ Missing colorized output'}")
        else:
            print(f"\nTemplate processing: {'✅ Success' if success else '❌ Failed'}")
        
        return 0 if success else 1
    else:
        success_count, failure_count, total_count = process_all_templates(check_only=args.check_only)
        
        print("\nTemplate Processing Summary:")
        if args.check_only:
            print(f"- Templates with colorized output: {success_count}/{total_count}")
            print(f"- Templates missing colorized output: {failure_count}/{total_count}")
        else:
            print(f"- Successfully processed: {success_count}/{total_count}")
            print(f"- Failed to process: {failure_count}/{total_count}")
        
        return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
#\!/usr/bin/env python3
"""
Fix for the text_embedding_test_template_text_embedding.py template.

This script attempts to fix the syntax errors in the text_embedding template.
"""

import json
import sys
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JSON DB path
JSON_DB_PATH = "../generators/templates/template_db.json"

def load_template_db(db_path):
    """Load the template database from a JSON file"""
    with open(db_path, 'r') as f:
        db = json.load(f)
    return db

def fix_text_embedding_template(db):
    """Fix the text_embedding template in the database"""
    template_id = "text_embedding_test_template_text_embedding.py"
    
    if template_id not in db['templates']:
        logger.error(f"Template {template_id} not found in database")
        return False
    
    # Get the original template content
    original_content = db['templates'][template_id].get('template', '')
    
    # Fix 1: Fix the test cases where brackets are mismatched
    fixed_content = original_content
    
    # Look for mismatched brackets in test cases
    # Pattern for test cases section
    test_cases_pattern = r'(self\.test_cases = \[.*?\])'
    test_cases_match = re.search(test_cases_pattern, original_content, re.DOTALL)
    
    if test_cases_match:
        test_cases_section = test_cases_match.group(1)
        
        # Fix the test cases section - replace curly braces with proper syntax
        fixed_test_cases = test_cases_section.replace(
            '"expected": {},', 
            '"expected": {"success": true},'
        ).replace(
            '"data": {}', 
            '"data": {"input": "This is a test sentence for embedding"}'
        )
        
        # Replace in the original content
        fixed_content = original_content.replace(test_cases_section, fixed_test_cases)
    
    # Fix 2: Fix any indentation issues
    # Split into lines for indentation fixing
    lines = fixed_content.split('\n')
    fixed_lines = []
    
    # Track indentation level
    indent_level = 0
    in_class = False
    in_method = False
    
    for i, line in enumerate(lines):
        # Check if this is a class definition
        if re.match(r'^\s*class\s+', line):
            in_class = True
            indent_level = 0
            fixed_lines.append(line)
            continue
            
        # Check if this is a method definition
        if in_class and re.match(r'^\s+def\s+', line):
            in_method = True
            method_indent = len(line) - len(line.lstrip())
            indent_level = method_indent + 4  # Standard 4-space indent
            fixed_lines.append(line)
            continue
            
        # Check if we're exiting a method
        if in_method and not line.strip():
            in_method = False
            fixed_lines.append(line)
            continue
            
        # Fix indentation for lines within methods
        if in_method and line.strip():
            current_indent = len(line) - len(line.lstrip())
            if current_indent < indent_level:
                # This line is under-indented, fix it
                line = ' ' * indent_level + line.lstrip()
            
        fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    # Fix 3: Fix missing model attribute
    if 'self.model = None' not in fixed_content:
        # Find the __init__ method
        init_pattern = r'def __init__\(self, model_path=None\):(.*?)def'
        init_match = re.search(init_pattern, fixed_content, re.DOTALL)
        
        if init_match:
            init_section = init_match.group(1)
            # Add the model attribute after tokenizer
            fixed_init = init_section.replace(
                'self.tokenizer = None', 
                'self.tokenizer = None\n        self.model = None'
            )
            fixed_content = fixed_content.replace(init_section, fixed_init)
    
    # Update the template in the database
    db['templates'][template_id]['template'] = fixed_content
    
    # Save the fixed template to a local file for inspection
    with open('fixed_text_embedding.py', 'w') as f:
        f.write(fixed_content)
    
    return True

def save_template_db(db, db_path):
    """Save the template database to a JSON file"""
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)
    return True

def main():
    """Main function"""
    try:
        # Load the template database
        db = load_template_db(JSON_DB_PATH)
        
        # Fix the text_embedding template
        if fix_text_embedding_template(db):
            logger.info("Successfully fixed text_embedding template. Saved to fixed_text_embedding.py")
            
            # Save the updated database
            #if save_template_db(db, JSON_DB_PATH):
            #    logger.info(f"Successfully saved updated database to {JSON_DB_PATH}")
            
            # We'll comment out the actual save to prevent modifying the database until we're ready
            logger.info("NOTE: Database not actually updated. Uncomment the save_template_db call to update.")
            
            return 0
        else:
            logger.error("Failed to fix text_embedding template")
            return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

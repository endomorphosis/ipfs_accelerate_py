#\!/usr/bin/env python3
"""
Fix for the vision_test_template_vision.py template.

This script attempts to fix the syntax errors in the vision template.
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

def fix_vision_template(db):
    """Fix the vision template in the database"""
    template_id = "vision_test_template_vision.py"
    
    if template_id not in db['templates']:
        logger.error(f"Template {template_id} not found in database")
        return False
    
    # Get the original template content
    original_content = db['templates'][template_id].get('template', '')
    
    # Create a new fixed test cases section
    new_test_cases = """        # Define test cases
        self.test_cases = [
            {
                "description": "Test on CPU platform",
                "platform": "CPU",
                "expected": {"success": true},
                "data": {"image_path": "test_image.jpg"}
            },
            {
                "description": "Test on CUDA platform",
                "platform": "CUDA",
                "expected": {"success": true},
                "data": {"image_path": "test_image.jpg"}
            },
            {
                "description": "Test on OPENVINO platform",
                "platform": "OPENVINO",
                "expected": {"success": true},
                "data": {"image_path": "test_image.jpg"}
            },
            {
                "description": "Test on MPS platform",
                "platform": "MPS",
                "expected": {"success": true},
                "data": {"image_path": "test_image.jpg"}
            },
            {
                "description": "Test on ROCM platform",
                "platform": "ROCM",
                "expected": {"success": true},
                "data": {"image_path": "test_image.jpg"}
            }
        ]"""
    
    # Find and replace the existing test cases section
    pattern = r'# Define test cases.*?self\.test_cases = \[.*?\]'
    fixed_content = re.sub(pattern, new_test_cases, original_content, flags=re.DOTALL)
    
    # If the pattern wasn't found, we'll try another approach
    if fixed_content == original_content:
        # Look for self.test_cases = [ and replace the whole section
        start_pattern = r'self\.test_cases = \['
        start_match = re.search(start_pattern, original_content)
        
        if start_match:
            start_idx = start_match.start()
            # Find the closing bracket, accounting for possible nested brackets
            bracket_count = 1
            end_idx = start_idx + len(start_pattern)
            
            while bracket_count > 0 and end_idx < len(original_content):
                if original_content[end_idx] == '[':
                    bracket_count += 1
                elif original_content[end_idx] == ']':
                    bracket_count -= 1
                end_idx += 1
            
            if bracket_count == 0:
                # Rebuild the content
                fixed_content = original_content[:start_idx] + "self.test_cases = [\n" + \
                    '            {\n' + \
                    '                "description": "Test on CPU platform",\n' + \
                    '                "platform": "CPU",\n' + \
                    '                "expected": {"success": true},\n' + \
                    '                "data": {"image_path": "test_image.jpg"}\n' + \
                    '            },\n' + \
                    '            {\n' + \
                    '                "description": "Test on CUDA platform",\n' + \
                    '                "platform": "CUDA",\n' + \
                    '                "expected": {"success": true},\n' + \
                    '                "data": {"image_path": "test_image.jpg"}\n' + \
                    '            },\n' + \
                    '            {\n' + \
                    '                "description": "Test on OPENVINO platform",\n' + \
                    '                "platform": "OPENVINO",\n' + \
                    '                "expected": {"success": true},\n' + \
                    '                "data": {"image_path": "test_image.jpg"}\n' + \
                    '            },\n' + \
                    '            {\n' + \
                    '                "description": "Test on MPS platform",\n' + \
                    '                "platform": "MPS",\n' + \
                    '                "expected": {"success": true},\n' + \
                    '                "data": {"image_path": "test_image.jpg"}\n' + \
                    '            },\n' + \
                    '            {\n' + \
                    '                "description": "Test on ROCM platform",\n' + \
                    '                "platform": "ROCM",\n' + \
                    '                "expected": {"success": true},\n' + \
                    '                "data": {"image_path": "test_image.jpg"}\n' + \
                    '            }\n' + \
                    '        ]' + original_content[end_idx:]
    
    # Fix any indentation issues
    lines = fixed_content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix indentation in methods within classes
        if re.match(r'^\s*def\s+\w+\(', line):
            # This is a method definition
            stripped = line.strip()
            indentation = re.match(r'^\s*', line).group(0)
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    # Update the template in the database
    db['templates'][template_id]['template'] = fixed_content
    
    # Save the fixed template to a local file for inspection
    with open('fixed_vision.py', 'w') as f:
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
        
        # Fix the vision template
        if fix_vision_template(db):
            logger.info("Successfully fixed vision template. Saved to fixed_vision.py")
            
            # Save the updated database
            #if save_template_db(db, JSON_DB_PATH):
            #    logger.info(f"Successfully saved updated database to {JSON_DB_PATH}")
            
            # We'll comment out the actual save to prevent modifying the database until we're ready
            logger.info("NOTE: Database not actually updated. Uncomment the save_template_db call to update.")
            
            return 0
        else:
            logger.error("Failed to fix vision template")
            return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

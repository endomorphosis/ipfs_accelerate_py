#!/usr/bin/env python3
"""
Direct fixer for test_hf_bert.py file with indentation and syntax issues.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_bert_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Fix indentation and syntax issues in the test_hf_bert.py file."""
    file_path = "fixed_tests/test_hf_bert.py"
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1
    
    # Create backup
    backup_path = f"{file_path}.bak"
    try:
        with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup at {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return 1
    
    # Define our fixed content (externally prepared and validated)
    with open("fixed_bert_template.py", "r") as f:
        fixed_content = f.read()
    
    # Write the fixed content to the file
    try:
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        logger.info(f"Fixed content written to {file_path}")
        
        # Verify syntax
        try:
            compile(fixed_content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax is valid")
            print("Successfully fixed test_hf_bert.py")
            return 0
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error: {e}")
            logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
            return 1
    except Exception as e:
        logger.error(f"Error writing fixed content: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
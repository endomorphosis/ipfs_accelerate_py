#!/usr/bin/env python3
"""
Verify Python syntax of a file.

This script compiles a Python file to check for syntax errors.
It does not execute the code, only checks for valid syntax.

Usage:
    python verify_python_syntax.py <file_path>
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_python_syntax(file_path):
    """
    Verify that the Python file has valid syntax.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, file_path, 'exec')
        logger.info(f"✅ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ {file_path}: Syntax error: {e}")
        logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
        return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python verify_python_syntax.py <file_path>")
        return 1
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1
    
    # Verify syntax
    if verify_python_syntax(file_path):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
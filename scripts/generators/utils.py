"""
Common utilities for generators.
"""

import os
import logging
import sys

def setup_logger(name, level=logging.INFO):
    """Set up a logger for generator diagnostics."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def ensure_directory(directory):
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)
    
def template_variable_substitution(template, variables):
    """Substitute variables in a template string."""
    result = template
    for key, value in variables.items():
        placeholder = f"{{{key}}}"
        result = result.replace(placeholder, str(value))
    return result

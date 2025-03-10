#!/usr/bin/env python3
"""
Simple utility functions for the end-to-end testing framework.
"""

import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(logger_instance, level=logging.INFO):
    """Set up logging for a module."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_instance.addHandler(handler)
    logger_instance.setLevel(level)

def ensure_dir_exists(directory_path: str):
    """Ensure a directory exists, creating it if necessary."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {str(e)}")
        raise

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to be safe for all filesystems."""
    # Replace problematic characters
    sanitized = filename.replace('/', '_').replace('\\', '_')
    sanitized = sanitized.replace(':', '_').replace('*', '_')
    sanitized = sanitized.replace('?', '_').replace('"', '_')
    sanitized = sanitized.replace('<', '_').replace('>', '_')
    sanitized = sanitized.replace('|', '_').replace(' ', '_')
    
    return sanitized

def get_file_extension(file_path: str) -> str:
    """Get the extension of a file."""
    return os.path.splitext(file_path)[1].lower()

def human_readable_size(size_bytes: int) -> str:
    """Convert a size in bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
        
def is_valid_model_name(model_name: str) -> bool:
    """Check if a model name is valid."""
    if not model_name:
        return False
    
    # Check for invalid characters
    invalid_chars = ['\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in model_name for char in invalid_chars):
        return False
    
    return True

def get_timestamp_str() -> str:
    """Get the current timestamp as a string in the format YYYYMMDD_HHMMSS."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_timestamped_directory(base_dir: str, prefix: Optional[str] = None) -> str:
    """Create a timestamped directory within the base directory."""
    timestamp = get_timestamp_str()
    if prefix:
        dir_name = f"{prefix}_{timestamp}"
    else:
        dir_name = timestamp
    
    full_path = os.path.join(base_dir, dir_name)
    ensure_dir_exists(full_path)
    
    return full_path
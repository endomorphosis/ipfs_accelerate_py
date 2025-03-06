#!/usr/bin/env python3
"""
Update Hardware Template System for Phase 16

This script updates the hardware template system to integrate with the centralized
hardware detection module and improve consistency across all generated tests.

Usage:
  python update_generators/update_hardware_template_system.py
"""

import os
import re
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("update_hardware_templates")

# Path to centralized hardware detection module
HARDWARE_DETECTION_MODULE_PATH = Path(__file__).parent.parent / "centralized_hardware_detection"

# Path to hardware template directory
HARDWARE_TEMPLATES_DIR = Path(__file__).parent.parent / "hardware_test_templates"

def backup_file(file_path: Path) -> Path:
    """
    Create a backup of a file before modifying it.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak_{timestamp}")
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup of {file_path} at {backup_path}")
    return backup_path

def update_template_file(file_path: Path) -> bool:
    """
    Update a hardware template file to use centralized hardware detection.
    
    Args:
        file_path: Path to the template file to update
        
    Returns:
        True if file was updated
    """
    if not file_path.exists():
        logger.warning(f"File {file_path} not found, skipping")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_file(file_path)
    
    # Add import for centralized hardware detection
    import_pattern = r"import (?:os|sys|importlib|torch).*?(?:^\s*$|\n\n)"
    first_import_match = re.search(import_pattern, content, re.MULTILINE | re.DOTALL)
    
    if first_import_match:
        centralized_import = """
# Centralized hardware detection
from centralized_hardware_detection import (
    get_hardware_manager,
    get_capabilities,
    get_web_optimizations,
    get_browser_info,
    get_model_hardware_compatibility
)

"""
        # Insert after the first import block
        import_end = first_import_match.end()
        content_updated = content[:import_end] + centralized_import + content[import_end:]
        
        # Replace manual hardware detection with centralized detection
        # Look for class initialization where hardware capabilities are checked
        class_init_pattern = r"def __init__\s*\(self[^)]*\):\s*.*?self\.hardware_capabilities\s*=\s*{[^}]*}"
        class_init_match = re.search(class_init_pattern, content_updated, re.DOTALL)
        
        if class_init_match:
            init_code = class_init_match.group(0)
            
            # Add centralized hardware detection
            updated_init = init_code.replace(
                "self.hardware_capabilities = {", 
                "# Get hardware capabilities from centralized system\n        self.hardware_capabilities = get_capabilities()\n        # For backward compatibility\n        self.hardware_capabilities = {"
            )
            
            content_updated = content_updated.replace(init_code, updated_init)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content_updated)
        
        logger.info(f"Updated {file_path} to use centralized hardware detection")
        return True
    else:
        logger.warning(f"Could not find import section in {file_path}, skipping")
        return False

def update_template_database(db_path: Path) -> bool:
    """
    Update the template database to include centralized hardware detection information.
    
    Args:
        db_path: Path to the template database file
        
    Returns:
        True if database was updated
    """
    if not db_path.exists():
        logger.warning(f"Template database {db_path} not found, skipping")
        return False
    
    try:
        # Read the database
        with open(db_path, 'r') as f:
            db_content = json.load(f)
        
        # Create backup
        backup_file(db_path)
        
        # Add centralized hardware detection flag
        db_content["uses_centralized_hardware_detection"] = True
        
        # Update hardware capabilities to use centralized system
        if "hardware_capabilities" in db_content:
            db_content["hardware_capabilities"]["source"] = "centralized_hardware_detection"
        
        # Update each template record
        for template_name, template_data in db_content.get("templates", {}).items():
            template_data["uses_centralized_detection"] = True
        
        # Write updated database
        with open(db_path, 'w') as f:
            json.dump(db_content, f, indent=2)
        
        logger.info(f"Updated template database {db_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating template database {db_path}: {e}")
        return False

def main():
    """Main function to update hardware template system."""
    logger.info("Starting update of hardware template system for Phase 16")
    
    # Check if centralized hardware detection module exists
    if not HARDWARE_DETECTION_MODULE_PATH.exists():
        logger.error(f"Centralized hardware detection module not found at {HARDWARE_DETECTION_MODULE_PATH}")
        return 1
    
    # Check if hardware templates directory exists
    if not HARDWARE_TEMPLATES_DIR.exists():
        logger.error(f"Hardware templates directory not found at {HARDWARE_TEMPLATES_DIR}")
        return 1
    
    # Get all template files
    template_files = list(HARDWARE_TEMPLATES_DIR.glob("template_*.py"))
    logger.info(f"Found {len(template_files)} hardware template files")
    
    # Update each template file
    updated_files = 0
    for template_path in template_files:
        if update_template_file(template_path):
            updated_files += 1
    
    logger.info(f"Updated {updated_files}/{len(template_files)} hardware template files")
    
    # Update template database
    db_path = HARDWARE_TEMPLATES_DIR / "template_database.json"
    if update_template_database(db_path):
        logger.info("Updated template database")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
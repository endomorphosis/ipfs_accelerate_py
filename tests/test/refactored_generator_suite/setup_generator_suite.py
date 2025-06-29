#!/usr/bin/env python3
"""
Setup script for the refactored generator suite.

This script creates the necessary directory structure and ensures
all required files exist.
"""

import os
import shutil
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create the directory structure for the refactored generator suite."""
    # Base directories
    directories = [
        "generated",
        "templates/__pycache__",
        "hardware/__pycache__",
        "generators/__pycache__"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create __init__.py files in each module directory
    init_directories = ["templates", "hardware", "generators"]
    for directory in init_directories:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f"""#!/usr/bin/env python3
\"\"\"
{directory.capitalize()} module for the refactored generator suite.
\"\"\"
""")
            logger.info(f"Created {init_file}")


def check_required_files():
    """Check that all required files exist."""
    required_files = [
        "templates/base_hardware.py",
        "templates/base_architecture.py",
        "templates/base_pipeline.py",
        "templates/template_composer.py",
        "hardware/hardware_detection.py",
        "create_reference_implementations.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("The following required files are missing:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info("All required files exist.")
    return True


def main():
    """Setup the refactored generator suite."""
    logger.info("Setting up the refactored generator suite...")
    
    # Create directory structure
    create_directory_structure()
    
    # Check required files
    if not check_required_files():
        logger.error("Setup failed due to missing required files.")
        sys.exit(1)
    
    # Create output directory for generated files
    os.makedirs("generated", exist_ok=True)
    
    logger.info("Setup completed successfully.")
    logger.info("You can now use the create_reference_implementations.py script to generate model implementations.")


if __name__ == "__main__":
    main()
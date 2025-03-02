#!/usr/bin/env python
"""
Fix API Backend __init__.py file

This script modifies the __init__.py file to properly import 
the API backend classes directly, rather than the modules.
"""

import os
import sys
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_api_init")

# Backend classes to expose
API_BACKENDS = [
    "claude",
    "openai_api",
    "groq",
    "gemini",
    "ollama",
    "hf_tgi",
    "hf_tei",
    "llvm",
    "opea",
    "ovms",
    "s3_kit"
]

def fix_init_py(init_file_path):
    """Fix the __init__.py file to properly import classes directly"""
    # Create backup
    backup_path = f"{init_file_path}.bak2"
    shutil.copy2(init_file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Create new content
    init_content = """# ipfs_accelerate_py.api_backends module initialization
# This module exposes all the API backend classes directly

import logging
logger = logging.getLogger(__name__)

"""

    # Add direct imports for all backend classes
    for backend in API_BACKENDS:
        init_content += f"""try:
    from .{backend} import {backend}
except ImportError as e:
    logger.debug(f"Failed to import {backend}: {{e}}")
    {backend} = None

"""
    
    # Add __all__ list
    init_content += "# List of all backend classes\n"
    init_content += "__all__ = [\n    "
    init_content += ", ".join(f'"{backend}"' for backend in API_BACKENDS)
    init_content += "\n]\n"
    
    # Write the file
    with open(init_file_path, 'w') as f:
        f.write(init_content)
    
    logger.info(f"✅ Updated {init_file_path}")
    return True

def main():
    """Main function to run the script"""
    script_dir = Path(__file__).parent.parent
    api_backends_dir = script_dir / "ipfs_accelerate_py" / "api_backends"
    init_file = api_backends_dir / "__init__.py"
    
    if not api_backends_dir.exists():
        logger.error(f"API backends directory not found at {api_backends_dir}")
        return 1
    
    if not init_file.exists():
        logger.error(f"__init__.py file not found at {init_file}")
        return 1
    
    try:
        success = fix_init_py(init_file)
        if success:
            logger.info("✅ Successfully fixed API __init__.py file")
            return 0
        else:
            logger.error("❌ Failed to fix API __init__.py file")
            return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
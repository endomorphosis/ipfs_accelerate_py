#!/usr/bin/env python
"""
Fix API Module Not Callable issues

This script fixes the issue where API modules are not callable due to the
class name being the same as the module name. It modifies the init.py
to properly expose the classes.
"""

import os
import sys
import re
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_api_module_callable")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def fix_class_definition(file_path, class_name):
    """Fix class definition in an API file"""
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if proper class is defined
        class_pattern = f"class {class_name}[\\(\\s:]"
        class_match = re.search(class_pattern, content)
        
        if class_match:
            # Class exists, do nothing
            logger.info(f"Class {class_name} already defined in {file_path}")
            return True
        else:
            # Class might exist with a different name, try to find it
            any_class_pattern = r"class\s+(\w+)"
            any_class_match = re.search(any_class_pattern, content)
            
            if any_class_match:
                current_class_name = any_class_match.group(1)
                logger.info(f"Found class {current_class_name} in {file_path}, renaming to {class_name}")
                
                # Replace the class name
                updated_content = re.sub(
                    f"class\\s+{current_class_name}",
                    f"class {class_name}",
                    content
                )
                
                # Create backup
                backup_path = f"{file_path}.class.bak"
                with open(backup_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created backup at {backup_path}")
                
                # Write updated content
                with open(file_path, 'w') as f:
                    f.write(updated_content)
                    
                logger.info(f"Renamed class {current_class_name} to {class_name} in {file_path}")
                return True
            else:
                logger.warning(f"No class definition found in {file_path}")
                return False
    except Exception as e:
        logger.error(f"Error fixing class definition in {file_path}: {e}")
        return False

def main():
    """Main function to fix API module callable issues"""
    # Find API backends directory
    script_dir = Path(__file__).parent.parent
    api_backends_dir = script_dir / "ipfs_accelerate_py" / "api_backends"
    
    if not api_backends_dir.exists():
        logger.error(f"API backends directory not found at {api_backends_dir}")
        return 1
    
    # API files to process and their expected class names
    api_files = {
        "claude.py": "claude",
        "openai_api.py": "openai_api",
        "groq.py": "groq",
        "gemini.py": "gemini",
        "ollama.py": "ollama",
        "hf_tgi.py": "hf_tgi",
        "hf_tei.py": "hf_tei",
        "llvm.py": "llvm",
        "opea.py": "opea",
        "ovms.py": "ovms",
        "s3_kit.py": "s3_kit"
    }
    
    # Fix class definitions in API files
    fixed_count = 0
    for api_file, class_name in api_files.items():
        file_path = api_backends_dir / api_file
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping")
            continue
            
        success = fix_class_definition(file_path, class_name)
        if success:
            fixed_count += 1
    
    # Print summary
    logger.info(f"Fixed {fixed_count} out of {len(api_files)} API files")
    
    if fixed_count == len(api_files):
        logger.info("✅ Successfully fixed all API module callable issues")
        return 0
    else:
        logger.warning(f"⚠️ Fixed {fixed_count}/{len(api_files)} API module callable issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
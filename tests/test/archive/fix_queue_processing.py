#!/usr/bin/env python
"""
Fix Queue Processing Attribute Issues

This script fixes the missing queue_processing attribute in API backends.
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
logger = logging.getLogger("fix_queue_processing")

# Backends known to need queue_processing attribute
BACKENDS_TO_FIX = [
    "gemini.py",
    "ovms.py",
    "claude.py",
    "llvm.py",
    "opea.py",
    "s3_kit.py"
]

def fix_queue_processing(file_path):
    """Fix missing queue_processing attribute in a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = f"{file_path}.queue.bak"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        
        # Check if queue_processing is already initialized
        if "self.queue_processing" in content and "self.queue_processing = " not in content:
            # Find the __init__ method
            init_match = re.search(r"def __init__.*?\(.*?\):", content)
            if init_match:
                # Find where to insert the queue_processing initialization
                # Look for common initialization patterns like request_queue or queue_lock
                insertion_points = [
                    "self.request_queue =",
                    "self.queue_lock =",
                    "self.queue_processor =",
                    "self.active_requests ="
                ]
                
                for point in insertion_points:
                    pattern = f"({re.escape(point)}.*?\n)"
                    match = re.search(pattern, content)
                    if match:
                        # Get the indentation
                        indent_match = re.match(r"(\s*)", match.group(1))
                        indent = indent_match.group(1) if indent_match else "        "
                        
                        # Insert queue_processing initialization after this line
                        insert_after = match.end()
                        new_content = content[:insert_after] + f"{indent}self.queue_processing = False\n" + content[insert_after:]
                        
                        # Write updated content
                        with open(file_path, 'w') as f:
                            f.write(new_content)
                            
                        logger.info(f"✅ Added queue_processing attribute to {file_path}")
                        return True
            
            logger.warning(f"Could not find a place to insert queue_processing in {file_path}")
            return False
        else:
            logger.info(f"queue_processing already initialized or not used in {file_path}")
            return True
    except Exception as e:
        logger.error(f"Error fixing queue_processing in {file_path}: {e}")
        return False

def main():
    """Main function to fix queue processing attribute issues"""
    # Find API backends directory
    script_dir = Path(__file__).parent.parent
    api_backends_dir = script_dir / "ipfs_accelerate_py" / "api_backends"
    
    if not api_backends_dir.exists():
        logger.error(f"API backends directory not found at {api_backends_dir}")
        return 1
    
    # Fix each backend
    success_count = 0
    for backend in BACKENDS_TO_FIX:
        file_path = api_backends_dir / backend
        if not file_path.exists():
            logger.warning(f"Backend file {backend} not found, skipping")
            continue
            
        if fix_queue_processing(file_path):
            success_count += 1
    
    # Print summary
    logger.info(f"Fixed queue_processing attribute in {success_count} out of {len(BACKENDS_TO_FIX)} backends")
    
    if success_count == len(BACKENDS_TO_FIX):
        logger.info("✅ Successfully fixed all queue_processing attribute issues")
        return 0
    else:
        logger.warning(f"⚠️ Only fixed {success_count}/{len(BACKENDS_TO_FIX)} queue_processing attribute issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
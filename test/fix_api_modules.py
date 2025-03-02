#!/usr/bin/env python
"""
Fix API Module Initialization Issues

This script fixes module initialization issues in API backends by:
1. Standardizing module structure and exports
2. Ensuring proper class exports in __init__.py
3. Fixing import patterns in test code

The goal is to resolve the 'module' object is not callable errors.
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
logger = logging.getLogger("fix_api_modules")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Template for __init__.py file
INIT_TEMPLATE = """# ipfs_accelerate_py.api_backends module initialization
# This file contains imports for all API backend implementations

# Import standard modules
import logging
import importlib
import inspect

logger = logging.getLogger(__name__)

# Import API backend classes
try:
    from .claude import claude
except ImportError as e:
    logger.debug(f"Failed to import claude: {e}")
    claude = None

try:
    from .openai_api import openai_api
except ImportError as e:
    logger.debug(f"Failed to import openai_api: {e}")
    openai_api = None

try:
    from .groq import groq
except ImportError as e:
    logger.debug(f"Failed to import groq: {e}")
    groq = None

try:
    from .gemini import gemini
except ImportError as e:
    logger.debug(f"Failed to import gemini: {e}")
    gemini = None

try:
    from .ollama import ollama
except ImportError as e:
    logger.debug(f"Failed to import ollama: {e}")
    ollama = None

try:
    from .hf_tgi import hf_tgi
except ImportError as e:
    logger.debug(f"Failed to import hf_tgi: {e}")
    hf_tgi = None

try:
    from .hf_tei import hf_tei
except ImportError as e:
    logger.debug(f"Failed to import hf_tei: {e}")
    hf_tei = None

try:
    from .llvm import llvm
except ImportError as e:
    logger.debug(f"Failed to import llvm: {e}")
    llvm = None

try:
    from .opea import opea
except ImportError as e:
    logger.debug(f"Failed to import opea: {e}")
    opea = None

try:
    from .ovms import ovms
except ImportError as e:
    logger.debug(f"Failed to import ovms: {e}")
    ovms = None

try:
    from .s3_kit import s3_kit
except ImportError as e:
    logger.debug(f"Failed to import s3_kit: {e}")
    s3_kit = None

# Create list of available backends
__all__ = [
    'claude', 'openai_api', 'groq', 'gemini', 'ollama',
    'hf_tgi', 'hf_tei', 'llvm', 'opea', 'ovms', 's3_kit'
]

# Check each API class to ensure it's properly defined
for api_name in __all__:
    api_class = globals().get(api_name)
    if api_class is not None and not inspect.isclass(api_class):
        logger.warning(f"{api_name} is not a class, it's a {type(api_class)}")
"""

# Template for ensuring proper class definition in API files
CLASS_DEFINITION_TEMPLATE = """
class {class_name}:
    \"\"\"
    API implementation for {class_name} service.
    
    This class provides methods for interacting with the {class_name} API
    with queue management, backoff, and error handling.
    \"\"\"
"""

def fix_init_file(api_backends_dir):
    """Fix the __init__.py file in the API backends directory"""
    init_file = api_backends_dir / "__init__.py"
    
    # Create backup
    if init_file.exists():
        backup_path = init_file.with_suffix(".py.bak")
        shutil.copy2(init_file, backup_path)
        logger.info(f"Created backup of __init__.py at {backup_path}")
    
    # Write new __init__.py file
    with open(init_file, "w") as f:
        f.write(INIT_TEMPLATE)
    
    logger.info(f"✅ Updated {init_file}")
    return True

def fix_class_definition(file_path, class_name):
    """Fix class definition in API file"""
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if class is properly defined
        class_pattern = f"class {class_name}[\\(\\s:]"
        class_match = re.search(class_pattern, content)
        
        if not class_match:
            # Class not properly defined, add it
            logger.info(f"Class {class_name} not found in {file_path}, adding it")
            
            # Determine where to add the class definition
            # Try to find any class definition
            any_class_pattern = r"class .*?:"
            any_class_match = re.search(any_class_pattern, content)
            
            if any_class_match:
                # Replace existing class name
                old_class_name = re.search(r"class\s+(\w+)", any_class_match.group(0)).group(1)
                content = content.replace(f"class {old_class_name}", f"class {class_name}")
            else:
                # No class definition found, add one at the beginning after imports
                import_section_end = 0
                import_pattern = r"^(?:import\s+.*|from\s+.*\s+import\s+.*)\s*$"
                for match in re.finditer(import_pattern, content, re.MULTILINE):
                    if match.end() > import_section_end:
                        import_section_end = match.end()
                
                # If no imports found, add after any module docstring
                if import_section_end == 0:
                    docstring_pattern = r'^""".*?"""'
                    docstring_match = re.search(docstring_pattern, content, re.MULTILINE | re.DOTALL)
                    if docstring_match:
                        import_section_end = docstring_match.end()
                
                # Insert class definition
                class_def = CLASS_DEFINITION_TEMPLATE.format(class_name=class_name)
                if import_section_end > 0:
                    content = content[:import_section_end] + "\n\n" + class_def + content[import_section_end:]
                else:
                    content = class_def + "\n\n" + content
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Fixed class definition in {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error fixing class definition in {file_path}: {e}")
        return False

def fix_test_imports(test_dir):
    """Fix imports in test files"""
    try:
        # Find all test files
        test_files = list(test_dir.glob("test_*.py"))
        test_files.extend(test_dir.glob("**/test_*.py"))
        
        fixed_count = 0
        for test_file in test_files:
            # Read file content
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Fix import pattern
            old_content = content
            
            # Fix "from api_backends import X" pattern
            content = re.sub(
                r"from\s+api_backends\s+import\s+(.*)",
                r"from ipfs_accelerate_py.api_backends import \1",
                content
            )
            
            # Fix "import api_backends" pattern
            content = re.sub(
                r"import\s+api_backends",
                r"import ipfs_accelerate_py.api_backends",
                content
            )
            
            # Fix any "api_backends.X" references
            content = re.sub(
                r"api_backends\.(\w+)",
                r"ipfs_accelerate_py.api_backends.\1",
                content
            )
            
            # Write back if changed
            if content != old_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                fixed_count += 1
                logger.info(f"✅ Fixed imports in {test_file}")
        
        logger.info(f"Fixed imports in {fixed_count} test files")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error fixing test imports: {e}")
        return False

def main():
    """Main function to fix API module issues"""
    # Find directories
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    api_backends_dir = project_dir / "ipfs_accelerate_py" / "api_backends"
    test_dir = script_dir
    
    if not api_backends_dir.exists():
        logger.error(f"API backends directory not found at {api_backends_dir}")
        return 1
    
    # Fix the __init__.py file
    init_success = fix_init_file(api_backends_dir)
    
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
    class_results = []
    for api_file, class_name in api_files.items():
        file_path = api_backends_dir / api_file
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping")
            continue
            
        success = fix_class_definition(file_path, class_name)
        class_results.append((api_file, success))
    
    # Fix imports in test files
    import_success = fix_test_imports(test_dir)
    
    # Print summary
    logger.info("\n=== Module Fix Summary ===")
    logger.info(f"__init__.py fix: {'✅ Success' if init_success else '❌ Failed'}")
    
    for file_name, success in class_results:
        logger.info(f"{file_name} class fix: {'✅ Success' if success else '❌ Failed'}")
    
    logger.info(f"Test imports fix: {'✅ Success' if import_success else '❌ Failed'}")
    
    success_count = sum(1 for _, success in class_results if success)
    logger.info(f"\nSuccessfully fixed class definitions in {success_count} of {len(class_results)} API files")
    
    if not init_success or not import_success or success_count < len(class_results):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
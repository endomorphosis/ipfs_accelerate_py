#!/usr/bin/env python3
"""
Add environment variable-based mocking support to template files.

This script adds environment variable control for dependency mocking,
allowing for testing to simulate missing dependencies.

Usage:
    python add_env_mock_support.py [--check-only] [--template TEMPLATE_FILE]
"""

import os
import sys
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"env_mock_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def check_env_mock_support(content):
    """
    Check if environment variable-based mocking is implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        bool: True if env mock support is implemented, False otherwise
    """
    # Check for key environment mock patterns
    has_mock_transformers = "MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'" in content
    has_mock_torch = "MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'" in content
    has_mock_import_check = "if MOCK_TORCH:" in content
    
    return has_mock_transformers and has_mock_torch and has_mock_import_check

def add_env_mock_support(content):
    """
    Add environment variable-based mocking support to content.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with env mock support implemented
    """
    if check_env_mock_support(content):
        return content
    
    # Find import section
    import_matches = re.finditer(r"import os|from os import", content)
    if not import_matches:
        # If os import not found, add it
        import_section = "import os\n"
        if "import sys" in content:
            content = content.replace("import sys", "import os\nimport sys")
    
    # Add environment variable definitions
    if "# Try to import torch" in content:
        torch_import_index = content.find("# Try to import torch")
        if torch_import_index != -1:
            env_vars_definition = """
# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'
"""
            content = content[:torch_import_index] + env_vars_definition + content[torch_import_index:]
    
    # Update torch import
    torch_import_pattern = r"try:\s+import torch\s+HAS_TORCH = True\s+except ImportError:"
    torch_import_with_mock = """try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    HAS_TORCH = True
except ImportError:"""
    content = re.sub(torch_import_pattern, torch_import_with_mock, content)
    
    # Update transformers import
    transformers_import_pattern = r"try:\s+import transformers\s+HAS_TRANSFORMERS = True\s+except ImportError:"
    transformers_import_with_mock = """try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:"""
    content = re.sub(transformers_import_pattern, transformers_import_with_mock, content)
    
    # Update tokenizers import if present
    tokenizers_import_pattern = r"try:\s+import tokenizers\s+HAS_TOKENIZERS = True\s+except ImportError:"
    if tokenizers_import_pattern in content:
        tokenizers_import_with_mock = """try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:"""
        content = re.sub(tokenizers_import_pattern, tokenizers_import_with_mock, content)
    
    # Update sentencepiece import if present
    sentencepiece_import_pattern = r"try:\s+import sentencepiece\s+HAS_SENTENCEPIECE = True\s+except ImportError:"
    if sentencepiece_import_pattern in content:
        sentencepiece_import_with_mock = """try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:"""
        content = re.sub(sentencepiece_import_pattern, sentencepiece_import_with_mock, content)
    
    return content

def process_template(template_path, check_only=False, create_backup=True):
    """
    Process a template file to ensure it has env mock support implemented.
    
    Args:
        template_path: Path to the template file
        check_only: If True, only check for env mock support without modifying
        create_backup: If True, create a backup before modifying
        
    Returns:
        bool: True if template has or now has env mock support, False otherwise
    """
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check if env mock support is implemented
        has_env_mock_support = check_env_mock_support(content)
        
        if has_env_mock_support:
            logger.info(f"✅ {template_path}: Environment mock support is already implemented")
            return True
        else:
            logger.warning(f"❌ {template_path}: Environment mock support is missing")
            
            if check_only:
                return False
            
            # Create backup if requested
            if create_backup:
                backup_path = f"{template_path}.bak"
                with open(backup_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created backup at {backup_path}")
            
            # Add env mock support
            updated_content = add_env_mock_support(content)
            
            # Write updated content
            with open(template_path, 'w') as f:
                f.write(updated_content)
            
            # Verify update
            with open(template_path, 'r') as f:
                new_content = f.read()
            
            has_env_mock_support = check_env_mock_support(new_content)
            if has_env_mock_support:
                logger.info(f"✅ {template_path}: Successfully added environment mock support")
                return True
            else:
                logger.error(f"❌ {template_path}: Failed to add environment mock support")
                return False
            
    except Exception as e:
        logger.error(f"Error processing template {template_path}: {e}")
        return False

def process_all_templates(templates_dir="templates", check_only=False):
    """
    Process all template files in the given directory.
    
    Args:
        templates_dir: Directory containing template files
        check_only: If True, only check for env mock support without modifying
        
    Returns:
        Tuple of (success_count, failure_count, total_count)
    """
    success_count = 0
    failure_count = 0
    
    try:
        templates_path = os.path.join(os.path.dirname(__file__), templates_dir)
        template_files = []
        
        for file in os.listdir(templates_path):
            if file.endswith("_template.py"):
                template_files.append(os.path.join(templates_path, file))
        
        logger.info(f"Found {len(template_files)} template files to process")
        
        for template_path in template_files:
            if process_template(template_path, check_only):
                success_count += 1
            else:
                failure_count += 1
        
        return success_count, failure_count, len(template_files)
    
    except Exception as e:
        logger.error(f"Error processing templates: {e}")
        return success_count, failure_count, 0

def main():
    parser = argparse.ArgumentParser(description="Add environment variable-based mocking support to template files")
    parser.add_argument("--check-only", action="store_true", help="Only check for env mock support without modifying")
    parser.add_argument("--template", type=str, help="Process a specific template file")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    
    args = parser.parse_args()
    
    create_backup = not args.no_backup
    
    if args.template:
        template_path = args.template
        if not os.path.exists(template_path):
            template_path = os.path.join(os.path.dirname(__file__), "templates", args.template)
        
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return 1
        
        success = process_template(template_path, args.check_only, create_backup)
        
        if args.check_only:
            print(f"\nTemplate check: {'✅ Has environment mock support' if success else '❌ Missing environment mock support'}")
        else:
            print(f"\nTemplate processing: {'✅ Success' if success else '❌ Failed'}")
        
        return 0 if success else 1
    else:
        success_count, failure_count, total_count = process_all_templates(check_only=args.check_only)
        
        print("\nTemplate Processing Summary:")
        if args.check_only:
            print(f"- Templates with environment mock support: {success_count}/{total_count}")
            print(f"- Templates missing environment mock support: {failure_count}/{total_count}")
        else:
            print(f"- Successfully processed: {success_count}/{total_count}")
            print(f"- Failed to process: {failure_count}/{total_count}")
        
        return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
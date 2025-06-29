#!/usr/bin/env python3
"""
Add mock detection system to template files.

This script:
1. Checks if template files have the mock detection implemented
2. If not, adds it to the run_tests method
3. Also adds the visual indicators to the main function
4. Updates the metadata section with mock detection information

Usage:
    python add_mock_detection_to_templates.py [--check-only] [--template TEMPLATE_FILE]
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
        logging.FileHandler(f"template_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def check_mock_detection(content):
    """
    Check if mock detection system is implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        bool: True if mock detection is implemented, False otherwise
    """
    # Check for key mock detection patterns
    has_using_real_inference = "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH" in content
    has_using_mocks = "using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE" in content
    has_visual_indicators = "🚀 Using REAL INFERENCE with actual models" in content and "🔷 Using MOCK OBJECTS for CI/CD testing only" in content
    has_metadata = '"using_real_inference": using_real_inference,' in content and '"using_mocks": using_mocks,' in content
    
    return has_using_real_inference and has_using_mocks and has_visual_indicators and has_metadata

def ensure_mock_detection(content):
    """
    Ensure mock detection system is implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock detection implemented
    """
    if check_mock_detection(content):
        return content
        
    # Add mock detection logic in the run_tests method
    if "def run_tests(" in content and "return {" in content:
        # Find the return statement in run_tests method
        return_index = content.find("return {", content.find("def run_tests("))
        
        if return_index != -1:
            # Add mock detection logic before the return statement
            mock_detection_code = """
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
"""
            # Insert the mock detection code
            content = content[:return_index] + mock_detection_code + content[return_index:]
            
            # Now add the mock detection metadata to the return dictionary
            if '"metadata":' in content:
                metadata_index = content.find('"metadata":', return_index)
                if metadata_index != -1:
                    # Find the closing brace of the metadata dictionary
                    closing_brace_index = content.find("}", metadata_index)
                    if closing_brace_index != -1:
                        # Add the mock detection metadata
                        mock_metadata = """
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
"""
                        content = content[:closing_brace_index] + mock_metadata + content[closing_brace_index:]
            
    # Add the visual indicators to the main function
    if "def main():" in content and "print(f\"Successfully tested" in content:
        success_print_index = content.find("print(f\"Successfully tested", content.find("def main():"))
        
        if success_print_index != -1:
            # Find the appropriate location to insert the visual indicators
            # Look for Test Results Summary section
            summary_index = content.find("TEST RESULTS SUMMARY", success_print_index - 200, success_print_index)
            if summary_index != -1:
                next_line_index = content.find("\n", summary_index)
                if next_line_index != -1:
                    # Add the visual indicators
                    visual_indicators_code = """
    
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"🚀 Using REAL INFERENCE with actual models")
    else:
        print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
"""
                    content = content[:next_line_index] + visual_indicators_code + content[next_line_index:]
            
    return content

def process_template(template_path, check_only=False, create_backup=True):
    """
    Process a template file to ensure it has mock detection implemented.
    
    Args:
        template_path: Path to the template file
        check_only: If True, only check for mock detection without modifying
        create_backup: If True, create a backup before modifying
        
    Returns:
        bool: True if template has or now has mock detection, False otherwise
    """
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check if mock detection is implemented
        has_mock_detection = check_mock_detection(content)
        
        if has_mock_detection:
            logger.info(f"✅ {template_path}: Mock detection is already implemented")
            return True
        else:
            logger.warning(f"❌ {template_path}: Mock detection is missing")
            
            if check_only:
                return False
            
            # Create backup if requested
            if create_backup:
                backup_path = f"{template_path}.bak"
                with open(backup_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created backup at {backup_path}")
            
            # Add mock detection
            updated_content = ensure_mock_detection(content)
            
            # Write updated content
            with open(template_path, 'w') as f:
                f.write(updated_content)
            
            # Verify update
            with open(template_path, 'r') as f:
                content = f.read()
            
            has_mock_detection = check_mock_detection(content)
            if has_mock_detection:
                logger.info(f"✅ {template_path}: Successfully added mock detection")
                return True
            else:
                logger.error(f"❌ {template_path}: Failed to add mock detection")
                return False
            
    except Exception as e:
        logger.error(f"Error processing template {template_path}: {e}")
        return False

def process_all_templates(templates_dir="templates", check_only=False):
    """
    Process all template files in the given directory.
    
    Args:
        templates_dir: Directory containing template files
        check_only: If True, only check for mock detection without modifying
        
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
        return 0, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Add mock detection to template files")
    parser.add_argument("--check-only", action="store_true", help="Only check for mock detection without modifying")
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
            print(f"\nTemplate check: {'✅ Has mock detection' if success else '❌ Missing mock detection'}")
        else:
            print(f"\nTemplate processing: {'✅ Success' if success else '❌ Failed'}")
        
        return 0 if success else 1
    else:
        success_count, failure_count, total_count = process_all_templates(check_only=args.check_only)
        
        print("\nTemplate Processing Summary:")
        if args.check_only:
            print(f"- Templates with mock detection: {success_count}/{total_count}")
            print(f"- Templates missing mock detection: {failure_count}/{total_count}")
        else:
            print(f"- Successfully processed: {success_count}/{total_count}")
            print(f"- Failed to process: {failure_count}/{total_count}")
        
        return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
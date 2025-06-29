#!/usr/bin/env python3
"""
Update all template files to include mock object detection.

This script:
1. Reads the encoder_only_template.py which has the updated mock detection code
2. Extracts the relevant sections for mock detection
3. Updates all other template files with the same mock detection code
"""

import os
import sys
import re
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"update_templates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Path constants
SKILLS_DIR = Path(__file__).parent
TEMPLATES_DIR = SKILLS_DIR / "templates"
ENCODER_ONLY_TEMPLATE = TEMPLATES_DIR / "encoder_only_template.py"

def extract_mock_detection_code(template_path):
    """Extract the mock detection code from the encoder_only_template.py file."""
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Extract the results metadata section
    metadata_pattern = r"# Determine if real inference or mock objects were used.*?return \{.*?\"test_type\":.*?\}"
    metadata_match = re.search(metadata_pattern, content, re.DOTALL)
    if not metadata_match:
        logger.error("Could not find metadata section in encoder_only_template.py")
        return None, None
    
    metadata_code = metadata_match.group(0)
    
    # Extract the print summary section
    summary_pattern = r"# Print summary.*?print\(\"\\nFor detailed results.*?\)"
    summary_match = re.search(summary_pattern, content, re.DOTALL)
    if not summary_match:
        logger.error("Could not find print summary section in encoder_only_template.py")
        return None, None
    
    summary_code = summary_match.group(0)
    
    return metadata_code, summary_code

def update_template_file(file_path, metadata_code, summary_code):
    """Update a template file with the new mock detection code."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Try various patterns for the metadata section
    metadata_patterns = [
        r"# Build final results.*?return \{.*?\"has_sentencepiece\": HAS_SENTENCEPIECE\n.*?\}",
        r"return \{.*?\"metadata\":.*?\"has_sentencepiece\": HAS_SENTENCEPIECE\n.*?\}",
        r"# Build final results.*?return \{.*?\"hardware\": HW_CAPABILITIES,\n.*?\"metadata\":.*?\}",
        r"return \{.*?\"hardware\": HW_CAPABILITIES,\n.*?\"metadata\":.*?\}",
        r"# Build final results.*?return \{.*?\"metadata\":.*?\n.*?\}"
    ]
    
    metadata_matched = False
    for pattern in metadata_patterns:
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, metadata_code, content, flags=re.DOTALL)
            metadata_matched = True
            break
    
    if not metadata_matched:
        # Hard-coded replacement for specific patterns
        if "return {" in content and '"hardware": HW_CAPABILITIES,' in content and '"metadata":' in content:
            # Find the return statement by detecting the surrounding context
            start_idx = content.find("# Build final results")
            if start_idx == -1:
                start_idx = content.find("return {")
                if start_idx > 0:
                    # Go back to find the beginning of the method
                    prev_line_idx = content.rfind("\n", 0, start_idx)
                    if prev_line_idx > 0:
                        start_idx = prev_line_idx + 1
            
            if start_idx > 0:
                # Find the end of the return statement
                end_pattern = r"\n\s*\}"
                end_match = re.search(end_pattern, content[start_idx:])
                if end_match:
                    end_idx = start_idx + end_match.end()
                    # Replace the entire section
                    content = content[:start_idx] + metadata_code + content[end_idx:]
                    metadata_matched = True
    
    if not metadata_matched:
        logger.error(f"Could not find metadata section in {file_path}")
        return False
    
    # Try various patterns for the summary section
    summary_patterns = [
        r"# Print summary.*?print\(\"\\nFor detailed results.*?\)",
        r"print\(\"\\nTEST RESULTS SUMMARY:.*?print\(\"\\nFor detailed results.*?\)",
        r"# Print summary.*?success =.*?print\(\"\\nFor detailed results.*?\)",
        r"success =.*?print\(\"\\nTEST RESULTS SUMMARY:.*?print\(\"\\nFor detailed results.*?\)"
    ]
    
    summary_matched = False
    for pattern in summary_patterns:
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, summary_code, content, flags=re.DOTALL)
            summary_matched = True
            break
    
    if not summary_matched:
        # Handle specific patterns by searching for known segments
        if 'print("\\nTEST RESULTS SUMMARY:")' in content and 'print("\\nFor detailed results' in content:
            start_idx = content.find('print("\\nTEST RESULTS SUMMARY:")')
            if start_idx > 0:
                # Find the end of the print statements
                end_idx = content.find('print("\\nFor detailed results', start_idx)
                if end_idx > 0:
                    end_idx = content.find(')', end_idx) + 1
                    # Replace the entire section
                    content = content[:start_idx] + summary_code + content[end_idx:].replace('print("\\nFor detailed results', '', 1)
                    summary_matched = True
    
    if not summary_matched:
        logger.error(f"Could not find print summary section in {file_path}")
        return False
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True

def main():
    # Extract mock detection code from encoder_only_template.py
    metadata_code, summary_code = extract_mock_detection_code(ENCODER_ONLY_TEMPLATE)
    if not metadata_code or not summary_code:
        logger.error("Failed to extract mock detection code from encoder_only_template.py")
        return 1
    
    # Get all template files except encoder_only_template.py
    template_files = [f for f in TEMPLATES_DIR.glob("*_template.py") if f.name != "encoder_only_template.py"]
    logger.info(f"Found {len(template_files)} template files to update")
    
    # Update each template file
    updated = 0
    failed = 0
    
    for template_file in template_files:
        logger.info(f"Updating {template_file.name}")
        if update_template_file(template_file, metadata_code, summary_code):
            updated += 1
            logger.info(f"Successfully updated {template_file.name}")
        else:
            failed += 1
            logger.error(f"Failed to update {template_file.name}")
    
    # Print summary
    print(f"\nTemplate Update Summary:")
    print(f"- Successfully updated: {updated} templates")
    print(f"- Failed: {failed} templates")
    print(f"- Total: {len(template_files)} templates")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Generate test files for skillset implementations.

This script generates test files for skillset implementations based on
the model files in the skillset directory.
"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model_name_from_filename(filename: str) -> str:
    """Extract model name from a skillset filename.
    
    Args:
        filename: The filename to extract from (e.g., 'hf_bert.py')
        
    Returns:
        The extracted model name (e.g., 'bert')
    """
    basename = os.path.basename(filename)
    if basename.startswith('hf_') and basename.endswith('.py'):
        return basename[3:-3]  # Strip hf_ prefix and .py suffix
    return basename


def get_skillset_files(skillset_dir: str) -> List[str]:
    """Get a list of all skillset files in the directory.
    
    Args:
        skillset_dir: Directory containing skillset files
        
    Returns:
        List of skillset filenames
    """
    pattern = os.path.join(skillset_dir, 'hf_*.py')
    return glob.glob(pattern)


def generate_test_for_skillset(model_name: str, template_path: str, output_dir: str) -> Tuple[bool, str]:
    """Generate a test file for a skillset model.
    
    Args:
        model_name: The model name (e.g., 'bert', 'vision-encoder-decoder')
        template_path: Path to the template file
        output_dir: Directory to write the test file to
        
    Returns:
        Tuple of (success, output_file_path)
    """
    try:
        # Read template
        with open(template_path, 'r') as f:
            template = f.read()
            
        # Replace placeholders
        # Convert model_name to a valid Python identifier for class names
        class_safe_name = model_name.replace('-', '_')
        
        # Create class name with proper formatting
        class_name_upper = class_safe_name.title().replace('_', '')
        
        # Replace in template
        content = template.replace('{model_type}', model_name)
        content = content.replace('{model_type_upper}', class_name_upper)
        content = content.replace('{model_type_safe}', class_safe_name)
        
        # Determine output file path
        output_file = os.path.join(output_dir, f"test_hf_{model_name}.py")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(content)
            
        logger.info(f"✅ Generated test file: {output_file}")
        return True, output_file
    except Exception as e:
        logger.error(f"❌ Error generating test file for {model_name}: {e}")
        return False, ""


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate test files for skillset implementations")
    parser.add_argument(
        "--skillset-dir", 
        type=str, 
        default="../ipfs_accelerate_py/worker/skillset",
        help="Directory containing skillset files"
    )
    parser.add_argument(
        "--template-path", 
        type=str, 
        default="templates/skillset_test_template.py",
        help="Path to the template file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="../skillset",
        help="Directory to write test files to"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Specific model to generate test for (e.g., 'bert')"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get skillset files
    if args.model:
        # Generate for specific model
        success, output_file = generate_test_for_skillset(
            args.model, 
            args.template_path, 
            args.output_dir
        )
        return 0 if success else 1
    else:
        # Get all skillset files
        skillset_files = get_skillset_files(args.skillset_dir)
        if not skillset_files:
            logger.error(f"No skillset files found in {args.skillset_dir}")
            return 1
            
        # Sort skillset files for predictable order
        skillset_files.sort()
            
        logger.info(f"Found {len(skillset_files)} skillset files to generate tests for")
        
        # Generate tests for each file
        success_count = 0
        failed_count = 0
        
        for file_path in skillset_files:
            model_name = get_model_name_from_filename(file_path)
            success, _ = generate_test_for_skillset(
                model_name, 
                args.template_path, 
                args.output_dir
            )
            
            if success:
                success_count += 1
            else:
                failed_count += 1
                
        logger.info(f"Generated {success_count} test files, {failed_count} failed")
        return 0 if failed_count == 0 else 1
        

if __name__ == "__main__":
    sys.exit(main())
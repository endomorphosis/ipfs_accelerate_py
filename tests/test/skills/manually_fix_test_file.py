#!/usr/bin/env python3
"""
Manually fix a test file with hyphenated model names.

This script does a more direct and intensive repair of problematic test files.
"""

import os
import sys
import re
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def to_valid_identifier(text):
    """Convert text to a valid Python identifier."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text

def fix_file(file_path, model_type=None):
    """
    Fix the test file by extracting model type from filename and applying corrections.
    
    Args:
        file_path: Path to the file to fix
        model_type: Optional model type override
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract model type from filename if not provided
        if not model_type:
            filename = os.path.basename(file_path)
            model_type = filename.replace('test_hf_', '').replace('.py', '')
        
        logger.info(f"Fixing file: {file_path} for model type: {model_type}")
        
        # Create valid identifiers
        model_valid = to_valid_identifier(model_type)
        model_upper = model_valid.upper()
        
        # For capitalized names (like class names)
        model_capitalized = model_type.split('-')[0].capitalize()
        if len(model_type.split('-')) > 1:
            model_capitalized += ''.join(part.capitalize() for part in model_type.split('-')[1:])
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = f"{file_path}.manual.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created backup at: {backup_path}")
        
        # Fix registry key
        registry_pattern = r'([A-Z\-_]+)_MODELS_REGISTRY'
        if re.search(registry_pattern, content):
            content = re.sub(registry_pattern, f"{model_upper}_MODELS_REGISTRY", content)
            logger.info(f"Fixed registry key to: {model_upper}_MODELS_REGISTRY")
        
        # Fix class name
        class_pattern = r'class\s+Test([A-Za-z0-9\-_]+)([A-Za-z0-9]+):'
        if re.search(class_pattern, content):
            content = re.sub(class_pattern, f"class Test{model_capitalized}Models:", content)
            logger.info(f"Fixed class name to: Test{model_capitalized}Models")
        
        # Fix constructor calls
        constructor_pattern = r'([a-z0-9\-_]+)_tester\s*=\s*Test([A-Za-z0-9\-_]+)([A-Za-z0-9]+)\('
        if re.search(constructor_pattern, content):
            content = re.sub(constructor_pattern, f"{model_valid}_tester = Test{model_capitalized}Models(", content)
            logger.info(f"Fixed constructor to: {model_valid}_tester = Test{model_capitalized}Models(")
        
        # Fix filename references in save_results
        filename_pattern = r'hf_([a-z0-9\-_]+)_'
        if re.search(filename_pattern, content):
            content = re.sub(filename_pattern, f"hf_{model_valid}_", content)
            logger.info(f"Fixed filename references to: hf_{model_valid}_")
        
        # Fix model references
        content = content.replace("BERT model", f"{model_capitalized} model")
        content = content.replace("bert model", f"{model_type} model")
        content = content.replace("BERT", model_capitalized)
        content = content.replace("bert-base-uncased", f"{model_type}-base-uncased")
        
        # Fix all remaining hyphenated mentions of the model name
        content = content.replace(model_type, model_valid)
        
        # Special case for model class
        if "BertForMaskedLM" in content:
            content = content.replace("BertForMaskedLM", f"{model_capitalized}ForMaskedLM")
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify syntax
        try:
            compile(content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax is valid after fixes")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error after fixes: {e}")
            # If syntax error, restore from backup
            logger.info(f"Restoring from backup...")
            with open(backup_path, 'r') as f:
                original = f.read()
            with open(file_path, 'w') as f:
                f.write(original)
            return False
            
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manually fix a test file with syntax issues")
    parser.add_argument("--file", type=str, required=True, help="Path to the file to fix")
    parser.add_argument("--model", type=str, help="Model type (extracted from filename if not provided)")
    
    args = parser.parse_args()
    
    success = fix_file(args.file, args.model)
    
    if success:
        print(f"\nSuccessfully fixed file: {args.file}")
    else:
        print(f"\nFailed to fix file: {args.file}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
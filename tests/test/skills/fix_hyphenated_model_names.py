#!/usr/bin/env python3
"""
Fix hyphenated model names and other syntax errors in test files.

This script specifically addresses syntax errors in test files that involve:
1. Hyphenated model names causing invalid identifiers in variable or class names
2. Unescaped special characters in strings
3. Other common syntax issues

Usage:
    python fix_hyphenated_model_names.py --file FILE_PATH
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path

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

def fix_hyphenated_registry_key(file_content, model_type):
    """Fix hyphenated registry key in the models registry."""
    # Fix the registry key (should be a valid Python identifier)
    model_valid = to_valid_identifier(model_type)
    model_upper = model_valid.upper()
    
    # Replace registry variable
    pattern = r'([A-Za-z\-]+)_MODELS_REGISTRY'
    registry_with_hyphen = re.search(pattern, file_content)
    if registry_with_hyphen:
        old_registry = registry_with_hyphen.group(0)
        new_registry = f"{model_upper}_MODELS_REGISTRY"
        file_content = file_content.replace(old_registry, new_registry)
        logger.info(f"Fixed registry key: {old_registry} -> {new_registry}")
    
    return file_content

def fix_hyphenated_class_name(file_content, model_type):
    """Fix hyphenated class name."""
    # Find class definition with a hyphen
    class_pattern = r'class\s+Test([A-Za-z\-]+)([A-Za-z]+):'
    class_match = re.search(class_pattern, file_content)
    
    if class_match:
        original_class_name = class_match.group(0)
        model_valid = to_valid_identifier(model_type)
        model_capitalized = model_type.split('-')[0].capitalize()
        if len(model_type.split('-')) > 1:
            model_capitalized += ''.join(part.capitalize() for part in model_type.split('-')[1:])
        
        new_class_name = f"class Test{model_capitalized}Models:"
        file_content = file_content.replace(original_class_name, new_class_name)
        logger.info(f"Fixed class name: {original_class_name} -> {new_class_name}")
    
    return file_content

def fix_hyphenated_constructor(file_content, model_type):
    """Fix hyphenated constructor name."""
    # Find broken constructor reference
    constructor_pattern = r'([a-z\-]+)_tester = Test([A-Za-z\-]+)([A-Za-z]+)\('
    constructor_match = re.search(constructor_pattern, file_content)
    
    if constructor_match:
        original_constructor = constructor_match.group(0)
        model_valid = to_valid_identifier(model_type)
        model_capitalized = model_type.split('-')[0].capitalize()
        if len(model_type.split('-')) > 1:
            model_capitalized += ''.join(part.capitalize() for part in model_type.split('-')[1:])
        
        new_constructor = f"{model_valid}_tester = Test{model_capitalized}Models("
        file_content = file_content.replace(original_constructor, new_constructor)
        logger.info(f"Fixed constructor: {original_constructor} -> {new_constructor}")
    
    return file_content

def fix_filename_references(file_content, model_type):
    """Fix references to the filename in the file content."""
    model_valid = to_valid_identifier(model_type)
    
    # Fix filename references in save_results function
    pattern = r'hf_([a-z\-]+)_'
    filename_matches = re.finditer(pattern, file_content)
    
    for match in filename_matches:
        original_reference = match.group(0)
        new_reference = f"hf_{model_valid}_"
        file_content = file_content.replace(original_reference, new_reference)
        logger.info(f"Fixed filename reference: {original_reference} -> {new_reference}")
        
    return file_content

def fix_docstrings(file_content):
    """Fix multiline docstrings that may have syntax errors."""
    # Look for docstring patterns that might be malformed
    docstring_pattern = r'"""(.*?)"""'
    docstring_matches = re.finditer(docstring_pattern, file_content, re.DOTALL)
    
    fixed_content = file_content
    for match in docstring_matches:
        docstring = match.group(0)
        if '\\' in docstring and not '\\\\' in docstring:
            # Replace unescaped backslashes with double backslashes
            fixed_docstring = docstring.replace('\\', '\\\\')
            fixed_content = fixed_content.replace(docstring, fixed_docstring)
            logger.info(f"Fixed unescaped backslashes in docstring")
    
    return fixed_content

def fix_unterminated_string_literals(file_content):
    """Fix common issues with unterminated string literals."""
    # First, fix triple-quoted docstrings
    triple_quote_pattern = r'("""[^"]*)(""")?'
    docstring_matches = re.finditer(triple_quote_pattern, file_content, re.DOTALL)
    
    fixed_content = file_content
    for match in docstring_matches:
        if match.group(2) is None:  # Missing closing triple quotes
            start = match.start()
            docstring_start = match.group(1)
            fixed_content = fixed_content[:start] + docstring_start + '"""' + fixed_content[match.end():]
            logger.info(f"Fixed unterminated triple-quoted docstring")
    
    # Then check line by line for simpler string issues
    lines = fixed_content.split('\n')
    fixed_lines = []
    
    in_triple_quote = False
    triple_quote_type = None
    
    for line in lines:
        # Track triple quotes
        if '"""' in line:
            count = line.count('"""')
            if count % 2 == 1:  # Odd number of triple quotes
                in_triple_quote = not in_triple_quote
                triple_quote_type = '"""'
        
        if "'''" in line:
            count = line.count("'''")
            if count % 2 == 1:  # Odd number of triple quotes
                in_triple_quote = not in_triple_quote
                triple_quote_type = "'''"
        
        # Only fix single/double quotes if not inside a triple quote
        if not in_triple_quote:
            if line.count('"') % 2 == 1 and '#' not in line:
                # Unbalanced double quotes, try to fix
                if line.rstrip().endswith('\\'):
                    # Line ends with backslash, add a quote
                    fixed_lines.append(line + '"')
                    logger.info(f"Fixed unterminated string literal: {line}")
                else:
                    # Add a closing quote
                    fixed_lines.append(line + '"')
                    logger.info(f"Fixed unterminated string literal: {line}")
            elif line.count("'") % 2 == 1 and '#' not in line:
                # Unbalanced single quotes, try to fix
                if line.rstrip().endswith('\\'):
                    # Line ends with backslash, add a quote
                    fixed_lines.append(line + "'")
                    logger.info(f"Fixed unterminated string literal: {line}")
                else:
                    # Add a closing quote
                    fixed_lines.append(line + "'")
                    logger.info(f"Fixed unterminated string literal: {line}")
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Close any unclosed triple quotes
    if in_triple_quote:
        fixed_lines.append(triple_quote_type)
        logger.info(f"Added missing closing triple quote at end of file")
    
    return '\n'.join(fixed_lines)

def fix_model_references(file_content, model_type):
    """Fix model type references and other specific replacements."""
    model_valid = to_valid_identifier(model_type)
    model_capitalized = model_type.split('-')[0].capitalize()
    if len(model_type.split('-')) > 1:
        model_capitalized += ''.join(part.capitalize() for part in model_type.split('-')[1:])
    
    # Replace model references in function arguments and text
    replacements = {
        f"Testing {model_type.upper()}": f"Testing {model_capitalized}",
        f"Testing BERT model": f"Testing {model_capitalized} model",
        f"bert-base-uncased": f"{model_type}-base-uncased",
        "BERT Large": f"{model_capitalized} Large",
        "BERT Base": f"{model_capitalized} Base",
        "bert HuggingFace": f"{model_type} HuggingFace",
        "BERT models": f"{model_capitalized} models",
        "BERT model": f"{model_capitalized} model",
    }
    
    for old, new in replacements.items():
        if old in file_content:
            file_content = file_content.replace(old, new)
            logger.info(f"Fixed model reference: {old} -> {new}")
    
    return file_content

def fix_variable_references(file_content, model_type):
    """Fix variable references to the model."""
    model_valid = to_valid_identifier(model_type)
    
    # Find instances of hyphenated variables
    var_pattern = r'\b([a-z\-]+)_tester\b'
    var_matches = re.finditer(var_pattern, file_content)
    
    for match in var_matches:
        original_var = match.group(0)
        if "-" in original_var:
            new_var = f"{model_valid}_tester"
            file_content = file_content.replace(original_var, new_var)
            logger.info(f"Fixed variable reference: {original_var} -> {new_var}")
    
    # Fix other references
    replacements = {
        f"bert model": f"{model_type} model",
        f"bert_tester": f"{model_valid}_tester",
        f"bert-tester": f"{model_valid}_tester",
    }
    
    for old, new in replacements.items():
        if old in file_content:
            file_content = file_content.replace(old, new)
            logger.info(f"Fixed variable reference: {old} -> {new}")
    
    return file_content

def fix_file(file_path):
    """Fix issues in the given file."""
    try:
        # Extract model type from filename
        filename = os.path.basename(file_path)
        model_type = filename.replace('test_hf_', '').replace('.py', '')
        
        logger.info(f"Fixing file: {file_path} for model type: {model_type}")
        
        # Read file content
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        # Create a backup
        backup_path = f"{file_path}.syntax.bak"
        with open(backup_path, 'w') as f:
            f.write(file_content)
            
        logger.info(f"Created backup at: {backup_path}")
        
        # Apply fixes in the correct order
        file_content = fix_hyphenated_registry_key(file_content, model_type)
        file_content = fix_hyphenated_class_name(file_content, model_type)
        file_content = fix_hyphenated_constructor(file_content, model_type)
        file_content = fix_filename_references(file_content, model_type)
        file_content = fix_docstrings(file_content)
        file_content = fix_model_references(file_content, model_type)
        file_content = fix_variable_references(file_content, model_type)
        file_content = fix_unterminated_string_literals(file_content)
        
        # Check for any remaining issues with invalid identifiers
        lines = file_content.split('\n')
        for i, line in enumerate(lines):
            if model_type in line and '-' in line:
                model_valid = to_valid_identifier(model_type)
                lines[i] = line.replace(model_type, model_valid)
                logger.info(f"Fixed remaining hyphenated model name in line {i+1}")
        
        file_content = '\n'.join(lines)
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(file_content)
        
        # Verify syntax
        try:
            compile(file_content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax is valid after fixes")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error after fixes: {e}")
            # Log the problematic line for debugging
            line_no = e.lineno
            if line_no and line_no <= len(lines):
                logger.error(f"Problematic line {line_no}: {lines[line_no-1]}")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix hyphenated model names and other syntax errors in test files")
    parser.add_argument("--file", type=str, help="Path to the file to fix")
    parser.add_argument("--dir", type=str, help="Directory containing files to fix")
    parser.add_argument("--model", type=str, help="Specific model type to target (e.g., xlm-roberta)")
    parser.add_argument("--hyphenated-only", action="store_true", help="Only fix files with hyphenated names")
    parser.add_argument("--rename-files", action="store_true", help="Rename files to replace hyphens with underscores")
    parser.add_argument("--apply-mock-fixes", action="store_true", help="Also apply mock detection fixes")
    
    args = parser.parse_args()
    
    # Import mock fixes if needed
    if args.apply_mock_fixes:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from fix_mock_detection_errors import fix_file as fix_mock_detection
            logger.info("Imported mock detection fixes")
        except ImportError:
            logger.warning("Could not import fix_mock_detection_errors.py, will not apply those fixes")
            args.apply_mock_fixes = False
    
    if args.file:
        # Fix a single file
        success = fix_file(args.file)
        if success and args.apply_mock_fixes:
            try:
                success = fix_mock_detection(args.file)
            except Exception as e:
                logger.error(f"Error applying mock fixes to {args.file}: {e}")
                success = False
                
        if success:
            print(f"\nSuccessfully fixed file: {args.file}")
        else:
            print(f"\nFailed to fix file: {args.file}")
            return 1
            
        # Rename file if requested
        if args.rename_files and '-' in os.path.basename(args.file):
            new_name = args.file.replace('-', '_')
            try:
                os.rename(args.file, new_name)
                print(f"Renamed file to: {new_name}")
            except Exception as e:
                logger.error(f"Error renaming file {args.file}: {e}")
                
    elif args.dir:
        # Fix all files in directory
        fixed_count = 0
        failed_count = 0
        failed_files = []
        files_to_fix = []
        renamed_count = 0
        
        # Find all Python files in the directory
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # Filter by model type if specified
                    if args.model and args.model not in file:
                        continue
                        
                    # Filter for hyphenated files if requested
                    if args.hyphenated_only and '-' not in file:
                        continue
                        
                    files_to_fix.append(file_path)
        
        logger.info(f"Found {len(files_to_fix)} files to fix")
        
        # Fix each file
        for file_path in files_to_fix:
            file_fixed = fix_file(file_path)
            
            # Apply mock fixes if requested and initial fix was successful
            if file_fixed and args.apply_mock_fixes:
                try:
                    mock_fixed = fix_mock_detection(file_path)
                    if not mock_fixed:
                        logger.warning(f"Mock detection fixes failed for {file_path}")
                except Exception as e:
                    logger.error(f"Error applying mock fixes to {file_path}: {e}")
                    mock_fixed = False
                    
                # Only count as fixed if both fixes were successful
                file_fixed = file_fixed and (mock_fixed if args.apply_mock_fixes else True)
            
            if file_fixed:
                fixed_count += 1
                
                # Rename file if requested
                if args.rename_files and '-' in os.path.basename(file_path):
                    new_name = file_path.replace('-', '_')
                    try:
                        os.rename(file_path, new_name)
                        renamed_count += 1
                        logger.info(f"Renamed {file_path} to {new_name}")
                    except Exception as e:
                        logger.error(f"Error renaming file {file_path}: {e}")
            else:
                failed_count += 1
                failed_files.append(file_path)
        
        print(f"\nFixed {fixed_count} files successfully")
        if args.rename_files:
            print(f"Renamed {renamed_count} files to replace hyphens with underscores")
            
        if failed_count > 0:
            print(f"Failed to fix {failed_count} files:")
            for file in failed_files:
                print(f"  - {file}")
            return 1
    else:
        print("Please specify either --file or --dir")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
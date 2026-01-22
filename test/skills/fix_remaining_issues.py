#!/usr/bin/env python3
"""
Fix remaining issues in test files.

This script fixes common syntax errors and missing imports
that may have been introduced during the regeneration process.

Usage:
    python fix_remaining_issues.py --file FILE_PATH
    python fix_remaining_issues.py --dir DIRECTORY
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def fix_docstring_quotes(content):
    """Fix incorrect triple quotes in docstrings."""
    # Pattern to find incorrect docstring triple quotes
    patterns = [
        (r'""""', '"""'),  # Four quotes instead of three
        (r"''''", "'''"),  # Four quotes instead of three
    ]
    
    fixed_content = content
    for pattern, replacement in patterns:
        if pattern in fixed_content:
            fixed_content = fixed_content.replace(pattern, replacement)
            logger.info(f"Fixed incorrect docstring quotes: {pattern} -> {replacement}")
    
    # Check for unterminated triple quotes
    lines = fixed_content.split('\n')
    triple_quote_count = 0
    fixed_lines = []
    
    for line in lines:
        if '"""' in line:
            # Count the number of triple quotes
            triple_quote_count += line.count('"""')
            # If odd number of triple quotes on the line
            if line.count('"""') % 2 == 1 and line.strip().endswith('"""'):
                # Fix extra quote
                if line.strip().endswith('""""'):
                    line = line[:-1]
                    logger.info(f"Fixed extra double quote in docstring: {line}")
        fixed_lines.append(line)
    
    # If odd number of triple quotes overall, add closing triple quotes at end
    if triple_quote_count % 2 == 1:
        fixed_lines.append('"""')
        logger.info("Added missing closing triple quotes")
    
    return '\n'.join(fixed_lines)

def fix_missing_imports(content):
    """Fix missing import definitions."""
    # Check if required variables are accessed but not defined
    required_vars = {
        "HAS_TOKENIZERS": "try:\n    import tokenizers\n    HAS_TOKENIZERS = True\nexcept ImportError:\n    tokenizers = MagicMock()\n    HAS_TOKENIZERS = False\n    logger.warning(\"tokenizers not available, using mock\")",
        "HAS_SENTENCEPIECE": "try:\n    import sentencepiece\n    HAS_SENTENCEPIECE = True\nexcept ImportError:\n    sentencepiece = MagicMock()\n    HAS_SENTENCEPIECE = False\n    logger.warning(\"sentencepiece not available, using mock\")"
    }
    
    lines = content.split("\n")
    fixed_content = content
    
    # Check if variables are accessed but not defined
    for var, import_code in required_vars.items():
        if re.search(rf"\b{var}\b", content) and not re.search(rf"\b{var}\s*=", content):
            logger.info(f"Adding missing definition for {var}")
            
            # Find a good position to insert the import
            # After imports but before main code
            insert_pos = 0
            for i, line in enumerate(lines):
                if "MOCK_TRANSFORMERS" in line:
                    insert_pos = i + 1
                    break
            
            if insert_pos > 0:
                lines.insert(insert_pos, "")
                lines.insert(insert_pos + 1, import_code)
                fixed_content = "\n".join(lines)
    
    return fixed_content

def fix_model_ids(content):
    """Fix invalid model IDs."""
    # Map of model_type to valid model_id
    model_id_map = {
        "decoder-only": "gpt2",
        "decoder_only": "gpt2",
        "encoder-decoder": "t5-small",
        "encoder_decoder": "t5-small",
    }
    
    # Check for invalid model IDs
    for model_type, model_id in model_id_map.items():
        pattern = f"{model_type}(-small)?"
        if re.search(rf"Testing {pattern} with", content):
            fixed_content = re.sub(
                rf"{model_type}(-small)?", 
                model_id, 
                content
            )
            logger.info(f"Fixed invalid model ID: {model_type}(-small)? -> {model_id}")
            return fixed_content
    
    return content

def fix_file(file_path):
    """Fix issues in the given file."""
    try:
        logger.info(f"Fixing issues in {file_path}")
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create a backup
        backup_path = f"{file_path}.issues.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created backup at: {backup_path}")
        
        # Apply fixes
        fixed_content = content
        fixed_content = fix_docstring_quotes(fixed_content)
        fixed_content = fix_missing_imports(fixed_content)
        fixed_content = fix_model_ids(fixed_content)
        
        # Only write if changes were made
        if fixed_content != content:
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            logger.info(f"Fixed issues in {file_path}")
        else:
            logger.info(f"No issues found in {file_path}")
        
        # Verify syntax
        try:
            compile(fixed_content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax is valid after fixes")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error after fixes: {e}")
            # If still has syntax errors, look for and fix specific issues
            if "unterminated string literal" in str(e):
                line_num = e.lineno
                lines = fixed_content.split('\n')
                if line_num <= len(lines):
                    problem_line = lines[line_num - 1]
                    logger.info(f"Attempting to fix unterminated string on line {line_num}: {problem_line}")
                    
                    # Fix the specific issue in this line
                    # Count quotes to see if we're missing one
                    double_quotes = problem_line.count('"')
                    single_quotes = problem_line.count("'")
                    
                    if double_quotes % 2 == 1:
                        lines[line_num - 1] = problem_line + '"'
                        logger.info(f"Added missing double quote to line {line_num}")
                    elif single_quotes % 2 == 1:
                        lines[line_num - 1] = problem_line + "'"
                        logger.info(f"Added missing single quote to line {line_num}")
                    
                    # Try again
                    fixed_content = '\n'.join(lines)
                    with open(file_path, 'w') as f:
                        f.write(fixed_content)
                    
                    try:
                        compile(fixed_content, file_path, 'exec')
                        logger.info(f"✅ {file_path}: Syntax is valid after additional fixes")
                        return True
                    except SyntaxError as e2:
                        logger.error(f"❌ {file_path}: Still has syntax error after additional fixes: {e2}")
                        # Restore from backup
                        with open(backup_path, 'r') as f:
                            original = f.read()
                        with open(file_path, 'w') as f:
                            f.write(original)
                        return False
            else:
                # Restore from backup
                with open(backup_path, 'r') as f:
                    original = f.read()
                with open(file_path, 'w') as f:
                    f.write(original)
                return False
    
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix remaining issues in test files")
    parser.add_argument("--file", type=str, help="Path to the file to fix")
    parser.add_argument("--dir", type=str, help="Directory containing files to fix")
    
    args = parser.parse_args()
    
    if args.file:
        # Fix a single file
        success = fix_file(args.file)
        if success:
            print(f"\nSuccessfully fixed {args.file}")
        else:
            print(f"\nFailed to fix {args.file}")
            return 1
    
    elif args.dir:
        # Fix all files in directory
        success_count = 0
        failure_count = 0
        failed_files = []
        
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if fix_file(file_path):
                        success_count += 1
                    else:
                        failure_count += 1
                        failed_files.append(file_path)
        
        print(f"\nFixed {success_count} files successfully")
        if failure_count > 0:
            print(f"Failed to fix {failure_count} files:")
            for file in failed_files:
                print(f"  - {file}")
            return 1
    
    else:
        print("Please specify either --file or --dir")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
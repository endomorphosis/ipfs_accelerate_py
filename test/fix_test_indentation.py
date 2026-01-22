#!/usr/bin/env python3
"""
Script to fix indentation issues in Hugging Face test files.
This script focuses on the most critical indentation problems that prevent
the files from passing syntax validation.
"""

import os
import sys
import re
import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_indentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def verify_python_syntax(file_path):
    """
    Verify the Python syntax of a file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the code to check syntax
        compile(content, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"{e.__class__.__name__}: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def fix_if_conditionals(content):
    """Fix conditional statements that have incorrect indentation."""
    # Fix if not HAS_TOKENIZERS with class definition
    content = re.sub(r'(\s+)if not HAS_TOKENIZERS:(\s+)class', r'\1if not HAS_TOKENIZERS:\n\1    class', content)
    
    # Fix if not HAS_SENTENCEPIECE with class definition
    content = re.sub(r'(\s+)if not HAS_SENTENCEPIECE:(\s+)class', r'\1if not HAS_SENTENCEPIECE:\n\1    class', content)
    
    # Fix other if statements with missing indentation
    content = re.sub(r'(\s+)if HW_CAPABILITIES\["(\w+)"\]:(\s+)(\w+)', 
                    r'\1if HW_CAPABILITIES["\2"]:\n\1    \4', content)
    
    return content

def fix_tokenizer_mock(content):
    """Fix the tokenizer mock class that has indentation issues."""
    # Fix logger and conditional on same line 
    content = re.sub(r'logger\.warning\("tokenizers not available, using mock"\)\s+if not HAS_TOKENIZERS:', 
                    r'logger.warning("tokenizers not available, using mock")\n\n    if not HAS_TOKENIZERS:', content)
    
    # Fix logger statements on import error lines
    content = re.sub(r'HAS_TORCH = False\s+logger\.warning\("torch not available, using mock"\)', 
                   r'HAS_TORCH = False\n    logger.warning("torch not available, using mock")', content)
    content = re.sub(r'HAS_TRANSFORMERS = False\s+logger\.warning\("transformers not available, using mock"\)', 
                   r'HAS_TRANSFORMERS = False\n    logger.warning("transformers not available, using mock")', content)
    content = re.sub(r'HAS_TOKENIZERS = False\s+logger\.warning\("tokenizers not available, using mock"\)', 
                   r'HAS_TOKENIZERS = False\n    logger.warning("tokenizers not available, using mock")', content)
    content = re.sub(r'HAS_SENTENCEPIECE = False\s+logger\.warning\("sentencepiece not available, using mock"\)', 
                   r'HAS_SENTENCEPIECE = False\n    logger.warning("sentencepiece not available, using mock")', content)
    content = re.sub(r'HAS_PIL = False\s+logger\.warning\("PIL or requests not available, using mock"\)', 
                   r'HAS_PIL = False\n    logger.warning("PIL or requests not available, using mock")', content)
    
    # Fix decode method indentation in MockTokenizer
    if "def decode(self, ids, **kwargs):" in content and "def __init__(" in content:
        # First find where the decode method appears
        decode_index = content.find("def decode(self, ids, **kwargs):")
        init_index = content.rfind("def __init__(", 0, decode_index)
        
        # If we found both methods, check if they're at the same level
        if decode_index > 0 and init_index > 0:
            init_line = content[content.rfind("\n", 0, init_index)+1:init_index].strip()
            decode_line = content[content.rfind("\n", 0, decode_index)+1:decode_index].strip()
            
            # If decode has no indentation but init does, we need to fix it
            if not decode_line and init_line.startswith(" "):
                indent = " " * len(init_line)
                content = content.replace("def decode(self, ids, **kwargs):", 
                                         f"{indent}def decode(self, ids, **kwargs):")
    
    # Explicit fix for common pattern - directly replace
    content = content.replace(
        "def decode(self, ids, **kwargs):",
        "        def decode(self, ids, **kwargs):"
    )
    
    return content

def fix_method_parameter_indentation(content):
    """Fix indentation in method parameters that appear with duplicated self parameters."""
    # Fix for test_pipeline method parameters
    content = re.sub(r' def test_pipeline\(self,\(self,\(self,', r'    def test_pipeline(self,', content)
    content = re.sub(r'def test_pipeline\(self,\(self,\(self,', r'def test_pipeline(self,', content)
    
    # Fix for test_from_pretrained method parameters
    content = re.sub(r' def test_from_pretrained\(self,\(self,\(self,', r'    def test_from_pretrained(self,', content)
    content = re.sub(r'def test_from_pretrained\(self,\(self,\(self,', r'def test_from_pretrained(self,', content)
    
    # Fix for run_tests method parameters
    content = re.sub(r' def run_tests\(self,\(self,\(self,', r'    def run_tests(self,', content)
    content = re.sub(r'def run_tests\(self,\(self,\(self,', r'def run_tests(self,', content)
    
    # Also fix when it's missing proper indentation
    content = re.sub(r'\n def test_pipeline\(self', r'\n    def test_pipeline(self', content)
    content = re.sub(r'\n def test_from_pretrained\(self', r'\n    def test_from_pretrained(self', content)
    content = re.sub(r'\n def run_tests\(self', r'\n    def run_tests(self', content)
    
    return content

def fix_try_except_indentation(content):
    """Fix indentation in try/except blocks."""
    # Fix for try blocks followed by indented code
    content = re.sub(r'(\s+)try:(\s+)import openvino', r'\1try:\n\1    import openvino', content)
    
    # Fix other try blocks
    content = re.sub(r'(\s+)try:(\s+)with torch.no_grad', r'\1try:\n\1    with torch.no_grad', content)
    content = re.sub(r'(\s+)try:(\s+)_', r'\1try:\n\1    _', content)
    
    # Fix for nested blocks in openvino try block
    content = re.sub(r'(\s+)try:\s+import openvino\s+capabilities\["openvino"\]', 
                    r'\1try:\n\1    import openvino\n\1    capabilities["openvino"]', 
                    content)
    
    return content

def fix_indented_line_continuation(content):
    """Fix line continuations with incorrect indentation."""
    # Fix continuation lines that are incorrectly indented
    content = re.sub(r'(\s+)if HW_CAPABILITIES\["cuda"\]:', r'        if HW_CAPABILITIES["cuda"]:', content)
    content = re.sub(r'(\s+)if hasattr\(tokenizer, .pad_token.\)', r'        if hasattr(tokenizer, "pad_token")', content)
    
    # Fix for the logger line often indented incorrectly
    content = re.sub(r'(\s+)logger.info\(f"Testing model:', r'        logger.info(f"Testing model:', content)
    
    return content

def fix_if_else_indentation(content):
    """Fix indentation in if/else/elif blocks."""
    content = re.sub(r'(\s+)if "cuda" in error_str.*?:\n(\s+)results', 
                    r'\1if "cuda" in error_str or "cuda" in traceback_str:\n\1    results', content)
    content = re.sub(r'(\s+)elif "memory" in error_str:\n(\s+)results', 
                    r'\1elif "memory" in error_str:\n\1    results', content)
    content = re.sub(r'(\s+)elif "no module named" in error_str:\n(\s+)results', 
                    r'\1elif "no module named" in error_str:\n\1    results', content)
    content = re.sub(r'(\s+)else:\n(\s+)results\["', 
                    r'\1else:\n\1    results["', content)
    return content

def fix_mock_class_indentation(content):
    """Fix indentation in mock class definitions."""
    # Fix sentencepiece mock class
    pattern1 = r'class MockSentencePieceProcessor:(.*?)def __init__'
    replacement1 = r'class MockSentencePieceProcessor:\n        def __init__'
    content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    
    # Fix tokenizer mock class
    pattern2 = r'class MockTokenizer:(.*?)def __init__'
    replacement2 = r'class MockTokenizer:\n        def __init__'
    content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
    
    return content

def fix_method_content_indentation(content):
    """Fix indentation in method content blocks."""
    # Find if/else statements in methods and ensure proper indentation
    pattern = r'(\s+)(if|elif|else)([^\n]*?):\n(\s+)(\w+)'
    replacement = lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}:\n{m.group(1)}    {m.group(5)}"
    content = re.sub(pattern, replacement, content)
    
    # Fix logger statements in methods
    content = re.sub(r'(\s+)logger\.info\(', r'        logger.info(', content)
    content = re.sub(r'(\s+)logger\.warning\(', r'        logger.warning(', content)
    content = re.sub(r'(\s+)logger\.error\(', r'        logger.error(', content)
    
    return content

def fix_main_indentation_issues(file_path, backup=True):
    """
    Fix the most critical indentation issues in test files.
    
    Args:
        file_path: Path to the test file to fix
        backup: If True, create a backup of the file before fixing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.bak"
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply fixes one by one
        logger.info(f"Applying indentation fixes to {file_path}...")
        
        # Fix conditional indentation first
        content = fix_if_conditionals(content)
        
        # Fix tokenizer mock blocks
        content = fix_tokenizer_mock(content)
        
        # Fix method parameters
        content = fix_method_parameter_indentation(content)
        
        # Fix try/except blocks
        content = fix_try_except_indentation(content)
        
        # Fix line continuation
        content = fix_indented_line_continuation(content)
        
        # Fix if/else indentation
        content = fix_if_else_indentation(content)
        
        # Fix mock class methods
        content = fix_mock_class_indentation(content)
        
        # Fix method content indentation
        content = fix_method_content_indentation(content)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify the syntax
        is_valid, error = verify_python_syntax(file_path)
        if is_valid:
            logger.info(f"✅ Successfully fixed indentation in {file_path}")
            return True
        else:
            logger.error(f"❌ Syntax error in {file_path}: {error}")
            
            # In case of failure, try a more aggressive approach - directly fix known issues
            logger.info("Trying more aggressive fixes...")
            
            # Read the file again
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Process lines one by one for specific fixes
            fixed_lines = []
            skip_next = False
            for i, line in enumerate(lines):
                if skip_next:
                    skip_next = False
                    continue
                    
                # Check for specific line-by-line fixes
                if "def decode(self, ids, **kwargs):" in line and not line.strip().startswith("def"):
                    fixed_lines.append("        def decode(self, ids, **kwargs):\n")
                elif "if not HAS_SENTENCEPIECE:" in line and i+1 < len(lines) and "class" in lines[i+1]:
                    fixed_lines.append(line)
                    if not lines[i+1].strip().startswith("class"):
                        fixed_lines.append("    class MockSentencePieceProcessor:\n")
                        skip_next = True
                elif "HAS_TORCH = False" in line and "logger" in line:
                    # Fix the combined line with logger
                    fixed_lines.append("    HAS_TORCH = False\n")
                    fixed_lines.append("    logger.warning(\"torch not available, using mock\")\n")
                elif "HAS_TRANSFORMERS = False" in line and "logger" in line:
                    # Fix the combined line with logger
                    fixed_lines.append("    HAS_TRANSFORMERS = False\n")
                    fixed_lines.append("    logger.warning(\"transformers not available, using mock\")\n")
                elif "HAS_TOKENIZERS = False" in line and "logger" in line:
                    # Fix the combined line with logger
                    fixed_lines.append("    HAS_TOKENIZERS = False\n")
                    fixed_lines.append("    logger.warning(\"tokenizers not available, using mock\")\n")
                elif "HAS_SENTENCEPIECE = False" in line and "logger" in line:
                    # Fix the combined line with logger
                    fixed_lines.append("    HAS_SENTENCEPIECE = False\n")
                    fixed_lines.append("    logger.warning(\"sentencepiece not available, using mock\")\n")
                elif "HAS_PIL = False" in line and "logger" in line:
                    # Fix the combined line with logger
                    fixed_lines.append("    HAS_PIL = False\n")
                    fixed_lines.append("    logger.warning(\"PIL or requests not available, using mock\")\n")
                else:
                    fixed_lines.append(line)
            
            # Write the aggressively fixed content
            with open(file_path, 'w') as f:
                f.writelines(fixed_lines)
            
            # Verify again
            is_valid, error = verify_python_syntax(file_path)
            if is_valid:
                logger.info(f"✅ Successfully fixed indentation with aggressive approach: {file_path}")
                return True
            else:
                logger.error(f"❌ Still has syntax errors after aggressive fixes: {error}")
                return False
            
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix indentation issues in HuggingFace test files")
    parser.add_argument("files", nargs="+", help="Files to fix")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups")
    parser.add_argument("--check-only", action="store_true", help="Only check for syntax errors, don't fix")
    
    args = parser.parse_args()
    
    # Process each file
    success_count = 0
    failure_count = 0
    
    for file_path in args.files:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            failure_count += 1
            continue
        
        # Check only mode
        if args.check_only:
            is_valid, error = verify_python_syntax(file_path)
            if is_valid:
                logger.info(f"✅ Syntax is valid: {file_path}")
                success_count += 1
            else:
                logger.error(f"❌ Syntax error in {file_path}: {error}")
                failure_count += 1
            continue
        
        # Fix mode
        if fix_main_indentation_issues(file_path, backup=not args.no_backup):
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"- Files processed: {len(args.files)}")
    logger.info(f"- Successful: {success_count}")
    logger.info(f"- Failed: {failure_count}")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
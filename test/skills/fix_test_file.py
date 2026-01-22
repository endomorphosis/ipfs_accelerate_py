#!/usr/bin/env python3

import os
import sys
import re
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_test_file(file_path):
    """Fix severe issues in a test file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # First fix the try-except blocks
        content = fix_try_except_blocks(content)
        
        # Fix method definitions
        content = fix_method_definitions(content)
        
        # Fix main method
        content = fix_main_method(content)
        
        # Fix class initialization
        content = fix_class_init(content)
        
        # Fix triple quotes
        content = fix_triple_quotes(content)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify the syntax
        try:
            compile(content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax is valid after fixes")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax errors remain: {e}")
            logger.error(f"At line {e.lineno}: {e.text}")
            return False
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False

def fix_try_except_blocks(content):
    """Fix broken try-except blocks."""
    # Fix the broken try-except block for transformers import
    import_try_except_pattern = re.compile(
        r'try:\s*\n\s*import transformers\s*\nimport transformers\s*\n\s*HAS_TRANSFORMERS = True',
        re.MULTILINE
    )
    if import_try_except_pattern.search(content):
        logger.info("Fixing broken transformers import try-except block")
        content = re.sub(
            r'try:\s*\n\s*import transformers\s*\nimport transformers\s*\n\s*HAS_TRANSFORMERS = True',
            'try:\n    import transformers\n    HAS_TRANSFORMERS = True',
            content
        )

    # Fix the broken try-except block in test_pipeline method
    pipeline_try_except_pattern = re.compile(
        r'try:\s*\n\s*if not HAS_TRANSFORMERS:\s*\n\s*if not HAS_TRANSFORMERS:',
        re.MULTILINE
    )
    if pipeline_try_except_pattern.search(content):
        logger.info("Fixing broken try-except block in test_pipeline method")
        content = re.sub(
            r'try:\s*\n\s*if not HAS_TRANSFORMERS:\s*\n\s*if not HAS_TRANSFORMERS:',
            'try:\n            if not HAS_TRANSFORMERS:',
            content
        )

    # Fix doubled error handler
    doubled_error_handler = re.compile(
        r'except Exception as e:\s*\n\s*logger\.error\(.*\)\s*\n\s*logger\.error\(',
        re.MULTILINE
    )
    if doubled_error_handler.search(content):
        logger.info("Fixing doubled error handler")
        content = re.sub(
            r'except Exception as e:\s*\n\s*logger\.error\(.*\)\s*\n\s*logger\.error\(',
            'except Exception as e:\n            logger.error(',
            content
        )

    return content

def fix_method_definitions(content):
    """Fix broken method definitions."""
    # Fix the run_tests method that got attached to test_pipeline
    run_tests_pattern = re.compile(
        r'return {"success": False, "error": str\(e\)}\s*def run_tests\(',
        re.MULTILINE
    )
    if run_tests_pattern.search(content):
        logger.info("Fixing broken run_tests method definition")
        content = re.sub(
            r'return {"success": False, "error": str\(e\)}\s*def run_tests\(',
            'return {"success": False, "error": str(e)}\n\n    def run_tests(',
            content
        )

    # Fix the return results line
    return_results_pattern = re.compile(
        r'}\s*return results',
        re.MULTILINE
    )
    if return_results_pattern.search(content):
        logger.info("Fixing broken return statement")
        content = re.sub(
            r'}\s*return results',
            '}\n        return results',
            content
        )

    return content

def fix_main_method(content):
    """Fix issues in the main method."""
    # Fix the broken if __name__ block
    main_pattern = re.compile(
        r'return 0 if success else 1\s*\n\s*if __name__ == "__main__":\s*\n\s*sys\.exit\(main\(\)\)',
        re.MULTILINE
    )
    if main_pattern.search(content):
        logger.info("Fixing broken main block")
        content = re.sub(
            r'return 0 if success else 1\s*\n\s*if __name__ == "__main__":\s*\n\s*sys\.exit\(main\(\)\)',
            'return 0 if success else 1\n\nif __name__ == "__main__":\n    sys.exit(main())',
            content
        )

    # Fix TestBertModels reference that should be TestXLMRobertaModels
    test_class_pattern = re.compile(r'xlm_roberta_tester = TestBertModels\(', re.MULTILINE)
    if test_class_pattern.search(content):
        logger.info("Fixing test class reference")
        content = re.sub(
            r'xlm_roberta_tester = TestBertModels\(',
            'xlm_roberta_tester = TestXLMRobertaModels(',
            content
        )

    # Fix bert_tester reference that should be xlm_roberta_tester
    tester_pattern = re.compile(r'Device: {bert_tester\.device}', re.MULTILINE)
    if tester_pattern.search(content):
        logger.info("Fixing tester reference")
        content = re.sub(
            r'Device: {bert_tester\.device}',
            'Device: {xlm_roberta_tester.device}',
            content
        )

    return content

def fix_class_init(content):
    """Fix missing __init__ method."""
    # Check if __init__ is missing
    class_def_pattern = re.compile(
        r'class TestXLMRobertaModels:\s*\n\s*"""\s*\n\s*Test class for BERT models\.\s*\n\s*"""\s*\n\s*Initialize',
        re.MULTILINE
    )
    if class_def_pattern.search(content):
        logger.info("Fixing missing __init__ method")
        content = re.sub(
            r'class TestXLMRobertaModels:\s*\n\s*"""\s*\n\s*Test class for BERT models\.\s*\n\s*"""\s*\n\s*Initialize',
            'class TestXLMRobertaModels:\n    """\n    Test class for BERT models.\n    """\n\n    def __init__(self, model_id="xlm-roberta-base", device=None):\n        """\n        Initialize',
            content
        )

    return content

def fix_triple_quotes(content):
    """Remove extra triple quotes at the end of the file."""
    # Remove extra triple quotes at EOF
    content = re.sub(r'"""(?:\s*""")*\s*$', '', content.rstrip()) + '\n'
    
    # Count triple quotes to ensure they're balanced
    if content.count('"""') % 2 != 0:
        logger.info("Adding missing closing triple quote")
        content += '"""\n'
    
    return content

def main():
    parser = argparse.ArgumentParser(description="Fix severe issues in test files")
    parser.add_argument("--file", type=str, required=True, help="The file to fix")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return 1
    
    if fix_test_file(args.file):
        logger.info(f"Successfully fixed {args.file}")
        return 0
    else:
        logger.error(f"Failed to fix {args.file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
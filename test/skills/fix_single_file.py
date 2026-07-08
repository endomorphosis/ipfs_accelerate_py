#!/usr/bin/env python3
"""
Fix HuggingFace Test Files - Single File Version

Simplified script to fix mock detection issues in a single test file.
This version applies only essential fixes and has additional safeguards.

Usage:
    python fix_single_file.py --file PATH_TO_FILE
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
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def check_mock_detection(content):
    """
    Check if mock detection system is properly implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        dict: Dictionary with check results
    """
    results = {
        "has_env_vars": {
            "torch": "MOCK_TORCH" in content,
            "transformers": "MOCK_TRANSFORMERS" in content,
            "tokenizers": "MOCK_TOKENIZERS" in content,
            "sentencepiece": "MOCK_SENTENCEPIECE" in content
        },
        "has_imports": {
            "torch": "import torch" in content,
            "transformers": "import transformers" in content,
            "tokenizers": "import tokenizers" in content,
            "sentencepiece": "import sentencepiece" in content
        },
        "has_detection": {
            "torch": "HAS_TORCH" in content,
            "transformers": "HAS_TRANSFORMERS" in content,
            "tokenizers": "HAS_TOKENIZERS" in content,
            "sentencepiece": "HAS_SENTENCEPIECE" in content
        },
        "has_mock_control": {
            "torch": "if MOCK_TORCH:" in content,
            "transformers": "if MOCK_TRANSFORMERS:" in content,
            "tokenizers": "if MOCK_TOKENIZERS:" in content,
            "sentencepiece": "if MOCK_SENTENCEPIECE:" in content
        },
        "has_inference_detection": "using_real_inference =" in content,
        "has_mock_detection": "using_mocks =" in content
    }
    
    # Derive overall results
    results["complete"] = (
        all(results["has_env_vars"].values()) and
        all(results["has_detection"].values()) and
        all(results["has_mock_control"].values()) and
        results["has_inference_detection"] and
        results["has_mock_detection"]
    )
    
    return results

def add_env_vars(content):
    """
    Add environment variable declarations for mock control if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with environment variable declarations
    """
    env_vars_to_add = []
    
    # Check which env vars are missing
    if "MOCK_TORCH" not in content:
        env_vars_to_add.append("MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'")
    
    if "MOCK_TRANSFORMERS" not in content:
        env_vars_to_add.append("MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'")
    
    if "MOCK_TOKENIZERS" not in content:
        env_vars_to_add.append("MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'")
    
    if "MOCK_SENTENCEPIECE" not in content:
        env_vars_to_add.append("MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'")
    
    if not env_vars_to_add:
        return content
    
    # Find insertion point: after MagicMock import
    if "from unittest.mock import MagicMock" in content:
        insert_after = "from unittest.mock import MagicMock"
        insert_point = content.find(insert_after) + len(insert_after)
        insert_point = content.find('\n', insert_point) + 1
        
        # Insert env vars
        env_vars_block = "\n# Check if we should mock specific dependencies\n" + "\n".join(env_vars_to_add) + "\n"
        content = content[:insert_point] + env_vars_block + content[insert_point:]
        logger.info(f"Added environment variables: {', '.join([var.split(' =')[0] for var in env_vars_to_add])}")
    else:
        logger.warning("Could not find suitable location to add environment variables")
    
    return content

def fix_mock_import_controls(content):
    """
    Fix mock import controls by adding mock checks to imports.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with fixed mock imports
    """
    # Fix torch import if needed
    if "import torch" in content and "if MOCK_TORCH:" not in content:
        torch_import_pattern = r"try:\s+import torch\s+HAS_TORCH = True\s+except ImportError:"
        torch_import_with_mock = """try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    HAS_TORCH = True
except ImportError:"""
        content = re.sub(torch_import_pattern, torch_import_with_mock, content)
        logger.info("Added mock control to torch import")
    
    # Fix transformers import if needed
    if "import transformers" in content and "if MOCK_TRANSFORMERS:" not in content:
        transformers_import_pattern = r"try:\s+import transformers\s+HAS_TRANSFORMERS = True\s+except ImportError:"
        transformers_import_with_mock = """try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:"""
        content = re.sub(transformers_import_pattern, transformers_import_with_mock, content)
        logger.info("Added mock control to transformers import")
    
    # Fix tokenizers import if needed
    if "import tokenizers" in content and "if MOCK_TOKENIZERS:" not in content:
        tokenizers_import_pattern = r"try:\s+import tokenizers\s+HAS_TOKENIZERS = True\s+except ImportError:"
        tokenizers_import_with_mock = """try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:"""
        content = re.sub(tokenizers_import_pattern, tokenizers_import_with_mock, content)
        logger.info("Added mock control to tokenizers import")
    
    # Fix sentencepiece import if needed
    if "import sentencepiece" in content and "if MOCK_SENTENCEPIECE:" not in content:
        sentencepiece_import_pattern = r"try:\s+import sentencepiece\s+HAS_SENTENCEPIECE = True\s+except ImportError:"
        sentencepiece_import_with_mock = """try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:"""
        content = re.sub(sentencepiece_import_pattern, sentencepiece_import_with_mock, content)
        logger.info("Added mock control to sentencepiece import")
    
    return content

def add_missing_imports(content):
    """
    Add missing mock imports for tokenizers and sentencepiece.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock imports added
    """
    # Check if necessary imports are missing
    needs_tokenizers = "import tokenizers" not in content and "HAS_TOKENIZERS" not in content
    needs_sentencepiece = "import sentencepiece" not in content and "HAS_SENTENCEPIECE" not in content
    
    if not needs_tokenizers and not needs_sentencepiece:
        return content
    
    # Find appropriate insertion point after transformers import
    transformers_import = re.search(r'# Try to import transformers.*?logger\.warning\("transformers not available, using mock"\)', content, re.DOTALL)
    
    if not transformers_import:
        logger.warning("Couldn't find transformers import section to add mock imports")
        return content
    
    insert_point = transformers_import.end()
    
    # Add tokenizers import if needed
    if needs_tokenizers:
        tokenizers_import = """

# Try to import tokenizers
try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")"""
        
        content = content[:insert_point] + tokenizers_import + content[insert_point:]
        insert_point += len(tokenizers_import)
        logger.info("Added tokenizers mock import")
    
    # Add sentencepiece import if needed
    if needs_sentencepiece:
        sentencepiece_import = """

# Try to import sentencepiece
try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")"""
        
        content = content[:insert_point] + sentencepiece_import + content[insert_point:]
        logger.info("Added sentencepiece mock import")
    
    return content

def add_mock_detection(content):
    """
    Add mock detection logic if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock detection added
    """
    if "using_real_inference =" in content and "using_mocks =" in content:
        return content
    
    # Find run_tests method
    run_tests_match = re.search(r'def run_tests\([^)]*\):', content)
    if not run_tests_match:
        logger.warning("Could not find run_tests method to add mock detection")
        return content
    
    # Find return statement in run_tests method
    start_pos = run_tests_match.end()
    return_match = re.search(r'\s+return\s+{', content[start_pos:])
    if not return_match:
        logger.warning("Could not find return statement in run_tests method")
        return content
    
    insert_point = start_pos + return_match.start()
    
    # Extract indentation from the return line
    return_line = return_match.group(0)
    indentation = re.match(r'(\s+)', return_line).group(1)
    
    # Create mock detection code
    mock_detection_code = f"""
{indentation}# Determine if real inference or mock objects were used
{indentation}using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
{indentation}using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE

"""
    
    # Insert mock detection code
    content = content[:insert_point] + mock_detection_code + content[insert_point:]
    logger.info("Added mock detection logic")
    
    return content

def fix_file(file_path):
    """
    Apply essential fixes to a test file.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check current state
        check_results = check_mock_detection(content)
        
        if check_results["complete"]:
            logger.info(f"✅ {file_path}: Mock detection is already complete")
            return True
        
        # Create backup
        backup_path = f"{file_path}.fix.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply essential fixes
        content = add_env_vars(content)
        content = fix_mock_import_controls(content)
        content = add_missing_imports(content)
        content = add_mock_detection(content)
        
        # Verify syntax before writing
        try:
            compile(content, file_path, 'exec')
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in modified content: {e}")
            logger.info(f"Skipping changes to avoid syntax errors")
            return False
        
        # Write changes
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify mock detection is now complete
        with open(file_path, 'r') as f:
            updated_content = f.read()
        
        check_results = check_mock_detection(updated_content)
        
        if check_results["complete"]:
            logger.info(f"✅ {file_path}: Successfully fixed mock detection")
            return True
        else:
            logger.warning(f"⚠️ {file_path}: Fixes applied but mock detection still incomplete")
            missing = []
            
            # Check environment variables
            for key, has_it in check_results["has_env_vars"].items():
                if not has_it:
                    missing.append(f"MOCK_{key.upper()}")
            
            # Check imports
            for key, has_it in check_results["has_detection"].items():
                if not has_it:
                    missing.append(f"HAS_{key.upper()}")
            
            # Check mock control
            for key, has_it in check_results["has_mock_control"].items():
                if not has_it:
                    missing.append(f"Mock control for {key}")
            
            # Check other elements
            if not check_results["has_inference_detection"]:
                missing.append("Real inference detection")
            
            if not check_results["has_mock_detection"]:
                missing.append("Mock detection")
            
            logger.warning(f"Still missing: {', '.join(missing)}")
            return False
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix mock detection in a single test file")
    parser.add_argument("--file", type=str, required=True, help="Test file to fix")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"{RED}File not found: {args.file}{RESET}")
        return 1
    
    success = fix_file(args.file)
    
    if success:
        print(f"\n{GREEN}✅ Successfully fixed mock detection in {args.file}{RESET}")
    else:
        print(f"\n{YELLOW}⚠️ Partially fixed mock detection in {args.file} - manual review needed{RESET}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
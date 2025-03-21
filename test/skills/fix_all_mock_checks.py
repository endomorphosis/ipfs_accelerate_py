#!/usr/bin/env python3
"""
Fix all mock checks in test files.

This script specifically targets the mock check issues in the import sections,
ensuring that all dependencies have proper mock checks implemented.

Usage:
    python fix_all_mock_checks.py [--file FILE_PATH] [--dir DIRECTORY]
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

def fix_import_mock_checks(content):
    """
    Fix import mock checks for all dependencies.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with fixed mock checks
    """
    # Fix torch import
    torch_pattern = r"try:[^\n]*\s+import torch\s+HAS_TORCH = True"
    torch_replacement = """try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    HAS_TORCH = True"""
    content = re.sub(torch_pattern, torch_replacement, content)
    
    # Fix transformers import
    transformers_pattern = r"try:[^\n]*\s+import transformers\s+HAS_TRANSFORMERS = True"
    transformers_replacement = """try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    HAS_TRANSFORMERS = True"""
    content = re.sub(transformers_pattern, transformers_replacement, content)
    
    # Fix tokenizers import
    tokenizers_pattern = r"try:[^\n]*\s+import tokenizers\s+HAS_TOKENIZERS = True"
    tokenizers_replacement = """try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True"""
    content = re.sub(tokenizers_pattern, tokenizers_replacement, content)
    
    # Fix sentencepiece import
    sentencepiece_pattern = r"try:[^\n]*\s+import sentencepiece\s+HAS_SENTENCEPIECE = True"
    sentencepiece_replacement = """try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True"""
    content = re.sub(sentencepiece_pattern, sentencepiece_replacement, content)
    
    return content

def add_missing_imports(content):
    """
    Add missing imports for tokenizers and sentencepiece.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with added imports
    """
    # Check if environment variables are defined
    if 'MOCK_TOKENIZERS' not in content:
        # Find the existing environment variable section
        env_vars = re.search(r'MOCK_TORCH.*?MOCK_TRANSFORMERS.*?(?=\n# Try|import)', content, re.DOTALL)
        
        if env_vars:
            env_section = env_vars.group(0)
            if 'MOCK_TOKENIZERS' not in env_section:
                new_env_section = env_section + "\nMOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'"
                content = content.replace(env_section, new_env_section)
    
    # Add MOCK_SENTENCEPIECE if missing
    if 'MOCK_SENTENCEPIECE' not in content:
        # Find the updated environment variable section
        env_vars = re.search(r'MOCK_TORCH.*?MOCK_TRANSFORMERS.*?(?=\n# Try|import)', content, re.DOTALL)
        
        if env_vars:
            env_section = env_vars.group(0)
            if 'MOCK_SENTENCEPIECE' not in env_section:
                new_env_section = env_section + "\nMOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'"
                content = content.replace(env_section, new_env_section)
    
    # Check for tokenizers import
    if 'import tokenizers' not in content or 'HAS_TOKENIZERS' not in content:
        # Find location to insert tokenizers import
        transformers_import_end = re.search(r'logger\.warning\("transformers not available, using mock"\)', content)
        
        if transformers_import_end:
            insert_point = transformers_import_end.end()
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
    
    # Check for sentencepiece import
    if 'import sentencepiece' not in content or 'HAS_SENTENCEPIECE' not in content:
        # Find location to insert sentencepiece import
        tokenizers_import_end = re.search(r'logger\.warning\("tokenizers not available, using mock"\)', content)
        
        if tokenizers_import_end:
            insert_point = tokenizers_import_end.end()
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
    
    return content

def fix_detection_logic(content):
    """
    Fix or add detection logic for mock detection.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with fixed detection logic
    """
    # Check for detection logic
    if 'using_real_inference = HAS_TRANSFORMERS and HAS_TORCH' not in content:
        # Find return statement in run_tests method
        run_tests_return = re.search(r'def run_tests\([^)]*\):[^}]*return\s*{', content, re.DOTALL)
        
        if run_tests_return:
            insert_point = run_tests_return.end()
            detection_logic = """
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
"""
            content = content[:insert_point] + detection_logic + content[insert_point:]
    
    # Check for metadata in return statement
    if '"using_real_inference": using_real_inference' not in content:
        # Find metadata section in return statement
        metadata_pattern = r'"metadata":\s*{([^}]*)}'
        metadata_match = re.search(metadata_pattern, content)
        
        if metadata_match:
            existing_metadata = metadata_match.group(1)
            new_metadata = existing_metadata + """
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
"""
            new_metadata_section = f'"metadata": {{{new_metadata}}}'
            content = re.sub(metadata_pattern, new_metadata_section, content)
    
    return content

def fix_visual_indicators(content):
    """
    Fix or add visual indicators for mock detection.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with fixed visual indicators
    """
    # Add color codes if missing
    if 'GREEN = "\\033[32m"' not in content:
        # Find a good location to add color codes
        import_section_end = re.search(r'import\s+datetime', content)
        
        if import_section_end:
            insert_point = import_section_end.end()
            color_codes = """

# ANSI color codes for terminal output
GREEN = "\\033[32m"
BLUE = "\\033[34m"
RESET = "\\033[0m"
"""
            content = content[:insert_point] + color_codes + content[insert_point:]
    
    # Add visual indicators if missing
    if '🚀 Using REAL INFERENCE with actual models' not in content:
        # Find a good location to add visual indicators
        success_print = re.search(r'print\(f"Successfully tested', content)
        
        if success_print:
            # Move back to find a better insertion point
            section_before = content[:success_print.start()].rfind("\n\n")
            if section_before != -1:
                insert_point = section_before
                visual_indicators = """
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")

"""
                content = content[:insert_point] + visual_indicators + content[insert_point:]
    
    return content

def fix_file(file_path):
    """
    Apply all fixes to a file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if fixes were applied successfully, False otherwise
    """
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = f"{file_path}.mock_checks.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply fixes
        content = add_missing_imports(content)
        content = fix_import_mock_checks(content)
        content = fix_detection_logic(content)
        content = fix_visual_indicators(content)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify fixes
        with open(file_path, 'r') as f:
            updated_content = f.read()
        
        # Check if import mock checks are now present
        has_torch_check = "if MOCK_TORCH:" in updated_content
        has_transformers_check = "if MOCK_TRANSFORMERS:" in updated_content
        
        # Additional checks only if imports exist
        has_tokenizers_check = True
        if "import tokenizers" in updated_content:
            has_tokenizers_check = "if MOCK_TOKENIZERS:" in updated_content
        
        has_sentencepiece_check = True
        if "import sentencepiece" in updated_content:
            has_sentencepiece_check = "if MOCK_SENTENCEPIECE:" in updated_content
        
        # Check if all needed fixes were applied
        success = has_torch_check and has_transformers_check and has_tokenizers_check and has_sentencepiece_check
        
        if success:
            logger.info(f"✅ Successfully fixed all mock checks in {file_path}")
        else:
            logger.error(f"❌ Failed to fix all mock checks in {file_path}")
            
            if not has_torch_check:
                logger.error("  - Missing torch mock check")
            if not has_transformers_check:
                logger.error("  - Missing transformers mock check")
            if not has_tokenizers_check:
                logger.error("  - Missing tokenizers mock check")
            if not has_sentencepiece_check:
                logger.error("  - Missing sentencepiece mock check")
        
        return success
    
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix all mock checks in test files")
    parser.add_argument("--file", type=str, help="Path to a specific file to fix")
    parser.add_argument("--dir", type=str, default="fixed_tests", help="Directory containing files to fix")
    
    args = parser.parse_args()
    
    if args.file:
        # Fix a single file
        if os.path.exists(args.file):
            success = fix_file(args.file)
            return 0 if success else 1
        else:
            logger.error(f"File not found: {args.file}")
            return 1
    
    # Fix all files in directory
    if not os.path.exists(args.dir):
        logger.error(f"Directory not found: {args.dir}")
        return 1
    
    # Find all test files
    test_files = []
    for root, _, files in os.walk(args.dir):
        for file in files:
            if file.startswith("test_hf_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(test_files)} test files to fix")
    
    # Fix each file
    success_count = 0
    failure_count = 0
    
    for file_path in test_files:
        if fix_file(file_path):
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"- Successfully fixed: {success_count} files")
    logger.info(f"- Failed to fix: {failure_count} files")
    logger.info(f"- Total processed: {len(test_files)} files")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
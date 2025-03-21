#\!/usr/bin/env python3
"""
Integrate fixes to ensure consistent mock detection in test generator files.

This script:
1. Adds missing HAS_TOKENIZERS and HAS_SENTENCEPIECE imports in test files
2. Ensures the imports for non-BERT models are handled properly
3. Fixes model class names for GPT2, T5, ViT, etc.
4. Updates variables and class names for consistency

Usage:
    python integrate_generator_fixes.py [--file FILE_PATH]
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

def fix_missing_imports(content, file_path):
    """Fix missing import variables like HAS_TOKENIZERS and HAS_SENTENCEPIECE."""
    model_type = os.path.basename(file_path)[8:-3]  # Extract model type from test_hf_MODEL.py
    
    # First check if the imports are missing
    has_tokenizers_var = "HAS_TOKENIZERS" in content
    has_sentencepiece_var = "HAS_SENTENCEPIECE" in content
    
    if has_tokenizers_var and has_sentencepiece_var:
        logger.info(f"✅ {file_path}: All necessary import variables present")
        return content
    
    # Add missing import sections
    if not has_tokenizers_var:
        logger.info(f"❌ {file_path}: Missing HAS_TOKENIZERS variable, adding it")
        # Find where to add the tokenizers import section
        if "Try to import tokenizers" not in content:
            # Add after transformers import
            transformers_import_match = re.search(r'(# Try to import transformers.*?HAS_TRANSFORMERS = False.*?logger\.warning\("transformers not available, using mock"\))', content, re.DOTALL)
            if transformers_import_match:
                tokenizers_section = """
# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")
"""
                insert_pos = transformers_import_match.end()
                content = content[:insert_pos] + tokenizers_section + content[insert_pos:]
                
    if not has_sentencepiece_var:
        logger.info(f"❌ {file_path}: Missing HAS_SENTENCEPIECE variable, adding it")
        # Find where to add the sentencepiece import section
        if "Try to import sentencepiece" not in content:
            # Add after tokenizers import (or transformers import if we just added tokenizers)
            tokenizers_import_match = re.search(r'(# Try to import tokenizers.*?HAS_TOKENIZERS = False.*?logger\.warning\("tokenizers not available, using mock"\))', content, re.DOTALL)
            if tokenizers_import_match:
                sentencepiece_section = """
# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")
"""
                insert_pos = tokenizers_import_match.end()
                content = content[:insert_pos] + sentencepiece_section + content[insert_pos:]
            else:
                # If tokenizers section wasn't found and we're adding both, check if we can add after transformers
                transformers_import_match = re.search(r'(# Try to import transformers.*?HAS_TRANSFORMERS = False.*?logger\.warning\("transformers not available, using mock"\))', content, re.DOTALL)
                if transformers_import_match:
                    both_sections = """
# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")
"""
                    insert_pos = transformers_import_match.end()
                    content = content[:insert_pos] + both_sections + content[insert_pos:]
    
    return content

def fix_model_class_names(content, file_path):
    """Fix model class names (e.g., Gpt2LMHeadModel -> GPT2LMHeadModel)."""
    model_type = os.path.basename(file_path)[8:-3]  # Extract model type from test_hf_MODEL.py
    
    # Define the correct class names for common model types
    model_class_corrections = {
        "gpt2": ("Gpt2LMHeadModel", "GPT2LMHeadModel"),
        "t5": ("T5ForConditionalGeneration", "T5ForConditionalGeneration"),
        "vit": ("VitForImageClassification", "ViTForImageClassification"),
        "swin": ("SwinForImageClassification", "SwinForImageClassification"),
        "clip": ("ClipModel", "CLIPModel"),
        "bart": ("BartForConditionalGeneration", "BartForConditionalGeneration"),
        "whisper": ("WhisperForConditionalGeneration", "WhisperForConditionalGeneration"),
    }
    
    # Apply corrections specific to this model type
    for model_prefix, (incorrect, correct) in model_class_corrections.items():
        if model_type.startswith(model_prefix):
            old_line = f"model = transformers.{incorrect}.from_pretrained"
            new_line = f"model = transformers.{correct}.from_pretrained"
            if old_line in content:
                logger.info(f"❌ {file_path}: Incorrect model class name '{incorrect}', fixing to '{correct}'")
                content = content.replace(old_line, new_line)
    
    return content

def fix_run_tests_function(content, file_path):
    """Fix the run_tests function to correctly handle mock detection."""
    # Check if the run_tests function already has proper mock detection
    mock_detection_pattern = r"using_real_inference\s*=\s*HAS_TRANSFORMERS\s+and\s+HAS_TORCH"
    mocks_pattern = r"using_mocks\s*=\s*not\s+using_real_inference\s+or\s+not\s+HAS_TOKENIZERS\s+or\s+not\s+HAS_SENTENCEPIECE"
    
    has_detection = re.search(mock_detection_pattern, content) is not None
    has_mocks = re.search(mocks_pattern, content) is not None
    
    if has_detection and has_mocks:
        return content  # Already fixed
        
    # Find the run_tests function
    run_tests_match = re.search(r'(def run_tests\(self, all_hardware=False\):.*?return results\s*$)', content, re.DOTALL | re.MULTILINE)
    if not run_tests_match:
        logger.warning(f"❗ {file_path}: Could not find run_tests function, skipping mock detection fix")
        return content
        
    # Get the function content
    run_tests_content = run_tests_match.group(1)
    
    # Check for missing mock detection code
    if not has_detection or not has_mocks:
        # Find where to add the mock detection logic (before the metadata dict)
        metadata_match = re.search(r'(\s+# Add metadata\s+results\["metadata"\] = {)', run_tests_content)
        if metadata_match:
            # Add the missing mock detection logic
            mock_detection_logic = """
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
"""
            prefix = run_tests_content[:metadata_match.start()]
            suffix = run_tests_content[metadata_match.start():]
            updated_function = prefix + mock_detection_logic + suffix
            
            # Replace the function in the full content
            content = content.replace(run_tests_content, updated_function)
            logger.info(f"✅ {file_path}: Added mock detection logic in run_tests function")
    
    # Now check if the metadata section includes mock detection keys
    metadata_keys = [
        '"has_transformers": HAS_TRANSFORMERS', 
        '"has_torch": HAS_TORCH',
        '"has_tokenizers": HAS_TOKENIZERS',
        '"has_sentencepiece": HAS_SENTENCEPIECE',
        '"using_real_inference": using_real_inference',
        '"using_mocks": using_mocks',
        '"test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"'
    ]
    
    # Check for each key
    missing_keys = []
    for key in metadata_keys:
        if key not in content:
            missing_keys.append(key)
    
    if missing_keys:
        # Find the metadata dictionary
        metadata_dict_match = re.search(r'(\s+results\["metadata"\] = {.*?\n\s+})', content, re.DOTALL)
        if metadata_dict_match:
            metadata_dict = metadata_dict_match.group(1)
            # Find the end of the dictionary (last }, before first return)
            dict_closing_match = re.search(r'\n(\s+})(?=\s*\n\s+return)', metadata_dict)
            if dict_closing_match:
                # Add missing keys before the closing brace
                indent = dict_closing_match.group(1).replace('}', '')
                missing_keys_str = ',\n'.join(f"{indent}{key}" for key in missing_keys)
                new_dict = metadata_dict[:dict_closing_match.start()] + ",\n" + missing_keys_str + metadata_dict[dict_closing_match.start():]
                content = content.replace(metadata_dict, new_dict)
                logger.info(f"✅ {file_path}: Added missing mock detection keys to metadata dict")
    
    return content

def fix_main_function(content, file_path):
    """Fix the main function to correctly display mock detection status."""
    # Check if the main function already has proper status display
    status_pattern = r'using_real_inference = results\["metadata"\]\["using_real_inference"\].*?using_mocks = results\["metadata"\]\["using_mocks"\]'
    indicator_pattern = r'if using_real_inference and not using_mocks:.*?print\(f"\{GREEN\}🚀 Using REAL INFERENCE with actual models\{RESET\}"\)'
    
    has_status = re.search(status_pattern, content, re.DOTALL) is not None
    has_indicator = re.search(indicator_pattern, content, re.DOTALL) is not None
    
    if has_status and has_indicator:
        return content  # Already fixed
        
    # Find the main function
    main_match = re.search(r'(def main\(\):.*?)(\n\s*if __name__ == "__main__")', content, re.DOTALL)
    if not main_match:
        logger.warning(f"❗ {file_path}: Could not find main function, skipping status display fix")
        return content
        
    # Fix the main function to add mock detection display
    main_function = main_match.group(1)
    
    # Find the summary section
    summary_section_match = re.search(r'(\s+# Print a summary\s+print\("\n" \+ "="\*50\)\s+print\("TEST RESULTS SUMMARY"\)\s+print\("="\*50\)\s+)', main_function)
    if summary_section_match:
        # Add the mock status display right after the summary header
        mock_status_code = """
    # Indicate real vs mock inference clearly
    using_real_inference = results["metadata"]["using_real_inference"]
    using_mocks = results["metadata"]["using_mocks"]
    
    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
"""
        prefix = main_function[:summary_section_match.end()]
        suffix = main_function[summary_section_match.end():]
        updated_main = prefix + mock_status_code + suffix
        
        # Replace the main function in the full content
        content = content.replace(main_function, updated_main)
        logger.info(f"✅ {file_path}: Added mock status display in main function")
    
    return content

def fix_file(file_path):
    """Apply all fixes to a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Create backup
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply fixes
        content = fix_missing_imports(content, file_path)
        content = fix_model_class_names(content, file_path)
        content = fix_run_tests_function(content, file_path)
        content = fix_main_function(content, file_path)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify the fix works
        try:
            # Basic syntax check
            compile(content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax check passed")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error after fixes: {e}")
            # Restore from backup
            with open(backup_path, 'r') as f:
                original = f.read()
            with open(file_path, 'w') as f:
                f.write(original)
            logger.info(f"Restored from backup due to syntax error")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix mock detection issues in test files")
    parser.add_argument("--file", type=str, help="Path to specific file to fix")
    parser.add_argument("--dir", type=str, default="fixed_tests", help="Directory containing test files to fix")
    
    args = parser.parse_args()
    
    if args.file:
        if os.path.exists(args.file):
            success = fix_file(args.file)
            if success:
                print(f"Successfully fixed {args.file}")
            else:
                print(f"Failed to fix {args.file}")
        else:
            print(f"File not found: {args.file}")
    else:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return 1
            
        # Process all test files in the directory
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.startswith("test_hf_") and f.endswith(".py")]
        
        success_count = 0
        failure_count = 0
        
        for file_path in files:
            print(f"Processing {file_path}...")
            if fix_file(file_path):
                success_count += 1
            else:
                failure_count += 1
                
        print(f"\nSummary:")
        print(f"- Successfully fixed: {success_count} files")
        print(f"- Failed to fix: {failure_count} files")
        print(f"- Total: {len(files)} files")
        
        if failure_count > 0:
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

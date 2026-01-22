#!/usr/bin/env python3
"""
Fix common errors in generated test files.

This script fixes issues with test files that cause mock detection to fail:
1. Adds missing import sections for tokenizers and sentencepiece
2. Fixes model class names (e.g., Gpt2LMHeadModel -> GPT2LMHeadModel)
3. Adds environment variable control to tokenizers and sentencepiece imports
4. Ensures consistency across test files

Usage:
    python fix_mock_detection_errors.py [--file FILE_PATH]
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

def fix_model_class_names(content, model_type):
    """Fix model class names to match transformers library."""
    model_class_fixes = {
        "gpt2": ("Gpt2LMHeadModel", "GPT2LMHeadModel"),
        "t5": ("T5ForConditionalGeneration", "T5ForConditionalGeneration"),  # Already correct
        "vit": ("VitForImageClassification", "ViTForImageClassification"),
        "bart": ("BartForConditionalGeneration", "BartForConditionalGeneration"),  # Already correct
        "bert": ("BertForMaskedLM", "BertForMaskedLM"),  # Already correct
        "clip": ("ClipModel", "CLIPModel"),
        "swin": ("SwinForImageClassification", "SwinForImageClassification"),  # Already correct
    }
    
    for prefix, (wrong, correct) in model_class_fixes.items():
        if model_type.startswith(prefix):
            # Fix class references in model registry
            content = content.replace(f'"class": "{wrong}"', f'"class": "{correct}"')
            
            # Fix class references in code
            content = content.replace(f'self.class_name == "{wrong}"', f'self.class_name == "{correct}"')
            content = content.replace(f'transformers.{wrong}', f'transformers.{correct}')
    
    return content

def add_mock_imports(content):
    """Add missing mock imports for tokenizers and sentencepiece."""
    # Check if tokenizers import is missing
    if 'import tokenizers' not in content or 'HAS_TOKENIZERS' not in content:
        # Find import section for transformers
        transformers_import = re.search(r'# Try to import transformers.*?logger\.warning\("transformers not available, using mock"\)', content, re.DOTALL)
        
        if transformers_import:
            insert_point = transformers_import.end()
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
    
    # Check if sentencepiece import is missing
    if 'import sentencepiece' not in content or 'HAS_SENTENCEPIECE' not in content:
        # Find import section for tokenizers or transformers as fallback
        tokenizers_import = re.search(r'# Try to import tokenizers.*?logger\.warning\("tokenizers not available, using mock"\)', content, re.DOTALL)
        
        if tokenizers_import:
            insert_point = tokenizers_import.end()
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

def add_environment_vars(content):
    """Add missing environment variable declaration for mock control."""
    if 'MOCK_TOKENIZERS' not in content:
        # Find the existing environment variable section
        env_vars = re.search(r'MOCK_TORCH.*?MOCK_TRANSFORMERS.*?(?=\n# Try)', content, re.DOTALL)
        
        if env_vars:
            env_section = env_vars.group(0)
            if 'MOCK_TOKENIZERS' not in env_section:
                new_env_section = env_section + "\nMOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'"
                content = content.replace(env_section, new_env_section)
    
    if 'MOCK_SENTENCEPIECE' not in content:
        # Find the existing environment variable section again (it might have changed)
        env_vars = re.search(r'MOCK_TORCH.*?MOCK_TRANSFORMERS.*?(?=\n# Try)', content, re.DOTALL)
        
        if env_vars:
            env_section = env_vars.group(0)
            if 'MOCK_SENTENCEPIECE' not in env_section:
                if 'MOCK_TOKENIZERS' in env_section:
                    new_env_section = env_section + "\nMOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'"
                else:
                    new_env_section = env_section + "\nMOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'\nMOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'"
                content = content.replace(env_section, new_env_section)
    
    return content

def fix_file(file_path):
    """Apply all fixes to a given file."""
    try:
        # Extract model type from filename
        model_type = os.path.basename(file_path)[8:-3]  # Extract from test_hf_MODEL.py
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = f"{file_path}.fix.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Apply fixes
        content = add_environment_vars(content)
        content = add_mock_imports(content)
        content = fix_model_class_names(content, model_type)
        
        # Write changes
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify changes
        try:
            # Basic syntax check
            compile(content, file_path, 'exec')
            logger.info(f"✅ Fixed file: {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax error after fixes in {file_path}: {e}")
            # Restore backup
            with open(backup_path, 'r') as f:
                original = f.read()
            with open(file_path, 'w') as f:
                f.write(original)
            logger.info(f"Restored backup for {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix common errors in test files")
    parser.add_argument("--file", type=str, help="Specific file to fix")
    parser.add_argument("--dir", type=str, default="fixed_tests", help="Directory containing test files")
    
    args = parser.parse_args()
    
    if args.file:
        # Fix a specific file
        if os.path.exists(args.file):
            success = fix_file(args.file)
            print(f"Fixed file: {args.file} - {'Success' if success else 'Failed'}")
            return 0 if success else 1
        else:
            print(f"File not found: {args.file}")
            return 1
    else:
        # Fix all files in the directory
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return 1
        
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                if f.startswith("test_hf_") and f.endswith(".py")]
        
        success_count = 0
        failure_count = 0
        
        for file_path in files:
            if fix_file(file_path):
                success_count += 1
            else:
                failure_count += 1
        
        print(f"\nSummary:")
        print(f"- Successfully fixed: {success_count} files")
        print(f"- Failed to fix: {failure_count} files")
        print(f"- Total: {len(files)} files")
        
        return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Integrate mock detection and other enhancements into HuggingFace test files.

This script:
1. Adds the mock detection system to all HuggingFace test files
2. Implements architecture-specific template selection
3. Fixes indentation and code formatting issues
4. Ensures consistent dependency checking

Usage:
    python integrate_generator_fixes.py [--check-only] [--all] [--test TEST_FILE]
"""

import os
import sys
import argparse
import logging
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generator_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "albert", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gptj", "gpt-neo", "gpt_neo", "gpt_neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "opt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "mt5", "longt5", "led", "marian"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
}

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    model_type_lower = model_type.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown

def check_mock_detection(file_path):
    """
    Check if a file has the mock detection system implemented.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        bool: True if mock detection is implemented, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check for key mock detection patterns
        has_using_real_inference = "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH" in content
        has_using_mocks = "using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE" in content
        has_visual_indicators = "🚀 Using REAL INFERENCE with actual models" in content and "🔷 Using MOCK OBJECTS for CI/CD testing only" in content
        has_metadata = '"using_real_inference": using_real_inference,' in content and '"using_mocks": using_mocks,' in content
        
        if has_using_real_inference and has_using_mocks and has_visual_indicators and has_metadata:
            return True
        else:
            missing = []
            if not has_using_real_inference: missing.append("using_real_inference definition")
            if not has_using_mocks: missing.append("using_mocks definition")
            if not has_visual_indicators: missing.append("visual indicators")
            if not has_metadata: missing.append("metadata enrichment")
            
            logger.warning(f"❌ {file_path}: Mock detection is missing: {', '.join(missing)}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking mock detection in {file_path}: {e}")
        return False

def fix_indentation(file_path):
    """
    Fix indentation issues in a file using complete_indentation_fix.py.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), "complete_indentation_fix.py")
        if not os.path.exists(script_path):
            logger.error(f"Indentation fix script not found: {script_path}")
            return False
            
        cmd = [sys.executable, script_path, file_path, "--verify"]
        
        logger.info(f"Running indentation fix on {file_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error fixing indentation: {result.stderr}")
            return False
            
        logger.info(f"Successfully fixed indentation in {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error running indentation fix: {e}")
        return False

def regenerate_test(model_type):
    """
    Regenerate a test file using regenerate_fixed_tests.py.
    
    Args:
        model_type: The model type to regenerate (e.g., bert, gpt2)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), "regenerate_fixed_tests.py")
        cmd = [sys.executable, script_path, "--model", model_type, "--verify"]
        
        logger.info(f"Regenerating test for {model_type}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error regenerating test: {result.stderr or result.stdout}")
            return False
            
        logger.info(f"Successfully regenerated test for {model_type}")
        return True
        
    except Exception as e:
        logger.error(f"Error regenerating test: {e}")
        return False

def add_mock_detection_to_generator(generator_path):
    """
    Add mock detection system to the test generator.
    
    Args:
        generator_path: Path to the test generator file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Check if mock detection is already added
        if "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH" in content and \
           "using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE" in content:
            logger.info(f"Mock detection already implemented in {generator_path}")
            return True
        
        # Find the run_tests method
        run_tests_start = content.find("def run_tests(")
        if run_tests_start == -1:
            logger.error("Could not find run_tests method in generator")
            return False
        
        # Find the return statement in run_tests
        return_start = content.find("return {", run_tests_start)
        if return_start == -1:
            logger.error("Could not find return statement in run_tests method")
            return False
        
        # Add mock detection code before return
        mock_detection_code = """
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
"""
        content = content[:return_start] + mock_detection_code + content[return_start:]
        
        # Find metadata in return dictionary
        metadata_start = content.find('"metadata":', return_start)
        if metadata_start == -1:
            logger.error("Could not find metadata in return dictionary")
            return False
        
        # Find closing brace of metadata
        closing_brace = content.find("}", metadata_start)
        if closing_brace == -1:
            logger.error("Could not find closing brace of metadata dictionary")
            return False
        
        # Add mock detection metadata
        mock_metadata = """
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
"""
        content = content[:closing_brace] + mock_metadata + content[closing_brace:]
        
        # Find the main function
        main_start = content.find("def main(")
        if main_start == -1:
            logger.error("Could not find main function in generator")
            return False
        
        # Find TEST RESULTS SUMMARY
        summary_section = content.find("TEST RESULTS SUMMARY", main_start)
        if summary_section == -1:
            logger.warning("Could not find TEST RESULTS SUMMARY section in main function")
            # Find the success print statement
            success_print = content.find('print(f"✅ Successfully tested', main_start)
            if success_print != -1:
                # Find the newline after SUCCESS_TEST
                next_line = content.find("\n", success_print)
                if next_line != -1:
                    summary_section = next_line
        
        if summary_section != -1:
            # Add visual indicators
            visual_indicators_code = """
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"🚀 Using REAL INFERENCE with actual models")
    else:
        print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
"""
            content = content[:summary_section] + visual_indicators_code + content[summary_section:]
        
        # Write updated content
        with open(generator_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Added mock detection to {generator_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error adding mock detection to generator: {e}")
        traceback.print_exc()
        return False

def fix_all_tests(test_dir="fixed_tests"):
    """
    Fix all test files in the specified directory.
    
    Args:
        test_dir: Path to the directory containing test files
        
    Returns:
        Tuple of (success_count, failure_count, total_count)
    """
    success_count = 0
    failure_count = 0
    
    # Get all test files
    test_files = []
    
    try:
        test_dir_path = os.path.join(os.path.dirname(__file__), test_dir)
        for file in os.listdir(test_dir_path):
            if file.startswith("test_hf_") and file.endswith(".py") and not file.endswith(".bak.py"):
                test_files.append(os.path.join(test_dir_path, file))
    except Exception as e:
        logger.error(f"Error listing test files: {e}")
        return 0, 0, 0
    
    logger.info(f"Found {len(test_files)} test files to process")
    
    # Process each file
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        model_type = file_name[8:-3]  # Extract model type from test_hf_MODEL.py
        
        logger.info(f"Processing {file_name} (model: {model_type})...")
        
        # Check if mock detection is implemented
        has_mock_detection = check_mock_detection(file_path)
        
        if not has_mock_detection:
            # Regenerate the test from template
            if regenerate_test(model_type):
                # Fix indentation
                if fix_indentation(file_path):
                    # Check mock detection again
                    if check_mock_detection(file_path):
                        success_count += 1
                    else:
                        failure_count += 1
                else:
                    failure_count += 1
            else:
                failure_count += 1
        else:
            # Just fix indentation if needed
            if fix_indentation(file_path):
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count, len(test_files)

def fix_specific_test(test_file):
    """
    Fix a specific test file.
    
    Args:
        test_file: Name of the test file (e.g., test_hf_bert.py)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not test_file.startswith("test_hf_"):
        test_file = f"test_hf_{test_file}.py"
    elif not test_file.endswith(".py"):
        test_file = f"{test_file}.py"
    
    test_dir_path = os.path.join(os.path.dirname(__file__), "fixed_tests")
    file_path = os.path.join(test_dir_path, test_file)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    model_type = test_file[8:-3]  # Extract model type from test_hf_MODEL.py
    
    # Check if mock detection is implemented
    has_mock_detection = check_mock_detection(file_path)
    
    if not has_mock_detection:
        # Regenerate the test from template
        if regenerate_test(model_type):
            # Fix indentation
            if fix_indentation(file_path):
                # Check mock detection again
                return check_mock_detection(file_path)
            return False
        return False
    else:
        # Just fix indentation if needed
        return fix_indentation(file_path)

def integrate_all_fixes(create_backup_file=True):
    """
    Integrate all fixes to test files and generator.
    
    Args:
        create_backup_file: Whether to create backup files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # First, add mock detection to generator
        generator_path = os.path.join(os.path.dirname(__file__), "test_generator_fixed.py")
        if not os.path.exists(generator_path):
            logger.warning(f"Test generator not found at {generator_path}")
            generator_path = os.path.join(os.path.dirname(__file__), "test_generator.py")
            if not os.path.exists(generator_path):
                logger.error("Could not find test generator file")
                return False
        
        # Create backup if requested
        if create_backup_file:
            backup_file = f"{generator_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                with open(generator_path, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Created backup of generator at {backup_file}")
            except Exception as e:
                logger.error(f"Failed to create backup of generator: {e}")
        
        # Add mock detection to generator
        if not add_mock_detection_to_generator(generator_path):
            logger.error("Failed to add mock detection to generator")
            return False
        
        # Fix indentation in generator
        if not fix_indentation(generator_path):
            logger.error("Failed to fix indentation in generator")
            return False
        
        # Fix all test files
        success_count, failure_count, total_count = fix_all_tests()
        
        if failure_count > 0:
            logger.warning(f"Failed to fix {failure_count} of {total_count} test files")
            return False
        
        logger.info(f"Successfully fixed all {total_count} test files")
        return True
    
    except Exception as e:
        logger.error(f"Error integrating all fixes: {e}")
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Integrate mock detection and other fixes")
    parser.add_argument("--check-only", action="store_true", help="Only check files, don't apply fixes")
    parser.add_argument("--all", action="store_true", help="Fix all test files and generator")
    parser.add_argument("--test", type=str, help="Specific test file to fix (e.g., bert or test_hf_bert.py)")
    parser.add_argument("--generator", action="store_true", help="Only fix the test generator")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    
    args = parser.parse_args()
    
    create_backup_file = not args.no_backup
    
    if args.test:
        success = fix_specific_test(args.test)
        
        if success:
            logger.info(f"Successfully fixed test file: {args.test}")
            print(f"\n✅ Successfully fixed test file: {args.test}")
            return 0
        else:
            logger.error(f"Failed to fix test file: {args.test}")
            print(f"\n❌ Failed to fix test file: {args.test}")
            return 1
    
    elif args.generator:
        generator_path = os.path.join(os.path.dirname(__file__), "test_generator_fixed.py")
        if not os.path.exists(generator_path):
            generator_path = os.path.join(os.path.dirname(__file__), "test_generator.py")
        
        # Create backup if requested
        if create_backup_file:
            backup_file = f"{generator_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                with open(generator_path, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Created backup of generator at {backup_file}")
            except Exception as e:
                logger.error(f"Failed to create backup of generator: {e}")
        
        success = add_mock_detection_to_generator(generator_path) and fix_indentation(generator_path)
        
        if success:
            logger.info("Successfully fixed test generator")
            print("\n✅ Successfully fixed test generator")
            return 0
        else:
            logger.error("Failed to fix test generator")
            print("\n❌ Failed to fix test generator")
            return 1
    
    elif args.check_only:
        # Check test files and generator without fixing
        logger.info("Checking all test files for mock detection")
        
        # Check generator
        generator_path = os.path.join(os.path.dirname(__file__), "test_generator_fixed.py")
        if not os.path.exists(generator_path):
            generator_path = os.path.join(os.path.dirname(__file__), "test_generator.py")
        
        generator_has_mock = check_mock_detection(generator_path)
        if generator_has_mock:
            logger.info(f"✅ Generator {generator_path} has mock detection")
        else:
            logger.warning(f"❌ Generator {generator_path} is missing mock detection")
        
        # Check all test files
        test_files_missing = 0
        test_files_total = 0
        test_dir_path = os.path.join(os.path.dirname(__file__), "fixed_tests")
        
        try:
            for file in os.listdir(test_dir_path):
                if file.startswith("test_hf_") and file.endswith(".py") and not file.endswith(".bak.py"):
                    test_files_total += 1
                    file_path = os.path.join(test_dir_path, file)
                    if not check_mock_detection(file_path):
                        test_files_missing += 1
        except Exception as e:
            logger.error(f"Error checking test files: {e}")
        
        # Print summary
        print("\nMock Detection Check Summary:")
        print(f"- Generator: {'✅ Implemented' if generator_has_mock else '❌ Missing'}")
        print(f"- Test Files: {test_files_total - test_files_missing}/{test_files_total} implemented")
        
        # Return success only if all have mock detection
        if generator_has_mock and test_files_missing == 0:
            print("\n✅ All files have mock detection implemented")
            return 0
        else:
            print("\n❌ Some files are missing mock detection")
            return 1
    
    elif args.all:
        # Apply all fixes
        success = integrate_all_fixes(create_backup_file)
        
        if success:
            logger.info("Successfully integrated all fixes")
            print("\n✅ Successfully integrated all fixes")
            return 0
        else:
            logger.error("Failed to integrate all fixes")
            print("\n❌ Failed to integrate all fixes")
            return 1
    
    else:
        # No option specified, show help
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
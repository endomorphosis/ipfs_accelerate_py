#!/usr/bin/env python3
"""
Regenerate Tests with Enhanced Hardware Support (March 2025)

This script uses the enhanced test and skillset generators to regenerate tests 
with full cross-platform hardware support for all models, ensuring they pass
all tests and benchmarks.

Features:
- Uses the fixed_merged_test_generator_enhanced.py with full hardware support
- Focuses on key model families that need comprehensive hardware compatibility
- Ensures all generated tests have REAL implementations for all platforms
- Updates both test files and skillset implementations for consistency
- Preserves original files with backups
"""

import os
import sys
import time
import shutil
import datetime
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("regenerate_hardware_tests")

# Import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.dirname(current_dir)
sys.path.append(test_dir)

# Constants
PROJECT_ROOT = Path(os.path.dirname(test_dir))
SKILLS_DIR = PROJECT_ROOT / "test" / "skills"
OUTPUT_DIR = PROJECT_ROOT / "test" / "regenerated_skills"
BACKUPS_DIR = PROJECT_ROOT / "test" / "backups"

# Key model families that need enhanced hardware support
KEY_MODELS = [
    "bert", "t5", "llama", "llama3", "clip", "vit", "clap", "whisper", 
    "wav2vec2", "llava", "llava-next", "xclip", "qwen2", "qwen3", "gemma",
    "gemma2", "gemma3", "detr"
]

# Additional models that would benefit from enhanced hardware support
ADDITIONAL_MODELS = [
    "t5", "gpt2", "bert-base-uncased", "vit-base", "clip-vit-base-patch32",
    "resnet", "whisper-base", "wav2vec2-base", "sentence-transformers",
    "distilbert", "roberta", "electra", "convnext", "swin", "hubert",
    "speecht5", "flava", "vilt", "git", "sbert", "blenderbot", "bart",
    "pegasus"
]

def setup_directories():
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {OUTPUT_DIR}")
    logger.info(f"Created backups directory: {BACKUPS_DIR}")

def backup_original_files():
    """Backup original test files before regenerating."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUPS_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    
    # Backup all existing test files
    for test_file in SKILLS_DIR.glob("test_hf_*.py"):
        shutil.copy(test_file, backup_dir / test_file.name)
    
    logger.info(f"Backed up {len(list(backup_dir.glob('*.py')))} test files to {backup_dir}")
    return backup_dir

def find_failing_tests() -> List[str]:
    """Find test files that are failing and need regeneration."""
    # This would ideally use test results database, but for now we'll 
    # regenerate all key model tests as we've updated hardware support for all
    failing_tests = []
    
    for model in KEY_MODELS:
        # Find all test files that match this model prefix
        for test_file in SKILLS_DIR.glob(f"test_hf_{model}*.py"):
            model_name = test_file.stem.replace("test_hf_", "")
            failing_tests.append(model_name)
    
    # Add any additional models
    for model in ADDITIONAL_MODELS:
        normalized_name = model.replace("-", "_").replace(".", "_").lower()
        test_file = SKILLS_DIR / f"test_hf_{normalized_name}.py"
        if test_file.exists():
            failing_tests.append(normalized_name)
    
    logger.info(f"Found {len(failing_tests)} tests to regenerate")
    return failing_tests

def regenerate_test(model_name: str) -> bool:
    """Regenerate a test file with enhanced hardware support."""
    # Build the command to call the enhanced test generator
    generator_path = os.path.join(current_dir, "fixed_merged_test_generator_enhanced.py")
    
    # Make sure the generator file exists
    if not os.path.exists(generator_path):
        logger.error(f"Enhanced generator not found: {generator_path}")
        return False
    
    # Make sure the generator is executable
    os.chmod(generator_path, 0o755)
    
    # Build the command
    cmd = [
        sys.executable, 
        generator_path, 
        "--generate", model_name.replace("_", "-"),
        "--output-dir", str(OUTPUT_DIR),
        "--force",
        "--platform", "all"  # Generate with support for all platforms
    ]
    
    # Run the generator
    try:
        logger.info(f"Regenerating test for {model_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully regenerated test for {model_name}")
            return True
        else:
            logger.error(f"Failed to regenerate test for {model_name}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error regenerating test for {model_name}: {e}")
        return False

def regenerate_skillset(model_name: str) -> bool:
    """Regenerate a skillset implementation with enhanced hardware support."""
    # Build the command to call the enhanced skillset generator
    generator_path = os.path.join(current_dir, "integrated_skillset_generator_enhanced.py")
    
    # Make sure the generator file exists
    if not os.path.exists(generator_path):
        logger.error(f"Enhanced skillset generator not found: {generator_path}")
        return False
    
    # Make sure the generator is executable
    os.chmod(generator_path, 0o755)
    
    # Denormalize model name (replace underscores with hyphens)
    denormalized_name = model_name.replace("_", "-")
    
    # Build the command
    cmd = [
        sys.executable, 
        generator_path,
        "--model", denormalized_name,
        "--output-dir", str(PROJECT_ROOT / "generated_skillsets"),
        "--force",
        "--hardware", "all",
        "--cross-platform"
    ]
    
    # Run the generator
    try:
        logger.info(f"Regenerating skillset for {model_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully regenerated skillset for {model_name}")
            return True
        else:
            logger.error(f"Failed to regenerate skillset for {model_name}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error regenerating skillset for {model_name}: {e}")
        return False

def copy_regenerated_tests_to_skills():
    """Copy regenerated tests to the skills directory."""
    # Count number of files copied
    copied_count = 0
    
    # Copy all regenerated tests to the skills directory
    for test_file in OUTPUT_DIR.glob("test_hf_*.py"):
        dest_path = SKILLS_DIR / test_file.name
        try:
            # Make regenerated tests executable
            os.chmod(test_file, 0o755)
            
            # Copy the file
            shutil.copy(test_file, dest_path)
            copied_count += 1
            logger.info(f"Copied {test_file.name} to {dest_path}")
        except Exception as e:
            logger.error(f"Error copying {test_file.name} to {dest_path}: {e}")
    
    logger.info(f"Copied {copied_count} regenerated tests to {SKILLS_DIR}")

def main():
    """Main function."""
    logger.info("Starting test regeneration with enhanced hardware support")
    
    # Set up directories
    setup_directories()
    
    # Backup original files
    backup_dir = backup_original_files()
    
    # Find tests to regenerate
    tests_to_regenerate = find_failing_tests()
    
    # Track successful regenerations
    successful_tests = []
    successful_skillsets = []
    
    # Regenerate tests
    for model_name in tests_to_regenerate:
        test_success = regenerate_test(model_name)
        if test_success:
            successful_tests.append(model_name)
            
            # Also regenerate the skillset implementation
            skillset_success = regenerate_skillset(model_name)
            if skillset_success:
                successful_skillsets.append(model_name)
    
    # Copy regenerated tests to skills directory
    copy_regenerated_tests_to_skills()
    
    # Print summary
    logger.info("\nRegeneration Summary:")
    logger.info(f"Total tests to regenerate: {len(tests_to_regenerate)}")
    logger.info(f"Successfully regenerated tests: {len(successful_tests)}")
    logger.info(f"Successfully regenerated skillsets: {len(successful_skillsets)}")
    logger.info(f"Original files backed up to: {backup_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
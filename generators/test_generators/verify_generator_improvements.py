#!/usr/bin/env python3
"""
Verify Generator Improvements

This script verifies that the generator improvements have been successfully applied
by generating test files for key models on all hardware platforms. It ensures that
the generated tests have proper hardware support for all platforms.

Usage:
  python verify_generator_improvements.py [--model MODEL_NAME] [--all-models]
"""

import os
import sys
import logging
import argparse
import importlib
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Current directory
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Key models for verification
KEY_MODELS = [
    "bert", "t5", "llama", "vit", "clip", "detr", 
    "wav2vec2", "whisper", "clap", "llava", "llava_next", 
    "xclip", "qwen2"
]

# Hardware platforms to verify
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "openvino", "mps", "rocm", "qualcomm", "webnn", "webgpu"
]

def verify_model(model_name, output_dir=None):
    """Verify that a model can be generated with full hardware support."""
    if output_dir is None:
        # Use a temporary directory
        output_dir = tempfile.mkdtemp()
    
    try:
        # Import the fixed merged test generator
        sys.path.insert(0, str(CURRENT_DIR))
        from fixed_merged_test_generator import TestGenerator
        
        # Initialize the generator
        generator = TestGenerator()
        
        # Output file path
        output_file = os.path.join(output_dir, f"test_hf_{model_name}_all_hardware.py")
        
        # Generate a test file with all hardware platforms
        logger.info(f"Generating test for {model_name} with all hardware platforms...")
        generator.generate_test_file(
            model=model_name,
            output_path=output_file,
            platforms=",".join(HARDWARE_PLATFORMS)
        )
        
        # Verify that the file exists and contains proper hardware support
        if not os.path.exists(output_file):
            logger.error(f"Failed to generate test file for {model_name}")
            return False
        
        # Check the content of the file for hardware support
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Check for hardware detection
        hardware_detection_ok = "detect_hardware" in content or "HW_CAPABILITIES" in content
        if not hardware_detection_ok:
            logger.error(f"Missing hardware detection in test file for {model_name}")
            return False
        
        # Check for each hardware platform
        platforms_found = []
        for platform in HARDWARE_PLATFORMS:
            if f"init_{platform}" in content:
                platforms_found.append(platform)
        
        # Log the results
        if len(platforms_found) == len(HARDWARE_PLATFORMS):
            logger.info(f"✅ {model_name}: All {len(HARDWARE_PLATFORMS)} hardware platforms supported")
            return True
        else:
            missing = set(HARDWARE_PLATFORMS) - set(platforms_found)
            logger.warning(f"⚠️ {model_name}: {len(platforms_found)}/{len(HARDWARE_PLATFORMS)} platforms supported, missing: {missing}")
            return False
    
    except Exception as e:
        logger.error(f"Error verifying model {model_name}: {e}")
        return False
    finally:
        # Clean up if we used a temporary directory
        if output_dir != str(CURRENT_DIR):
            shutil.rmtree(output_dir)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify generator improvements")
    parser.add_argument("--model", type=str, help="Specific model to verify")
    parser.add_argument("--all-models", action="store_true", help="Verify all key models")
    parser.add_argument("--output-dir", type=str, help="Directory to output test files")
    args = parser.parse_args()
    
    output_dir = args.output_dir or str(CURRENT_DIR)
    
    # Check if a specific model was requested
    if args.model:
        verify_model(args.model, output_dir)
        return 0
    
    # Otherwise, check all key models
    if args.all_models:
        models_to_check = KEY_MODELS
    else:
        # Default to checking a subset of key models for quicker testing
        models_to_check = ["bert", "vit", "whisper"]
    
    # Log what we're checking
    logger.info(f"Verifying {len(models_to_check)} models with {len(HARDWARE_PLATFORMS)} hardware platforms...")
    logger.info(f"Models: {', '.join(models_to_check)}")
    logger.info(f"Platforms: {', '.join(HARDWARE_PLATFORMS)}")
    
    # Verify each model
    success_count = 0
    for model in models_to_check:
        success = verify_model(model, output_dir)
        if success:
            success_count += 1
    
    # Log the summary
    logger.info(f"Verification complete: {success_count}/{len(models_to_check)} models passed")
    
    return 0 if success_count == len(models_to_check) else 1

if __name__ == "__main__":
    sys.exit(main())
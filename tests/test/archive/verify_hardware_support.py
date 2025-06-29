#!/usr/bin/env python3
"""
Verify hardware support for key models
This script checks that the hardware_template_integration.py file has the correct
hardware support map for all key models, and that the fixed_merged_test_generator.py
file uses the same configuration.
"""

import os
import sys
import importlib.util
from pprint import pprint
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hardware_verification")

# Key models to check
KEY_MODELS = [
    "bert", "t5", "llama", "vit", "clip", "detr", "clap", 
    "wav2vec2", "whisper", "llava", "llava_next", "xclip", "qwen2"
]

# Hardware platforms to check
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "rocm", "mps", "openvino", 
    "qualcomm", "webnn", "webgpu"
]

def load_hardware_template_module():
    """Load the hardware_template_integration module."""
    try:
        # Get the absolute path to the hardware_template_integration.py file
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardware_template_integration.py")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("hardware_template_integration", file_path)
        hardware_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hardware_module)
        
        logger.info("Successfully loaded hardware_template_integration module")
        return hardware_module
    except Exception as e:
        logger.error(f"Error loading hardware_template_integration module: {e}")
        return None

def load_test_generator_module():
    """Load the fixed_merged_test_generator module."""
    try:
        # Get the absolute path to the fixed_merged_test_generator.py file
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixed_merged_test_generator.py")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("fixed_merged_test_generator", file_path)
        generator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator_module)
        
        logger.info("Successfully loaded fixed_merged_test_generator module")
        return generator_module
    except Exception as e:
        logger.error(f"Error loading fixed_merged_test_generator module: {e}")
        return None

def verify_hardware_maps():
    """Verify that the hardware maps are consistent between modules."""
    # Load modules
    hardware_module = load_hardware_template_module()
    generator_module = load_test_generator_module()
    
    if not hardware_module or not generator_module:
        logger.error("Failed to load required modules")
        return False
    
    # Get the hardware maps
    hardware_map = hardware_module.KEY_MODEL_HARDWARE_MAP
    generator_map = generator_module.KEY_MODEL_HARDWARE_CONFIG
    
    # Check that all key models are in both maps
    for model in KEY_MODELS:
        if model not in hardware_map:
            logger.error(f"Model {model} missing from hardware_template_integration.KEY_MODEL_HARDWARE_MAP")
            return False
        
        if model not in generator_map:
            logger.error(f"Model {model} missing from fixed_merged_test_generator.KEY_MODEL_HARDWARE_CONFIG")
            return False
    
    # Check that the hardware maps match for all key models
    discrepancies = []
    for model in KEY_MODELS:
        for platform in HARDWARE_PLATFORMS:
            hw_value = hardware_map[model].get(platform, "UNKNOWN")
            gen_value = generator_map[model].get(platform, "UNKNOWN")
            
            if hw_value != gen_value:
                discrepancies.append(f"Model {model}, Platform {platform}: hardware={hw_value}, generator={gen_value}")
    
    if discrepancies:
        logger.error("Found discrepancies between hardware maps:")
        for discrepancy in discrepancies:
            logger.error(f"  {discrepancy}")
        return False
    
    # Verify that all hardware maps have March 2025 updates (all REAL)
    non_real_entries = []
    for model in KEY_MODELS:
        for platform in HARDWARE_PLATFORMS:
            hw_value = hardware_map[model].get(platform, "UNKNOWN")
            if hw_value != "REAL":
                non_real_entries.append(f"Model {model}, Platform {platform}: {hw_value}")
    
    if non_real_entries:
        logger.warning("Found non-REAL entries in hardware maps (March 2025 should have all REAL):")
        for entry in non_real_entries:
            logger.warning(f"  {entry}")
    
    # Check model category detection
    try:
        categories = {}
        for model in KEY_MODELS:
            category = hardware_module.detect_model_modality(model)
            categories[model] = category
        
        logger.info("Model categories:")
        for model, category in categories.items():
            logger.info(f"  {model}: {category}")
    except Exception as e:
        logger.error(f"Error detecting model categories: {e}")
    
    logger.info("Hardware maps verification completed successfully")
    return True

def main():
    """Main function."""
    logger.info("Verifying hardware support for key models")
    
    success = verify_hardware_maps()
    
    if success:
        logger.info("All hardware maps are consistent and up to date")
        return 0
    else:
        logger.error("Hardware maps verification failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
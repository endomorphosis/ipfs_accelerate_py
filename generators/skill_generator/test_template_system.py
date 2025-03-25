#!/usr/bin/env python3
"""
Test script for the modular template system.

This script verifies that the template composer can generate model implementations
by testing with a simple BERT model.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the template system
from create_reference_implementations import (
    create_hardware_templates,
    create_architecture_templates,
    create_pipeline_templates,
    get_arch_type_for_model,
    get_model_type_from_autoconfig
)
from templates.template_composer import TemplateComposer


def setup_output_directory():
    """Set up the output directory for generated files."""
    output_dir = "generated_reference"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def test_template_composer():
    """Test the template composer with a simple BERT model."""
    # Set up output directory
    output_dir = setup_output_directory()
    
    # Create templates
    logger.info("Creating templates...")
    hardware_templates = create_hardware_templates()
    architecture_templates = create_architecture_templates()
    pipeline_templates = create_pipeline_templates()
    
    # Create template composer
    logger.info("Creating template composer...")
    composer = TemplateComposer(
        hardware_templates=hardware_templates,
        architecture_templates=architecture_templates,
        pipeline_templates=pipeline_templates,
        output_dir=output_dir
    )
    
    # Test with BERT model
    model_name = "bert-base-uncased"
    logger.info(f"Testing with model: {model_name}")
    
    # Try to detect model type with AutoConfig
    model_type = get_model_type_from_autoconfig(model_name)
    if model_type:
        logger.info(f"Detected model type: {model_type}")
    else:
        model_type = "bert"
        logger.info(f"Using default model type: {model_type}")
    
    # Determine architecture type
    arch_type = get_arch_type_for_model(model_name)
    logger.info(f"Architecture type: {arch_type}")
    
    # Generate implementation for CPU only (for testing)
    logger.info("Generating implementation for CPU only...")
    success, output_file = composer.generate_model_implementation(
        model_name=model_type,
        arch_type=arch_type,
        hardware_types=["cpu"],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated implementation at: {output_file}")
        logger.info(f"File size: {os.path.getsize(output_file)} bytes")
        
        # Check if the file exists and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info("✅ Test passed: File exists and has content")
        else:
            logger.error("❌ Test failed: File does not exist or is empty")
            return False
    else:
        logger.error(f"❌ Test failed: Could not generate implementation")
        return False
    
    # Additionally, try to generate a vision model implementation
    vision_model = "google/vit-base-patch16-224"
    logger.info(f"Testing with vision model: {vision_model}")
    
    # Try to detect model type with AutoConfig
    vision_model_type = get_model_type_from_autoconfig(vision_model)
    if vision_model_type:
        logger.info(f"Detected model type: {vision_model_type}")
    else:
        vision_model_type = "vit"
        logger.info(f"Using default model type: {vision_model_type}")
    
    # Determine architecture type
    vision_arch_type = get_arch_type_for_model(vision_model)
    logger.info(f"Architecture type: {vision_arch_type}")
    
    # Generate implementation for CPU only (for testing)
    logger.info("Generating implementation for vision model on CPU...")
    success, output_file = composer.generate_model_implementation(
        model_name=vision_model_type,
        arch_type=vision_arch_type,
        hardware_types=["cpu"],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated vision implementation at: {output_file}")
        logger.info(f"File size: {os.path.getsize(output_file)} bytes")
        
        # Check if the file exists and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info("✅ Test passed: Vision file exists and has content")
        else:
            logger.error("❌ Test failed: Vision file does not exist or is empty")
            return False
    else:
        logger.error(f"❌ Test failed: Could not generate vision implementation")
        return False
    
    logger.info("All tests passed!")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the modular template system")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the test
    if test_template_composer():
        logger.info("✅ Template system verified successfully")
        sys.exit(0)
    else:
        logger.error("❌ Template system verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
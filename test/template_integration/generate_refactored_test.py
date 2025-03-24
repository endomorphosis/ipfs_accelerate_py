#!/usr/bin/env python3
"""
Generate refactored test files for models using the template integration system.

This script combines the template system with the refactored test suite structure
to generate test files that follow the new standardized approach.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import template utilities
try:
    from fix_template_issues import customize_template, verify_test_file
except ImportError:
    logger.error("Could not import from fix_template_issues.py")
    sys.exit(1)

# Model architecture mapping
MODEL_ARCHITECTURE_MAPPING = {
    "bert": "encoder_only",
    "gpt": "decoder_only",
    "vit": "vision",
    "t5": "encoder_decoder",
    "clip": "multimodal",
    "wav2vec": "speech",
    "wav2vec2": "speech",
    "whisper": "speech",
    "hubert": "speech",
    "clap": "speech",
    "encodec": "speech",
    "data2vec_audio": "speech",
    "llama": "decoder_only",
    "opt": "decoder_only",
    "sam": "vision",
    "blip": "multimodal",
    "flava": "multimodal",
    # Add more mappings as needed
}

# Base class mapping
BASE_CLASS_MAPPING = {
    "encoder_only": "ModelTest",
    "decoder_only": "ModelTest",
    "vision": "ModelTest",
    "vision_text": "ModelTest",
    "speech": "ModelTest",
    "text": "ModelTest",
    "encoder_decoder": "ModelTest",
    "multimodal": "ModelTest",
}

def determine_architecture(model_name):
    """Determine the model architecture based on the model name."""
    model_name_lower = model_name.lower()
    
    for key, arch in MODEL_ARCHITECTURE_MAPPING.items():
        if key in model_name_lower:
            return arch
    
    # Default to encoder_only if unknown
    logger.warning(f"Could not determine architecture for {model_name}, using encoder_only")
    return "encoder_only"

def sanitize_model_name(model_name):
    """Sanitize model name to avoid syntax errors with hyphens, etc."""
    # Get the model name without organization prefix
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    # Replace hyphens with underscores for class names
    sanitized = model_name.replace("-", "_")
    
    return sanitized

def get_template_path(architecture, refactored=True):
    """Get the path to the template file for the given architecture."""
    # Directory containing templates
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    
    # Use refactored templates if requested
    if refactored:
        template_name = f"refactored_{architecture}_template.py"
    else:
        template_name = f"{architecture}_template.py"
    
    template_path = os.path.join(template_dir, template_name)
    
    # If the template doesn't exist, try to find it in the skills template directory
    if not os.path.exists(template_path):
        skills_template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          "skills", "templates")
        skills_template_path = os.path.join(skills_template_dir, template_name)
        
        # If refactored version doesn't exist, fall back to standard template
        if not os.path.exists(skills_template_path) and refactored:
            logger.warning(f"Refactored template {template_name} not found, falling back to standard template")
            return get_template_path(architecture, refactored=False)
        
        template_path = skills_template_path
    
    if not os.path.exists(template_path):
        logger.error(f"Template file {template_path} not found")
        raise FileNotFoundError(f"Template file for {architecture} not found")
    
    return template_path

def generate_output_path(model_name, architecture, refactored=True):
    """Generate the output path for the test file."""
    # Strip any organization prefix (e.g., "google/")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    # Generate test filename
    test_filename = f"test_{model_name}.py"
    
    if refactored:
        # For refactored tests, use the refactored directory structure
        # Determine the appropriate subdirectory based on architecture
        if architecture in ["vision", "vit"]:
            subdir = "models/vision"
        elif architecture in ["vision_text", "multimodal"]:
            subdir = "models/multimodal"
        elif architecture in ["speech", "audio", "whisper", "wav2vec"]:
            subdir = "models/audio"
        else:
            subdir = "models/text"
        
        # Create the full path
        refactored_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "refactored_test_suite")
        output_path = os.path.join(refactored_dir, subdir, test_filename)
    else:
        # For standard tests, use the normal output directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "skills", "fixed_tests")
        output_path = os.path.join(output_dir, test_filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return output_path

def generate_test_file(model_name, architecture=None, refactored=True, output_path=None):
    """Generate a test file for the given model."""
    # Determine architecture if not provided
    if architecture is None:
        architecture = determine_architecture(model_name)
    
    logger.info(f"Generating {'refactored' if refactored else 'standard'} test file for {model_name} with architecture {architecture}")
    
    # Get template path
    template_path = get_template_path(architecture, refactored)
    logger.info(f"Using template: {template_path}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = generate_output_path(model_name, architecture, refactored)
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get sanitized model name for class names to avoid syntax errors
    sanitized_model_name = sanitize_model_name(model_name)
    
    # Customize the template
    model_params = {
        "model_name": model_name,
        "sanitized_model_name": sanitized_model_name,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "architecture": architecture,
    }
    
    # Specify base class if using refactored templates
    if refactored:
        base_class = BASE_CLASS_MAPPING.get(architecture, "ModelTest")
        model_params["base_class"] = base_class
    
    # Generate the test file
    customize_template(template_path, output_path, model_params)
    logger.info(f"Generated test file: {output_path}")
    
    # Verify the test file
    verify_result = verify_test_file(output_path)
    if verify_result["valid"]:
        logger.info("Test file validation successful")
    else:
        logger.error(f"Test file validation failed: {verify_result['error']}")
    
    return output_path, verify_result["valid"]

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate test files for models")
    parser.add_argument("--model", type=str, required=True, help="Model name/ID to generate test for")
    parser.add_argument("--architecture", type=str, help="Model architecture (encoder_only, decoder_only, vision, etc.)")
    parser.add_argument("--output", type=str, help="Output path for the generated test file")
    parser.add_argument("--no-refactor", action="store_true", help="Generate standard test file instead of refactored")
    
    args = parser.parse_args()
    
    # Generate the test file
    output_path, success = generate_test_file(
        args.model,
        architecture=args.architecture,
        refactored=not args.no_refactor,
        output_path=args.output
    )
    
    # Print result
    if success:
        print(f"✅ Successfully generated test file: {output_path}")
    else:
        print(f"❌ Failed to generate valid test file: {output_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
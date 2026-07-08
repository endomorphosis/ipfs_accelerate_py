#!/usr/bin/env python3
"""
Generate a multimodal test file directly without using the template system.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(script_dir))

def sanitize_model_name(model_name):
    """Convert model name to a Python class name."""
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    # Replace special characters
    sanitized = model_name.replace("-", "_").replace(".", "_")
    
    # Convert to CamelCase
    parts = sanitized.split("_")
    sanitized = "".join(p.capitalize() for p in parts)
    
    return sanitized

def generate_multimodal_test(model_id, output_dir=None):
    """Generate test file for a multimodal model."""
    # Get sanitized model name
    sanitized_name = sanitize_model_name(model_id)
    
    # Get model type (clip, blip, etc.)
    model_type = "clip"
    if "blip" in model_id.lower():
        model_type = "blip"
    elif "flava" in model_id.lower():
        model_type = "flava"
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(script_dir), "refactored_test_suite", "models", "multimodal")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"test_{model_id.split('/')[-1].replace('-', '_')}.py")
    
    # Check if file exists
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return False
    
    # Find existing test files to use as reference
    reference_file = None
    if model_type == "clip" and os.path.exists(os.path.join(output_dir, "test_clip_vit_base_patch32.py")):
        reference_file = os.path.join(output_dir, "test_clip_vit_base_patch32.py")
    elif model_type == "blip" and os.path.exists(os.path.join(output_dir, "test_blip_image_captioning_base.py")):
        reference_file = os.path.join(output_dir, "test_blip_image_captioning_base.py")
    # For FLAVA, use CLIP as the reference
    elif model_type == "flava" and os.path.exists(os.path.join(output_dir, "test_clip_vit_base_patch32.py")):
        reference_file = os.path.join(output_dir, "test_clip_vit_base_patch32.py")
    
    if reference_file:
        # Use reference file as template
        with open(reference_file, 'r') as f:
            content = f.read()
        
        # Replace model specifics
        content = content.replace("openai/clip-vit-base-patch32", model_id)
        content = content.replace("Salesforce/blip-image-captioning-base", model_id)
        
        # Replace class name
        content = content.replace("TestClipVitBasePatch32", f"Test{sanitized_name}")
        content = content.replace("TestBlipImageCaptioningBase", f"Test{sanitized_name}")
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"Generated test file for {model_id} at {output_file}")
        return True
    else:
        print(f"No reference file found for model type {model_type}")
        return False

def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Generate multimodal test files")
    parser.add_argument("--model", type=str, required=True, help="Model ID to generate test for")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Generate test file
    success = generate_multimodal_test(args.model, args.output_dir)
    
    if success:
        print(f"Successfully generated test file for {args.model}")
        return 0
    else:
        print(f"Failed to generate test file for {args.model}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
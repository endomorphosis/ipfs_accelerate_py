#!/usr/bin/env python3
"""
Generate and run modality-specific tests to verify the template system
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from merged_test_generator
from merged_test_generator import detect_model_modality, generate_modality_specific_template

# Define sample models for each modality with known good implementations
SAMPLE_MODELS = {
    "text": ["bert", "t5", "gpt2", "roberta", "distilbert"],
    "vision": ["vit", "resnet", "convnext", "swin", "deit"],
    "audio": ["whisper", "wav2vec2", "hubert", "speecht5"],
    "multimodal": ["clip", "blip", "vilt", "flava", "git"]
}

def generate_test_for_model(model_name, output_dir):
    """Generate a test file for a specific model."""
    modality = detect_model_modality(model_name)
    print(f"Generating test for {model_name} (modality: {modality})")
    
    # Generate template
    template = generate_modality_specific_template(model_name, modality)
    
    # Replace placeholders
    template = template.replace("MODEL_PLACEHOLDER", f"{model_name}-base")
    
    # Task placeholder - based on modality
    if modality == "text":
        task = "fill-mask"
        input_example = "The quick brown fox jumps over the lazy dog."
    elif modality == "vision":
        task = "image-classification"
        input_example = "test.jpg"
    elif modality == "audio":
        task = "automatic-speech-recognition"
        input_example = "test.mp3"
    elif modality == "multimodal":
        task = "image-to-text"
        # Fix the template directly to avoid string escape issues
        template = template.replace(
            'result = pipeline("MODEL_INPUT_PLACEHOLDER")',
            'result = pipeline({"image": "test.jpg", "text": "What is in this image?"})'
        )
        input_example = 'test.jpg'  # Simplified for other replacements
    else:
        task = "feature-extraction"
        input_example = "Example input"
    
    template = template.replace("MODEL_TASK_PLACEHOLDER", task)
    template = template.replace("MODEL_INPUT_PLACEHOLDER", input_example)
    
    # Handle special case for class name template
    class_name = ''.join(word.capitalize() for word in model_name.replace('-', '_').replace('.', '_').split('_'))
    template = template.replace(f"TestHF{{class_name}}", f"TestHF{class_name}")
    
    # Write output file
    os.makedirs(output_dir, exist_ok=True)
    test_file = os.path.join(output_dir, f"test_hf_{model_name}.py")
    
    with open(test_file, 'w') as f:
        f.write(template)
    
    # Make executable
    os.chmod(test_file, 0o755)
    print(f"Created test file: {test_file}")
    
    return test_file

def verify_test(test_file):
    """Verify that a generated test file is syntactically correct."""
    try:
        # Use Python to check syntax
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", test_file],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✅ Syntax check passed for {os.path.basename(test_file)}")
            return True
        else:
            print(f"❌ Syntax error in {os.path.basename(test_file)}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error verifying test file: {e}")
        return False

def run_sanity_check(test_file):
    """Run a basic sanity check on the test file."""
    try:
        # Just try to import the file to check basic functionality
        module_name = os.path.basename(test_file).replace('.py', '')
        test_dir = os.path.dirname(test_file)
        
        # Add the test directory to Python path
        sys.path.insert(0, test_dir)
        
        # Try importing the module
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}; print('Successfully imported {module_name}')"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": f"{test_dir}:{os.environ.get('PYTHONPATH', '')}"}
        )
        
        if result.returncode == 0:
            print(f"✅ Import check passed for {module_name}")
            return True
        else:
            print(f"❌ Import error in {module_name}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error running sanity check: {e}")
        return False

def generate_tests(modality, output_dir, verify=True):
    """Generate tests for a specific modality."""
    if modality not in SAMPLE_MODELS:
        print(f"Unknown modality: {modality}. Available modalities: {', '.join(SAMPLE_MODELS.keys())}")
        return []
    
    print(f"\n=== Generating {modality} model tests ===\n")
    generated_files = []
    
    for model in SAMPLE_MODELS[modality]:
        try:
            test_file = generate_test_for_model(model, output_dir)
            generated_files.append(test_file)
            
            if verify:
                verify_test(test_file)
                run_sanity_check(test_file)
        except Exception as e:
            print(f"Error generating test for {model}: {e}")
    
    return generated_files

def main():
    parser = argparse.ArgumentParser(description="Generate and verify modality-specific tests")
    parser.add_argument("--modality", choices=list(SAMPLE_MODELS.keys()) + ["all"], default="all", 
                        help="Which modality to generate tests for")
    parser.add_argument("--output-dir", default="modality_tests", help="Output directory for test files")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification of generated files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.modality == "all":
        # Generate tests for all modalities
        all_files = []
        for modality in SAMPLE_MODELS.keys():
            files = generate_tests(modality, args.output_dir, not args.no_verify)
            all_files.extend(files)
            
        print(f"\nGenerated {len(all_files)} test files across all modalities")
    else:
        # Generate tests for specific modality
        files = generate_tests(args.modality, args.output_dir, not args.no_verify)
        print(f"\nGenerated {len(files)} test files for {args.modality} modality")
    
    print("\nTest generation complete!")

if __name__ == "__main__":
    main()
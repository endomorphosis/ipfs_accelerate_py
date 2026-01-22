#!/usr/bin/env python3
"""
Generate sample modality-specific tests for evaluation
"""

import os
import sys
import argparse
from generators.test_generators.merged_test_generator import generate_modality_specific_template, detect_model_modality

# Define sample models for each modality
SAMPLE_MODELS = {
    "text": ["bert", "t5", "gpt2"],
    "vision": ["vit", "detr", "swin"],
    "audio": ["whisper", "wav2vec2", "clap"],
    "multimodal": ["clip", "llava", "blip"]
}

def generate_test_for_model(model_name, output_dir):
    """Generate a test file for a specific model."""
    modality = detect_model_modality(model_name)
    print(f"Generating test for {model_name} (modality: {modality})")
    
    # Generate template
    template = generate_modality_specific_template(model_name, modality)
    
    # Replace placeholders
    template = template.replace("MODEL_PLACEHOLDER", f"{model_name}-base")
    
    # Task placeholder
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
        input_example = '{"image": "test.jpg", "text": "What is in this image?"}'
    else:
        task = "feature-extraction"
        input_example = "Example input"
    
    template = template.replace("MODEL_TASK_PLACEHOLDER", task)
    template = template.replace("MODEL_INPUT_PLACEHOLDER", input_example)
    
    # Write output file
    os.makedirs(output_dir, exist_ok=True)
    test_file = os.path.join(output_dir, f"test_hf_{model_name}.py")
    
    with open(test_file, 'w') as f:
        f.write(template)
    
    # Make executable
    os.chmod(test_file, 0o755)
    print(f"Created test file: {test_file}")
    
    return test_file

def main():
    parser = argparse.ArgumentParser(description="Generate sample tests for different modalities")
    parser.add_argument("--output-dir", default="new_generated_tests", help="Output directory for test files")
    parser.add_argument("--modality", choices=list(SAMPLE_MODELS.keys()) + ["all"], default="all", 
                        help="Which modality to generate tests for")
    args = parser.parse_args()
    
    if args.modality == "all":
        # Generate tests for all modalities
        for modality, models in SAMPLE_MODELS.items():
            print(f"\nGenerating {modality} model tests:")
            for model in models:
                generate_test_for_model(model, args.output_dir)
    else:
        # Generate tests for specific modality
        print(f"\nGenerating {args.modality} model tests:")
        for model in SAMPLE_MODELS[args.modality]:
            generate_test_for_model(model, args.output_dir)
    
    print("\nAll test files generated successfully!")

if __name__ == "__main__":
    main()
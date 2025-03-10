#!/usr/bin/env python3
import os
import sys
import argparse
from generators.test_generators.merged_test_generator import generate_modality_specific_template, detect_model_modality

def generate_test_template(model_type):
    """Generate a test template for a model type."""
    modality = detect_model_modality(model_type)
    print(f"Detected modality for {model_type}: {modality}")
    
    # Use the modality-specific template generator
    template = generate_modality_specific_template(model_type, modality)
    
    # Replace placeholder values with model-specific ones
    template = template.replace("MODEL_PLACEHOLDER", f"{model_type}-base")
    template = template.replace("MODEL_TASK_PLACEHOLDER", "fill-mask" if modality == "text" else 
                               "image-classification" if modality == "vision" else
                               "automatic-speech-recognition" if modality == "audio" else
                               "image-to-text" if modality == "multimodal" else
                               "feature-extraction")
    
    # Replace input placeholder with modality-specific example
    if modality == "text":
        template = template.replace("MODEL_INPUT_PLACEHOLDER", "The quick brown fox jumps over the lazy dog.")
    elif modality == "vision":
        template = template.replace("MODEL_INPUT_PLACEHOLDER", "test.jpg")
    elif modality == "audio":
        template = template.replace("MODEL_INPUT_PLACEHOLDER", "test.mp3")
    elif modality == "multimodal":
        template = template.replace("MODEL_INPUT_PLACEHOLDER", '{"image": "test.jpg", "text": "What is in this image?"}')
    else:
        template = template.replace("MODEL_INPUT_PLACEHOLDER", "Example input")
    
    return template

def main():
    parser = argparse.ArgumentParser(description="Generate test template for a model type")
    parser.add_argument("model_type", help="Model type (e.g., bert, gpt2)")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--modality", "-m", help="Override detected modality", 
                        choices=["text", "vision", "audio", "multimodal", "specialized"])
    args = parser.parse_args()
    
    if args.modality:
        # Use specified modality
        template = generate_modality_specific_template(args.model_type, args.modality)
    else:
        # Auto-detect modality
        template = generate_test_template(args.model_type)
    
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(template)
        print(f"Template written to {args.output}")
        
        # Make the file executable
        os.chmod(args.output, 0o755)
    else:
        print(template)

if __name__ == "__main__":
    main()
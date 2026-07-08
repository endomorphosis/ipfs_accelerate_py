#!/usr/bin/env python3
"""
Generate Missing Models

This script generates test files for the missing high-priority models from the HF_MODEL_COVERAGE_ROADMAP.md.
"""

import os
import sys
import logging
import argparse
import datetime
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from generator_core.generator import GeneratorCore
from generator_core.config import ConfigManager
from model_selection.registry import ModelRegistry
from templates import get_template_for_architecture

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# High-priority models from the roadmap
HIGH_PRIORITY_MODELS = [
    # Decoder-only models
    {"name": "codellama/CodeLlama-7b-hf", "type": "codellama", "architecture": "decoder-only"},
    {"name": "Qwen/Qwen2-7B", "type": "qwen2", "architecture": "decoder-only"},
    {"name": "Qwen/Qwen3-7B-Q5_K_M", "type": "qwen3", "architecture": "decoder-only"},
    {"name": "EleutherAI/gpt-neo-1.3B", "type": "gpt-neo", "architecture": "decoder-only"},
    {"name": "EleutherAI/gpt-neox-20b", "type": "gpt-neox", "architecture": "decoder-only"},
    
    # Encoder-decoder models
    {"name": "google/long-t5-tglobal-base", "type": "longt5", "architecture": "encoder-decoder"},
    {"name": "google/pegasus-x-base", "type": "pegasus-x", "architecture": "encoder-decoder"},
    
    # Encoder-only models
    {"name": "studio-ousia/luke-base", "type": "luke", "architecture": "encoder-only"},
    {"name": "microsoft/mpnet-base", "type": "mpnet", "architecture": "encoder-only"},
    
    # Multimodal models
    {"name": "adept/fuyu-8b", "type": "fuyu", "architecture": "vision-text"},
    {"name": "microsoft/kosmos-2-patch14-224", "type": "kosmos-2", "architecture": "vision-text"},
    {"name": "llava-hf/llava-1.5-7b-hf", "type": "llava-next", "architecture": "vision-text"},
    {"name": "LanguageBind/Video-LLaVA-7B", "type": "video-llava", "architecture": "vision-text"},
    
    # Speech models
    {"name": "suno/bark-small", "type": "bark", "architecture": "speech"},
    
    # Vision models
    {"name": "google/mobilenet_v2_1.0_224", "type": "mobilenet-v2", "architecture": "vision"},
    
    # Vision-text models
    {"name": "Salesforce/blip2-opt-2.7b", "type": "blip-2", "architecture": "vision-text"},
    {"name": "OFA-Sys/chinese-clip-vit-base-patch16", "type": "chinese-clip", "architecture": "vision-text"},
    {"name": "CIDAS/clipseg-rd64-refined", "type": "clipseg", "architecture": "vision-text"},
]

def create_output_directory(output_dir):
    """
    Create the output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def initialize_generator():
    """
    Initialize the generator.
    
    Returns:
        GeneratorCore instance
    """
    config = ConfigManager()
    registry = ModelRegistry(config)
    generator = GeneratorCore(config=config)
    return generator

def generate_model_test(generator, model_info, output_dir):
    """
    Generate a test file for a model.
    
    Args:
        generator: GeneratorCore instance
        model_info: Model information dictionary
        output_dir: Output directory path
        
    Returns:
        Path to the generated file
    """
    template_class = get_template_for_architecture(model_info["architecture"])
    
    if not template_class:
        logger.error(f"No template found for architecture: {model_info['architecture']}")
        return None
    
    template = template_class()
    
    # Prepare model info
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context = {
        "model_info": model_info,
        "timestamp": timestamp,
        "has_openvino": model_info.get("has_openvino", False)
    }
    
    # Generate the file
    file_content = template.render(context)
    
    # Create filename
    model_type = model_info["type"]
    filename = f"test_hf_{model_type}.py"
    file_path = os.path.join(output_dir, filename)
    
    # Save the file
    with open(file_path, "w") as f:
        f.write(file_content)
        
    logger.info(f"Generated test file for {model_info['name']} at {file_path}")
    return file_path

def generate_missing_models(output_dir, selected_models=None):
    """
    Generate test files for all missing high-priority models.
    
    Args:
        output_dir: Output directory path
        selected_models: List of model types to generate (if None, generate all)
        
    Returns:
        List of generated file paths
    """
    # Create output directory
    output_path = create_output_directory(output_dir)
    
    # Initialize generator
    generator = initialize_generator()
    
    # Generate files
    generated_files = []
    
    for model_info in HIGH_PRIORITY_MODELS:
        # Skip if not in selected models (if specified)
        if selected_models and model_info["type"] not in selected_models:
            continue
            
        # Generate file
        file_path = generate_model_test(generator, model_info, output_dir)
        
        if file_path:
            generated_files.append(file_path)
    
    return generated_files

def main():
    parser = argparse.ArgumentParser(description="Generate missing HuggingFace model tests")
    parser.add_argument("--output-dir", default="generated_tests", help="Directory to output generated files")
    parser.add_argument("--models", nargs="*", help="Specific model types to generate (if not specified, generate all)")
    
    args = parser.parse_args()
    
    logger.info(f"Generating missing model tests in {args.output_dir}")
    if args.models:
        logger.info(f"Generating only models: {', '.join(args.models)}")
    
    generated_files = generate_missing_models(args.output_dir, args.models)
    
    logger.info(f"Generated {len(generated_files)} test files:")
    for file_path in generated_files:
        logger.info(f"  - {file_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
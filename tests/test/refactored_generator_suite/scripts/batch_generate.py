#!/usr/bin/env python3
"""
Batch Generate Missing Models

This script generates test files for all missing models in batches.
"""

import os
import sys
import logging
import argparse
import datetime
import json
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

def load_models_from_report(report_path):
    """
    Load model information from a coverage report.
    
    Args:
        report_path: Path to coverage report JSON file
        
    Returns:
        Dictionary with model lists by priority
    """
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
            
        # Extract missing models by priority
        high_priority = [model for model in data.get("high_priority_models", []) if not model.get("implemented", False)]
        medium_priority = [model for model in data.get("medium_priority_models", []) if not model.get("implemented", False)]
        
        # Also get models by architecture from missing_by_architecture
        by_architecture = data.get("missing_by_architecture", {})
        
        return {
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "by_architecture": by_architecture
        }
    except Exception as e:
        logger.error(f"Error loading models from report: {e}")
        return {
            "high_priority": [],
            "medium_priority": [],
            "by_architecture": {}
        }

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
    architecture = model_info.get("architecture")
    if not architecture:
        logger.error(f"No architecture specified for model: {model_info}")
        return None
        
    template_class = get_template_for_architecture(architecture)
    
    if not template_class:
        logger.error(f"No template found for architecture: {architecture}")
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

def batch_generate_models(models, output_dir, batch_size=10, start_index=0, priority="high"):
    """
    Generate test files for models in batches.
    
    Args:
        models: List of model information dictionaries
        output_dir: Output directory path
        batch_size: Number of models to generate in this batch
        start_index: Index to start from
        priority: Priority level for logging
        
    Returns:
        Tuple of (list of generated file paths, next start index)
    """
    # Create output directory
    output_path = create_output_directory(output_dir)
    
    # Initialize generator
    generator = initialize_generator()
    
    # Generate files
    generated_files = []
    
    end_index = min(start_index + batch_size, len(models))
    batch_models = models[start_index:end_index]
    
    logger.info(f"Generating batch of {len(batch_models)} {priority} priority models ({start_index+1}-{end_index} of {len(models)})")
    
    for model_info in batch_models:
        # Generate file
        file_path = generate_model_test(generator, model_info, output_dir)
        
        if file_path:
            generated_files.append(file_path)
    
    return generated_files, end_index

def main():
    parser = argparse.ArgumentParser(description="Batch generate missing HuggingFace model tests")
    parser.add_argument("--report", required=True, help="Path to coverage report JSON file")
    parser.add_argument("--output-dir", default="generated_tests", help="Directory to output generated files")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of models to generate in this batch")
    parser.add_argument("--start-index", type=int, default=0, help="Index to start from")
    parser.add_argument("--architecture", help="Generate models for a specific architecture only")
    parser.add_argument("--priority", choices=["high", "medium"], default="high", help="Priority level to generate")
    
    args = parser.parse_args()
    
    # Load models from report
    models_data = load_models_from_report(args.report)
    
    if args.architecture:
        # Generate models for a specific architecture
        architecture_models = models_data["by_architecture"].get(args.architecture, [])
        logger.info(f"Generating models for architecture: {args.architecture} ({len(architecture_models)} models)")
        
        if not architecture_models:
            logger.error(f"No missing models found for architecture: {args.architecture}")
            return 1
            
        generated_files, next_index = batch_generate_models(
            architecture_models, 
            args.output_dir, 
            args.batch_size, 
            args.start_index,
            "architecture-specific"
        )
    else:
        # Generate models by priority
        priority_models = models_data.get(f"{args.priority}_priority", [])
        logger.info(f"Generating {args.priority} priority models ({len(priority_models)} models)")
        
        if not priority_models:
            logger.error(f"No missing models found for {args.priority} priority")
            return 1
            
        generated_files, next_index = batch_generate_models(
            priority_models, 
            args.output_dir, 
            args.batch_size, 
            args.start_index,
            args.priority
        )
    
    logger.info(f"Generated {len(generated_files)} test files:")
    for file_path in generated_files:
        logger.info(f"  - {file_path}")
    
    # Print next command to run
    if next_index < (len(models_data["by_architecture"].get(args.architecture, [])) if args.architecture 
                    else len(models_data.get(f"{args.priority}_priority", []))):
        logger.info("\nTo generate the next batch, run:")
        if args.architecture:
            logger.info(f"python {sys.argv[0]} --report {args.report} --output-dir {args.output_dir} --batch-size {args.batch_size} --start-index {next_index} --architecture {args.architecture}")
        else:
            logger.info(f"python {sys.argv[0]} --report {args.report} --output-dir {args.output_dir} --batch-size {args.batch_size} --start-index {next_index} --priority {args.priority}")
    else:
        logger.info(f"\nAll {args.priority} priority models have been generated!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Batch generate test files for multiple models using the template integration system.

This script allows for batch generation of test files for multiple models
of different architectures, using the refactored templates.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

# Configure logging
log_filename = f"batch_generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import template utilities
try:
    from template_integration.template_integration_workflow import generate_test_file
    from template_integration.generate_refactored_test import (
        determine_architecture, MODEL_ARCHITECTURE_MAPPING
    )
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    sys.exit(1)

# Model groups by architecture
MODEL_GROUPS = {
    "vision": [
        "google/vit-base-patch16-224",
        "facebook/deit-base-patch16-224",
        "microsoft/beit-base-patch16-224",
        "facebook/convnext-base-224-22k",
        "facebook/dinov2-base",
    ],
    "encoder_only": [
        "bert-base-uncased",
        "roberta-base",
        "google/electra-base-discriminator",
        "xlm-roberta-base",
        "google/fnet-base",
    ],
    "decoder_only": [
        "gpt2",
        "facebook/opt-350m",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-j-6b",
        "meta-llama/Llama-2-7b-hf",
    ],
    "encoder_decoder": [
        "t5-base",
        "facebook/bart-base",
        "google/flan-t5-base",
        "google/mt5-base",
        "facebook/mbart-large-50",
    ],
    "speech": [
        "openai/whisper-tiny",
        "facebook/wav2vec2-base-960h",
        "facebook/hubert-base-ls960",
        "facebook/data2vec-audio-base-960h",
        "laion/clap-htsat-unfused",
    ],
    "multimodal": [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-vqa-base",
        "facebook/flava-full",
    ],
}

def generate_file_list(model_list: List[str], output_dir: str) -> Dict[str, str]:
    """Generate a mapping of model IDs to their output file paths."""
    file_mapping = {}
    
    for model_id in model_list:
        # Determine architecture
        architecture = determine_architecture(model_id)
        
        # Determine output path
        if "clip" in model_id.lower() or "blip" in model_id.lower() or "flava" in model_id.lower():
            # Special case for multimodal models
            subdir = "models/multimodal"
        elif architecture in ["vision", "vit"]:
            subdir = "models/vision"
        elif architecture in ["vision_text", "multimodal"]:
            subdir = "models/multimodal"
        elif architecture in ["speech", "audio", "whisper", "wav2vec"]:
            subdir = "models/audio"
        else:
            subdir = "models/text"
        
        # Generate output file name
        if "/" in model_id:
            model_name = model_id.split("/")[-1]
        else:
            model_name = model_id
        
        # Replace hyphens with underscores for Python file naming
        model_name = model_name.replace("-", "_")
        
        # Create full output path
        output_path = os.path.join(output_dir, subdir, f"test_{model_name}.py")
        
        # Add to mapping
        file_mapping[model_id] = output_path
    
    return file_mapping

def batch_generate(
    models: Optional[List[str]] = None,
    architectures: Optional[List[str]] = None,
    output_dir: str = None,
    skip_existing: bool = True,
    dry_run: bool = False
) -> Dict[str, bool]:
    """
    Generate test files for multiple models.
    
    Args:
        models: List of specific model IDs to generate tests for.
        architectures: List of architectures to generate tests for.
        output_dir: Output directory for the test files.
        skip_existing: Skip generation if the file already exists.
        dry_run: Just print what would be done, don't actually generate files.
        
    Returns:
        Dictionary of model IDs to generation success status.
    """
    results = {}
    
    # Default to the refactored test suite directory if not specified
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "refactored_test_suite")
    
    # Determine models to generate
    models_to_generate = []
    
    if models:
        # Add specified models
        models_to_generate.extend(models)
    
    if architectures:
        # Add models from specified architectures
        for arch in architectures:
            if arch in MODEL_GROUPS:
                models_to_generate.extend(MODEL_GROUPS[arch])
    
    if not models and not architectures:
        # If neither specified, use all models
        for models_list in MODEL_GROUPS.values():
            models_to_generate.extend(models_list)
    
    # Remove duplicates
    models_to_generate = list(set(models_to_generate))
    
    # Generate file mapping
    file_mapping = generate_file_list(models_to_generate, output_dir)
    
    if dry_run:
        logger.info("Dry run mode - showing what would be generated:")
        for model_id, output_path in file_mapping.items():
            logger.info(f"Would generate: {model_id} -> {output_path}")
        return {model_id: True for model_id in models_to_generate}
    
    # Process each model
    for model_id in models_to_generate:
        output_path = file_mapping[model_id]
        
        # Skip if file exists and skip_existing is True
        if skip_existing and os.path.exists(output_path):
            logger.info(f"Skipping existing file: {output_path}")
            results[model_id] = True
            continue
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get architecture
        architecture = determine_architecture(model_id)
        
        # Generate the test file
        logger.info(f"Generating test file for {model_id} with architecture {architecture}")
        
        try:
            success = generate_test_file(model_id, architecture, debug=False)
            results[model_id] = success
            
            if success:
                logger.info(f"Successfully generated test file: {output_path}")
            else:
                logger.error(f"Failed to generate test file for {model_id}")
                
        except Exception as e:
            logger.error(f"Error generating test for {model_id}: {e}")
            results[model_id] = False
    
    # Summarize results
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    logger.info(f"Generation complete: {successful}/{total} successful")
    
    return results

def save_results(results: Dict[str, bool], output_file: str = None):
    """Save batch generation results to a file."""
    if output_file is None:
        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Add timestamp
    results_with_meta = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "successful": sum(1 for success in results.values() if success),
            "failed": sum(1 for success in results.values() if not success),
        },
        "results": results
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return output_file

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Batch generate test files for multiple models")
    
    # Model selection
    model_selection = parser.add_argument_group("Model Selection")
    model_selection.add_argument("--models", type=str, nargs="+", help="List of specific model IDs")
    model_selection.add_argument("--architectures", type=str, nargs="+", 
                               choices=list(MODEL_GROUPS.keys()),
                               help="Architectures to generate tests for")
    
    # Output options
    output_options = parser.add_argument_group("Output Options")
    output_options.add_argument("--output-dir", type=str, help="Output directory for test files")
    output_options.add_argument("--results-file", type=str, help="File to save results to")
    output_options.add_argument("--no-skip-existing", action="store_true", 
                               help="Regenerate files even if they already exist")
    
    # Other options
    parser.add_argument("--dry-run", action="store_true", 
                       help="Don't generate files, just show what would be done")
    parser.add_argument("--list-models", action="store_true", 
                       help="List available models by architecture")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\nAvailable models by architecture:")
        for arch, models in MODEL_GROUPS.items():
            print(f"\n{arch.upper()}:")
            for model in models:
                print(f"  - {model}")
        return 0
    
    # Batch generate files
    results = batch_generate(
        models=args.models,
        architectures=args.architectures,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run
    )
    
    # Save results if not in dry run mode
    if not args.dry_run:
        save_results(results, args.results_file)
    
    # Exit with success if all generations succeeded
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
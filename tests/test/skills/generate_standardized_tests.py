#!/usr/bin/env python3
"""
Generate standardized test files for all supported model architectures.

This script uses the create_standardized_test.py script to generate test files
for all model architectures defined in the MODEL_INFO dictionary.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model architectures to generate tests for
MODEL_ARCHITECTURES = [
    'bert',
    'gpt2',
    't5',
    'vit',
    'gpt-j',
    'roberta',
    'albert',
    'distilbert',
    'bart',
    'electra',
    'deit',
    'wav2vec2',
    'clip',
    'blip',
    'llama',
    'mistral',
    'falcon',
    'gemma'
]

def generate_tests(output_dir, models=None, overwrite=False):
    """
    Generate standardized test files for specified models.
    
    Args:
        output_dir: Directory to store generated test files
        models: List of model architectures to generate tests for (default: all)
        overwrite: Whether to overwrite existing files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use all models if none specified
    if models is None:
        models = MODEL_ARCHITECTURES
    
    # Track results
    results = {
        'success': [],
        'skipped': [],
        'failed': []
    }
    
    # Generate test files for each model
    for model in models:
        # Determine output path
        model_valid_id = model.replace('-', '_')
        output_path = os.path.join(output_dir, f"test_hf_{model_valid_id}_standardized.py")
        
        # Check if file already exists and skip if not overwriting
        if os.path.exists(output_path) and not overwrite:
            logger.info(f"Skipping {model} (file already exists)")
            results['skipped'].append(model)
            continue
        
        # Run the create_standardized_test.py script
        logger.info(f"Generating test file for {model}...")
        try:
            result = subprocess.run(
                ['python', 'create_standardized_test.py', model, output_path],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Check if generation was successful
            if "Successfully created" in result.stdout or "Successfully created" in result.stderr:
                logger.info(f"✅ Successfully generated test file for {model}")
                results['success'].append(model)
            else:
                logger.error(f"❌ Failed to generate test file for {model}")
                logger.error(f"Output: {result.stdout}")
                logger.error(f"Error: {result.stderr}")
                results['failed'].append(model)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error generating test file for {model}: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            results['failed'].append(model)
    
    # Print summary
    print("\n=== GENERATION SUMMARY ===")
    print(f"✅ Successfully generated: {len(results['success'])} files")
    print(f"⏭️ Skipped (already exist): {len(results['skipped'])} files")
    print(f"❌ Failed to generate: {len(results['failed'])} files")
    
    if results['failed']:
        print("\nFailed models:")
        for model in results['failed']:
            print(f"  - {model}")
    
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate standardized test files for HuggingFace models")
    parser.add_argument("--output-dir", type=str, default="standardized_tests", help="Directory to store generated test files")
    parser.add_argument("--models", nargs="*", help="Specific models to generate tests for (default: all supported models)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    # Generate tests
    results = generate_tests(args.output_dir, args.models, args.overwrite)
    
    # Return success if all specified models were generated successfully
    successful = len(results['success']) + len(results['skipped'])
    total = successful + len(results['failed'])
    
    return 0 if len(results['failed']) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
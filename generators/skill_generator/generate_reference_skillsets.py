#!/usr/bin/env python3
"""
Generate reference implementations for model skillsets.

This script provides a command-line interface for generating
reference-pattern model skillsets based on the existing
implementation in ipfs_accelerate_py/worker/skillset.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import generator module
from generators.reference_model_generator import ReferenceModelGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"generate_reference_models_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
        )
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reference implementations for model skillsets"
    )
    
    # Model selection options
    model_group = parser.add_argument_group("Model Selection")
    model_selection = model_group.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--model", "-m", type=str,
        help="Specific model to generate (e.g., 'bert', 'gpt2', 'openai/clip-vit-base-patch32')"
    )
    model_selection.add_argument(
        "--priority", "-p", type=str, default="critical",
        choices=["critical", "high", "medium", "low"],
        help="Generate models with the specified priority level"
    )
    model_selection.add_argument(
        "--from-file", "-f", type=str,
        help="Read model names from the specified file (one per line)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", "-o", type=str,
        help="Directory to write generated skillsets to"
    )
    output_group.add_argument(
        "--force", action="store_true",
        help="Force overwrite of existing files"
    )
    output_group.add_argument(
        "--no-verify", action="store_true",
        help="Skip verification of generated files"
    )
    output_group.add_argument(
        "--summary-file", "-s", type=str,
        help="Write generation summary to the specified JSON file"
    )
    
    # Template options
    template_group = parser.add_argument_group("Template Options")
    template_group.add_argument(
        "--template-dir", "-t", type=str,
        help="Directory containing template files"
    )
    
    return parser.parse_args()

def get_models_from_file(file_path: str) -> List[str]:
    """
    Read model names from a file.
    
    Args:
        file_path: Path to file containing model names (one per line).
        
    Returns:
        List of model names.
    """
    try:
        with open(file_path, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Read {len(models)} models from {file_path}")
        return models
    except Exception as e:
        logger.error(f"Error reading model list from {file_path}: {e}")
        return []

def generate_reference_implementations(args):
    """
    Generate reference implementations based on command line arguments.
    
    Args:
        args: Parsed command line arguments.
    """
    start_time = time.time()
    
    # Create generator
    generator = ReferenceModelGenerator(args.template_dir, args.output_dir)
    
    # Determine which models to generate
    models_to_generate = []
    
    if args.model:
        # Single model
        models_to_generate = [args.model]
        logger.info(f"Generating reference implementation for model: {args.model}")
    
    elif args.from_file:
        # Models from file
        models_to_generate = get_models_from_file(args.from_file)
        logger.info(f"Generating reference implementations for {len(models_to_generate)} models from file")
    
    else:
        # Generate by priority
        from generators.model_generator import PRIORITY_MODELS
        models_to_generate = PRIORITY_MODELS.get(args.priority, [])
        logger.info(f"Generating reference implementations for {len(models_to_generate)} {args.priority} models")
    
    # Process each model
    results = {}
    overall_success = True
    
    for model in models_to_generate:
        success, files = generator.generate_reference_implementation(
            model, args.force, not args.no_verify
        )
        
        results[model] = {
            "success": success,
            "files": files
        }
        
        if not success:
            overall_success = False
    
    # Generate summary statistics
    total_files = sum(len(result["files"]) for result in results.values())
    total_successful = sum(1 for result in results.values() if result["success"])
    
    logger.info(f"Generation completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Generated {total_files} files for {len(models_to_generate)} models")
    logger.info(f"Successful: {total_successful}/{len(models_to_generate)}")
    
    # Write summary if requested
    if args.summary_file:
        try:
            with open(args.summary_file, 'w') as f:
                json.dump({
                    "success": overall_success,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": time.time() - start_time,
                    "total_models": len(models_to_generate),
                    "successful_models": total_successful,
                    "total_files": total_files,
                    "results": results
                }, f, indent=2)
            logger.info(f"Summary written to {args.summary_file}")
        except Exception as e:
            logger.error(f"Error writing summary to {args.summary_file}: {e}")
    
    return overall_success

if __name__ == "__main__":
    args = parse_args()
    
    # Banner
    print("""
    ╭────────────────────────────────────────────────────────╮
    │                                                        │
    │   IPFS Accelerate Reference Model Generator            │
    │                                                        │
    │   Generates model implementations in the reference     │
    │   format used in ipfs_accelerate_py/worker/skillset    │
    │                                                        │
    ╰────────────────────────────────────────────────────────╯
    """)
    
    # Generate reference implementations
    success = generate_reference_implementations(args)
    
    # Exit with appropriate status
    sys.exit(0 if success else 1)
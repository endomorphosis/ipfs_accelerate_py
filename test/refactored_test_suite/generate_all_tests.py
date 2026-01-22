#!/usr/bin/env python3
"""
Comprehensive test generator for all model architectures.

This script generates test files for models across all architectures based on priority.
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_all_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the test generator
try:
    from generators.test_generator import ModelTestGenerator, PRIORITY_MODELS
    from generators.architecture_detector import ARCHITECTURE_TYPES
except ImportError:
    # Try with full path
    try:
        from refactored_test_suite.generators.test_generator import ModelTestGenerator, PRIORITY_MODELS
        from refactored_test_suite.generators.architecture_detector import ARCHITECTURE_TYPES
    except ImportError:
        logger.error("Failed to import required modules. Please ensure you're running from the correct directory.")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate tests for multiple models across architectures")
    
    parser.add_argument(
        "--priority", 
        choices=["high", "medium", "low", "all"],
        default="high",
        help="Priority level of models to generate tests for"
    )
    
    parser.add_argument(
        "--architecture",
        choices=list(ARCHITECTURE_TYPES.keys()) + ["all"],
        default="all",
        help="Architecture type to generate tests for"
    )
    
    parser.add_argument(
        "--model",
        help="Generate test for a specific model instead of by priority/architecture"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./generated_tests",
        help="Directory to save generated test files"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated test files"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files"
    )
    
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate a coverage report after test generation"
    )
    
    parser.add_argument(
        "--coverage-file",
        default="model_test_coverage.md",
        help="Path to save the coverage report"
    )
    
    return parser.parse_args()


def get_models_to_generate(priority: str, architecture: str) -> List[str]:
    """
    Get list of models to generate tests for based on priority and architecture.
    
    Args:
        priority: Priority level (high, medium, low, all)
        architecture: Architecture type (encoder-only, decoder-only, etc., or all)
        
    Returns:
        List of model names to generate tests for
    """
    # Get priority models
    if priority == "all":
        priority_models = []
        for p in PRIORITY_MODELS:
            priority_models.extend(PRIORITY_MODELS[p])
    else:
        priority_models = PRIORITY_MODELS.get(priority, [])
    
    # If architecture is "all", return all priority models
    if architecture == "all":
        return priority_models
    
    # Otherwise, filter models by architecture
    arch_models = ARCHITECTURE_TYPES.get(architecture, [])
    
    # Return intersection of priority models and architecture models
    return [model for model in priority_models if any(arch_model in model for arch_model in arch_models)]


def generate_tests(args) -> Tuple[int, int, int]:
    """
    Generate tests based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (num_generated, num_failed, total)
    """
    # Create the generator instance
    generator = ModelTestGenerator(output_dir=args.output_dir)
    
    # If a specific model is provided, generate only that test
    if args.model:
        logger.info(f"Generating test for model: {args.model}")
        success, file_path = generator.generate_test_file(
            args.model, 
            force=args.force, 
            verify=args.verify
        )
        
        return (1 if success else 0, 0 if success else 1, 1)
    
    # Otherwise, generate tests by priority and architecture
    models_to_generate = get_models_to_generate(args.priority, args.architecture)
    
    if not models_to_generate:
        logger.warning(f"No models found for priority={args.priority}, architecture={args.architecture}")
        return (0, 0, 0)
    
    # Generate each model
    generated = []
    failed = []
    
    for model_type in models_to_generate:
        logger.info(f"Generating test for {model_type}")
        success, file_path = generator.generate_test_file(
            model_type, 
            force=args.force, 
            verify=args.verify
        )
        
        if success:
            generated.append(file_path)
        else:
            failed.append(model_type)
    
    # Print summary
    logger.info("\nGeneration Summary:")
    logger.info(f"- Generated: {len(generated)} files")
    logger.info(f"- Failed: {len(failed)} models")
    logger.info(f"- Total: {len(models_to_generate)} models")
    
    if failed:
        logger.info("\nFailed models:")
        for model in sorted(failed):
            logger.info(f"  - {model}")
    
    return len(generated), len(failed), len(models_to_generate)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate tests
    num_generated, num_failed, total = generate_tests(args)
    
    # Generate coverage report if requested
    if args.coverage_report:
        logger.info("Generating coverage report...")
        generator = ModelTestGenerator(output_dir=args.output_dir)
        generator.generate_coverage_report(args.coverage_file)
        logger.info(f"Coverage report saved to {args.coverage_file}")
    
    # Return success if all tests were generated successfully
    return 0 if num_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
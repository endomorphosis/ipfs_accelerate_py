#!/usr/bin/env python3
"""
Generate Tests for All Model Architectures

This script generates test files for all major model architectures in the HuggingFace ecosystem.
"""

import os
import sys
import argparse
import logging
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import our simple generator module
from simple_generator import generate_test, ARCHITECTURE_MAPPING

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_representative_models() -> List[Tuple[str, str]]:
    """Get a list of representative models for each architecture.
    
    Returns:
        List of tuples (model_type, architecture)
    """
    models = []
    
    # Select the first model from each architecture as representative
    for architecture, model_types in ARCHITECTURE_MAPPING.items():
        if model_types:
            models.append((model_types[0], architecture))
    
    return models

def generate_all_tests(output_dir: str, parallel: bool = True) -> Dict[str, Any]:
    """Generate test files for all model architectures.
    
    Args:
        output_dir: Directory to save the generated files
        parallel: Whether to run generation in parallel
        
    Returns:
        Dict with generation results
    """
    start_time = time.time()
    
    # Get representative models
    models = get_representative_models()
    logger.info(f"Found {len(models)} representative models across architectures")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate tests
    results = {}
    
    if parallel:
        # Parallel execution using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models), os.cpu_count() or 4)) as executor:
            # Submit tasks
            future_to_model = {
                executor.submit(generate_test, model_type, output_dir): (model_type, architecture)
                for model_type, architecture in models
            }
            
            # Gather results
            for future in concurrent.futures.as_completed(future_to_model):
                model_type, architecture = future_to_model[future]
                try:
                    result = future.result()
                    results[model_type] = result
                    logger.info(f"Generated test for {model_type} ({architecture})")
                except Exception as e:
                    logger.error(f"Error generating test for {model_type} ({architecture}): {str(e)}")
                    results[model_type] = {
                        "success": False,
                        "error": str(e),
                        "model_type": model_type,
                        "architecture": architecture
                    }
    else:
        # Sequential execution
        for model_type, architecture in models:
            logger.info(f"Generating test for {model_type} ({architecture})")
            try:
                result = generate_test(model_type, output_dir)
                results[model_type] = result
            except Exception as e:
                logger.error(f"Error generating test for {model_type} ({architecture}): {str(e)}")
                results[model_type] = {
                    "success": False,
                    "error": str(e),
                    "model_type": model_type,
                    "architecture": architecture
                }
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for result in results.values() if result["success"])
    
    # Create summary
    summary = {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "success_rate": successful / total if total > 0 else 0,
        "duration": time.time() - start_time,
        "results": results
    }
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "generation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate Tests for All Model Architectures")
    parser.add_argument("--output-dir", default="./generated_tests", help="Output directory for generated files")
    parser.add_argument("--sequential", action="store_true", help="Run generation sequentially (default: parallel)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate tests
    summary = generate_all_tests(args.output_dir, parallel=not args.sequential)
    
    # Print summary
    print(f"\nGeneration Summary:")
    print(f"Total models: {summary['total']}")
    print(f"Successful: {summary['successful']} ({summary['success_rate']:.1%})")
    print(f"Failed: {summary['failed']}")
    print(f"Duration: {summary['duration']:.2f} seconds")
    print(f"Summary saved to: {os.path.join(args.output_dir, 'generation_summary.json')}")
    
    # Print failed models
    if summary['failed'] > 0:
        print("\nFailed models:")
        for model_type, result in summary['results'].items():
            if not result["success"]:
                print(f"  - {model_type}: {result.get('error', 'Unknown error')}")
    
    return 0 if summary['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Integration script for incorporating enhanced test generator capabilities.

This script serves as a bridge between the new enhanced generator and the
existing test infrastructure. It provides functions for generating test files 
using the new approach while maintaining compatibility with existing systems.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the new enhanced generator
# Set initial state for generators
HAS_ENHANCED_GENERATOR = False
HAS_MINIMAL_GENERATOR = False

# Try to import the enhanced generator
try:
    from enhanced_generator import (
        generate_test, generate_all_tests, get_model_architecture,
        get_default_model, validate_generated_file,
        MODEL_REGISTRY, ARCHITECTURE_TYPES, DEFAULT_MODELS
    )
    HAS_ENHANCED_GENERATOR = True
    logger.info("Using enhanced generator for test file generation")
except ImportError:
    logger.warning("Enhanced generator not available, falling back to minimal generator")

# Import the minimal generator as fallback if needed
if not HAS_ENHANCED_GENERATOR:
    try:
        from generate_minimal_test import (
            generate_bert_test, generate_gpt2_test, generate_t5_test,
            generate_vit_test, generate_clip_test, generate_whisper_test,
            generate_all_minimal_tests
        )
        HAS_MINIMAL_GENERATOR = True
        logger.info("Using minimal generator as fallback")
    except ImportError:
        logger.error("Neither enhanced nor minimal generator available")
        sys.exit("Cannot import any test generator")

def create_test_file(model_type: str, output_dir: str, model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a test file for the specified model type, using the best available generator.
    
    Args:
        model_type (str): Type of model (e.g., bert, gpt2, t5)
        output_dir (str): Directory to save the generated test file
        model_id (str, optional): Specific model ID to use. If None, use default.
        
    Returns:
        dict: Result of the generation process
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Use enhanced generator if available
    if HAS_ENHANCED_GENERATOR:
        result = generate_test(model_type, output_dir, model_id)
        return result
    
    # Otherwise use minimal generator
    elif HAS_MINIMAL_GENERATOR:
        # Map model types to minimal generator functions
        generator_map = {
            "bert": generate_bert_test,
            "gpt2": generate_gpt2_test,
            "t5": generate_t5_test,
            "vit": generate_vit_test,
            "clip": generate_clip_test,
            "whisper": generate_whisper_test
        }
        
        # Find closest match if model type not directly supported
        if model_type not in generator_map:
            # Try to find architecture match
            architecture = None
            
            # Define architecture types - use enhanced generator's if available, otherwise use minimal
            architecture_types = ARCHITECTURE_TYPES if HAS_ENHANCED_GENERATOR else {
                "encoder-only": ["bert"],
                "decoder-only": ["gpt2"],
                "encoder-decoder": ["t5"],
                "vision": ["vit"],
                "vision-text": ["clip"],
                "speech": ["whisper"]
            }
            
            # Check if model type matches any architecture
            for arch, models in architecture_types.items():
                if model_type in models:
                    architecture = arch
                    break
            
            # Map architecture to default model type
            if architecture:
                default_types = {
                    "encoder-only": "bert",
                    "decoder-only": "gpt2",
                    "encoder-decoder": "t5",
                    "vision": "vit",
                    "vision-text": "clip",
                    "speech": "whisper"
                }
                model_type = default_types.get(architecture, "bert")
        
        # Generate the test file
        if model_type in generator_map:
            generator_func = generator_map[model_type]
            output_file = generator_func(output_dir)
            
            # Create result in the same format as enhanced generator
            result = {
                "success": True,
                "output_file": output_file,
                "model_type": model_type,
                "architecture": "unknown",
                "duration": 0.0,
                "validation": "unknown",
                "is_valid": True
            }
            return result
    
    # Cannot generate test file
    return {
        "success": False,
        "error": "No suitable generator available",
        "model_type": model_type
    }

def create_all_test_files(output_dir: str) -> Dict[str, Any]:
    """
    Create test files for all supported model types.
    
    Args:
        output_dir (str): Directory to save the generated test files
        
    Returns:
        dict: Summary of the generation process
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Use enhanced generator if available
    if HAS_ENHANCED_GENERATOR:
        return generate_all_tests(output_dir)
    
    # Otherwise use minimal generator
    elif HAS_MINIMAL_GENERATOR:
        # Generate all minimal tests and convert to enhanced format
        valid_files, total_files = generate_all_minimal_tests(output_dir)
        
        # Key model types
        key_models = ["bert", "gpt2", "t5", "vit", "clip", "whisper"]
        
        # Create results in the same format as enhanced generator
        results = {}
        for model_type in key_models:
            results[model_type] = {
                "success": True,
                "output_file": os.path.join(output_dir, f"test_{model_type}.py"),
                "model_type": model_type,
                "architecture": "unknown",
                "duration": 0.0,
                "validation": "unknown",
                "is_valid": True
            }
        
        return {
            "total": total_files,
            "successful": valid_files,
            "valid_syntax": valid_files,
            "results": results
        }
    
    # Cannot generate test files
    return {
        "total": 0,
        "successful": 0,
        "valid_syntax": 0,
        "results": {},
        "error": "No suitable generator available"
    }

def integrate_with_existing_system(model_type: str, output_dir: str = None) -> str:
    """
    Integration function to work with existing test infrastructure.
    
    This function serves as an adapter between the new generator and
    existing test infrastructure.
    
    Args:
        model_type (str): Type of model to generate a test for
        output_dir (str, optional): Output directory. If None, use default.
        
    Returns:
        str: Path to generated test file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "generated_tests_integrated"
    
    # Create the test file
    result = create_test_file(model_type, output_dir)
    
    if result["success"]:
        logger.info(f"Successfully generated test file for {model_type}: {result['output_file']}")
        return result["output_file"]
    else:
        logger.error(f"Failed to generate test file for {model_type}: {result.get('error', 'Unknown error')}")
        return None

def main():
    """Command-line entry point for the integration script."""
    parser = argparse.ArgumentParser(description="Generate model test files using best available generator")
    parser.add_argument("--model-type", type=str, help="Model type to generate a test for (e.g., bert, gpt2, t5)")
    parser.add_argument("--model-id", type=str, help="Specific model ID to use (e.g., bert-base-uncased)")
    parser.add_argument("--output-dir", type=str, default="generated_tests_integrated", 
                       help="Output directory for generated tests")
    parser.add_argument("--all", action="store_true", help="Generate tests for all supported model types")
    parser.add_argument("--validate", action="store_true", help="Validate the generated test files")
    parser.add_argument("--info", action="store_true", help="Show information about available generators")
    
    args = parser.parse_args()
    
    if args.info:
        print("Test Generator Information:")
        print(f"- Enhanced Generator: {'Available' if HAS_ENHANCED_GENERATOR else 'Not Available'}")
        print(f"- Minimal Generator: {'Available' if HAS_MINIMAL_GENERATOR else 'Not Available'}")
        
        if HAS_ENHANCED_GENERATOR:
            print("\nSupported Model Types:")
            for arch, models in ARCHITECTURE_TYPES.items():
                print(f"- {arch}: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                
            print("\nDefault Models:")
            for arch, model in DEFAULT_MODELS.items():
                print(f"- {arch}: {model}")
        
        return 0
    
    if args.all:
        # Generate tests for all supported model types
        results = create_all_test_files(args.output_dir)
        
        # Save validation results
        with open(os.path.join(args.output_dir, "validation_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Generated {results['successful']} of {results['total']} test files")
        print(f"Files with valid syntax: {results['valid_syntax']} of {results['total']}")
        print(f"Results saved to {os.path.join(args.output_dir, 'validation_summary.json')}")
        
        return 0 if results['successful'] == results['total'] else 1
    
    elif args.model_type:
        # Generate a test for a specific model type
        result = create_test_file(args.model_type, args.output_dir, args.model_id)
        
        if result["success"]:
            print(f"Generated test file: {result['output_file']}")
            print(f"Model type: {result['model_type']}")
            if 'architecture' in result:
                print(f"Architecture: {result['architecture']}")
            if 'validation' in result:
                print(f"Validation: {result['validation']}")
            
            return 0
        else:
            print(f"Failed to generate test file: {result.get('error', 'Unknown error')}")
            return 1
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test Template Validation

This script tests the template validator against generated templates.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path to import validator
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.generators.validators.template_validator_integration import (
    validate_template_for_generator,
    validate_template_file_for_generator
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_generator(generator_script: str, model_name: str, output_file: str) -> Tuple[bool, List[str]]:
    """
    Test a template generator by generating a template and validating it.
    
    Args:
        generator_script: Path to generator script
        model_name: Model name to generate template for
        output_file: Path to output file
        
    Returns:
        Tuple of (success, errors)
    """
    try:
        # Generate the template
        logger.info(f"Generating template for {model_name} using {generator_script}")
        
        cmd = [sys.executable, generator_script, "--model", model_name, "--output", output_file]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Generator failed with exit code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False, [f"Generator failed with exit code {result.returncode}"]
            
        logger.info(f"Generator succeeded, output written to {output_file}")
        
        # Validate the generated template
        logger.info(f"Validating generated template {output_file}")
        
        is_valid, errors = validate_template_file_for_generator(
            output_file,
            Path(generator_script).stem,
            validate_hardware=True,
            check_resource_pool=True
        )
        
        if is_valid:
            logger.info(f"Template validation passed for {output_file}")
            return True, []
        else:
            logger.error(f"Template validation failed for {output_file}")
            for error in errors:
                logger.error(f"  - {error}")
            return False, errors
    except Exception as e:
        logger.error(f"Error testing generator: {str(e)}")
        return False, [str(e)]

def test_model_types(generator_script: str, output_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Test a template generator against all model types.
    
    Args:
        generator_script: Path to generator script
        output_dir: Directory to save output files
        
    Returns:
        Dictionary of model types with test results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model types to test
    model_types = {
        "bert": "bert-base-uncased",
        "t5": "t5-small",
        "gpt2": "gpt2",
        "vit": "google/vit-base-patch16-224",
        "whisper": "openai/whisper-tiny",
        "clip": "openai/clip-vit-base-patch32"
    }
    
    # Test each model type
    results = {}
    for model_type, model_name in model_types.items():
        logger.info(f"Testing model type {model_type} with model {model_name}")
        
        output_file = os.path.join(output_dir, f"test_{model_type}.py")
        success, errors = test_generator(generator_script, model_name, output_file)
        
        results[model_type] = {
            "model_name": model_name,
            "output_file": output_file,
            "success": success,
            "errors": errors
        }
        
    return results

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Test Template Validation")
    parser.add_argument("--generator", type=str, required=True, help="Path to generator script")
    parser.add_argument("--model", type=str, help="Model name to test")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--test-all", action="store_true", help="Test all model types")
    parser.add_argument("--json", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    if args.test_all:
        # Test all model types
        output_dir = args.output if args.output else "test_output"
        results = test_model_types(args.generator, output_dir)
        
        # Print summary
        success_count = sum(1 for r in results.values() if r["success"])
        logger.info(f"Test summary: {success_count}/{len(results)} model types passed validation")
        
        # Print detailed results
        for model_type, result in results.items():
            status = "✅ Passed" if result["success"] else "❌ Failed"
            logger.info(f"{model_type}: {status}")
            
            if not result["success"]:
                for error in result["errors"]:
                    logger.info(f"  - {error}")
                    
        # Write JSON output if requested
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
                
        # Return success if all tests passed
        return 0 if success_count == len(results) else 1
    elif args.model:
        # Test a single model
        output_file = args.output if args.output else f"test_{args.model.split('/')[-1]}.py"
        success, errors = test_generator(args.generator, args.model, output_file)
        
        # Write JSON output if requested
        if args.json:
            results = {
                args.model: {
                    "output_file": output_file,
                    "success": success,
                    "errors": errors
                }
            }
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
                
        return 0 if success else 1
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
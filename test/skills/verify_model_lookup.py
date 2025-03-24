#!/usr/bin/env python3

"""
Verify the HuggingFace model lookup integration with the test generator.

This script tests:
1. The direct API integration by calling get_recommended_default_model
2. The test generator integration by calling get_model_from_registry
3. The minimal test generation by creating test files with the right models

Usage:
    python verify_model_lookup.py [--model-type TYPE] [--generate]
"""

import os
import sys
import json
import logging
import argparse
import importlib.util
import tempfile
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = CURRENT_DIR / "huggingface_model_types.json"

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.warning(f"Could not find module spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning(f"Could not import {module_name} from {file_path}: {e}")
        return None

def test_direct_api(model_type):
    """Test direct API integration."""
    try:
        logger.info(f"Testing direct API integration for {model_type}")
        
        # Import find_models.py
        find_models_path = CURRENT_DIR / "find_models.py"
        find_models = import_module_from_path("find_models", find_models_path)
        if not find_models:
            logger.error("Could not import find_models.py")
            return False
        
        # Call get_recommended_default_model
        default_model = find_models.get_recommended_default_model(model_type)
        logger.info(f"✅ Direct API call successful: {model_type} → {default_model}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Direct API call failed: {e}")
        return False

def test_generator_integration(model_type):
    """Test test generator integration."""
    try:
        logger.info(f"Testing test generator integration for {model_type}")
        
        # Import test_generator_fixed.py
        test_generator_path = CURRENT_DIR / "test_generator_fixed.py"
        test_generator = import_module_from_path("test_generator_fixed", test_generator_path)
        if not test_generator:
            logger.error("Could not import test_generator_fixed.py")
            return False
        
        # Call get_model_from_registry
        default_model = test_generator.get_model_from_registry(model_type)
        logger.info(f"✅ Generator integration successful: {model_type} → {default_model}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Generator integration failed: {e}")
        return False

def test_minimal_generation(model_type, output_dir=None):
    """Test minimal test generation."""
    try:
        logger.info(f"Testing minimal test generation for {model_type}")
        
        # Create temporary output directory if not specified
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="model_lookup_test_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Import generate_minimal_test.py
        generate_minimal_path = CURRENT_DIR / "generate_minimal_test.py"
        generate_minimal = import_module_from_path("generate_minimal_test", generate_minimal_path)
        if not generate_minimal:
            logger.error("Could not import generate_minimal_test.py")
            return False
        
        # Call generate_minimal_test
        result = generate_minimal.generate_minimal_test(model_type, output_dir)
        
        if result:
            # Check if file was created
            test_file = os.path.join(output_dir, f"test_hf_{model_type.replace('-', '_')}.py")
            if os.path.exists(test_file):
                logger.info(f"✅ Test generation successful: {test_file}")
                
                # Execute the test file to verify it works
                try:
                    # Only print the model (--help doesn't run the model lookup)
                    cmd = [sys.executable, test_file, "--help"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Generated test file syntax is valid")
                        return True
                    else:
                        logger.error(f"❌ Generated test file execution failed: {result.stderr}")
                        return False
                except Exception as e:
                    logger.error(f"❌ Error executing generated test file: {e}")
                    return False
            else:
                logger.error(f"❌ Test file not created: {test_file}")
                return False
        else:
            logger.error(f"❌ Test generation failed")
            return False
    
    except Exception as e:
        logger.error(f"❌ Minimal test generation failed: {e}")
        return False

def verify_model_lookup(model_type="bert", output_dir=None, generate=False):
    """Verify model lookup integration."""
    results = {
        "direct_api": test_direct_api(model_type),
        "generator_integration": test_generator_integration(model_type),
    }
    
    if generate:
        results["minimal_generation"] = test_minimal_generation(model_type, output_dir)
    
    # Print summary
    print("\n# Verification Summary")
    print(f"\nModel Type: {model_type}\n")
    for test, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test}")
    
    # Overall result
    all_passed = all(results.values())
    print(f"\nOverall Result: {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    return all_passed

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify HuggingFace model lookup integration")
    parser.add_argument("--model-type", type=str, default="bert", help="Model type to verify")
    parser.add_argument("--output-dir", type=str, help="Output directory for generated test files")
    parser.add_argument("--generate", action="store_true", help="Generate test files")
    parser.add_argument("--all", action="store_true", help="Verify with all core model types")
    
    args = parser.parse_args()
    
    if args.all:
        # Verify with core model types
        core_types = ["bert", "gpt2", "t5", "vit"]
        results = {}
        
        for model_type in core_types:
            logger.info(f"\n=== Verifying {model_type} ===\n")
            results[model_type] = verify_model_lookup(
                model_type=model_type,
                output_dir=args.output_dir,
                generate=args.generate
            )
        
        # Print overall summary
        print("\n# Overall Verification Summary\n")
        for model_type, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {model_type}")
        
        all_passed = all(results.values())
        print(f"\nFinal Result: {'✅ PASS' if all_passed else '❌ FAIL'}")
        
        return 0 if all_passed else 1
    else:
        # Verify with single model type
        success = verify_model_lookup(
            model_type=args.model_type,
            output_dir=args.output_dir,
            generate=args.generate
        )
        
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Runner script for hardware testing against multiple models.
This script executes tests for specified models on selected hardware platforms.
"""

import os
import sys
import json
import time
import argparse
import logging
import importlib.util
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_module_from_file(file_path, module_name=None):
    """Dynamically import a module from a file path."""
    if module_name is None:
        module_name = os.path.basename(file_path).replace('.py', '')
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_test_class(module, model_type):
    """Get the appropriate test class from the module based on model type."""
    # Look for a class with "Test" in the name
    test_classes = [obj for name, obj in module.__dict__.items() if name.endswith('TestBase') or name.startswith('Test')]
    
    if not test_classes:
        raise ValueError(f"No test class found in module for {model_type}")
    
    return test_classes[0]

def run_test(model_file, model_name, platform, output_dir):
    """Run a test for the specified model and platform."""
    logger.info(f"Testing {model_name} on {platform} platform...")
    
    try:
        # Import the module
        model_type = os.path.basename(model_file).replace('test_hf_', '').replace('.py', '')
        module = import_module_from_file(model_file)
        
        # Get the test class
        TestClass = get_test_class(module, model_type)
        
        # Create test instance
        test_instance = TestClass(model_id=model_name)
        
        # Run the test
        if platform.lower() == "all":
            results = test_instance.test()
        else:
            # Run for a specific platform
            platform_results = test_instance.run_test(platform)
            results = {
                "results": {platform: platform_results},
                "examples": test_instance.examples,
                "metadata": {
                    "model_id": test_instance.model_id,
                    "model_path": test_instance.model_path,
                    "model_config": getattr(test_instance, "model_config", {}),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        output_file = os.path.join(output_dir, f"{model_type}_{platform.lower()}_test.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print(f"\n{model_type.upper()} MODEL TEST RESULTS ({model_name}):")
        for p, p_results in results["results"].items():
            success = p_results.get("success", False)
            impl_type = p_results.get("implementation_type", "UNKNOWN")
            error = p_results.get("error", "")
            
            if success:
                print(f"{p.upper()}: ✅ Success ({impl_type})")
            else:
                print(f"{p.upper()}: ❌ Failed ({error})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing {model_name} on {platform}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run hardware tests for specified models."""
    parser = argparse.ArgumentParser(description="Run hardware tests for models")
    parser.add_argument("--models-dir", default="./updated_models", help="Directory containing model test files")
    parser.add_argument("--models", nargs="+", default=["bert", "t5"], help="Models to test (e.g., bert t5)")
    parser.add_argument("--platforms", nargs="+", default=["cpu", "cuda", "mps", "openvino", "rocm", "webnn", "webgpu"], 
                        help="Platforms to test on")
    parser.add_argument("--output-dir", default="./hardware_test_results", help="Directory to save test results")
    parser.add_argument("--model-names", nargs="+", default=[], help="Specific model names for each model (e.g., bert-base-uncased)")
    args = parser.parse_args()
    
    # Default model names if not provided
    if not args.model_names:
        model_names = {
            "bert": "bert-base-uncased",
            "t5": "t5-small",
            "llama": "facebook/opt-125m",
            "clip": "openai/clip-vit-base-patch32",
            "vit": "google/vit-base-patch16-224",
            "clap": "laion/clap-htsat-unfused",
            "whisper": "openai/whisper-tiny",
            "wav2vec2": "facebook/wav2vec2-base",
            "llava": "llava-hf/llava-1.5-7b-hf",
            "xclip": "microsoft/xclip-base-patch32",
            "qwen2": "Qwen/Qwen2-1.5B-Instruct",
            "detr": "facebook/detr-resnet-50"
        }
    else:
        # If specific model names are provided, ensure they match the number of models
        if len(args.model_names) != len(args.models):
            logger.error("Number of model names must match number of models")
            sys.exit(1)
        model_names = dict(zip(args.models, args.model_names))
    
    # Run tests for each model
    success_count = 0
    failure_count = 0
    
    for model in args.models:
        model_file = os.path.join(args.models_dir, f"test_hf_{model}.py")
        
        if not os.path.exists(model_file):
            logger.error(f"Model file not found: {model_file}")
            failure_count += 1
            continue
        
        model_name = model_names.get(model, f"{model}-default")
        
        # Test on all platforms or specific platform
        if "all" in args.platforms:
            success = run_test(model_file, model_name, "all", args.output_dir)
            if success:
                success_count += 1
            else:
                failure_count += 1
        else:
            for platform in args.platforms:
                success = run_test(model_file, model_name, platform, args.output_dir)
                if success:
                    success_count += 1
                else:
                    failure_count += 1
    
    # Print summary
    logger.info(f"\nTest Summary: {success_count} successful, {failure_count} failed")
    
    # Return success if all tests passed
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
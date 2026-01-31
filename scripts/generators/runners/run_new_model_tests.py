#!/usr/bin/env python3
"""
Script to run tests for newly implemented Hugging Face model tests.
This script will execute tests for recently implemented model test files.
"""

import os
import sys
import json
import time
import datetime
import subprocess
import importlib.util
import traceback
from pathlib import Path

# List of newly implemented high-priority models to test
NEW_MODEL_TESTS = [
    "blip",
    "data2vec_audio",
    "data2vec_vision",
    "dpt",
    "encodec",
    "fuyu",
    "layoutlmv3",
    "mask2former",
    "mobilevit",
    "owlvit",
    "pix2struct",
    "seamless_m4t",
    "segformer",
    "speecht5",
    "vilt",
    "vision_encoder_decoder",
    "wavlm"
]

def run_test(model_name):
    """Run a specific model test and collect results.
    
    Args:
        model_name (str): Name of the model to test (without test_hf_ prefix and .py suffix)
        
    Returns:
        dict: Test results or error information
    """
    test_file = f"skills/test_hf_{model_name}.py"
    
    # Check if the test file exists
    if not os.path.exists(test_file):
        return {
            "status": "error",
            "error": f"Test file not found: {test_file}",
            "model": model_name
        }
    
    try:
        # Try to import the module and run the test
        print(f"\n{'='*50}")
        print(f"Running test for model: {model_name}")
        print(f"{'='*50}")
        
        # Define the test module name
        module_name = f"test_hf_{model_name}"
        test_class_name = f"test_hf_{model_name}"
        
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Get the test class
        test_class = getattr(test_module, test_class_name)
        
        # Create an instance of the test class
        test_instance = test_class()
        
        # Run the test
        start_time = time.time()
        results = test_instance.test()
        elapsed_time = time.time() - start_time
        
        # Extract status information
        status_dict = results.get("status", {})
        platform_status = {}
        
        for key, value in status_dict.items():
            if "cpu_" in key and "Success" in value:
                platform_status["cpu"] = "success"
            elif "cuda_" in key and "Success" in value:
                platform_status["cuda"] = "success"
            elif "openvino_" in key and "Success" in value:
                platform_status["openvino"] = "success"
        
        return {
            "status": "success",
            "model": model_name,
            "elapsed_time": elapsed_time,
            "platform_status": platform_status,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        # Capture any exception that might occur
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "model": model_name
        }

def save_results(results_dict):
    """Save test results to a JSON file.
    
    Args:
        results_dict (dict): Dictionary containing test results
    """
    # Create results directory if it doesn't exist
    results_dir = Path("new_model_results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"new_model_tests_{timestamp}.json"
    
    # Save results to file
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

def main():
    """Main function to run all tests"""
    print(f"Starting tests at {datetime.datetime.now().isoformat()}")
    
    # Setup results dictionary
    all_results = {
        "summary": {
            "total": len(NEW_MODEL_TESTS),
            "success": 0,
            "failure": 0
        },
        "tests": []
    }
    
    # Run tests for each model
    for model_name in NEW_MODEL_TESTS:
        try:
            result = run_test(model_name)
            all_results["tests"].append(result)
            
            # Update summary
            if result["status"] == "success":
                all_results["summary"]["success"] += 1
            else:
                all_results["summary"]["failure"] += 1
                
            # Print status
            if result["status"] == "success":
                print(f"✅ {model_name}: SUCCESS")
                if "platform_status" in result:
                    platforms = []
                    if result["platform_status"].get("cpu") == "success":
                        platforms.append("CPU")
                    if result["platform_status"].get("cuda") == "success":
                        platforms.append("CUDA")
                    if result["platform_status"].get("openvino") == "success":
                        platforms.append("OpenVINO")
                    print(f"   Successful platforms: {', '.join(platforms)}")
            else:
                print(f"❌ {model_name}: FAILED")
                print(f"   Error: {result['error']}")
        except Exception as e:
            # Handle any unexpected exceptions
            error_result = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "model": model_name
            }
            all_results["tests"].append(error_result)
            all_results["summary"]["failure"] += 1
            print(f"❌ {model_name}: EXCEPTION: {str(e)}")
    
    # Calculate overall percentages
    total = all_results["summary"]["total"]
    success = all_results["summary"]["success"]
    failure = all_results["summary"]["failure"]
    success_rate = (success / total) * 100 if total > 0 else 0
    
    all_results["summary"]["success_rate"] = f"{success_rate:.1f}%"
    
    # Print summary
    print("\n" + "="*50)
    print(f"TEST SUMMARY")
    print(f"Total tests: {total}")
    print(f"Successful: {success} ({success_rate:.1f}%)")
    print(f"Failed: {failure} ({100-success_rate:.1f}%)")
    print("="*50)
    
    # Save results
    save_results(all_results)

if __name__ == "__main__":
    main()
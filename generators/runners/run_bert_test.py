#!/usr/bin/env python3
"""
Simple test script to run BERT model tests on different hardware platforms.
This is a direct example of how to use the enhanced hardware detection
and model testing capabilities.
"""

import os
import sys
import time
import argparse
import importlib.util
from pathlib import Path

def load_module_from_file(file_path, module_name):
    """Load a Python module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    """Run BERT model tests"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run BERT model tests")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model to test (default: bert-base-uncased)")
    parser.add_argument("--hardware", type=str, default="all",
                       help="Hardware platform (cpu, cuda, openvino, all)")
    args = parser.parse_args()
    
    # Find the test file
    test_file = Path("skills/test_hf_bert.py")
    if not test_file.exists():
        test_file = Path("modality_tests/test_hf_bert.py")
        if not test_file.exists():
            print(f"Error: Could not find test file test_hf_bert.py")
            sys.exit(1)
    
    # Load the test module
    print(f"Loading test module from {test_file}")
    test_module = load_module_from_file(test_file, "bert_test")
    
    # Create a tester instance
    tester = test_module.TestBertModels(args.model)
    
    # Setup hardware to test
    hardware_to_test = []
    if args.hardware == "all":
        hardware_to_test = ["cpu"]
        if hasattr(test_module, "HW_CAPABILITIES"):
            if test_module.HW_CAPABILITIES.get("cuda", False):
                hardware_to_test.append("cuda")
            if test_module.HW_CAPABILITIES.get("openvino", False):
                hardware_to_test.append("openvino")
            if test_module.HW_CAPABILITIES.get("mps", False):
                hardware_to_test.append("mps")
    else:
        hardware_to_test = [args.hardware]
    
    print(f"Testing model {args.model} on hardware: {', '.join(hardware_to_test)}")
    
    # Run the tests
    results = {}
    for device in hardware_to_test:
        print(f"\nTesting on {device.upper()}:")
        
        # Set environment variable for device
        os.environ["TEST_HARDWARE_PLATFORM"] = device
        if device == "cpu":
            # Make sure CUDA is disabled for CPU tests
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Run the test using the pipeline method
        start_time = time.time()
        try:
            pipeline_result = test_module.test_pipeline(tester, device=device)
            
            # Store results
            results[device] = {
                "success": pipeline_result.get("pipeline_success", False),
                "elapsed_time": time.time() - start_time,
                "result": pipeline_result
            }
            
            # Print result summary
            if pipeline_result.get("pipeline_success", False):
                print(f"  ✅ Success")
                if "timings" in pipeline_result:
                    print(f"  ⏱️ Average inference time: {pipeline_result['timings'].get('avg_inference_time', 0):.3f}s")
                if "examples" in pipeline_result:
                    print(f"  📋 Examples saved: {len(pipeline_result.get('examples', []))}")
                if "implementation_type" in pipeline_result:
                    print(f"  🔧 Implementation: {pipeline_result.get('implementation_type', 'UNKNOWN')}")
            else:
                print(f"  ❌ Failed: {pipeline_result.get('pipeline_error_type', 'Unknown error')}")
                print(f"  🚫 Error: {pipeline_result.get('pipeline_error', 'Unknown error')}")
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results[device] = {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    # Print summary
    print("\n" + "="*50)
    print(f"Test Summary for {args.model}:")
    print("="*50)
    
    successful = sum(1 for r in results.values() if r["success"])
    print(f"Successful: {successful}/{len(results)} platforms")
    
    for device, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        time_str = f"{result['elapsed_time']:.2f}s"
        print(f"{device.upper()}: {status} ({time_str})")
    
    print("="*50)
    
    # Return success if at least one test passed
    return successful > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
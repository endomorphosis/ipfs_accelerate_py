#!/usr/bin/env python3
"""
Script to run a specific test and collect performance metrics.
"""

import os
import sys
import time
import json
import datetime
import importlib
import traceback
from typing import Dict, Any

def run_test(test_file: str) -> Dict[str, Any]:
    """
    Run a single test file and collect metrics.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        Dict containing test results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Running test: {test_file}")
    print(f"{'='*80}")
    
    try:
        # Get module name from file path
        module_path = test_file.replace('/', '.').replace('.py', '')
        
        # Extract the class name from the file name (assuming naming convention)
        file_name = os.path.basename(test_file)
        class_name = file_name.replace('.py', '')
        
        # Import the module
        print(f"Importing module: {module_path}")
        module = importlib.import_module(module_path)
        
        # Get the test class
        print(f"Loading test class: {class_name}")
        test_class = getattr(module, class_name)
        
        # Create an instance of the test class
        print(f"Creating test instance")
        test_instance = test_class()
        
        # Run the test and capture results
        print(f"Running test...")
        start_time = time.time()
        results = test_instance.test()
        total_time = time.time() - start_time
        
        # Add total execution time to results
        results["total_execution_time"] = total_time
        
        print(f"\nTest completed in {total_time:.2f} seconds")
        
        # Save results to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), "performance_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract model name from test file path
        model_name = file_name.replace('test_', '').replace('.py', '')
        output_file = f"{model_name}_performance_{timestamp}.json"
        output_path = os.path.join(output_dir, output_file)
        
        # Add metadata
        results_with_metadata = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "test_file": test_file,
                "model": model_name,
                "execution_time": total_time
            },
            "results": results
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        # Print a basic summary of the results
        platforms = ["cpu", "cuda", "openvino"]
        print("\n" + "-"*80)
        print("TEST SUMMARY")
        print("-"*80)
        
        for platform in platforms:
            impl_type = None
            success = False
            
            for key, value in results.items():
                if key.startswith(f"{platform}_") and "Success" in str(value):
                    success = True
                    if "(" in value and ")" in value:
                        impl_type = value.split("(")[1].split(")")[0]
                    break
            
            status = "Success" if success else "Failed/Not Available"
            print(f"{platform.upper()}: {status} - Implementation: {impl_type or 'Unknown'}")
        
        print("-"*80)
        
        return {
            "status": "Success",
            "results": results,
            "output_path": output_path
        }
        
    except Exception as e:
        print(f"Error running test {test_file}: {e}")
        traceback.print_exc()
        return {
            "status": "Error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_single_test.py <test_file_path>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    print(f"Running test: {test_file}")
    run_test(test_file)
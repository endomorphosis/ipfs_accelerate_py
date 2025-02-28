#!/usr/bin/env python3
"""
Test script to check CUDA implementation status across all skills.
"""

import os
import sys
import json
import glob
import subprocess
import time

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
COLLECTED_RESULTS_DIR = os.path.join(SKILLS_DIR, "collected_results")

def get_test_files():
    """Get a list of all test files in the skills directory"""
    test_files = []
    # Look for Python files that start with "test_"
    for file_path in glob.glob(os.path.join(SKILLS_DIR, "test_*.py")):
        test_files.append(os.path.basename(file_path))
    return sorted(test_files)

def check_cuda_status():
    """Check CUDA implementation status for all models"""
    test_files = get_test_files()
    
    print(f"Found {len(test_files)} test files to check")
    
    # Prepare results table
    results = []
    
    for test_file in test_files:
        model_name = test_file.replace("test_", "").replace(".py", "")
        module_name = f"skills.{test_file[:-3]}"  # Remove .py extension
        
        print(f"\nTesting {model_name}...")
        
        # Run the test in a subprocess
        start_time = time.time()
        subprocess.run(
            ["python3", "-m", module_name],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        elapsed_time = time.time() - start_time
        
        # Check the results file
        result_file = os.path.join(COLLECTED_RESULTS_DIR, f"{model_name}_test_results.json")
        if not os.path.exists(result_file):
            print(f"  WARNING: No results file found for {model_name}")
            results.append({
                "model": model_name,
                "cuda_available": "Unknown",
                "cuda_status": "Unknown",
                "implementation_type": "Unknown",
                "time": elapsed_time
            })
            continue
            
        try:
            with open(result_file, 'r') as f:
                test_results = json.load(f)
                
            # Check CUDA status
            status = test_results.get("status", {})
            cuda_status = "Not found"
            cuda_implementation = "Unknown"
            
            # Check for cuda_init or cuda_tests
            if "cuda_init" in status:
                cuda_status = status["cuda_init"]
                if "(REAL)" in cuda_status:
                    cuda_implementation = "REAL"
                elif "(MOCK)" in cuda_status:
                    cuda_implementation = "MOCK"
            elif "cuda_tests" in status:
                cuda_status = status["cuda_tests"]
                # Special case for "CUDA not available"
                if cuda_status == "CUDA not available":
                    cuda_implementation = "Not available"
                
            # Check if CUDA is available
            cuda_available = "Yes" if "cuda_init" in status and "Failed" not in status.get("cuda_init", "") else "No"
            
            # Check for implementation type in the examples
            examples = test_results.get("examples", [])
            for example in examples:
                if example.get("platform") == "CUDA":
                    cuda_implementation = example.get("implementation_type", cuda_implementation)
                    break
            
            results.append({
                "model": model_name,
                "cuda_available": cuda_available,
                "cuda_status": cuda_status,
                "implementation_type": cuda_implementation,
                "time": elapsed_time
            })
            
            print(f"  CUDA Status: {cuda_status}")
            print(f"  Implementation Type: {cuda_implementation}")
            
        except Exception as e:
            print(f"  ERROR: Failed to parse results for {model_name}: {e}")
            results.append({
                "model": model_name,
                "cuda_available": "Error",
                "cuda_status": f"Error: {str(e)}",
                "implementation_type": "Error",
                "time": elapsed_time
            })
    
    return results

def print_results_table(results):
    """Print a formatted table of results"""
    print("\n\n======= CUDA IMPLEMENTATION STATUS =======")
    print(f"{'Model':<20} {'CUDA Available':<15} {'Implementation':<15} {'Status':<30} {'Time (s)':<10}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['model']:<20} {result['cuda_available']:<15} {result['implementation_type']:<15} {result['cuda_status']:<30} {result['time']:.2f}")
    
    print("=" * 90)
    
    # Count real vs mock implementations
    real_count = sum(1 for r in results if r['implementation_type'] == "REAL" or "(REAL)" in str(r['implementation_type']))
    mock_count = sum(1 for r in results if r['implementation_type'] == "MOCK" or "(MOCK)" in str(r['implementation_type']))
    na_count = sum(1 for r in results if r['implementation_type'] == "Not available")
    error_count = sum(1 for r in results if r['implementation_type'] == "Error" or r['implementation_type'] == "Unknown")
    
    print(f"\nSummary: {len(results)} models tested")
    print(f"- REAL implementations: {real_count}")
    print(f"- MOCK implementations: {mock_count}")
    print(f"- Not available: {na_count}")
    print(f"- Error/Unknown: {error_count}")

if __name__ == "__main__":
    print("Testing CUDA implementation status for all models...")
    
    # Import torch and check CUDA availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    except ImportError:
        print("PyTorch not available, cannot check CUDA status directly")
    
    # Check CUDA status for all models
    results = check_cuda_status()
    
    # Print results table
    print_results_table(results)
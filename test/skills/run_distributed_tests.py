#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Distributed Tests

This script runs HuggingFace model tests in distributed mode using the
Distributed Testing Framework.

Usage:
    python run_distributed_tests.py --workers 4 --model-family bert
    python run_distributed_tests.py --all --workers 8
"""

import os
import sys
import json
import argparse
import importlib
import time
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from distributed_testing_framework import (
        Worker, 
        ResultCollector, 
        TaskDistributor, 
        HardwareDetector,
        get_registered_tests,
        get_tests_by_model_type
    )
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    print("Distributed Testing Framework not available. Installing stub implementation...")
    # Create the stub framework
    framework_dir = "distributed_testing_framework"
    if not os.path.exists(framework_dir):
        os.makedirs(framework_dir, exist_ok=True)
        os.system(f"python update_for_distributed_testing.py --create-framework")
    
    # Try importing again
    try:
        from distributed_testing_framework import (
            Worker, 
            ResultCollector, 
            TaskDistributor, 
            HardwareDetector,
            get_registered_tests,
            get_tests_by_model_type
        )
        FRAMEWORK_AVAILABLE = True
    except ImportError:
        print("Failed to install stub framework. Exiting.")
        sys.exit(1)

def find_test_files(directory="fixed_tests"):
    """Find all test files in the specified directory"""
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return []
    
    test_files = []
    for file in os.listdir(directory):
        if file.startswith("test_hf_") and file.endswith(".py"):
            test_files.append(os.path.join(directory, file))
    
    return sorted(test_files)

def get_model_families():
    """Get all model families from test files"""
    families = set()
    for test_file in find_test_files():
        # Extract the model family from the filename (test_hf_bert.py -> bert)
        family = os.path.basename(test_file).replace("test_hf_", "").replace(".py", "")
        families.add(family)
    return sorted(families)

def get_hardware_summary():
    """Get a summary of available hardware"""
    detector = HardwareDetector()
    hardware = detector.detect()
    
    summary = []
    for hw_type, available in hardware.items():
        if hw_type == "cpu":
            continue  # CPU is always available
        if isinstance(available, dict):
            if available.get("available", False):
                summary.append(hw_type)
        elif available:
            summary.append(hw_type)
    
    if not summary:
        return "CPU only"
    else:
        return f"CPU + {', '.join(summary)}"

def run_tests_parallel(test_files, workers=4, timeout=600):
    """Run tests in parallel using ThreadPoolExecutor"""
    results = []
    
    def run_test(test_file):
        """Run a single test file and return the result"""
        test_name = os.path.basename(test_file).replace(".py", "")
        print(f"Running test: {test_name}")
        
        try:
            # Get the module name from the file path
            module_name = os.path.basename(test_file).replace(".py", "")
            
            # Add the directory to the path if needed
            test_dir = os.path.dirname(test_file)
            if test_dir and test_dir not in sys.path:
                sys.path.insert(0, test_dir)
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the model name from the module
            model_name = module_name.replace("test_hf_", "")
            
            # Run the test function with distributed=True
            if hasattr(module, "run_test"):
                start_time = time.time()
                result = module.run_test(model_name, distributed=True, num_workers=workers)
                execution_time = time.time() - start_time
                
                return {
                    "test_name": test_name,
                    "model_name": model_name,
                    "success": True,
                    "execution_time": execution_time,
                    "result": result,
                    "hardware": get_hardware_summary()
                }
            else:
                return {
                    "test_name": test_name,
                    "model_name": module_name.replace("test_hf_", ""),
                    "success": False,
                    "error": "No run_test function found"
                }
                
        except Exception as e:
            return {
                "test_name": test_name,
                "model_name": os.path.basename(test_file).replace("test_hf_", "").replace(".py", ""),
                "success": False,
                "error": str(e)
            }
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_test, test_file): test_file for test_file in test_files}
        
        for future in as_completed(futures):
            test_file = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    print(f"✅ Completed: {result['test_name']} in {result.get('execution_time', 0):.2f}s")
                else:
                    print(f"❌ Failed: {result['test_name']} - {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                print(f"❌ Exception running {os.path.basename(test_file)}: {str(e)}")
                results.append({
                    "test_name": os.path.basename(test_file).replace(".py", ""),
                    "model_name": os.path.basename(test_file).replace("test_hf_", "").replace(".py", ""),
                    "success": False,
                    "error": str(e)
                })
    
    return results

def generate_dashboard_data(results, output_dir="distributed_results"):
    """Generate dashboard data visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"distributed_test_results_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    summary_file = os.path.join(output_dir, f"distributed_test_summary_{timestamp}.md")
    
    successful_tests = [r for r in results if r.get("success", False)]
    failed_tests = [r for r in results if not r.get("success", False)]
    
    with open(summary_file, "w") as f:
        f.write(f"# Distributed Test Run Summary\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Overview\n\n")
        f.write(f"- **Total Tests**: {len(results)}\n")
        f.write(f"- **Successful Tests**: {len(successful_tests)} ({len(successful_tests)/len(results)*100:.1f}%)\n")
        f.write(f"- **Failed Tests**: {len(failed_tests)} ({len(failed_tests)/len(results)*100:.1f}%)\n")
        f.write(f"- **Hardware Used**: {get_hardware_summary()}\n\n")
        
        # Performance statistics
        if successful_tests:
            execution_times = [r.get("execution_time", 0) for r in successful_tests if "execution_time" in r]
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                
                f.write(f"## Performance Statistics\n\n")
                f.write(f"- **Average Execution Time**: {avg_time:.2f}s\n")
                f.write(f"- **Maximum Execution Time**: {max_time:.2f}s\n")
                f.write(f"- **Minimum Execution Time**: {min_time:.2f}s\n\n")
                
                # Find slowest and fastest tests
                slowest_test = max(successful_tests, key=lambda r: r.get("execution_time", 0))
                fastest_test = min(successful_tests, key=lambda r: r.get("execution_time", 0))
                
                f.write(f"**Slowest Test**: {slowest_test['test_name']} ({slowest_test.get('execution_time', 0):.2f}s)\n")
                f.write(f"**Fastest Test**: {fastest_test['test_name']} ({fastest_test.get('execution_time', 0):.2f}s)\n\n")
        
        # Failed tests details
        if failed_tests:
            f.write(f"## Failed Tests\n\n")
            f.write("| Test | Error |\n")
            f.write("|------|-------|\n")
            
            for test in failed_tests:
                f.write(f"| {test['test_name']} | {test.get('error', 'Unknown error')} |\n")
            
            f.write("\n")
        
        # Successful tests
        f.write(f"## Successful Tests\n\n")
        f.write("| Test | Execution Time (s) | Hardware |\n")
        f.write("|------|-------------------|----------|\n")
        
        # Sort by execution time
        sorted_tests = sorted(successful_tests, key=lambda r: r.get("execution_time", 0), reverse=True)
        
        for test in sorted_tests:
            execution_time = test.get("execution_time", "N/A")
            if isinstance(execution_time, (int, float)):
                execution_time = f"{execution_time:.2f}"
            
            hardware = test.get("hardware", "Unknown")
            f.write(f"| {test['test_name']} | {execution_time} | {hardware} |\n")
    
    print(f"Results saved to: {results_file}")
    print(f"Summary report generated: {summary_file}")
    
    return summary_file

def run_distributed_tests(model_family=None, workers=4, timeout=600, all_models=False, test_dir="fixed_tests"):
    """Run tests in distributed mode"""
    if not FRAMEWORK_AVAILABLE:
        print("Distributed Testing Framework not available.")
        return
    
    start_time = time.time()
    test_files = find_test_files(test_dir)
    
    if not test_files:
        print(f"No test files found in {test_dir}.")
        return
    
    print(f"Found {len(test_files)} test files in {test_dir}.")
    
    # Filter test files by model family if specified
    if model_family and not all_models:
        test_files = [f for f in test_files if f"test_hf_{model_family}" in f]
        if not test_files:
            print(f"No test files found for model family: {model_family}")
            return
    
    print(f"Running {len(test_files)} tests with {workers} workers...")
    
    # Run tests in parallel
    results = run_tests_parallel(test_files, workers, timeout)
    
    # Generate dashboard data
    summary_file = generate_dashboard_data(results)
    
    # Print final summary
    total_time = time.time() - start_time
    successful_tests = [r for r in results if r.get("success", False)]
    failed_tests = [r for r in results if not r.get("success", False)]
    
    print(f"\nDistributed Test Run Complete:")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Summary written to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Run HuggingFace tests in distributed mode")
    
    # Model selection args
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-family", type=str, help="Model family to test (e.g., bert, gpt2)")
    model_group.add_argument("--all", action="store_true", help="Test all available models")
    
    # Distributed testing args
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for distributed testing")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for each test")
    parser.add_argument("--test-dir", type=str, default="fixed_tests", help="Directory containing test files")
    
    # Framework management
    parser.add_argument("--update-tests", action="store_true", help="Update test files for distributed testing")
    parser.add_argument("--list-models", action="store_true", help="List available model families")
    parser.add_argument("--hardware-check", action="store_true", help="Check available hardware")
    
    args = parser.parse_args()
    
    # Update test files if requested
    if args.update_tests:
        os.system(f"python update_for_distributed_testing.py --dir {args.test_dir} --verify")
        return
    
    # List model families if requested
    if args.list_models:
        families = get_model_families()
        print("Available model families:")
        for family in families:
            print(f"- {family}")
        return
    
    # Check hardware if requested
    if args.hardware_check:
        detector = HardwareDetector()
        hardware = detector.detect()
        
        print("Available Hardware:")
        for hw_type, info in hardware.items():
            if isinstance(info, dict):
                status = "Available" if info.get("available", False) else "Not Available"
                print(f"- {hw_type}: {status}")
                if hw_type != "cpu" and info.get("available", False) and "name" in info:
                    print(f"  - Name: {info['name']}")
            else:
                print(f"- {hw_type}: {'Available' if info else 'Not Available'}")
        print(f"\nHardware Summary: {get_hardware_summary()}")
        return
    
    # Run distributed tests
    run_distributed_tests(
        model_family=args.model_family,
        workers=args.workers,
        timeout=args.timeout,
        all_models=args.all,
        test_dir=args.test_dir
    )

if __name__ == "__main__":
    main()
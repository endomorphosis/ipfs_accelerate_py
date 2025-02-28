#!/usr/bin/env python3
"""
Script to run performance tests for all skills on CPU, OpenVINO, and CUDA platforms.
Collects and reports performance metrics for each skill and platform.
"""

import os
import sys
import time
import json
import datetime
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Test files to run (excluding special test files or utility modules)
SKILL_TEST_FILES = [
    "skills/test_hf_bert.py",
    "skills/test_hf_clip.py",
    "skills/test_hf_llama.py",
    "skills/test_hf_t5.py",
    "skills/test_hf_wav2vec2.py",
    "skills/test_hf_whisper.py",
    "skills/test_hf_xclip.py",
    "skills/test_hf_clap.py",
    "skills/test_default_embed.py",
    "skills/test_default_lm.py",
    "skills/test_hf_llava.py",
    "skills/test_hf_llava_next.py"
]

def run_test_file(test_file: str) -> Dict[str, Any]:
    """
    Run a single test file and capture its results.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        Dict containing test results and performance metrics
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
        module = importlib.import_module(module_path)
        
        # Get the test class
        test_class = getattr(module, class_name)
        
        # Create an instance of the test class
        test_instance = test_class()
        
        # Run the test and capture results
        start_time = time.time()
        results = test_instance.test()
        total_time = time.time() - start_time
        
        # Add total execution time to results
        results["total_execution_time"] = total_time
        
        print(f"\nTest completed in {total_time:.2f} seconds")
        return {
            "test_file": test_file,
            "status": "Success",
            "results": results,
            "execution_time": total_time
        }
        
    except Exception as e:
        print(f"Error running test {test_file}: {e}")
        traceback.print_exc()
        return {
            "test_file": test_file,
            "status": "Error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def extract_performance_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract performance metrics from test results.
    
    Args:
        results: Test results dictionary
        
    Returns:
        Dict containing extracted performance metrics
    """
    metrics = {}
    
    # Skip if there was an error
    if results.get("status") != "Success" or not results.get("results"):
        return {"error": "No valid results to extract metrics from"}
    
    test_results = results["results"]
    
    # Extract platform-specific metrics
    platforms = ["cpu", "cuda", "openvino", "apple", "qualcomm"]
    
    for platform in platforms:
        platform_metrics = {}
        
        # Check for platform initialization status
        for key in test_results:
            if key.startswith(f"{platform}_"):
                platform_metrics[key] = test_results[key]
        
        # Extract implementation type
        implementation_type = None
        for key in test_results:
            if key.startswith(f"{platform}_") and "Success" in str(test_results[key]):
                # Parse implementation type from status string (e.g., "Success (REAL)")
                status = test_results[key]
                if "(" in status and ")" in status:
                    implementation_type = status.split("(")[1].split(")")[0]
                    break
        
        if implementation_type:
            platform_metrics["implementation_type"] = implementation_type
        
        # Extract platform examples (may contain performance metrics)
        examples = []
        for key in test_results:
            if key.endswith("_example") and platform in key:
                examples.append(test_results[key])
        
        if examples:
            platform_metrics["examples"] = examples
            
            # Extract performance data from examples
            for example in examples:
                if "performance" in example and example["performance"]:
                    platform_metrics["performance_metrics"] = example["performance"]
                    break
        
        # Extract capabilities info if available
        if f"{platform}_capabilities" in test_results:
            platform_metrics["capabilities"] = test_results[f"{platform}_capabilities"]
        
        # Add to metrics if we found any data
        if platform_metrics:
            metrics[platform] = platform_metrics
    
    return metrics

def run_all_tests() -> Dict[str, Any]:
    """
    Run all tests and collect performance metrics.
    
    Returns:
        Dict containing all test results and metrics
    """
    all_results = {}
    
    for test_file in SKILL_TEST_FILES:
        print(f"\nTesting {test_file}...")
        
        # Run test and get results
        results = run_test_file(test_file)
        
        # Extract performance metrics
        performance_metrics = extract_performance_metrics(results)
        
        # Store results and metrics
        all_results[test_file] = {
            "status": results.get("status"),
            "execution_time": results.get("execution_time"),
            "performance_metrics": performance_metrics
        }
        
        # If there was an error, store the error details
        if results.get("status") == "Error":
            all_results[test_file]["error"] = results.get("error")
            all_results[test_file]["traceback"] = results.get("traceback")
    
    return all_results

def save_results(results: Dict[str, Any], filename: str = None) -> str:
    """
    Save test results to a JSON file.
    
    Args:
        results: Test results to save
        filename: Optional filename to use
        
    Returns:
        Path to the saved file
    """
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_results_{timestamp}.json"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "performance_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to output file
    output_path = os.path.join(output_dir, filename)
    
    # Add metadata
    results_with_metadata = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "test_count": len(results),
            "test_files": list(results.keys())
        },
        "results": results
    }
    
    # Save results to file
    with open(output_path, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output_path

def summarize_results(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Generate a summary of test results.
    
    Args:
        results: Test results
        
    Returns:
        Dict containing summary information
    """
    summary = {
        "cpu": {"success": 0, "error": 0, "real_impl": 0, "mock_impl": 0},
        "cuda": {"success": 0, "error": 0, "real_impl": 0, "mock_impl": 0},
        "openvino": {"success": 0, "error": 0, "real_impl": 0, "mock_impl": 0},
        "total_execution_time": 0
    }
    
    for test_file, test_data in results.items():
        # Add execution time to total
        if "execution_time" in test_data:
            summary["total_execution_time"] += test_data["execution_time"]
        
        # Process each platform's metrics
        for platform in ["cpu", "cuda", "openvino"]:
            if platform not in test_data.get("performance_metrics", {}):
                continue
            
            platform_metrics = test_data["performance_metrics"][platform]
            
            # Check for success or failure in platform metrics
            success = False
            for key, value in platform_metrics.items():
                if isinstance(value, str) and "Success" in value:
                    success = True
                    summary[platform]["success"] += 1
                    break
            
            if not success and not platform_metrics.get("error"):
                for key, value in platform_metrics.items():
                    if isinstance(value, str) and "Failed" in value or "Error" in value:
                        summary[platform]["error"] += 1
                        break
            
            # Check implementation type
            impl_type = platform_metrics.get("implementation_type", "")
            if impl_type == "REAL":
                summary[platform]["real_impl"] += 1
            elif impl_type == "MOCK":
                summary[platform]["mock_impl"] += 1
    
    return summary

def print_summary(summary: Dict[str, Dict[str, Any]]):
    """
    Print a summary of test results.
    
    Args:
        summary: Summary information
    """
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    # Print platform-specific summaries
    for platform in ["cpu", "cuda", "openvino"]:
        platform_summary = summary[platform]
        print(f"\n{platform.upper()} Implementation:")
        print(f"  Success: {platform_summary['success']}")
        print(f"  Errors: {platform_summary['error']}")
        print(f"  Real Implementations: {platform_summary['real_impl']}")
        print(f"  Mock Implementations: {platform_summary['mock_impl']}")
    
    # Print total execution time
    print(f"\nTotal Execution Time: {summary['total_execution_time']:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    print("\nRunning performance tests for all skills...\n")
    
    # Run all tests and collect results
    all_results = run_all_tests()
    
    # Save results to file
    output_path = save_results(all_results)
    
    # Summarize and print results
    summary = summarize_results(all_results)
    print_summary(summary)
    
    print(f"\nDetailed results saved to: {output_path}")
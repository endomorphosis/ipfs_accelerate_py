#!/usr/bin/env python3
"""
Run performance tests for all models to ensure they use real implementations.
This script tests CPU, CUDA and OpenVINO backends for all models.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import datetime
from pathlib import Path

# Define the base directory
BASE_DIR = Path("/home/barberb/ipfs_accelerate_py/test")
SKILLS_DIR = BASE_DIR / "skills"
APIS_DIR = BASE_DIR / "apis"
PERFORMANCE_RESULTS_DIR = BASE_DIR / "performance_results"

# Create performance results directory if it doesn't exist
PERFORMANCE_RESULTS_DIR.mkdir(exist_ok=True)

# Define the list of skill tests to run
SKILL_TESTS = [
    "test_hf_bert.py",
    "test_hf_clip.py",
    "test_hf_llama.py",
    "test_hf_t5.py",
    "test_hf_wav2vec2.py",
    "test_hf_whisper.py",
    "test_hf_xclip.py",
    "test_hf_clap.py",
    "test_default_embed.py",
    "test_default_lm.py",
]

# Define the list of API tests to run
API_TESTS = [
    "test_claude.py",
    "test_gemini.py",
    "test_groq.py",
    "test_ollama.py",
    "test_openai_api.py",
]

def run_test(test_file, test_dir, timeout=600):
    """Run a single test and return its results
    
    Args:
        test_file (str): Name of the test file
        test_dir (Path): Directory containing the test file
        timeout (int): Timeout in seconds
        
    Returns:
        dict: Test results or error message
    """
    print(f"Running test: {test_file}", flush=True)
    test_path = test_dir / test_file
    
    try:
        # Run the test with timeout
        start_time = time.time()
        cmd = [sys.executable, str(test_path)]
        
        # Create a new process to run the test
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        
        # Wait for the process to complete or timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return_code = -1
            
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            print(f"✅ Test passed: {test_file} in {elapsed_time:.2f}s", flush=True)
        else:
            print(f"❌ Test failed: {test_file} (code {return_code})", flush=True)
            
        # Try to extract JSON results from stdout
        result = {
            "test_file": test_file,
            "elapsed_time": elapsed_time,
            "return_code": return_code,
            "stdout": stdout[:1000] + "..." if len(stdout) > 1000 else stdout,
            "stderr": stderr[:1000] + "..." if len(stderr) > 1000 else stderr,
        }
        
        # Try to parse any JSON in the output
        try:
            # Look for JSON content in the output
            json_start = stdout.find("{")
            json_end = stdout.rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                json_content = stdout[json_start:json_end+1]
                parsed_json = json.loads(json_content)
                result["json_result"] = parsed_json
        except Exception as json_err:
            result["json_parse_error"] = str(json_err)
        
        return result
        
    except Exception as e:
        print(f"Error running test {test_file}: {e}", flush=True)
        return {
            "test_file": test_file,
            "error": str(e)
        }

def run_all_tests(args):
    """Run all the specified tests and save results
    
    Args:
        args: Command-line arguments
    """
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "command": " ".join(sys.argv),
    }
    
    # Run skill tests if selected
    if args.skills:
        skill_results = {}
        for test_file in SKILL_TESTS:
            if args.filter and args.filter not in test_file:
                continue
            skill_results[test_file] = run_test(test_file, SKILLS_DIR, args.timeout)
        results["skill_tests"] = skill_results
    
    # Run API tests if selected
    if args.apis:
        api_results = {}
        for test_file in API_TESTS:
            if args.filter and args.filter not in test_file:
                continue
            api_results[test_file] = run_test(test_file, APIS_DIR, args.timeout)
        results["api_tests"] = api_results
    
    # Save results with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_type = []
    if args.skills:
        test_type.append("skills")
    if args.apis:
        test_type.append("apis")
    
    filter_suffix = f"_{args.filter}" if args.filter else ""
    results_filename = f"performance_test_{'_'.join(test_type)}{filter_suffix}_{timestamp}.json"
    results_path = PERFORMANCE_RESULTS_DIR / results_filename
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\nTest Summary:")
    
    if args.skills:
        print("\nSkill Tests:")
        for test_file, result in results["skill_tests"].items():
            status = "✅ PASSED" if result.get("return_code", -1) == 0 else "❌ FAILED"
            time_str = f"({result.get('elapsed_time', 0):.2f}s)" if "elapsed_time" in result else ""
            print(f"{status} {test_file} {time_str}")
    
    if args.apis:
        print("\nAPI Tests:")
        for test_file, result in results["api_tests"].items():
            status = "✅ PASSED" if result.get("return_code", -1) == 0 else "❌ FAILED"
            time_str = f"({result.get('elapsed_time', 0):.2f}s)" if "elapsed_time" in result else ""
            print(f"{status} {test_file} {time_str}")
    
    # Generate performance metrics report
    generate_report(results, args)

def get_implementation_status(result):
    """Extract implementation status from test results
    
    Args:
        result (dict): Test result
        
    Returns:
        tuple: (cpu_status, cuda_status, openvino_status)
    """
    # Default to unknown/mock status
    cpu_status = "UNKNOWN"
    cuda_status = "UNKNOWN"
    openvino_status = "UNKNOWN"
    
    try:
        # Try to get status from nested JSON result
        json_result = result.get("json_result", {})
        
        # First check for status dict
        status = json_result.get("status", {})
        if status:
            # Look for CPU status
            for key in status:
                if "cpu_init" in key or "cpu_handler" in key:
                    if "REAL" in status[key]:
                        cpu_status = "REAL"
                    elif "MOCK" in status[key]:
                        cpu_status = "MOCK"
                        
            # Look for CUDA status
            for key in status:
                if "cuda_init" in key or "cuda_handler" in key:
                    if "REAL" in status[key]:
                        cuda_status = "REAL"
                    elif "MOCK" in status[key]:
                        cuda_status = "MOCK"
                        
            # Look for OpenVINO status
            for key in status:
                if "openvino_init" in key or "openvino_handler" in key:
                    if "REAL" in status[key]:
                        openvino_status = "REAL"
                    elif "MOCK" in status[key]:
                        openvino_status = "MOCK"
        
        # Fallback to checking in examples
        examples = json_result.get("examples", [])
        if examples:
            for example in examples:
                platform = example.get("platform", "")
                impl_type = example.get("implementation_type", example.get("implementation", ""))
                
                if platform == "CPU" and "REAL" in impl_type:
                    cpu_status = "REAL"
                elif platform == "CUDA" and "REAL" in impl_type:
                    cuda_status = "REAL"
                elif platform == "OpenVINO" and "REAL" in impl_type:
                    openvino_status = "REAL"
    except Exception as e:
        print(f"Error extracting implementation status: {e}")
    
    return (cpu_status, cuda_status, openvino_status)

def generate_report(results, args):
    """Generate a report on all test results
    
    Args:
        results (dict): Combined test results
        args: Command-line arguments
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"# Performance Test Report - {timestamp}\n\n"
    report += "## Overview\n\n"
    report += "| Model | CPU Status | CUDA Status | OpenVINO Status | Notes |\n"
    report += "|-------|------------|-------------|-----------------|-------|\n"
    
    # Add skill tests to report
    skill_results = results.get("skill_tests", {})
    for test_file, result in skill_results.items():
        # Extract model name from test file
        model_name = test_file.replace("test_", "").replace(".py", "")
        model_name = model_name.replace("hf_", "").replace("default_", "")
        model_name = model_name.upper()
        
        # Get implementation status
        cpu_status, cuda_status, openvino_status = get_implementation_status(result)
        
        # Add to report table
        report += f"| {model_name} | {cpu_status} | {cuda_status} | {openvino_status} | |\n"
    
    # Add additional sections
    report += "\n## Implementation Details\n\n"
    report += "The following models had improvements in their implementations:\n\n"
    
    for test_file, result in skill_results.items():
        model_name = test_file.replace("test_", "").replace(".py", "")
        model_name = model_name.replace("hf_", "").replace("default_", "")
        model_name = model_name.upper()
        
        cpu_status, cuda_status, openvino_status = get_implementation_status(result)
        
        # Check if any of them is REAL
        if "REAL" in [cpu_status, cuda_status, openvino_status]:
            report += f"### {model_name}\n"
            if cpu_status == "REAL":
                report += "- Successfully using REAL implementation on CPU\n"
            if cuda_status == "REAL":
                report += "- Successfully using REAL implementation on CUDA\n"
            if openvino_status == "REAL":
                report += "- Successfully using REAL implementation on OpenVINO\n"
            
            # Try to extract more details from the result
            try:
                json_result = result.get("json_result", {})
                
                # Get model name actually used
                metadata = json_result.get("metadata", {})
                test_model = metadata.get("test_model", "")
                if test_model:
                    report += f"- Using model: {test_model}\n"
                
                # Check performance metrics
                examples = json_result.get("examples", [])
                if examples:
                    for example in examples:
                        platform = example.get("platform", "")
                        if platform and "elapsed_time" in example:
                            report += f"- {platform} Execution time: {example['elapsed_time']:.4f}s\n"
            except Exception as e:
                print(f"Error extracting additional details: {e}")
            
            report += "\n"
    
    # Save the report
    timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_suffix = f"_{args.filter}" if args.filter else ""
    report_filename = f"performance_report{filter_suffix}_{timestamp_file}.md"
    report_path = PERFORMANCE_RESULTS_DIR / report_filename
    
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Performance report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run performance tests for all models")
    parser.add_argument("--skills", action="store_true", help="Run skill tests")
    parser.add_argument("--apis", action="store_true", help="Run API tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--filter", type=str, help="Only run tests containing this string")
    parser.add_argument("--timeout", type=int, default=600, help="Test timeout in seconds (default: 600)")
    
    args = parser.parse_args()
    
    # If no specific test type is selected, run all tests
    if not args.skills and not args.apis and not args.all:
        args.skills = True  # Default to skills
    
    # If --all is specified, run both skill and API tests
    if args.all:
        args.skills = True
        args.apis = True
    
    run_all_tests(args)

if __name__ == "__main__":
    main()
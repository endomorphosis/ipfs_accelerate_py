#!/usr/bin/env python3
"""
Script to run tests for critical Hugging Face models with unique capabilities.
This script targets models that implement unique pipeline tasks or special features.
"""

import os
import sys
import json
import glob
import datetime
import subprocess
import argparse
from pathlib import Path

# List of critical models with unique capabilities to test
CRITICAL_MODELS = [
    "tapas",              # table-question-answering (only model)
    "esm",                # protein-folding (only model)
    "patchtst",           # time-series-prediction (most efficient)
    "informer",           # time-series-prediction (popular forecasting)
    "autoformer",         # time-series-prediction (alternative architecture)
    "depth_anything",     # depth-estimation (zero-shot)
    "dpt",                # depth-estimation (general purpose)
    "zoedepth",           # depth-estimation (state-of-the-art)
    "visual_bert",        # visual-question-answering (foundation)
]

def run_tests(models=None, timeout=300, skip_missing=True, output_dir="critical_model_results"):
    """
    Run tests for specified models or all critical models.
    
    Args:
        models (list, optional): List of model names to test
        timeout (int): Timeout in seconds for each test
        skip_missing (bool): Skip models without test files
        output_dir (str): Directory to save results
        
    Returns:
        dict: Test results
    """
    # Use provided models or default to all critical models
    target_models = models if models else CRITICAL_MODELS
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "test_timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total": len(target_models),
            "tested": 0,
            "passed": 0,
            "failed": 0,
            "missing": 0
        },
        "model_results": {}
    }
    
    # Run tests for each model
    for model in target_models:
        normalized_name = model.replace('-', '_').replace('.', '_').lower()
        test_file = f"skills/test_hf_{normalized_name}.py"
        
        print(f"\n{'='*60}")
        print(f"Testing {model}...")
        print(f"{'='*60}")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            results["summary"]["missing"] += 1
            results["model_results"][model] = {
                "status": "missing",
                "message": f"Test file not found: {test_file}"
            }
            if skip_missing:
                continue
        
        # Run the test
        try:
            start_time = datetime.datetime.now()
            
            # Run the test file as a subprocess with timeout
            process = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Check if test passed
            if process.returncode == 0:
                print(f"✅ Test passed for {model}")
                results["summary"]["passed"] += 1
                results["summary"]["tested"] += 1
                
                # Parse the output to extract platform status
                platform_status = {
                    "cpu": "unknown",
                    "cuda": "unknown",
                    "openvino": "unknown"
                }
                
                # Look for status indicators in output
                output = process.stdout
                for line in output.splitlines():
                    if "CPU_STATUS:" in line:
                        platform_status["cpu"] = line.split("CPU_STATUS:")[1].strip()
                    elif "CUDA_STATUS:" in line:
                        platform_status["cuda"] = line.split("CUDA_STATUS:")[1].strip()
                    elif "OPENVINO_STATUS:" in line:
                        platform_status["openvino"] = line.split("OPENVINO_STATUS:")[1].strip()
                
                # Look for structured_results JSON if present
                structured_data = {}
                if "structured_results" in output:
                    try:
                        json_part = output.split("structured_results")[1].strip()
                        structured_data = json.loads(json_part)
                    except Exception as e:
                        print(f"Warning: Failed to parse structured results: {e}")
                
                # Save result
                results["model_results"][model] = {
                    "status": "passed",
                    "return_code": process.returncode,
                    "elapsed_time": elapsed_time,
                    "platform_status": platform_status,
                    "structured_data": structured_data
                }
                
                # Display platform status
                for platform, status in platform_status.items():
                    status_icon = "✅" if status == "REAL" else "⚠️" if status == "MOCK" else "❌"
                    print(f"  {status_icon} {platform.upper()}: {status}")
            else:
                print(f"❌ Test failed for {model}")
                results["summary"]["failed"] += 1
                results["summary"]["tested"] += 1
                
                # Save failure info
                results["model_results"][model] = {
                    "status": "failed",
                    "return_code": process.returncode,
                    "elapsed_time": elapsed_time,
                    "stdout": process.stdout,
                    "stderr": process.stderr
                }
                
                # Display error
                if process.stderr:
                    error_lines = process.stderr.splitlines()
                    print(f"  Error: {error_lines[-1] if error_lines else 'Unknown error'}")
        
        except subprocess.TimeoutExpired:
            print(f"⏱️ Test timed out for {model} after {timeout} seconds")
            results["summary"]["failed"] += 1
            results["summary"]["tested"] += 1
            
            results["model_results"][model] = {
                "status": "timeout",
                "elapsed_time": timeout,
                "message": f"Test timed out after {timeout} seconds"
            }
        
        except Exception as e:
            print(f"❌ Exception running test for {model}: {e}")
            results["summary"]["failed"] += 1
            results["summary"]["tested"] += 1
            
            results["model_results"][model] = {
                "status": "error",
                "message": str(e)
            }
    
    # Calculate summary percentages
    total = results["summary"]["total"]
    tested = results["summary"]["tested"]
    passed = results["summary"]["passed"]
    if tested > 0:
        results["summary"]["pass_rate"] = f"{(passed / tested) * 100:.1f}%"
    else:
        results["summary"]["pass_rate"] = "0.0%"
    
    # Save results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"critical_model_tests_{timestamp}.json")
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    # Generate summary report
    report_file = os.path.join(output_dir, f"critical_model_report_{timestamp}.md")
    generate_report(results, report_file)
    
    print(f"Report saved to: {report_file}")
    
    return results

def generate_report(results, output_file):
    """
    Generate a Markdown report from test results.
    
    Args:
        results (dict): Test results dictionary
        output_file (str): Path to save the report
    """
    # Extract summary stats
    total = results["summary"]["total"]
    tested = results["summary"]["tested"]
    passed = results["summary"]["passed"]
    failed = results["summary"]["failed"]
    missing = results["summary"]["missing"]
    pass_rate = results["summary"]["pass_rate"]
    timestamp = results["test_timestamp"]
    
    # Create report header
    report = f"# Critical Hugging Face Model Test Report\n\n"
    report += f"Generated: {timestamp}\n\n"
    
    # Add summary section
    report += f"## Summary\n\n"
    report += f"- **Total Models**: {total}\n"
    report += f"- **Tested**: {tested} ({tested/total*100:.1f}% of total)\n"
    report += f"- **Passed**: {passed} ({pass_rate} of tested)\n"
    report += f"- **Failed**: {failed} ({failed/tested*100:.1f}% of tested if tested > 0 else 0}%)\n"
    report += f"- **Missing Tests**: {missing} ({missing/total*100:.1f}% of total)\n\n"
    
    # Add results table
    report += f"## Model Results\n\n"
    report += "| Model | Status | CPU | CUDA | OpenVINO | Elapsed Time |\n"
    report += "|-------|--------|-----|------|----------|---------------|\n"
    
    for model, result in results["model_results"].items():
        status = result["status"]
        
        if status == "passed":
            status_icon = "✅"
            platform_status = result["platform_status"]
            cpu = platform_status.get("cpu", "unknown")
            cuda = platform_status.get("cuda", "unknown")
            openvino = platform_status.get("openvino", "unknown")
            elapsed = f"{result.get('elapsed_time', 0):.1f}s"
        elif status == "failed":
            status_icon = "❌"
            cpu = "N/A"
            cuda = "N/A"
            openvino = "N/A"
            elapsed = f"{result.get('elapsed_time', 0):.1f}s"
        elif status == "timeout":
            status_icon = "⏱️"
            cpu = "Timeout"
            cuda = "Timeout"
            openvino = "Timeout"
            elapsed = f"{result.get('elapsed_time', 0):.1f}s"
        else:  # missing
            status_icon = "⚠️"
            cpu = "N/A"
            cuda = "N/A"
            openvino = "N/A"
            elapsed = "N/A"
        
        report += f"| {model} | {status_icon} {status.capitalize()} | {cpu} | {cuda} | {openvino} | {elapsed} |\n"
    
    # Add details section for failures
    failures = {model: result for model, result in results["model_results"].items() 
               if result["status"] in ["failed", "timeout", "error"]}
    
    if failures:
        report += f"\n## Failure Details\n\n"
        
        for model, result in failures.items():
            report += f"### {model}\n\n"
            report += f"- **Status**: {result['status'].capitalize()}\n"
            
            if result["status"] == "failed":
                # Extract error message from stderr
                stderr = result.get("stderr", "")
                error_lines = stderr.splitlines()
                if error_lines:
                    report += f"- **Error**: `{error_lines[-1]}`\n"
                    
                # Include relevant traceback snippet
                if len(error_lines) > 5:
                    report += f"\n```python\n"
                    report += "\n".join(error_lines[-5:])
                    report += f"\n```\n"
            
            elif result["status"] == "timeout":
                report += f"- **Error**: Test exceeded timeout of {result.get('elapsed_time')} seconds\n"
            
            elif result["status"] == "error":
                report += f"- **Error**: {result.get('message', 'Unknown error')}\n"
                
            report += "\n"
    
    # Add missing tests section
    missing_tests = {model: result for model, result in results["model_results"].items() 
                    if result["status"] == "missing"}
    
    if missing_tests:
        report += f"\n## Missing Tests\n\n"
        report += "The following models need test implementations:\n\n"
        
        for model in missing_tests:
            report += f"- `{model}`\n"
    
    # Write report to file
    with open(output_file, "w") as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description="Run tests for critical Hugging Face models")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to test (default: all critical models)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each test (default: 300)")
    parser.add_argument("--skip-missing", action="store_true", help="Skip models without test files")
    parser.add_argument("--output-dir", type=str, default="critical_model_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    print(f"Starting critical model tests at {datetime.datetime.now().isoformat()}")
    results = run_tests(
        models=args.models,
        timeout=args.timeout,
        skip_missing=args.skip_missing,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total models: {results['summary']['total']}")
    print(f"Tested: {results['summary']['tested']}")
    print(f"Passed: {results['summary']['passed']} ({results['summary']['pass_rate']})")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Missing: {results['summary']['missing']}")
    print("="*60)
    print("Complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Combined test runner for queue and backoff implementation tests.
This script runs:
1. API backoff tests for all available APIs
2. Comprehensive Ollama queue and backoff tests
3. Enhanced API multiplexing tests
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def get_all_apis():
    """Get all available API backends"""
    return [
        "openai", "groq", "claude", "gemini", 
        "ollama", "hf_tgi", "hf_tei", 
        "vllm", "opea", "ovms", "s3_kit"
    ]

def run_test_script(script_path, args=None, description=None):
    """Run a test script and return its result code"""
    if description:
        print(f"\n=== {description} ===")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # Run the script as a subprocess
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds with exit code {process.returncode}")
        
        # Print stdout and stderr
        if process.stdout:
            print("\nOutput:")
            print(process.stdout)
        
        if process.returncode != 0 and process.stderr:
            print("\nErrors:")
            print(process.stderr)
        
        return process.returncode
    
    except Exception as e:
        print(f"Error running script: {str(e)}")
        return 1

def run_api_backoff_tests(args):
    """Run API backoff tests for selected APIs"""
    print("\n=== Running API Backoff Queue Tests ===")
    
    # Default to all APIs if none specified
    apis_to_test = args.apis if args.apis else get_all_apis()
    
    if args.skip_apis:
        apis_to_test = [api for api in apis_to_test if api not in args.skip_apis]
    
    print(f"Testing APIs: {', '.join(apis_to_test)}")
    
    results = {}
    
    for api in apis_to_test:
        print(f"\n--- Testing {api.upper()} API backoff ---")
        
        # Build command line arguments
        test_args = ["--api", api]
        
        # Add API key if available in environment
        env_var = None
        if api == "openai":
            env_var = "OPENAI_API_KEY"
        elif api == "groq":
            env_var = "GROQ_API_KEY"
        elif api == "claude":
            env_var = "ANTHROPIC_API_KEY"
        elif api == "gemini":
            env_var = "GOOGLE_API_KEY"
        
        api_key = os.environ.get(env_var, "") if env_var else ""
        if api_key:
            test_args.extend(["--key", api_key])
        
        # Run the test
        result = run_test_script(
            "test_api_backoff_queue.py",
            test_args,
            f"Testing {api.upper()} API backoff and queue"
        )
        
        results[api] = {
            "exit_code": result,
            "status": "Success" if result == 0 else "Failed"
        }
    
    # Generate summary
    success_count = sum(1 for api, data in results.items() if data["status"] == "Success")
    print(f"\nAPI Backoff Tests Summary: {success_count}/{len(results)} APIs successful")
    
    for api, data in results.items():
        status_symbol = "✓" if data["status"] == "Success" else "✗"
        print(f"{status_symbol} {api.upper()}: {data['status']}")
    
    return results

def run_ollama_backoff_tests(args):
    """Run comprehensive Ollama backoff tests"""
    print("\n=== Running Comprehensive Ollama Backoff Tests ===")
    
    # Check if Ollama tests are enabled
    if args.skip_apis and "ollama" in args.skip_apis:
        print("Ollama tests are disabled with --skip-apis")
        return {"status": "Skipped"}
    
    # Build command line arguments
    test_args = []
    
    # Add model if specified
    if args.ollama_model:
        test_args.extend(["--model", args.ollama_model])
    
    # Add host if specified
    if args.ollama_host:
        test_args.extend(["--host", args.ollama_host])
    
    # Run the test
    result = run_test_script(
        "test_ollama_backoff_comprehensive.py",
        test_args,
        "Comprehensive Ollama API queue and backoff tests"
    )
    
    return {
        "exit_code": result,
        "status": "Success" if result == 0 else "Failed"
    }

def run_multiplexing_tests(args):
    """Run enhanced API multiplexing tests"""
    print("\n=== Running Enhanced API Multiplexing Tests ===")
    
    # Check if required APIs are available
    required_apis = ["openai", "groq"]
    
    if args.skip_apis and all(api in args.skip_apis for api in required_apis):
        print("API multiplexing tests require at least one of these APIs: " + 
              ", ".join(required_apis))
        return {"status": "Skipped"}
    
    # Build command line arguments
    test_args = []
    
    # Add quiet flag if specified
    if args.quiet:
        test_args.append("--quiet")
    
    # Run the test
    result = run_test_script(
        "test_api_multiplexing_enhanced.py",
        test_args,
        "Enhanced API key multiplexing tests"
    )
    
    return {
        "exit_code": result,
        "status": "Success" if result == 0 else "Failed"
    }

def main():
    parser = argparse.ArgumentParser(description="Run queue and backoff implementation tests")
    parser.add_argument("--apis", "-a", nargs="+", choices=get_all_apis(),
                       help="Specific APIs to test (default: all)")
    parser.add_argument("--skip-apis", "-s", nargs="+", choices=get_all_apis(),
                       help="APIs to skip")
    parser.add_argument("--ollama-model", "-m", default="llama3",
                       help="Model to use for Ollama tests (default: llama3)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama API host (default: http://localhost:11434)")
    parser.add_argument("--no-api-tests", action="store_true",
                       help="Skip standard API backoff tests")
    parser.add_argument("--no-ollama-tests", action="store_true",
                       help="Skip Ollama comprehensive tests")
    parser.add_argument("--no-multiplexing-tests", action="store_true",
                       help="Skip API multiplexing tests")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output in multiplexing tests")
    
    args = parser.parse_args()
    
    # Process skip_apis from no_* flags
    if args.skip_apis is None:
        args.skip_apis = []
    
    if args.no_ollama_tests and "ollama" not in args.skip_apis:
        args.skip_apis.append("ollama")
    
    # Prepare results container
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Run the selected tests
    if not args.no_api_tests:
        results["tests"]["api_backoff"] = run_api_backoff_tests(args)
    
    if not args.no_ollama_tests:
        results["tests"]["ollama_comprehensive"] = run_ollama_backoff_tests(args)
    
    if not args.no_multiplexing_tests:
        results["tests"]["api_multiplexing"] = run_multiplexing_tests(args)
    
    # Calculate overall success
    success_count = sum(1 for test in results["tests"].values() 
                     if test.get("status") == "Success")
    total_tests = sum(1 for test in results["tests"].values() 
                    if test.get("status") != "Skipped")
    
    results["success_count"] = success_count
    results["total_tests"] = total_tests
    results["success_rate"] = success_count / total_tests if total_tests > 0 else 0
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"queue_backoff_tests_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Test Run Complete ===")
    print(f"Results saved to: {results_file}")
    print(f"Success rate: {success_count}/{total_tests} tests passed ({results['success_rate']:.0%})")
    
    # Return exit code based on success
    if success_count == total_tests:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
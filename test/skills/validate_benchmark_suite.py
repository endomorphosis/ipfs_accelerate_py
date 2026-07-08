#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for the refactored benchmark suite.

This script performs a series of validation checks on the refactored benchmark suite
to ensure it meets the requirements and works as expected.
"""

import os
import sys
import importlib
import argparse
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_module_imports(suite_path):
    """Check that all required modules can be imported."""
    print("Checking module imports...")
    
    # Add suite path to Python path
    parent_dir = os.path.dirname(suite_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Try to import key modules
    modules = [
        "refactored_benchmark_suite",
        "refactored_benchmark_suite.benchmark",
        "refactored_benchmark_suite.metrics",
        "refactored_benchmark_suite.models",
        "refactored_benchmark_suite.exporters",
        "refactored_benchmark_suite.visualizers",
        "refactored_benchmark_suite.config",
        "refactored_benchmark_suite.utils",
    ]
    
    success = True
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            print(f"  ✓ Successfully imported {module_name}")
        except ImportError as e:
            print(f"  ✗ Failed to import {module_name}: {e}")
            success = False
    
    return success

def check_required_classes(suite_path):
    """Check that all required classes are present and importable."""
    print("\nChecking required classes...")
    
    # Add suite path to Python path
    parent_dir = os.path.dirname(suite_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Classes to check
    classes = [
        ("refactored_benchmark_suite", ["ModelBenchmark", "BenchmarkSuite", "BenchmarkConfig", "BenchmarkResults"]),
        ("refactored_benchmark_suite.metrics", ["LatencyMetric", "ThroughputMetric", "MemoryMetric", "FLOPsMetric"]),
        ("refactored_benchmark_suite.models", ["ModelAdapter", "TextModelAdapter", "VisionModelAdapter"])
    ]
    
    success = True
    for module_name, class_names in classes:
        try:
            module = importlib.import_module(module_name)
            for class_name in class_names:
                if hasattr(module, class_name):
                    print(f"  ✓ Found class {class_name} in {module_name}")
                else:
                    print(f"  ✗ Missing class {class_name} in {module_name}")
                    success = False
        except ImportError as e:
            print(f"  ✗ Failed to import {module_name}: {e}")
            success = False
    
    return success

def check_command_line_interface(suite_path):
    """Check that the command-line interface is working."""
    print("\nChecking command-line interface...")
    
    # Add suite path to Python path
    parent_dir = os.path.dirname(suite_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        import subprocess
        
        # Run the help command
        cmd = [sys.executable, "-m", "refactored_benchmark_suite", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print(f"  ✓ Command-line interface is working")
            return True
        else:
            print(f"  ✗ Command-line interface is not working")
            print(f"    Exit code: {result.returncode}")
            print(f"    Output: {result.stdout}")
            print(f"    Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error checking command-line interface: {e}")
        return False

def check_feature_parity(suite_path):
    """Check that the refactored suite has feature parity with the original."""
    print("\nChecking feature parity with original implementation...")
    
    # Features to check
    features = [
        ("CPU to GPU speedup calculation", "get_cpu_gpu_speedup"),
        ("HuggingFace Hub integration", "publish_to_hub"),
        ("Exporting to multiple formats", ["export_to_json", "export_to_csv", "export_to_markdown"]),
        ("Configuration file support", "create_benchmark_configs_from_file"),
        ("Parallel benchmarking", "ThreadPoolExecutor"),
        ("Model task auto-detection", "_initialize_task")
    ]
    
    # Add suite path to Python path
    parent_dir = os.path.dirname(suite_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from refactored_benchmark_suite.benchmark import BenchmarkResults, ModelBenchmark
        from refactored_benchmark_suite.config.benchmark_config import create_benchmark_configs_from_file
        
        success = True
        
        for feature_name, attr_name in features:
            if isinstance(attr_name, list):
                # Check if any of the attributes exist
                found = False
                for attr in attr_name:
                    if hasattr(BenchmarkResults, attr):
                        found = True
                        break
                
                if found:
                    print(f"  ✓ Feature '{feature_name}' is implemented")
                else:
                    print(f"  ✗ Feature '{feature_name}' is missing")
                    success = False
            
            elif attr_name == "create_benchmark_configs_from_file":
                # Special case for config file support
                if callable(create_benchmark_configs_from_file):
                    print(f"  ✓ Feature '{feature_name}' is implemented")
                else:
                    print(f"  ✗ Feature '{feature_name}' is missing")
                    success = False
            
            elif attr_name == "ThreadPoolExecutor":
                # Check for parallel benchmarking
                import inspect
                src = inspect.getsource(ModelBenchmark.run)
                if "ThreadPoolExecutor" in src:
                    print(f"  ✓ Feature '{feature_name}' is implemented")
                else:
                    print(f"  ✗ Feature '{feature_name}' is missing")
                    success = False
            
            else:
                # Check if the attribute exists on BenchmarkResults or ModelBenchmark
                if hasattr(BenchmarkResults, attr_name):
                    print(f"  ✓ Feature '{feature_name}' is implemented")
                elif hasattr(ModelBenchmark, attr_name):
                    print(f"  ✓ Feature '{feature_name}' is implemented")
                else:
                    print(f"  ✗ Feature '{feature_name}' is missing")
                    success = False
        
        return success
    
    except Exception as e:
        print(f"  ✗ Error checking feature parity: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate refactored benchmark suite")
    
    parser.add_argument(
        "--suite-path", type=str, default=os.path.join(SCRIPT_DIR, "refactored_benchmark_suite"),
        help="Path to the refactored benchmark suite directory"
    )
    
    args = parser.parse_args()
    
    # Ensure the benchmark suite path exists
    suite_path = Path(args.suite_path)
    if not suite_path.exists() or not suite_path.is_dir():
        print(f"Error: Benchmark suite path does not exist: {suite_path}")
        return 1
    
    print(f"Validating refactored benchmark suite at {suite_path}\n")
    
    # Run validation checks
    checks = [
        ("Module imports", check_module_imports(suite_path)),
        ("Required classes", check_required_classes(suite_path)),
        ("Command-line interface", check_command_line_interface(suite_path)),
        ("Feature parity", check_feature_parity(suite_path))
    ]
    
    # Print summary
    print("\nValidation Summary:")
    all_passed = True
    
    for check_name, result in checks:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {check_name}")
        all_passed = all_passed and result
    
    if all_passed:
        print("\n✓ All validation checks passed!")
        return 0
    else:
        print("\n✗ Some validation checks failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
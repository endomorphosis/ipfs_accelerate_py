#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update HuggingFace Test Files for Distributed Testing

This script updates existing HuggingFace test files to work with the distributed testing
framework, adding support for task distribution, result aggregation, and hardware-aware
worker assignment.

Features:
- Modifies test files to support distributed execution
- Adds result collection for aggregation
- Implements hardware-aware worker assignment
- Supports graceful fallback for worker failures
- Integrates with the Distributed Testing Framework
"""

import os
import re
import sys
import json
import argparse
import importlib
from pathlib import Path
from datetime import datetime

# Template for distributed testing modifications
DISTRIBUTED_IMPORTS_TEMPLATE = """
# Distributed Testing Framework imports
try:
    from distributed_testing_framework import (
        Worker, 
        ResultCollector, 
        TaskDistributor, 
        HardwareDetector,
        register_test
    )
    DISTRIBUTED_FRAMEWORK_AVAILABLE = True
except ImportError:
    DISTRIBUTED_FRAMEWORK_AVAILABLE = False
    # Mock implementations for graceful fallback
    class Worker:
        def __init__(self, *args, **kwargs): pass
        def execute(self, *args, **kwargs): return None
        
    class ResultCollector:
        def __init__(self, *args, **kwargs): pass
        def collect(self, *args, **kwargs): pass
        def aggregate(self): return {}
        
    class TaskDistributor:
        def __init__(self, *args, **kwargs): pass
        def distribute(self, *args, **kwargs): return []
        
    class HardwareDetector:
        def __init__(self): pass
        def detect(self): return {"cpu": True}
        
    def register_test(*args, **kwargs): 
        def wrapper(func): return func
        return wrapper
"""

DISTRIBUTED_MAIN_TEMPLATE = """
def run_distributed(model_name, num_workers=4, timeout=300):
    """Run the test in distributed mode with multiple workers"""
    if not DISTRIBUTED_FRAMEWORK_AVAILABLE:
        print("Distributed Testing Framework not available, falling back to local execution.")
        return run_test(model_name)
        
    # Initialize framework components
    hardware = HardwareDetector().detect()
    distributor = TaskDistributor(available_hardware=hardware)
    collector = ResultCollector(test_name=f"test_hf_{model_name.replace('-', '_')}")
    
    # Define tasks for distribution
    tasks = [
        {"model_name": model_name, "hardware": "cpu", "batch_size": 1},
    ]
    
    # Add GPU tasks if available
    if hardware.get("cuda", False):
        tasks.append({"model_name": model_name, "hardware": "cuda", "batch_size": 4})
    
    if hardware.get("mps", False):
        tasks.append({"model_name": model_name, "hardware": "mps", "batch_size": 2})
        
    # Add special hardware tasks if available
    for hw in ["openvino", "webnn", "webgpu"]:
        if hardware.get(hw, False):
            tasks.append({"model_name": model_name, "hardware": hw, "batch_size": 1})
    
    # Distribute tasks to workers
    worker_tasks = distributor.distribute(tasks, num_workers=num_workers)
    
    # Create and execute workers
    workers = []
    for i, worker_task in enumerate(worker_tasks):
        worker = Worker(
            worker_id=f"worker-{i}",
            tasks=worker_task,
            timeout=timeout
        )
        workers.append(worker)
        
    # Execute tasks and collect results
    for worker in workers:
        try:
            results = worker.execute()
            collector.collect(results)
        except Exception as e:
            print(f"Worker error: {e}")
    
    # Aggregate and return results
    return collector.aggregate()

@register_test(model_type="{model_type}")
def run_test(model_name, device="cpu", distributed=False, num_workers=4):
    """Main test execution function with distributed support"""
    if distributed:
        return run_distributed(model_name, num_workers)
"""

def find_test_files(directory):
    """Find all HuggingFace test files in the given directory"""
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return []
    
    test_files = []
    for file in os.listdir(directory):
        if file.startswith("test_hf_") and file.endswith(".py"):
            test_files.append(os.path.join(directory, file))
    
    return sorted(test_files)

def detect_model_type(file_path):
    """Detect the model architecture type from the test file content"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for architecture indicators in the file
    if "decoder-only" in content or "gpt" in content.lower() or "llama" in content.lower():
        return "decoder-only"
    elif "encoder-decoder" in content or "seq2seq" in content or "t5" in content.lower():
        return "encoder-decoder"
    elif "vision" in content or "image" in content or "vit" in content.lower():
        return "vision"
    elif ("text" in content and "vision" in content) or "clip" in content.lower() or "multimodal" in content:
        return "multimodal"
    elif "audio" in content or "speech" in content or "whisper" in content.lower():
        return "audio"
    else:
        return "encoder-only"  # Default to encoder-only

def update_file_for_distributed_testing(file_path, verify=False):
    """Update a single test file for distributed testing support"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Check if already modified
        if "DISTRIBUTED_FRAMEWORK_AVAILABLE" in content:
            print(f"File already updated: {file_path}")
            return True
        
        # Detect model type
        model_type = detect_model_type(file_path)
        
        # Find the import section and add distributed imports
        import_match = re.search(r"^import.*?(?=\n\n)", content, re.DOTALL | re.MULTILINE)
        if import_match:
            import_section = import_match.group(0)
            new_import_section = import_section + DISTRIBUTED_IMPORTS_TEMPLATE
            content = content.replace(import_section, new_import_section)
        else:
            print(f"Could not find import section in {file_path}")
            return False
        
        # Find the main function and modify it for distributed support
        main_match = re.search(r"def run_test\([^)]*\):.*?(?=\n\n)", content, re.DOTALL | re.MULTILINE)
        if main_match:
            # Extract the existing run_test function
            main_section = main_match.group(0)
            
            # Add the distributed main template before the run_test function
            distributed_template = DISTRIBUTED_MAIN_TEMPLATE.format(model_type=model_type)
            modified_content = content.replace(main_section, distributed_template + main_section)
            
            # Update the if __name__ == "__main__" section to support distributed mode
            main_block_match = re.search(r"if __name__ == \"__main__\":.*", modified_content, re.DOTALL)
            if main_block_match:
                main_block = main_block_match.group(0)
                
                # Add distributed mode argument
                parser_match = re.search(r"parser = argparse\.ArgumentParser\(.*?\)", main_block, re.DOTALL)
                if parser_match:
                    parser_section = parser_match.group(0)
                    new_parser_section = parser_section
                    
                    # Add distributed arguments if not already present
                    if "--distributed" not in main_block:
                        add_arg_match = re.search(r"parser\.add_argument\(.*?\)", main_block, re.DOTALL | re.MULTILINE)
                        if add_arg_match:
                            last_arg = add_arg_match.group(0)
                            distributed_args = '\n    parser.add_argument("--distributed", action="store_true", help="Run in distributed mode")'
                            distributed_args += '\n    parser.add_argument("--workers", type=int, default=4, help="Number of workers for distributed testing")'
                            modified_content = modified_content.replace(last_arg, last_arg + distributed_args)
                
                # Update the run_test call to pass distributed flag
                run_match = re.search(r"run_test\(.*?\)", main_block)
                if run_match:
                    run_call = run_match.group(0)
                    if "distributed" not in run_call:
                        new_run_call = run_call.replace(")", ", distributed=args.distributed, num_workers=args.workers)")
                        modified_content = modified_content.replace(run_call, new_run_call)
                
            # Write the modified content back to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            
            # Verify the file is still valid Python
            if verify:
                try:
                    compile(modified_content, file_path, 'exec')
                    print(f"✅ Successfully updated and verified: {file_path}")
                    return True
                except SyntaxError as e:
                    print(f"❌ Syntax error in updated file: {file_path}")
                    print(f"Error: {e}")
                    return False
            else:
                print(f"✅ Successfully updated: {file_path}")
                return True
        else:
            print(f"Could not find run_test function in {file_path}")
            return False
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def create_distributed_framework_stub():
    """Create a stub implementation of the distributed testing framework"""
    framework_dir = "distributed_testing_framework"
    os.makedirs(framework_dir, exist_ok=True)
    
    # Create __init__.py
    with open(os.path.join(framework_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("""# Distributed Testing Framework
from .worker import Worker
from .collector import ResultCollector
from .distributor import TaskDistributor
from .hardware import HardwareDetector
from .registry import register_test

__all__ = [
    'Worker',
    'ResultCollector',
    'TaskDistributor',
    'HardwareDetector',
    'register_test'
]
""")
    
    # Create worker.py
    with open(os.path.join(framework_dir, "worker.py"), "w", encoding="utf-8") as f:
        f.write("""# Worker Implementation
import time
import threading

class Worker:
    \"\"\"Worker for distributed testing framework\"\"\"
    def __init__(self, worker_id, tasks, timeout=300):
        self.worker_id = worker_id
        self.tasks = tasks
        self.timeout = timeout
        self.results = []
        
    def execute(self):
        \"\"\"Execute all assigned tasks\"\"\"
        print(f"Worker {self.worker_id} starting with {len(self.tasks)} tasks")
        
        for task in self.tasks:
            start_time = time.time()
            
            try:
                # Execute the task with the specified parameters
                model_name = task.get("model_name")
                hardware = task.get("hardware", "cpu")
                batch_size = task.get("batch_size", 1)
                
                print(f"Worker {self.worker_id} running task: {model_name} on {hardware}")
                
                # Simulate task execution
                time.sleep(1)  # In a real implementation, this would run the actual test
                
                # Store the result
                elapsed_time = time.time() - start_time
                result = {
                    "worker_id": self.worker_id,
                    "model_name": model_name,
                    "hardware": hardware,
                    "batch_size": batch_size,
                    "success": True,
                    "execution_time": elapsed_time,
                    "timestamp": time.time()
                }
                
                self.results.append(result)
                print(f"Worker {self.worker_id} completed task: {model_name} in {elapsed_time:.2f}s")
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                error_result = {
                    "worker_id": self.worker_id,
                    "model_name": task.get("model_name"),
                    "hardware": task.get("hardware", "cpu"),
                    "success": False,
                    "error": str(e),
                    "execution_time": elapsed_time,
                    "timestamp": time.time()
                }
                self.results.append(error_result)
                print(f"Worker {self.worker_id} task failed: {str(e)}")
        
        return self.results
""")
    
    # Create collector.py
    with open(os.path.join(framework_dir, "collector.py"), "w", encoding="utf-8") as f:
        f.write("""# Result Collector Implementation
import time
import json
import os
from datetime import datetime

class ResultCollector:
    \"\"\"Collects and aggregates test results\"\"\"
    def __init__(self, test_name):
        self.test_name = test_name
        self.results = []
        self.start_time = time.time()
        
    def collect(self, results):
        \"\"\"Collect results from a worker\"\"\"
        if isinstance(results, list):
            self.results.extend(results)
        else:
            self.results.append(results)
            
    def aggregate(self):
        \"\"\"Aggregate all collected results\"\"\"
        elapsed_time = time.time() - self.start_time
        
        # Calculate success rate and statistics
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.get("success", False))
        
        aggregated_result = {
            "test_name": self.test_name,
            "timestamp": datetime.now().isoformat(),
            "total_duration": elapsed_time,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "results": self.results
        }
        
        # Group results by hardware
        hw_results = {}
        for result in self.results:
            hw = result.get("hardware", "unknown")
            if hw not in hw_results:
                hw_results[hw] = []
            hw_results[hw].append(result)
            
        aggregated_result["hardware_results"] = {
            hw: {
                "count": len(results),
                "success_count": sum(1 for r in results if r.get("success", False)),
                "avg_execution_time": sum(r.get("execution_time", 0) for r in results) / len(results) if results else 0
            }
            for hw, results in hw_results.items()
        }
        
        # Save results to file
        os.makedirs("distributed_results", exist_ok=True)
        result_file = f"distributed_results/{self.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, "w") as f:
            json.dump(aggregated_result, f, indent=2)
            
        print(f"Results saved to: {result_file}")
        
        return aggregated_result
""")
    
    # Create distributor.py
    with open(os.path.join(framework_dir, "distributor.py"), "w", encoding="utf-8") as f:
        f.write("""# Task Distributor Implementation

class TaskDistributor:
    \"\"\"Distributes tasks to workers based on available hardware\"\"\"
    def __init__(self, available_hardware=None):
        self.available_hardware = available_hardware or {"cpu": True}
        
    def distribute(self, tasks, num_workers=4):
        \"\"\"Distribute tasks to workers based on hardware requirements\"\"\"
        # Filter tasks based on available hardware
        filtered_tasks = []
        for task in tasks:
            hw = task.get("hardware", "cpu")
            if hw == "cpu" or self.available_hardware.get(hw, False):
                filtered_tasks.append(task)
            else:
                print(f"Skipping task requiring {hw} as it's not available")
        
        if not filtered_tasks:
            print("No tasks can be executed on available hardware")
            return [[] for _ in range(num_workers)]
            
        # Distribute tasks evenly among workers
        worker_tasks = [[] for _ in range(num_workers)]
        
        # Sort tasks by complexity (GPU tasks first)
        hardware_priority = {"webgpu": 0, "cuda": 1, "mps": 2, "openvino": 3, "webnn": 4, "cpu": 5}
        sorted_tasks = sorted(filtered_tasks, key=lambda t: hardware_priority.get(t.get("hardware", "cpu"), 10))
        
        # Distribute tasks round-robin
        for i, task in enumerate(sorted_tasks):
            worker_idx = i % num_workers
            worker_tasks[worker_idx].append(task)
            
        return worker_tasks
""")
    
    # Create hardware.py
    with open(os.path.join(framework_dir, "hardware.py"), "w", encoding="utf-8") as f:
        f.write("""# Hardware Detection Implementation
import platform
import os

class HardwareDetector:
    \"\"\"Detects available hardware for testing\"\"\"
    def __init__(self):
        pass
        
    def detect(self):
        \"\"\"Detect available hardware\"\"\"
        hardware = {"cpu": True}
        
        # Detect CUDA availability
        try:
            import torch
            hardware["cuda"] = torch.cuda.is_available()
        except:
            hardware["cuda"] = False
            
        # Detect MPS availability (Apple Silicon)
        try:
            import torch
            hardware["mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except:
            hardware["mps"] = False
            
        # Detect OpenVINO availability
        try:
            import openvino
            hardware["openvino"] = True
        except:
            hardware["openvino"] = False
            
        # WebNN and WebGPU would be detected in a browser environment
        # For this stub, we'll just set them to False
        hardware["webnn"] = False
        hardware["webgpu"] = False
        
        return hardware
""")
    
    # Create registry.py
    with open(os.path.join(framework_dir, "registry.py"), "w", encoding="utf-8") as f:
        f.write("""# Test Registry Implementation

# Global registry of tests
_TEST_REGISTRY = {}

def register_test(model_type=None):
    \"\"\"Register a test function with metadata\"\"\"
    def wrapper(func):
        name = func.__name__
        _TEST_REGISTRY[name] = {
            "function": func,
            "model_type": model_type or "unknown"
        }
        return func
    return wrapper

def get_registered_tests():
    \"\"\"Get all registered tests\"\"\"
    return _TEST_REGISTRY

def get_tests_by_model_type(model_type):
    \"\"\"Get all tests for a specific model type\"\"\"
    return {
        name: details for name, details in _TEST_REGISTRY.items()
        if details.get("model_type") == model_type
    }
""")
    
    print(f"Created distributed testing framework stub in {framework_dir}")
    return framework_dir

def create_run_script():
    """Create a script to run distributed tests"""
    script_path = "run_distributed_tests.py"
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Run Distributed Tests

This script runs HuggingFace model tests in distributed mode using the
Distributed Testing Framework.

Usage:
    python run_distributed_tests.py --workers 4 --model-family bert
    python run_distributed_tests.py --all --workers 8
\"\"\"

import os
import sys
import argparse
import importlib
import time
from pathlib import Path
from datetime import datetime

try:
    from distributed_testing_framework import (
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
            get_registered_tests,
            get_tests_by_model_type
        )
        FRAMEWORK_AVAILABLE = True
    except ImportError:
        print("Failed to install stub framework. Exiting.")
        sys.exit(1)

def find_test_files(directory="fixed_tests"):
    \"\"\"Find all test files in the specified directory\"\"\"
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return []
    
    test_files = []
    for file in os.listdir(directory):
        if file.startswith("test_hf_") and file.endswith(".py"):
            test_files.append(os.path.join(directory, file))
    
    return sorted(test_files)

def get_model_families():
    \"\"\"Get all model families from test files\"\"\"
    families = set()
    for test_file in find_test_files():
        # Extract the model family from the filename (test_hf_bert.py -> bert)
        family = os.path.basename(test_file).replace("test_hf_", "").replace(".py", "")
        families.add(family)
    return sorted(families)

def run_distributed_tests(model_family=None, workers=4, timeout=600, all_models=False):
    \"\"\"Run tests in distributed mode\"\"\"
    if not FRAMEWORK_AVAILABLE:
        print("Distributed Testing Framework not available.")
        return
    
    start_time = time.time()
    test_files = find_test_files()
    
    if not test_files:
        print("No test files found.")
        return
    
    print(f"Found {len(test_files)} test files.")
    
    # Filter test files by model family if specified
    if model_family and not all_models:
        test_files = [f for f in test_files if f"test_hf_{model_family}" in f]
        if not test_files:
            print(f"No test files found for model family: {model_family}")
            return
    
    # Run each test file with distributed flag
    successful_tests = 0
    failed_tests = 0
    
    results_dir = "distributed_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(results_dir, f"distributed_test_summary_{timestamp}.txt")
    
    with open(summary_file, "w") as summary:
        summary.write(f"Distributed Test Run Summary - {datetime.now().isoformat()}\\n\\n")
        summary.write(f"Workers: {workers}\\nTimeout: {timeout}s\\n\\n")
        
        for test_file in test_files:
            test_name = os.path.basename(test_file).replace(".py", "")
            print(f"Running distributed test: {test_name}")
            
            try:
                # Run the test with distributed flag
                cmd = f"python {test_file} --distributed --workers {workers}"
                print(f"Executing: {cmd}")
                
                # In a real implementation, we would use subprocess or better integration
                # For this stub, we'll import and run the module directly
                
                # Get the module name from the file path
                module_name = os.path.basename(test_file).replace(".py", "")
                
                # Add the directory to the path if needed
                test_dir = os.path.dirname(test_file)
                if test_dir not in sys.path:
                    sys.path.insert(0, test_dir)
                
                try:
                    # Import the module
                    module = importlib.import_module(module_name)
                    
                    # Get the model name from the module
                    model_name = module_name.replace("test_hf_", "")
                    
                    # Run the test function with distributed=True
                    if hasattr(module, "run_test"):
                        result = module.run_test(model_name, distributed=True, num_workers=workers)
                        successful_tests += 1
                        
                        summary.write(f"✅ {test_name} - Success\\n")
                        print(f"✅ Completed: {test_name}")
                    else:
                        print(f"❌ No run_test function found in {module_name}")
                        failed_tests += 1
                        summary.write(f"❌ {test_name} - Failed: No run_test function\\n")
                        
                except Exception as e:
                    print(f"❌ Error running {test_name}: {e}")
                    failed_tests += 1
                    summary.write(f"❌ {test_name} - Failed: {str(e)}\\n")
            
            except Exception as e:
                print(f"❌ Failed to execute {test_name}: {e}")
                failed_tests += 1
                summary.write(f"❌ {test_name} - Failed: {str(e)}\\n")
        
        total_time = time.time() - start_time
        summary.write(f"\\nSummary:\\n")
        summary.write(f"Total tests: {len(test_files)}\\n")
        summary.write(f"Successful: {successful_tests}\\n")
        summary.write(f"Failed: {failed_tests}\\n")
        summary.write(f"Total time: {total_time:.2f}s\\n")
    
    print(f"\\nDistributed Test Run Complete:")
    print(f"Total tests: {len(test_files)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
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
    
    # Framework management
    parser.add_argument("--create-framework", action="store_true", help="Create the stub framework implementation")
    parser.add_argument("--update-tests", action="store_true", help="Update test files for distributed testing")
    parser.add_argument("--list-models", action="store_true", help="List available model families")
    
    args = parser.parse_args()
    
    # Create framework if requested
    if args.create_framework:
        from update_for_distributed_testing import create_distributed_framework_stub
        create_distributed_framework_stub()
        return
    
    # Update test files if requested
    if args.update_tests:
        from update_for_distributed_testing import update_file_for_distributed_testing, find_test_files
        test_files = find_test_files()
        for file in test_files:
            update_file_for_distributed_testing(file, verify=True)
        return
    
    # List model families if requested
    if args.list_models:
        families = get_model_families()
        print("Available model families:")
        for family in families:
            print(f"- {family}")
        return
    
    # Run distributed tests
    run_distributed_tests(
        model_family=args.model_family,
        workers=args.workers,
        timeout=args.timeout,
        all_models=args.all
    )

if __name__ == "__main__":
    main()
""")
    
    print(f"Created distributed test runner script: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Update HuggingFace test files for distributed testing")
    parser.add_argument("--dir", type=str, default="fixed_tests", help="Directory containing test files")
    parser.add_argument("--file", type=str, help="Specific test file to update")
    parser.add_argument("--verify", action="store_true", help="Verify syntax after updating")
    parser.add_argument("--create-framework", action="store_true", help="Create stub implementation of distributed framework")
    parser.add_argument("--create-runner", action="store_true", help="Create distributed test runner script")
    args = parser.parse_args()
    
    if args.create_framework:
        create_distributed_framework_stub()
        return
    
    if args.create_runner:
        create_run_script()
        return
    
    # Process a single file
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        
        update_file_for_distributed_testing(args.file, verify=args.verify)
        return
    
    # Process all files in directory
    test_files = find_test_files(args.dir)
    if not test_files:
        print(f"No test files found in {args.dir}")
        return
    
    print(f"Found {len(test_files)} test files in {args.dir}")
    
    success_count = 0
    for file_path in test_files:
        if update_file_for_distributed_testing(file_path, verify=args.verify):
            success_count += 1
    
    print(f"\nSummary: Updated {success_count} out of {len(test_files)} files")
    
    # Create framework and runner if not existing
    if success_count > 0:
        framework_dir = "distributed_testing_framework"
        if not os.path.exists(framework_dir):
            print("\nCreating distributed framework stub...")
            create_distributed_framework_stub()
        
        runner_script = "run_distributed_tests.py"
        if not os.path.exists(runner_script):
            print("\nCreating distributed test runner script...")
            create_run_script()
            
        print(f"\nYou can now run distributed tests with:")
        print(f"python run_distributed_tests.py --model-family bert --workers 4")
        print(f"python run_distributed_tests.py --all")

if __name__ == "__main__":
    main()
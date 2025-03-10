#!/usr/bin/env python
"""
Benchmark for IPFS Acceleration 

This script measures the performance of the IPFS Accelerate Python package
by testing core functionality and reporting metrics.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
import importlib
import platform
import datetime
import random
import string
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_benchmark")

def get_system_info():
    """Get information about the system"""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.architecture()[0],
        "node": platform.node(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    return info

def benchmark_loading():
    """Benchmark module loading time"""
    results = {}
    
    # For our flat module structure, we only need to test loading the main module
    # The other components are attributes, not submodules
    
    modules = ["ipfs_accelerate_py"]
    
    for module_name in modules:
        # Unload the module if already loaded
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Time the import
        start_time = time.time()
        try:
            module = importlib.import_module(module_name)
            end_time = time.time()
            load_time = (end_time - start_time) * 1000  # Convert to milliseconds
            logger.info(f"Module {module_name} loaded in {load_time:.2f} ms")
            results[module_name] = {
                "status": "Success",
                "load_time_ms": load_time
            }
            
            # For compatibility with the original script, simulate loading of component attributes
            # and measure the time to access them
            component_attrs = ["ipfs_accelerate", "backends", "config"]
            for attr_name in component_attrs:
                if hasattr(module, attr_name):
                    # Time accessing the attribute
                    start_time = time.time()
                    getattr(module, attr_name)
                    end_time = time.time()
                    access_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    
                    # Use a modified name to indicate this is an attribute access, not a submodule import
                    attr_key = f"{module_name}.{attr_name}"
                    results[attr_key] = {
                        "status": "Success",
                        "load_time_ms": access_time
                    }
                    logger.info(f"Attribute {attr_key} accessed in {access_time:.2f} ms")
                else:
                    attr_key = f"{module_name}.{attr_name}"
                    results[attr_key] = {
                        "status": "Failed",
                        "error": f"Attribute {attr_name} not found in module {module_name}"
                    }
                    logger.error(f"Failed to access attribute {attr_key}: attribute not found")
            
        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            results[module_name] = {
                "status": "Failed",
                "error": str(e)
            }
            
            # Mark all component attributes as failed since the main module couldn't be imported
            for attr_name in ["ipfs_accelerate", "backends", "config"]:
                attr_key = f"{module_name}.{attr_name}"
                results[attr_key] = {
                    "status": "Failed",
                    "error": f"Main module {module_name} could not be imported"
                }
    
    return results

def benchmark_basic_operations():
    """Benchmark basic operations"""
    results = {}
    
    try:
        import ipfs_accelerate_py
        
        # Benchmark creating a dummy config
        try:
            start_time = time.time()
            # We can only access the class, not instantiate it without a config file
            cfg_class = ipfs_accelerate_py.config
            end_time = time.time()
            access_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            results["config_access"] = {
                "status": "Success",
                "time_ms": access_time
            }
            logger.info(f"Config class accessed in {access_time:.2f} ms")
        except Exception as e:
            results["config_access"] = {
                "status": "Failed",
                "error": str(e)
            }
            logger.error(f"Error accessing config: {e}")
        
        # Benchmark backends class access
        try:
            start_time = time.time()
            backends_class = ipfs_accelerate_py.backends
            end_time = time.time()
            access_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            results["backends_access"] = {
                "status": "Success",
                "time_ms": access_time
            }
            logger.info(f"Backends class accessed in {access_time:.2f} ms")
        except Exception as e:
            results["backends_access"] = {
                "status": "Failed",
                "error": str(e)
            }
            logger.error(f"Error accessing backends: {e}")
            
        # Benchmark load_checkpoint_and_dispatch access
        try:
            start_time = time.time()
            dispatch_func = ipfs_accelerate_py.load_checkpoint_and_dispatch
            end_time = time.time()
            access_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            results["dispatch_access"] = {
                "status": "Success",
                "time_ms": access_time
            }
            logger.info(f"Dispatch function accessed in {access_time:.2f} ms")
        except Exception as e:
            results["dispatch_access"] = {
                "status": "Failed",
                "error": str(e)
            }
            logger.error(f"Error accessing dispatch function: {e}")
            
    except ImportError:
        logger.error("Failed to import ipfs_accelerate_py for benchmarks")
        results["import"] = {
            "status": "Failed",
            "error": "Failed to import ipfs_accelerate_py"
        }
    except Exception as e:
        logger.error(f"Error in benchmarks: {e}")
        results["general"] = {
            "status": "Failed",
            "error": str(e)
        }
        
    return results

def benchmark_parallel_loading(num_threads=5, iterations=3):
    """Benchmark parallel loading of the module"""
    results = {
        "threads": num_threads,
        "iterations": iterations,
        "thread_results": []
    }
    
    def load_module(thread_id, iteration):
        """Load module in a thread"""
        thread_start = time.time()
        
        # Unload modules to ensure fresh load
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('ipfs_accelerate_py'):
                del sys.modules[module_name]
        
        # Load modules
        try:
            import ipfs_accelerate_py
            import ipfs_accelerate_py.ipfs_accelerate
            import ipfs_accelerate_py.backends
            import ipfs_accelerate_py.config
            
            thread_end = time.time()
            load_time = (thread_end - thread_start) * 1000  # Convert to milliseconds
            
            return {
                "thread_id": thread_id,
                "iteration": iteration,
                "status": "Success",
                "load_time_ms": load_time
            }
        except Exception as e:
            return {
                "thread_id": thread_id,
                "iteration": iteration,
                "status": "Failed",
                "error": str(e)
            }
    
    # Create thread pool and execute loads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for iteration in range(iterations):
            for thread_id in range(num_threads):
                futures.append(executor.submit(load_module, thread_id, iteration))
        
        # Collect results
        for future in futures:
            results["thread_results"].append(future.result())
    
    # Calculate statistics
    load_times = [r["load_time_ms"] for r in results["thread_results"] if r["status"] == "Success"]
    if load_times:
        results["min_load_time_ms"] = min(load_times)
        results["max_load_time_ms"] = max(load_times)
        results["avg_load_time_ms"] = sum(load_times) / len(load_times)
        results["success_rate"] = len(load_times) / len(results["thread_results"])
        
        logger.info(f"Parallel loading statistics:")
        logger.info(f"  Min load time: {results['min_load_time_ms']:.2f} ms")
        logger.info(f"  Max load time: {results['max_load_time_ms']:.2f} ms")
        logger.info(f"  Avg load time: {results['avg_load_time_ms']:.2f} ms")
        logger.info(f"  Success rate: {results['success_rate'] * 100:.1f}%")
    else:
        logger.error("No successful parallel loads")
        results["success_rate"] = 0
    
    return results

def run_benchmarks(args):
    """Run all benchmarks and return results"""
    logger.info("Starting IPFS Accelerate Python benchmarks")
    
    results = {
        "benchmark_timestamp": datetime.datetime.now().isoformat(),
        "system_info": get_system_info()
    }
    
    # Run loading benchmark
    logger.info("Running module loading benchmark")
    loading_results = benchmark_loading()
    results["loading_benchmark"] = loading_results
    
    # Run basic operations benchmark
    logger.info("Running basic operations benchmark")
    operations_results = benchmark_basic_operations()
    results["operations_benchmark"] = operations_results
    
    # Run parallel loading benchmark
    if args.parallel:
        logger.info(f"Running parallel loading benchmark with {args.threads} threads")
        parallel_results = benchmark_parallel_loading(args.threads, args.iterations)
        results["parallel_benchmark"] = parallel_results
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark the IPFS Accelerate Python package")
    parser.add_argument("--output", "-o", default="benchmark_results.json", 
                       help="Output file for benchmark results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--parallel", "-p", action="store_true", 
                       help="Run parallel loading benchmark")
    parser.add_argument("--threads", "-t", type=int, default=5, 
                       help="Number of threads for parallel benchmark")
    parser.add_argument("--iterations", "-i", type=int, default=3, 
                       help="Number of iterations for each thread")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run benchmarks
    results = run_benchmarks(args)
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    
    # Module loading summary
    loading_success = all(r.get("status") == "Success" for r in results["loading_benchmark"].values())
    avg_load_time = sum(r.get("load_time_ms", 0) for r in results["loading_benchmark"].values() 
                      if r.get("status") == "Success") / len(results["loading_benchmark"])
    print(f"Module Loading: {'✅' if loading_success else '❌'}")
    print(f"  Average load time: {avg_load_time:.2f} ms")
    
    # Operations summary
    operations = results["operations_benchmark"]
    ops_success = all(op.get("status") == "Success" for op in operations.values())
    print(f"Basic Operations: {'✅' if ops_success else '⚠️'}")
    for op_name, op_result in operations.items():
        if "time_ms" in op_result:
            print(f"  {op_name}: {op_result['time_ms']:.2f} ms")
    
    # Parallel loading summary
    if args.parallel and "parallel_benchmark" in results:
        parallel = results["parallel_benchmark"]
        success_rate = parallel.get("success_rate", 0) * 100
        print(f"Parallel Loading ({args.threads} threads): {'✅' if success_rate == 100 else '⚠️'}")
        print(f"  Success rate: {success_rate:.1f}%")
        if "avg_load_time_ms" in parallel:
            print(f"  Average load time: {parallel['avg_load_time_ms']:.2f} ms")
            print(f"  Min/Max load time: {parallel['min_load_time_ms']:.2f}/{parallel['max_load_time_ms']:.2f} ms")
    
    # Save results
    output_path = Path(args.output)
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"\nError saving results: {e}")
    
    # Return success if basic benchmarks succeeded
    return 0 if loading_success and ops_success else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Benchmark for the efficientnet skillset implementation.

This benchmark measures the performance of the efficientnet model implementation
in the skillset directory.
"""

import os
import sys
import time
import logging
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import benchmark core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_core import BenchmarkBase, BenchmarkRegistry

@BenchmarkRegistry.register(
    name="efficientnet_inference_benchmark",
    category="inference",
    models=["efficientnet"],
    hardware=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"]
)
class EfficientnetInferenceBenchmark(BenchmarkBase):
    """Benchmark for efficientnet inference performance."""
    
    def setup(self):
        """Set up benchmark environment."""
        self.logger.info("Setting up efficientnet inference benchmark")
        
        # Extract configuration parameters
        self.batch_sizes = self.config.get("batch_sizes", [1, 2, 4, 8])
        self.warmup_runs = self.config.get("warmup_runs", 2)
        self.measurement_runs = self.config.get("measurement_runs", 5)
        
        # Find and import skillset module
        self.skillset_module = None
        self.skillset_class = None
        
        try:
            # Determine the path to the skillset directory
            test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            self.skillset_dir = os.path.join(test_dir, "ipfs_accelerate_py", "worker", "skillset")
            
            if not os.path.exists(self.skillset_dir):
                raise ImportError(f"Skillset directory not found: {{self.skillset_dir}}")
            
            # Import the efficientnet skillset
            self.model_name = "efficientnet"
            self.model_type_safe = "efficientnet"
            
            return True
            
        except ImportError as e:
            self.logger.error(f"Required libraries or files not available: {{e}}")
            raise
            
        except Exception as e:
            self.logger.error(f"Error setting up benchmark: {{e}}")
            raise
            
    def execute(self):
        """Execute the benchmark."""
        self.logger.info("Executing efficientnet inference benchmark")
        
        results = {{
            "model_name": self.model_name,
            "hardware": str(self.hardware),
            "batch_results": {{}},
            "import_time": 0,
            "instantiation_time": 0,
            "error": None
        }}
        
        try:
            # Import skillset module
            skillset_path = os.path.join(self.skillset_dir, f"hf_{{self.model_name}}.py")
            
            if not os.path.exists(skillset_path):
                raise ImportError(f"Skillset file not found: {{skillset_path}}")
            
            import_start = time.time()
            spec = importlib.util.spec_from_file_location(f"hf_{{self.model_name}}", skillset_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import_time = time.time() - import_start
            results["import_time"] = import_time
            
            # Get skillset class
            class_name = f"hf_{{self.model_type_safe}}"
            if not hasattr(module, class_name):
                raise AttributeError(f"Class '{{class_name}}' not found in module")
                
            skillset_class = getattr(module, class_name)
            
            # Instantiate skillset class
            instance_start = time.time()
            instance = skillset_class(resources={{"device": self.hardware.name}})
            instantiation_time = time.time() - instance_start
            results["instantiation_time"] = instantiation_time
            
            # Benchmark each initialization method
            backend_results = {{}}
            
            # Get the appropriate init method based on hardware
            if self.hardware.name == "cpu":
                init_method = instance.init_cpu
            elif self.hardware.name == "cuda":
                init_method = instance.init_cuda
            elif self.hardware.name == "rocm":
                init_method = instance.init_rocm
            elif self.hardware.name == "openvino":
                init_method = instance.init_openvino
            elif self.hardware.name == "mps":
                init_method = instance.init_apple
            elif self.hardware.name == "qnn":
                init_method = instance.init_qualcomm
            else:
                init_method = instance.init_cpu  # Fallback to CPU
                
            # Time the initialization method
            for batch_size in self.batch_sizes:
                batch_results = self._benchmark_batch(init_method, batch_size)
                backend_results[f"batch_size_{{batch_size}}"] = batch_results
                
            results["batch_results"] = backend_results
            
        except Exception as e:
            error_message = f"Error benchmarking model {{self.model_name}}: {{e}}"
            self.logger.error(error_message)
            results["error"] = error_message
            
        return results
        
    def _benchmark_batch(self, init_method, batch_size: int) -> Dict[str, Any]:
        """Benchmark a specific batch size."""
        batch_results = {{
            "batch_size": batch_size,
            "model_name": self.model_name,
            "hardware": str(self.hardware),
            "init_times_ms": [],
            "mean_init_time_ms": 0,
            "std_init_time_ms": 0,
            "error": None
        }}
        
        try:
            # Warmup runs
            for _ in range(self.warmup_runs):
                try:
                    # Call init method with model_name and device
                    model, tokenizer, handler, queue, _ = init_method(
                        model_name=self.model_name, 
                        device=self.hardware.name,
                        cpu_label=f"{{self.hardware.name}}_benchmark"
                    )
                    
                    # Clean up
                    del model, tokenizer, handler, queue
                    
                except Exception as e:
                    self.logger.warning(f"Warmup run failed: {{e}}")
                    # Continue with measurement runs even if warmup fails
                    
            # Measurement runs
            init_times = []
            for i in range(self.measurement_runs):
                try:
                    start_time = time.perf_counter()
                    
                    # Call init method with model_name and device
                    model, tokenizer, handler, queue, _ = init_method(
                        model_name=self.model_name, 
                        device=self.hardware.name,
                        cpu_label=f"{{self.hardware.name}}_benchmark"
                    )
                    
                    end_time = time.perf_counter()
                    init_time = (end_time - start_time) * 1000  # Convert to ms
                    init_times.append(init_time)
                    
                    # Clean up
                    del model, tokenizer, handler, queue
                    
                except Exception as e:
                    error_message = f"Measurement run {{i}} failed: {{e}}"
                    self.logger.error(error_message)
                    batch_results["error"] = error_message
                    break
                    
            # Calculate statistics
            if init_times:
                import numpy as np
                batch_results["init_times_ms"] = init_times
                batch_results["mean_init_time_ms"] = np.mean(init_times)
                batch_results["std_init_time_ms"] = np.std(init_times)
                
        except Exception as e:
            error_message = f"Error benchmarking batch size {{batch_size}}: {{e}}"
            self.logger.error(error_message)
            batch_results["error"] = error_message
            
        return batch_results
        
    def process_results(self, raw_results):
        """Process raw benchmark results."""
        self.logger.info("Processing benchmark results")
        
        # Create processed results
        processed_results = {{
            "success": not bool(raw_results.get("error")),
            "model": self.model_name,
            "hardware": str(self.hardware),
            "import_time": raw_results.get("import_time", 0),
            "instantiation_time": raw_results.get("instantiation_time", 0),
            "batch_results": {{}}
        }}
        
        if raw_results.get("error"):
            processed_results["error"] = raw_results.get("error")
            return processed_results
            
        # Process batch results
        batch_metrics = {{}}
        model_fastest_time = float('inf')
        all_init_times = []
        
        for batch_key, batch_data in raw_results.get("batch_results", {{}}).items():
            if batch_data.get("error"):
                batch_metrics[batch_key] = {{
                    "success": False,
                    "error": batch_data["error"]
                }}
                continue
                
            mean_init_time = batch_data.get("mean_init_time_ms", 0)
            all_init_times.append(mean_init_time)
            
            if mean_init_time < model_fastest_time:
                model_fastest_time = mean_init_time
                
            batch_metrics[batch_key] = {{
                "success": True,
                "batch_size": batch_data.get("batch_size", 0),
                "mean_init_time_ms": mean_init_time,
                "std_init_time_ms": batch_data.get("std_init_time_ms", 0)
            }}
        
        processed_results["batch_results"] = batch_metrics
        processed_results["fastest_init_time_ms"] = model_fastest_time
        
        # Calculate overall mean init time
        if all_init_times:
            import numpy as np
            processed_results["mean_init_time_ms"] = np.mean(all_init_times)
            
        return processed_results
        
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources")
        
        # Force garbage collection to release memory
        import gc
        gc.collect()
        
        try:
            # Additional cleanup for CUDA/ROCm
            if hasattr(self.hardware, "cleanup"):
                self.hardware.cleanup()
        except:
            pass


@BenchmarkRegistry.register(
    name="efficientnet_throughput_benchmark",
    category="throughput",
    models=["efficientnet"],
    hardware=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"]
)
class EfficientnetThroughputBenchmark(EfficientnetInferenceBenchmark):
    """Benchmark for efficientnet throughput performance."""
    
    def setup(self):
        """Set up benchmark environment."""
        # Use base class setup
        super().setup()
        
        # Additional setup for throughput benchmark
        self.concurrent_workers = self.config.get("concurrent_workers", 4)
        self.logger.info(f"Throughput benchmark with {{self.concurrent_workers}} concurrent workers")
        
        return True
        
    def execute(self):
        """Execute the benchmark."""
        self.logger.info("Executing efficientnet throughput benchmark")
        
        # Get base inference results
        inference_results = super().execute()
        
        # Run concurrent benchmark
        throughput_results = self._benchmark_concurrent(inference_results)
        
        # Combine results
        combined_results = {{
            "inference_results": inference_results,
            "throughput_results": throughput_results,
            "concurrent_workers": self.concurrent_workers
        }}
        
        return combined_results
        
    def _benchmark_concurrent(self, inference_results):
        """Benchmark concurrent execution of multiple instances."""
        self.logger.info(f"Running concurrent benchmark with {{self.concurrent_workers}} workers")
        
        throughput_results = {{
            "concurrent_execution": {{
                "workers": self.concurrent_workers,
                "total_time_ms": 0,
                "throughput_models_per_second": 0
            }}
        }}
        
        try:
            import time
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            
            # Import skillset module
            skillset_path = os.path.join(self.skillset_dir, f"hf_{{self.model_name}}.py")
            
            if not os.path.exists(skillset_path):
                raise ImportError(f"Skillset file not found: {{skillset_path}}")
                
            spec = importlib.util.spec_from_file_location(f"hf_{{self.model_name}}", skillset_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get skillset class
            class_name = f"hf_{{self.model_type_safe}}"
            if not hasattr(module, class_name):
                raise AttributeError(f"Class '{{class_name}}' not found in module")
                
            skillset_class = getattr(module, class_name)
            
            # Create a list of initialization functions for concurrent execution
            # This simulates multiple users/sessions initializing the model concurrently
            init_functions = []
            
            for i in range(self.concurrent_workers):
                # Instantiate skillset class
                instance = skillset_class(resources={{"device": self.hardware.name}})
                
                # Get the appropriate init method based on hardware
                if self.hardware.name == "cpu":
                    init_method = instance.init_cpu
                elif self.hardware.name == "cuda":
                    init_method = instance.init_cuda
                elif self.hardware.name == "rocm":
                    init_method = instance.init_rocm
                elif self.hardware.name == "openvino":
                    init_method = instance.init_openvino
                elif self.hardware.name == "mps":
                    init_method = instance.init_apple
                elif self.hardware.name == "qnn":
                    init_method = instance.init_qualcomm
                else:
                    init_method = instance.init_cpu  # Fallback to CPU
                
                # Create a callable that will run the init method
                def create_init_func(init_method, model_name, hardware_name, worker_id):
                    def init_func():
                        return init_method(
                            model_name=model_name, 
                            device=hardware_name,
                            cpu_label=f"{{hardware_name}}_worker_{{worker_id}}"
                        )
                    return init_func
                
                init_functions.append(create_init_func(init_method, self.model_name, self.hardware.name, i))
            
            # Warmup run
            self.logger.info("Performing warmup run")
            for i, init_func in enumerate(init_functions):
                try:
                    model, tokenizer, handler, queue, _ = init_func()
                    del model, tokenizer, handler, queue
                    self.logger.debug(f"Warmup for worker {{i}} successful")
                except Exception as e:
                    self.logger.warning(f"Warmup for worker {{i}} failed: {{e}}")
            
            # Measurement runs
            all_concurrent_times = []
            
            for run in range(self.measurement_runs):
                self.logger.info(f"Concurrent run {{run+1}}/{{self.measurement_runs}}")
                
                # Measure concurrent execution time
                start_time = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=self.concurrent_workers) as executor:
                    futures = [executor.submit(init_func) for init_func in init_functions]
                    
                    for i, future in enumerate(futures):
                        try:
                            model, tokenizer, handler, queue, _ = future.result()
                            del model, tokenizer, handler, queue
                        except Exception as e:
                            self.logger.error(f"Worker {{i}} failed: {{e}}")
                
                end_time = time.perf_counter()
                concurrent_time = (end_time - start_time) * 1000  # Convert to ms
                all_concurrent_times.append(concurrent_time)
                
                # Force garbage collection between runs
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
            
            # Calculate throughput
            mean_concurrent_time = np.mean(all_concurrent_times)
            throughput = self.concurrent_workers / (mean_concurrent_time / 1000)  # models per second
            
            throughput_results["concurrent_execution"]["total_time_ms"] = mean_concurrent_time
            throughput_results["concurrent_execution"]["throughput_models_per_second"] = throughput
            throughput_results["concurrent_execution"]["all_run_times_ms"] = all_concurrent_times
            
        except Exception as e:
            error_message = f"Error in concurrent benchmark: {{e}}"
            self.logger.error(error_message)
            throughput_results["error"] = error_message
        
        return throughput_results
        
    def process_results(self, raw_results):
        """Process raw benchmark results."""
        self.logger.info("Processing throughput benchmark results")
        
        # Process base inference results
        inference_results = raw_results.get("inference_results", {{}})
        base_processed = super().process_results(inference_results)
        
        # Process throughput results
        throughput_data = raw_results.get("throughput_results", {{}})
        concurrent_execution = throughput_data.get("concurrent_execution", {{}})
        
        # Create processed results
        processed_results = {{
            "success": not bool(throughput_data.get("error")),
            "base_results": base_processed,
            "throughput": {{
                "concurrent_workers": raw_results.get("concurrent_workers", 0),
                "total_time_ms": concurrent_execution.get("total_time_ms", 0),
                "throughput_models_per_second": concurrent_execution.get("throughput_models_per_second", 0),
            }}
        }}
        
        if throughput_data.get("error"):
            processed_results["throughput"]["error"] = throughput_data.get("error")
        
        # Calculate speedup over sequential execution
        sequential_time = base_processed.get("fastest_init_time_ms", 0) * raw_results.get("concurrent_workers", 0)
        concurrent_time = concurrent_execution.get("total_time_ms", 0)
        
        if sequential_time > 0 and concurrent_time > 0:
            speedup = sequential_time / concurrent_time
            processed_results["throughput"]["speedup_over_sequential"] = speedup
            processed_results["throughput"]["theoretical_sequential_time_ms"] = sequential_time
        
        return processed_results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Benchmark efficientnet skillset implementation")
    parser.add_argument("--type", type=str, choices=["inference", "throughput"], default="inference",
                      help="Type of benchmark to run")
    parser.add_argument("--hardware", type=str, choices=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"], default="cpu",
                      help="Hardware to benchmark on")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                      help="Comma-separated list of batch sizes")
    parser.add_argument("--concurrent-workers", type=int, default=4,
                      help="Number of concurrent workers for throughput benchmark")
    parser.add_argument("--warmup-runs", type=int, default=2,
                      help="Number of warmup runs")
    parser.add_argument("--measurement-runs", type=int, default=5,
                      help="Number of measurement runs")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Directory for benchmark results")
    
    args = parser.parse_args()
    
    # Import required modules
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from benchmark_core import BenchmarkRunner
    except ImportError as e:
        logger.error(f"Error importing benchmark modules: {{e}}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    # Create benchmark runner
    runner = BenchmarkRunner(config={{"output_dir": args.output_dir}})
    
    # Determine benchmark name
    benchmark_name = "efficientnet_inference_benchmark" if args.type == "inference" else "efficientnet_throughput_benchmark"
    
    # Create benchmark parameters
    params = {{
        "hardware": args.hardware,
        "batch_sizes": batch_sizes,
        "concurrent_workers": args.concurrent_workers,
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs
    }}
    
    # Log benchmark parameters
    logger.info(f"Running {{benchmark_name}} with parameters:")
    for key, value in params.items():
        logger.info(f"  - {{key}}: {{value}}")
    
    # Run benchmark
    try:
        start_time = time.time()
        results = runner.execute(benchmark_name, params)
        duration = time.time() - start_time
        
        logger.info(f"Benchmark completed in {{duration:.2f}} seconds")
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = runner.save_results(f"{args.type}_{args.hardware}_{timestamp}.json")
        
        logger.info(f"Results saved to {{result_file}}")
        
        # Print summary
        if args.type == "inference":
            logger.info(f"Benchmark Summary:")
            logger.info(f"Model: efficientnet")
            logger.info(f"Hardware: {{args.hardware}}")
            logger.info(f"Import time: {{results.get('import_time', 0):.4f}} seconds")
            logger.info(f"Instantiation time: {{results.get('instantiation_time', 0):.4f}} seconds")
            logger.info(f"Mean initialization time: {{results.get('mean_init_time_ms', 0):.2f}} ms")
        else:
            throughput = results.get("throughput", {{}})
            logger.info(f"Benchmark Summary:")
            logger.info(f"Concurrent workers: {{throughput.get('concurrent_workers', 0)}}")
            logger.info(f"Total time: {{throughput.get('total_time_ms', 0):.2f}} ms")
            logger.info(f"Throughput: {{throughput.get('throughput_models_per_second', 0):.2f}} models/s")
            if "speedup_over_sequential" in throughput:
                logger.info(f"Speedup over sequential: {{throughput.get('speedup_over_sequential', 0):.2f}}x")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running benchmark: {{e}}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

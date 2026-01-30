#!/usr/bin/env python3
"""
Skillset Benchmark Module

This module provides benchmarks for the skillset implementations found in
ipfs_accelerate_py/worker/skillset.
"""

import os
import sys
import time
import json
import logging
import random
import importlib.util
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import benchmark core
from benchmark_core import BenchmarkBase, BenchmarkRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@BenchmarkRegistry.register(
    name="skillset_inference_benchmark",
    category="inference",
    models="all",  # All models
    hardware=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"]
)
class SkillsetInferenceBenchmark(BenchmarkBase):
    """Benchmark for skillset inference performance."""
    
    def setup(self):
        """Set up benchmark environment."""
        self.logger.info("Setting up skillset inference benchmark")
        
        # Extract configuration parameters
        self.model_name = self.config.get("model", "bert")
        self.batch_sizes = self.config.get("batch_sizes", [1, 2, 4, 8])
        self.warmup_runs = self.config.get("warmup_runs", 2)
        self.measurement_runs = self.config.get("measurement_runs", 5)
        self.random_sample = self.config.get("random_sample", False)
        self.sample_size = self.config.get("sample_size", 10)
        
        # Find and import skillset module
        self.skillset_module = None
        self.skillset_class = None
        
        try:
            # Determine the path to the skillset directory
            test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            self.skillset_dir = os.path.join(test_dir, "ipfs_accelerate_py", "worker", "skillset")
            
            if not os.path.exists(self.skillset_dir):
                raise ImportError(f"Skillset directory not found: {self.skillset_dir}")
            
            # If model_name is "all", get list of all models
            if self.model_name == "all":
                self.model_list = self._get_all_models()
                if self.random_sample and len(self.model_list) > self.sample_size:
                    self.model_list = random.sample(self.model_list, self.sample_size)
                self.logger.info(f"Will benchmark {len(self.model_list)} models")
            else:
                self.model_list = [self.model_name]
                
            return True
            
        except ImportError as e:
            self.logger.error(f"Required libraries or files not available: {e}")
            raise
            
        except Exception as e:
            self.logger.error(f"Error setting up benchmark: {e}")
            raise
            
    def _get_all_models(self) -> List[str]:
        """Get a list of all available models in the skillset directory."""
        models = []
        for file in os.listdir(self.skillset_dir):
            if file.startswith("hf_") and file.endswith(".py"):
                model_name = file[3:-3]  # Remove 'hf_' prefix and '.py' suffix
                models.append(model_name)
        return models
            
    def execute(self):
        """Execute the benchmark."""
        self.logger.info("Executing skillset inference benchmark")
        
        results = {}
        
        for model_name in self.model_list:
            model_results = self._benchmark_model(model_name)
            results[model_name] = model_results
            
        return results
        
    def _benchmark_model(self, model_name: str) -> Dict[str, Any]:
        """Benchmark a specific model."""
        self.logger.info(f"Benchmarking model: {model_name}")
        
        model_results = {
            "model_name": model_name,
            "hardware": str(self.hardware),
            "batch_results": {},
            "import_time": 0,
            "instantiation_time": 0,
            "error": None
        }
        
        try:
            # Import skillset module
            model_type_safe = model_name.replace('-', '_')
            skillset_path = os.path.join(self.skillset_dir, f"hf_{model_name}.py")
            
            if not os.path.exists(skillset_path):
                raise ImportError(f"Skillset file not found: {skillset_path}")
            
            import_start = time.time()
            spec = importlib.util.spec_from_file_location(f"hf_{model_name}", skillset_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import_time = time.time() - import_start
            model_results["import_time"] = import_time
            
            # Get skillset class
            class_name = f"hf_{model_type_safe}"
            if not hasattr(module, class_name):
                raise AttributeError(f"Class '{class_name}' not found in module")
                
            skillset_class = getattr(module, class_name)
            
            # Instantiate skillset class
            instance_start = time.time()
            instance = skillset_class(resources={"device": self.hardware.name})
            instantiation_time = time.time() - instance_start
            model_results["instantiation_time"] = instantiation_time
            
            # Benchmark each initialization method
            backend_results = {}
            
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
                batch_results = self._benchmark_batch(init_method, model_name, batch_size)
                backend_results[f"batch_size_{batch_size}"] = batch_results
                
            model_results["batch_results"] = backend_results
            
        except Exception as e:
            error_message = f"Error benchmarking model {model_name}: {e}"
            self.logger.error(error_message)
            model_results["error"] = error_message
            
        return model_results
        
    def _benchmark_batch(self, init_method, model_name: str, batch_size: int) -> Dict[str, Any]:
        """Benchmark a specific batch size."""
        batch_results = {
            "batch_size": batch_size,
            "model_name": model_name,
            "hardware": str(self.hardware),
            "init_times_ms": [],
            "mean_init_time_ms": 0,
            "std_init_time_ms": 0,
            "error": None
        }
        
        try:
            # Warmup runs
            for _ in range(self.warmup_runs):
                try:
                    # Call init method with model_name and device
                    # The parameters may need adjusting based on the actual method signature
                    model, tokenizer, handler, queue, _ = init_method(
                        model_name=model_name, 
                        device=self.hardware.name,
                        cpu_label=f"{self.hardware.name}_benchmark"
                    )
                    
                    # Clean up
                    del model, tokenizer, handler, queue
                    
                except Exception as e:
                    self.logger.warning(f"Warmup run failed: {e}")
                    # Continue with measurement runs even if warmup fails
                    
            # Measurement runs
            init_times = []
            for i in range(self.measurement_runs):
                try:
                    start_time = time.perf_counter()
                    
                    # Call init method with model_name and device
                    model, tokenizer, handler, queue, _ = init_method(
                        model_name=model_name, 
                        device=self.hardware.name,
                        cpu_label=f"{self.hardware.name}_benchmark"
                    )
                    
                    end_time = time.perf_counter()
                    init_time = (end_time - start_time) * 1000  # Convert to ms
                    init_times.append(init_time)
                    
                    # Clean up
                    del model, tokenizer, handler, queue
                    
                except Exception as e:
                    error_message = f"Measurement run {i} failed: {e}"
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
            error_message = f"Error benchmarking batch size {batch_size}: {e}"
            self.logger.error(error_message)
            batch_results["error"] = error_message
            
        return batch_results
        
    def process_results(self, raw_results):
        """Process raw benchmark results."""
        self.logger.info("Processing benchmark results")
        
        # Create processed results
        processed_results = {
            "success": True,
            "models": {},
            "summary": {
                "total_models": len(raw_results),
                "successful_models": 0,
                "failed_models": 0,
                "fastest_model": None,
                "slowest_model": None,
                "fastest_init_time_ms": float('inf'),
                "slowest_init_time_ms": 0,
                "mean_init_time_ms": 0
            }
        }
        
        # Process each model's results
        all_init_times = []
        
        for model_name, model_data in raw_results.items():
            # Check if model had an error
            if model_data.get("error"):
                processed_results["models"][model_name] = {
                    "success": False,
                    "error": model_data["error"]
                }
                processed_results["summary"]["failed_models"] += 1
                continue
                
            # Model was successful
            processed_results["summary"]["successful_models"] += 1
            
            # Process batch results
            batch_metrics = {}
            model_fastest_time = float('inf')
            
            for batch_key, batch_data in model_data.get("batch_results", {}).items():
                if batch_data.get("error"):
                    batch_metrics[batch_key] = {
                        "success": False,
                        "error": batch_data["error"]
                    }
                    continue
                    
                mean_init_time = batch_data.get("mean_init_time_ms", 0)
                all_init_times.append(mean_init_time)
                
                if mean_init_time < model_fastest_time:
                    model_fastest_time = mean_init_time
                    
                batch_metrics[batch_key] = {
                    "success": True,
                    "batch_size": batch_data.get("batch_size", 0),
                    "mean_init_time_ms": mean_init_time,
                    "std_init_time_ms": batch_data.get("std_init_time_ms", 0)
                }
                
            # Update fastest/slowest model tracking
            if model_fastest_time < processed_results["summary"]["fastest_init_time_ms"]:
                processed_results["summary"]["fastest_init_time_ms"] = model_fastest_time
                processed_results["summary"]["fastest_model"] = model_name
                
            # Store model results
            processed_results["models"][model_name] = {
                "success": True,
                "import_time": model_data.get("import_time", 0),
                "instantiation_time": model_data.get("instantiation_time", 0),
                "fastest_init_time_ms": model_fastest_time,
                "batch_results": batch_metrics
            }
            
            # Find slowest model
            if model_fastest_time > processed_results["summary"]["slowest_init_time_ms"]:
                processed_results["summary"]["slowest_init_time_ms"] = model_fastest_time
                processed_results["summary"]["slowest_model"] = model_name
                
        # Calculate overall mean init time
        if all_init_times:
            import numpy as np
            processed_results["summary"]["mean_init_time_ms"] = np.mean(all_init_times)
            
        return processed_results
        
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources")
        
        # Force garbage collection to release memory
        import gc
        gc.collect()
        
        try:
            # Additional cleanup for CUDA/ROCm
            if hasattr(self, "torch") and hasattr(self.torch, "cuda"):
                self.torch.cuda.empty_cache()
                self.logger.info("CUDA cache cleared")
        except:
            pass
            
@BenchmarkRegistry.register(
    name="skillset_throughput_benchmark",
    category="throughput",
    models="all",
    hardware=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"]
)
class SkillsetThroughputBenchmark(SkillsetInferenceBenchmark):
    """Benchmark for skillset throughput performance."""
    
    def setup(self):
        """Set up benchmark environment."""
        # Use base class setup
        super().setup()
        
        # Additional setup for throughput benchmark
        self.concurrent_models = self.config.get("concurrent_models", 2)
        self.logger.info(f"Throughput benchmark with {self.concurrent_models} concurrent models")
        
        return True
        
    def execute(self):
        """Execute the benchmark."""
        self.logger.info("Executing skillset throughput benchmark")
        
        # Get base inference results
        inference_results = super().execute()
        
        # For throughput benchmarking, we'll select a subset of models to run concurrently
        # For demonstration, we'll select the top 5 models by initialization time
        if len(self.model_list) >= self.concurrent_models:
            # Sort models by initialization time (ascending)
            model_times = []
            for model_name, model_data in inference_results.items():
                if not model_data.get("error"):
                    # Get the fastest initialization time across batch sizes
                    fastest_time = float('inf')
                    for batch_key, batch_data in model_data.get("batch_results", {}).items():
                        if not batch_data.get("error"):
                            mean_time = batch_data.get("mean_init_time_ms", float('inf'))
                            if mean_time < fastest_time:
                                fastest_time = mean_time
                    model_times.append((model_name, fastest_time))
            
            # Sort by initialization time
            model_times.sort(key=lambda x: x[1])
            
            # Select the fastest models for concurrent execution
            concurrent_models = [model_name for model_name, _ in model_times[:self.concurrent_models]]
        else:
            # Use all models
            concurrent_models = self.model_list.copy()
            
        # Run concurrent benchmark
        throughput_results = self._benchmark_concurrent(concurrent_models, inference_results)
        
        # Combine results
        combined_results = {
            "inference_results": inference_results,
            "throughput_results": throughput_results,
            "concurrent_models": self.concurrent_models,
            "selected_models": concurrent_models
        }
        
        return combined_results
        
    def _benchmark_concurrent(self, model_list, inference_results):
        """Benchmark concurrent execution of multiple models."""
        self.logger.info(f"Running concurrent benchmark with {len(model_list)} models")
        
        throughput_results = {
            "concurrent_execution": {
                "models": model_list,
                "total_time_ms": 0,
                "throughput_models_per_second": 0
            },
            "model_results": {}
        }
        
        try:
            # Import time module
            import time
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            
            # Create a list of initialization functions
            init_functions = []
            
            for model_name in model_list:
                # Import skillset module
                model_type_safe = model_name.replace('-', '_')
                skillset_path = os.path.join(self.skillset_dir, f"hf_{model_name}.py")
                
                if not os.path.exists(skillset_path):
                    self.logger.warning(f"Skillset file not found: {skillset_path}")
                    continue
                
                spec = importlib.util.spec_from_file_location(f"hf_{model_name}", skillset_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get skillset class
                class_name = f"hf_{model_type_safe}"
                if not hasattr(module, class_name):
                    self.logger.warning(f"Class '{class_name}' not found in module")
                    continue
                    
                skillset_class = getattr(module, class_name)
                
                # Instantiate skillset class
                instance = skillset_class(resources={"device": self.hardware.name})
                
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
                
                # Create a callable that will run the init method for this model
                def create_init_func(init_method, model_name, hardware_name):
                    def init_func():
                        return init_method(
                            model_name=model_name, 
                            device=hardware_name,
                            cpu_label=f"{hardware_name}_concurrent_benchmark"
                        )
                    return init_func
                
                init_functions.append((model_name, create_init_func(init_method, model_name, self.hardware.name)))
            
            # Warmup run
            self.logger.info("Performing warmup runs")
            for model_name, init_func in init_functions:
                try:
                    model, tokenizer, handler, queue, _ = init_func()
                    del model, tokenizer, handler, queue
                except Exception as e:
                    self.logger.warning(f"Warmup run failed for {model_name}: {e}")
            
            # Measurement runs
            all_concurrent_times = []
            
            for run in range(self.measurement_runs):
                self.logger.info(f"Concurrent run {run+1}/{self.measurement_runs}")
                model_results = {}
                
                # Measure concurrent execution time
                start_time = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=len(init_functions)) as executor:
                    futures = {}
                    for model_name, init_func in init_functions:
                        futures[executor.submit(init_func)] = model_name
                    
                    for future in futures:
                        model_name = futures[future]
                        try:
                            model, tokenizer, handler, queue, _ = future.result()
                            del model, tokenizer, handler, queue
                            model_results[model_name] = {"success": True}
                        except Exception as e:
                            model_results[model_name] = {"success": False, "error": str(e)}
                
                end_time = time.perf_counter()
                concurrent_time = (end_time - start_time) * 1000  # Convert to ms
                all_concurrent_times.append(concurrent_time)
                
                # Force garbage collection between runs
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    if hasattr(self, "torch") and hasattr(self.torch, "cuda"):
                        self.torch.cuda.empty_cache()
                except:
                    pass
            
            # Calculate throughput
            mean_concurrent_time = np.mean(all_concurrent_times)
            throughput = len(model_list) / (mean_concurrent_time / 1000)  # models per second
            
            throughput_results["concurrent_execution"]["total_time_ms"] = mean_concurrent_time
            throughput_results["concurrent_execution"]["throughput_models_per_second"] = throughput
            throughput_results["concurrent_execution"]["all_run_times_ms"] = all_concurrent_times
            throughput_results["model_results"] = model_results
            
        except Exception as e:
            error_message = f"Error in concurrent benchmark: {e}"
            self.logger.error(error_message)
            throughput_results["error"] = error_message
        
        return throughput_results
        
    def process_results(self, raw_results):
        """Process raw benchmark results."""
        self.logger.info("Processing throughput benchmark results")
        
        # Process base inference results
        inference_results = raw_results.get("inference_results", {})
        base_processed = super().process_results(inference_results)
        
        # Process throughput results
        throughput_data = raw_results.get("throughput_results", {})
        concurrent_execution = throughput_data.get("concurrent_execution", {})
        model_results = throughput_data.get("model_results", {})
        
        # Create processed results
        processed_results = {
            "success": True,
            "base_results": base_processed,
            "throughput": {
                "concurrent_models": raw_results.get("concurrent_models", 0),
                "selected_models": raw_results.get("selected_models", []),
                "total_time_ms": concurrent_execution.get("total_time_ms", 0),
                "throughput_models_per_second": concurrent_execution.get("throughput_models_per_second", 0),
                "model_results": model_results
            }
        }
        
        # Calculate speedup over sequential execution
        if base_processed and "summary" in base_processed:
            fastest_init_time = base_processed["summary"].get("fastest_init_time_ms", 0)
            if fastest_init_time > 0:
                # Calculate theoretical sequential time (sum of individual times)
                theoretical_sequential_time = 0
                for model_name in raw_results.get("selected_models", []):
                    if model_name in base_processed.get("models", {}):
                        model_data = base_processed["models"][model_name]
                        if model_data.get("success", False):
                            theoretical_sequential_time += model_data.get("fastest_init_time_ms", 0)
                
                # Calculate actual concurrent time
                actual_concurrent_time = concurrent_execution.get("total_time_ms", 0)
                
                # Calculate speedup
                if actual_concurrent_time > 0:
                    speedup = theoretical_sequential_time / actual_concurrent_time
                    processed_results["throughput"]["speedup_over_sequential"] = speedup
                    processed_results["throughput"]["theoretical_sequential_time_ms"] = theoretical_sequential_time
        
        return processed_results


def main():
    """Example usage of the skillset benchmarks."""
    import sys
    import argparse
    from benchmark_core import BenchmarkRunner
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Skillset Benchmark Tool")
    parser.add_argument("--benchmark", type=str, choices=["inference", "throughput"], default="inference",
                       help="Type of benchmark to run")
    parser.add_argument("--hardware", type=str, choices=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"], default="cpu",
                       help="Hardware to benchmark on")
    parser.add_argument("--model", type=str, default="bert",
                       help="Model to benchmark (use 'all' for all models)")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                       help="Comma-separated list of batch sizes")
    parser.add_argument("--random-sample", action="store_true",
                       help="Use random sample of models when 'all' is specified")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Number of models to sample when using random sampling")
    parser.add_argument("--concurrent-models", type=int, default=2,
                       help="Number of concurrent models for throughput benchmark")
    parser.add_argument("--warmup-runs", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--measurement-runs", type=int, default=5,
                       help="Number of measurement runs")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Directory for benchmark results")
    parser.add_argument("--output-file", type=str,
                       help="Filename for benchmark results")
    
    args = parser.parse_args()
    
    # Convert batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    # Create runner
    runner = BenchmarkRunner(config={
        "output_dir": args.output_dir
    })
    
    # Determine benchmark type
    benchmark_name = "skillset_inference_benchmark" if args.benchmark == "inference" else "skillset_throughput_benchmark"
    
    # Run benchmark
    try:
        result = runner.execute(benchmark_name, {
            "hardware": args.hardware,
            "model": args.model,
            "batch_sizes": batch_sizes,
            "random_sample": args.random_sample,
            "sample_size": args.sample_size,
            "concurrent_models": args.concurrent_models,
            "warmup_runs": args.warmup_runs,
            "measurement_runs": args.measurement_runs
        })
        
        # Print summary
        if args.benchmark == "inference":
            summary = result.get("summary", {})
            print(f"\nSkillset Inference Benchmark Summary:")
            print(f"Total models: {summary.get('total_models', 0)}")
            print(f"Successful models: {summary.get('successful_models', 0)}")
            print(f"Failed models: {summary.get('failed_models', 0)}")
            print(f"\nFastest model: {summary.get('fastest_model', 'N/A')} "
                  f"({summary.get('fastest_init_time_ms', 0):.2f} ms)")
            print(f"Slowest model: {summary.get('slowest_model', 'N/A')} "
                  f"({summary.get('slowest_init_time_ms', 0):.2f} ms)")
            print(f"Mean initialization time: {summary.get('mean_init_time_ms', 0):.2f} ms")
        else:
            throughput = result.get("throughput", {})
            print(f"\nSkillset Throughput Benchmark Summary:")
            print(f"Concurrent models: {throughput.get('concurrent_models', 0)}")
            print(f"Models: {', '.join(throughput.get('selected_models', []))}")
            print(f"Total time: {throughput.get('total_time_ms', 0):.2f} ms")
            print(f"Throughput: {throughput.get('throughput_models_per_second', 0):.2f} models/s")
            if "speedup_over_sequential" in throughput:
                print(f"Speedup over sequential: {throughput.get('speedup_over_sequential', 0):.2f}x")
        
        # Save results
        results_path = runner.save_results(args.output_file)
        print(f"\nResults saved to: {results_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
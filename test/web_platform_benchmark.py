#!/usr/bin/env python3
"""
Web Platform Benchmarking for IPFS Accelerate Python

This module implements comprehensive benchmarking for WebNN and WebGPU backends
across different model types, sizes, and configurations, with detailed performance metrics.
"""

import os
import sys
import json
import time
import argparse
import importlib
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path to import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import required packages
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed, some functionality will be limited")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not installed, some functionality will be limited")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: Transformers not installed, some functionality will be limited")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed, progress bars will be disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("web_benchmark")

# Define modality types for categorization
MODALITY_TYPES = {
    "text": ["bert", "gpt", "t5", "llama", "roberta", "distilbert", "mistral", "phi"],
    "vision": ["vit", "resnet", "detr", "convnext", "swin", "sam"],
    "audio": ["whisper", "wav2vec", "clap", "hubert", "speecht5"],
    "multimodal": ["clip", "llava", "blip", "flava", "git", "pix2struct"]
}

# Define input shapes for different modalities
DEFAULT_INPUTS = {
    "text": "The quick brown fox jumps over the lazy dog.",
    "vision": "test.jpg",
    "audio": "test.mp3",
    "multimodal": {"image": "test.jpg", "text": "What is this?"}
}

@dataclass
class WebBenchmarkResult:
    """Store web platform benchmark results for a specific configuration"""
    model_name: str
    platform: str  # webnn or webgpu
    implementation_type: str  # REAL_WEBNN, REAL_WEBGPU, SIMULATED_WEBNN, SIMULATED_WEBGPU
    modality: str
    batch_size: int
    iteration_count: int = 10
    
    # Performance metrics
    inference_time_ms: float = 0.0
    first_inference_time_ms: float = 0.0  # First inference (cold start)
    avg_inference_time_ms: float = 0.0  # Average over all iterations after first
    peak_memory_mb: float = 0.0
    throughput: float = 0.0  # Items per second
    
    # Load metrics
    model_load_time_ms: float = 0.0
    tokenization_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    # Status
    initialized: bool = False
    error: Optional[str] = None
    
    def as_dict(self) -> Dict:
        """Convert result to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "platform": self.platform,
            "implementation_type": self.implementation_type,
            "modality": self.modality,
            "batch_size": self.batch_size,
            "iteration_count": self.iteration_count,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "first_inference_time_ms": round(self.first_inference_time_ms, 2),
            "avg_inference_time_ms": round(self.avg_inference_time_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "throughput": round(self.throughput, 2),
            "model_load_time_ms": round(self.model_load_time_ms, 2),
            "tokenization_time_ms": round(self.tokenization_time_ms, 2),
            "preprocessing_time_ms": round(self.preprocessing_time_ms, 2),
            "postprocessing_time_ms": round(self.postprocessing_time_ms, 2),
            "initialized": self.initialized,
            "error": self.error
        }


@dataclass
class WebBenchmarkSuite:
    """Main benchmark suite to run and collect results"""
    results: List[WebBenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: WebBenchmarkResult) -> None:
        """Add a benchmark result to the collection"""
        self.results.append(result)
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to JSON file"""
        with open(filename, 'w') as f:
            json.dump([result.as_dict() for result in self.results], f, indent=2)
    
    def load_results(self, filename: str) -> None:
        """Load benchmark results from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.results = [WebBenchmarkResult(**item) for item in data]
    
    def print_summary(self) -> None:
        """Print a summary table of benchmark results"""
        if not self.results:
            logger.warning("No benchmark results to display")
            return
        
        # Group by platform
        webnn_results = [r for r in self.results if r.platform == "webnn" and r.initialized]
        webgpu_results = [r for r in self.results if r.platform == "webgpu" and r.initialized]
        
        # Print WebNN results
        if webnn_results:
            print("\n--- WebNN Benchmark Results ---")
            headers = ["Model", "Type", "Batch", "First (ms)", "Avg (ms)", "Throughput"]
            rows = []
            
            for result in sorted(webnn_results, key=lambda x: x.model_name):
                rows.append([
                    result.model_name,
                    result.implementation_type,
                    result.batch_size,
                    f"{result.first_inference_time_ms:.2f}",
                    f"{result.avg_inference_time_ms:.2f}",
                    f"{result.throughput:.2f}"
                ])
            
            # Print tabulated results
            print("\n".join("  ".join(row) for row in [headers] + rows))
            
        # Print WebGPU results
        if webgpu_results:
            print("\n--- WebGPU Benchmark Results ---")
            headers = ["Model", "Type", "Batch", "First (ms)", "Avg (ms)", "Throughput"]
            rows = []
            
            for result in sorted(webgpu_results, key=lambda x: x.model_name):
                rows.append([
                    result.model_name,
                    result.implementation_type,
                    result.batch_size,
                    f"{result.first_inference_time_ms:.2f}",
                    f"{result.avg_inference_time_ms:.2f}",
                    f"{result.throughput:.2f}"
                ])
            
            # Print tabulated results
            print("\n".join("  ".join(row) for row in [headers] + rows))
        
        # Print errors if any
        failed_results = [r for r in self.results if not r.initialized]
        if failed_results:
            print("\n--- Failed Benchmarks ---")
            for result in failed_results:
                print(f"{result.model_name} ({result.platform}): {result.error}")
    
    def generate_comparison_chart(self, output_dir: str = "web_benchmark_charts") -> None:
        """Generate performance comparison charts between WebNN and WebGPU"""
        if not self.results:
            logger.warning("No benchmark results to chart")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Group results by model name
        model_results = {}
        for result in self.results:
            if result.initialized:  # Only include successful results
                if result.model_name not in model_results:
                    model_results[result.model_name] = []
                model_results[result.model_name].append(result)
        
        # Generate comparison chart for each model
        for model_name, results in model_results.items():
            # Only create comparisons if we have both WebNN and WebGPU results
            webnn_results = [r for r in results if r.platform == "webnn"]
            webgpu_results = [r for r in results if r.platform == "webgpu"]
            
            if not webnn_results or not webgpu_results:
                continue
                
            # Get batch sizes for both platforms
            batch_sizes = sorted(list(set([r.batch_size for r in results])))
            
            # Creating figure for inference time comparison
            plt.figure(figsize=(10, 6))
            
            # Width for the bars
            bar_width = 0.35
            opacity = 0.8
            
            # Positions for the bars
            index = np.arange(len(batch_sizes))
            
            # Get average inference times for each batch size
            webnn_times = []
            webgpu_times = []
            
            for batch_size in batch_sizes:
                webnn_batch_results = [r for r in webnn_results if r.batch_size == batch_size]
                webgpu_batch_results = [r for r in webgpu_results if r.batch_size == batch_size]
                
                webnn_time = webnn_batch_results[0].avg_inference_time_ms if webnn_batch_results else 0
                webgpu_time = webgpu_batch_results[0].avg_inference_time_ms if webgpu_batch_results else 0
                
                webnn_times.append(webnn_time)
                webgpu_times.append(webgpu_time)
            
            # Create the bar chart
            plt.bar(index, webnn_times, bar_width, alpha=opacity, color='b', label='WebNN')
            plt.bar(index + bar_width, webgpu_times, bar_width, alpha=opacity, color='g', label='WebGPU')
            
            plt.xlabel('Batch Size')
            plt.ylabel('Inference Time (ms)')
            plt.title(f'WebNN vs WebGPU Inference Time - {model_name}')
            plt.xticks(index + bar_width / 2, [str(b) for b in batch_sizes])
            plt.legend()
            plt.tight_layout()
            
            # Save the chart
            chart_path = os.path.join(output_dir, f"{model_name}_comparison_{timestamp}.png")
            plt.savefig(chart_path)
            plt.close()
            
            # Also create throughput comparison chart
            plt.figure(figsize=(10, 6))
            
            # Get throughput for each batch size
            webnn_throughput = []
            webgpu_throughput = []
            
            for batch_size in batch_sizes:
                webnn_batch_results = [r for r in webnn_results if r.batch_size == batch_size]
                webgpu_batch_results = [r for r in webgpu_results if r.batch_size == batch_size]
                
                webnn_tp = webnn_batch_results[0].throughput if webnn_batch_results else 0
                webgpu_tp = webgpu_batch_results[0].throughput if webgpu_batch_results else 0
                
                webnn_throughput.append(webnn_tp)
                webgpu_throughput.append(webgpu_tp)
            
            # Create the bar chart
            plt.bar(index, webnn_throughput, bar_width, alpha=opacity, color='b', label='WebNN')
            plt.bar(index + bar_width, webgpu_throughput, bar_width, alpha=opacity, color='g', label='WebGPU')
            
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (items/sec)')
            plt.title(f'WebNN vs WebGPU Throughput - {model_name}')
            plt.xticks(index + bar_width / 2, [str(b) for b in batch_sizes])
            plt.legend()
            plt.tight_layout()
            
            # Save the chart
            chart_path = os.path.join(output_dir, f"{model_name}_throughput_{timestamp}.png")
            plt.savefig(chart_path)
            plt.close()
        
        # Create modality comparison chart
        self._generate_modality_comparison_chart(output_dir, timestamp)
    
    def _generate_modality_comparison_chart(self, output_dir: str, timestamp: str) -> None:
        """Generate performance comparison by modality"""
        # Group results by modality
        modality_results = {}
        for result in self.results:
            if result.initialized:
                if result.modality not in modality_results:
                    modality_results[result.modality] = {"webnn": [], "webgpu": []}
                
                if result.platform == "webnn":
                    modality_results[result.modality]["webnn"].append(result)
                elif result.platform == "webgpu":
                    modality_results[result.modality]["webgpu"].append(result)
        
        # Creating figure for modality comparison
        plt.figure(figsize=(12, 8))
        
        # Get modalities with results for both platforms
        modalities = []
        webnn_avg_times = []
        webgpu_avg_times = []
        
        for modality, platforms in modality_results.items():
            if platforms["webnn"] and platforms["webgpu"]:
                modalities.append(modality)
                
                # Calculate average inference time for each platform
                webnn_avg = sum(r.avg_inference_time_ms for r in platforms["webnn"]) / len(platforms["webnn"])
                webgpu_avg = sum(r.avg_inference_time_ms for r in platforms["webgpu"]) / len(platforms["webgpu"])
                
                webnn_avg_times.append(webnn_avg)
                webgpu_avg_times.append(webgpu_avg)
        
        if not modalities:
            logger.warning("No modalities with both WebNN and WebGPU results")
            return
            
        # Width for the bars
        bar_width = 0.35
        opacity = 0.8
        
        # Positions for the bars
        index = np.arange(len(modalities))
        
        # Create the bar chart
        plt.bar(index, webnn_avg_times, bar_width, alpha=opacity, color='b', label='WebNN')
        plt.bar(index + bar_width, webgpu_avg_times, bar_width, alpha=opacity, color='g', label='WebGPU')
        
        plt.xlabel('Modality')
        plt.ylabel('Average Inference Time (ms)')
        plt.title('WebNN vs WebGPU Performance by Modality')
        plt.xticks(index + bar_width / 2, [m.capitalize() for m in modalities])
        plt.legend()
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(output_dir, f"modality_comparison_{timestamp}.png")
        plt.savefig(chart_path)
        plt.close()


class WebPlatformBenchmark:
    """
    Main class for benchmarking WebNN and WebGPU capabilities across different models.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize web platform benchmarking framework.
        
        Args:
            resources: Dictionary of shared resources
            metadata: Configuration metadata
        """
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Define web platforms to benchmark
        self.web_platforms = ["webnn", "webgpu"]
        
        # Import skill test modules
        self.skill_modules = self._import_skill_modules()
        
        # Setup paths for results
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.test_dir, "web_benchmark_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {}
        
    def _import_skill_modules(self):
        """Import all skill test modules from the skills folder."""
        skills_dir = os.path.join(self.test_dir, "skills")
        skill_modules = {}
        
        if not os.path.exists(skills_dir):
            logger.warning(f"Warning: Skills directory not found at {skills_dir}")
            return skill_modules
            
        for filename in os.listdir(skills_dir):
            if filename.startswith("test_hf_") and filename.endswith(".py"):
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"test.skills.{module_name}")
                    skill_modules[module_name] = module
                except ImportError as e:
                    logger.warning(f"Error importing {module_name}: {e}")
                    
        return skill_modules
        
    def detect_model_modality(self, model_name: str) -> str:
        """Detect model modality based on name patterns.
        
        Args:
            model_name: The model name to categorize
            
        Returns:
            String modality: "text", "vision", "audio", "multimodal", or "unknown"
        """
        model_name_lower = model_name.lower()
        
        for modality, patterns in MODALITY_TYPES.items():
            for pattern in patterns:
                if pattern.lower() in model_name_lower:
                    return modality
                    
        return "unknown"
        
    def benchmark_model_on_platform(self,
                                   model_name: str,
                                   platform: str,
                                   batch_sizes: List[int] = None,
                                   iterations: int = 10,
                                   warmup_iterations: int = 3) -> List[WebBenchmarkResult]:
        """Benchmark a model on a specific web platform.
        
        Args:
            model_name: Name of the model to benchmark
            platform: Web platform to benchmark on ("webnn" or "webgpu")
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations to run for each benchmark
            warmup_iterations: Number of warmup iterations before timing
            
        Returns:
            List of WebBenchmarkResult objects, one for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 8]  # Default batch sizes
            
        if platform not in self.web_platforms:
            logger.error(f"Unsupported platform: {platform}")
            return []
            
        # Clean up model name for module lookup
        module_name = model_name
        if model_name.startswith("test_"):
            module_name = model_name
        elif not model_name.startswith("test_hf_"):
            module_name = f"test_hf_{model_name}"
            
        # Get the test module
        if module_name not in self.skill_modules:
            logger.error(f"Test module not found for {model_name}")
            return []
            
        module = self.skill_modules[module_name]
        
        # Get the test class
        test_class = None
        for attr_name in dir(module):
            if attr_name.startswith("Test") and not attr_name.startswith("TestCase"):
                test_class = getattr(module, attr_name)
                break
                
        if test_class is None:
            logger.error(f"Test class not found in module {module_name}")
            return []
            
        # Run benchmarks for each batch size
        results = []
        
        # Detect modality
        modality = self.detect_model_modality(model_name)
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking {model_name} on {platform} with batch size {batch_size}")
            
            # Initialize a new benchmark result
            benchmark_result = WebBenchmarkResult(
                model_name=model_name,
                platform=platform,
                implementation_type="UNKNOWN",
                modality=modality,
                batch_size=batch_size,
                iteration_count=iterations
            )
            
            try:
                # Initialize the test instance
                test_instance = test_class()
                
                # Record model load time
                start_load_time = time.time()
                
                # Initialize the model on the appropriate platform
                if platform == "webnn":
                    if hasattr(test_instance, "init_webnn"):
                        platform_func = "init_webnn"
                    else:
                        benchmark_result.error = "Model does not support WebNN"
                        results.append(benchmark_result)
                        continue
                elif platform == "webgpu":
                    if hasattr(test_instance, "init_webgpu"):
                        platform_func = "init_webgpu"
                    else:
                        benchmark_result.error = "Model does not support WebGPU"
                        results.append(benchmark_result)
                        continue
                
                # Initialize the model
                init_func = getattr(test_instance, platform_func)
                endpoint, processor, handler, queue, _ = init_func()
                
                end_load_time = time.time()
                benchmark_result.model_load_time_ms = (end_load_time - start_load_time) * 1000
                
                # Get test input based on modality
                if modality in DEFAULT_INPUTS:
                    test_input = DEFAULT_INPUTS[modality]
                else:
                    test_input = "Example input"
                
                # Create batched input if needed
                if batch_size > 1:
                    if isinstance(test_input, str):
                        test_input = [test_input] * batch_size
                    elif isinstance(test_input, dict):
                        # For multimodal input, batch each component
                        batched_input = {}
                        for key, value in test_input.items():
                            batched_input[key] = [value] * batch_size
                        test_input = batched_input
                
                # Measure preprocessing time
                start_preprocess = time.time()
                if hasattr(processor, "preprocess"):
                    processed_input = processor.preprocess(test_input)
                else:
                    processed_input = test_input
                end_preprocess = time.time()
                benchmark_result.preprocessing_time_ms = (end_preprocess - start_preprocess) * 1000
                
                # Warmup iterations
                logger.info(f"Running {warmup_iterations} warmup iterations")
                for _ in range(warmup_iterations):
                    _ = handler(processed_input)
                
                # First inference (cold start) timing
                start_first = time.time()
                first_result = handler(processed_input)
                end_first = time.time()
                benchmark_result.first_inference_time_ms = (end_first - start_first) * 1000
                
                # Get implementation type from result if available
                if isinstance(first_result, dict) and "implementation_type" in first_result:
                    benchmark_result.implementation_type = first_result["implementation_type"]
                
                # Main benchmark iterations
                logger.info(f"Running {iterations} benchmark iterations")
                start_time = time.time()
                
                iteration_times = []
                for i in range(iterations):
                    iter_start = time.time()
                    _ = handler(processed_input)
                    iter_end = time.time()
                    iteration_times.append((iter_end - iter_start) * 1000)
                
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                total_items = iterations * batch_size
                
                benchmark_result.inference_time_ms = total_time * 1000
                benchmark_result.avg_inference_time_ms = sum(iteration_times) / len(iteration_times)
                benchmark_result.throughput = total_items / total_time
                benchmark_result.initialized = True
                
                # Estimate memory usage (if available from the result)
                if isinstance(first_result, dict) and "memory_usage_mb" in first_result:
                    benchmark_result.peak_memory_mb = first_result["memory_usage_mb"]
                else:
                    # Placeholder for memory usage
                    benchmark_result.peak_memory_mb = 0
                
                results.append(benchmark_result)
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name} on {platform}: {str(e)}")
                benchmark_result.error = str(e)
                benchmark_result.initialized = False
                results.append(benchmark_result)
                traceback.print_exc()
        
        return results
    
    def run_benchmark_suite(self,
                           models: List[str],
                           batch_sizes: List[int] = None,
                           platforms: List[str] = None,
                           parallel: bool = False,
                           max_workers: int = 4,
                           iterations: int = 10,
                           output_file: str = None) -> WebBenchmarkSuite:
        """Run benchmarks for multiple models on specified web platforms.
        
        Args:
            models: List of model names to benchmark
            batch_sizes: List of batch sizes to test
            platforms: List of platforms to test ("webnn", "webgpu")
            parallel: Whether to run benchmarks in parallel
            max_workers: Maximum number of parallel workers
            iterations: Number of iterations for each benchmark
            output_file: Path to save benchmark results
            
        Returns:
            WebBenchmarkSuite with all benchmark results
        """
        # Set defaults
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        
        if platforms is None:
            platforms = self.web_platforms
            
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.results_dir, f"web_benchmark_{timestamp}.json")
            
        # Initialize benchmark suite
        suite = WebBenchmarkSuite()
        
        # Determine total number of benchmark configurations
        total_configs = len(models) * len(platforms) * len(batch_sizes)
        logger.info(f"Running {total_configs} benchmark configurations across {len(models)} models")
        
        # Run benchmarks
        if parallel and total_configs > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for model in models:
                    for platform in platforms:
                        future = executor.submit(
                            self.benchmark_model_on_platform,
                            model_name=model,
                            platform=platform,
                            batch_sizes=batch_sizes,
                            iterations=iterations
                        )
                        futures.append((future, model, platform))
                
                for future, model, platform in tqdm(futures) if HAS_TQDM else futures:
                    try:
                        results = future.result()
                        for result in results:
                            suite.add_result(result)
                    except Exception as e:
                        logger.error(f"Error benchmarking {model} on {platform}: {str(e)}")
        else:
            for model in models:
                for platform in platforms:
                    results = self.benchmark_model_on_platform(
                        model_name=model,
                        platform=platform,
                        batch_sizes=batch_sizes,
                        iterations=iterations
                    )
                    for result in results:
                        suite.add_result(result)
        
        # Save results
        logger.info(f"Saving benchmark results to {output_file}")
        suite.save_results(output_file)
        
        return suite
    
    def run_comparative_benchmark(self,
                                 model_filter: str = None,
                                 modality: str = None,
                                 batch_sizes: List[int] = None,
                                 iterations: int = 10,
                                 parallel: bool = False) -> WebBenchmarkSuite:
        """Run comparative benchmarks with configurable filters.
        
        Args:
            model_filter: Filter models by name (substring match)
            modality: Filter models by modality
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations for each benchmark
            parallel: Whether to run benchmarks in parallel
            
        Returns:
            WebBenchmarkSuite with all benchmark results
        """
        # Get all available models
        all_models = list(self.skill_modules.keys())
        models = []
        
        # Apply filters
        for module_name in all_models:
            # Convert module name to model name
            if module_name.startswith("test_hf_"):
                model_name = module_name[8:]  # Remove "test_hf_" prefix
            else:
                model_name = module_name
                
            # Apply model name filter
            if model_filter and model_filter.lower() not in model_name.lower():
                continue
                
            # Apply modality filter
            if modality and modality != "all":
                detected_modality = self.detect_model_modality(model_name)
                if detected_modality != modality:
                    continue
                    
            # Check if model supports web platforms
            module = self.skill_modules[module_name]
            test_class = None
            for attr_name in dir(module):
                if attr_name.startswith("Test") and not attr_name.startswith("TestCase"):
                    test_class = getattr(module, attr_name)
                    break
                    
            if test_class:
                try:
                    test_instance = test_class()
                    has_webnn = hasattr(test_instance, "init_webnn")
                    has_webgpu = hasattr(test_instance, "init_webgpu")
                    
                    if has_webnn or has_webgpu:
                        models.append(model_name)
                except Exception:
                    # Skip models that fail to initialize
                    continue
        
        if not models:
            logger.warning("No models found matching the specified filters")
            return WebBenchmarkSuite()
            
        logger.info(f"Running comparative benchmark on {len(models)} models")
        
        # Run benchmark suite
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.results_dir, f"web_benchmark_comparative_{timestamp}.json")
        
        suite = self.run_benchmark_suite(
            models=models,
            batch_sizes=batch_sizes,
            platforms=self.web_platforms,
            parallel=parallel,
            iterations=iterations,
            output_file=output_file
        )
        
        return suite
        

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Web Platform Benchmarking Tool")
    
    # Main benchmark selection
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument("--model", type=str, help="Benchmark a specific model")
    benchmark_group.add_argument("--modality", type=str, 
                              choices=["text", "vision", "audio", "multimodal", "all"],
                              help="Benchmark models from a specific modality")
    benchmark_group.add_argument("--comparative", action="store_true", 
                              help="Run comparative benchmark across platforms")
    
    # Benchmark parameters
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32],
                      help="Batch sizes to benchmark (default: 1, 8, 32)")
    parser.add_argument("--iterations", type=int, default=10,
                      help="Number of iterations per benchmark (default: 10)")
    parser.add_argument("--warmup", type=int, default=3,
                      help="Number of warmup iterations (default: 3)")
    parser.add_argument("--parallel", action="store_true",
                      help="Run benchmarks in parallel")
    
    # Output options
    parser.add_argument("--output", type=str,
                      help="Custom output file for benchmark results")
    parser.add_argument("--chart-dir", type=str, default="web_benchmark_charts",
                      help="Directory for benchmark charts")
    parser.add_argument("--no-charts", action="store_true",
                      help="Disable chart generation")
    
    # List available models
    parser.add_argument("--list-models", action="store_true",
                      help="List available models for benchmarking")
                      
    # Platform selection
    parser.add_argument("--platform", type=str, choices=["webnn", "webgpu", "both"],
                      default="both", help="Web platform to benchmark")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Create benchmarking framework
    benchmark = WebPlatformBenchmark()
    
    # List models if requested
    if args.list_models:
        # Get all available models grouped by modality
        all_modules = benchmark.skill_modules
        available_models = {}
        
        for module_name in all_modules:
            if module_name.startswith("test_hf_"):
                model_name = module_name[8:]  # Remove "test_hf_" prefix
            else:
                model_name = module_name
                
            modality = benchmark.detect_model_modality(model_name)
            
            if modality not in available_models:
                available_models[modality] = []
                
            available_models[modality].append(model_name)
        
        # Print models by modality
        print("Available models for benchmarking:")
        for modality, models in available_models.items():
            print(f"\n{modality.upper()} ({len(models)}):")
            for model in sorted(models):
                # Check web platform support
                module = all_modules[f"test_hf_{model}"]
                test_class = None
                for attr_name in dir(module):
                    if attr_name.startswith("Test") and not attr_name.startswith("TestCase"):
                        test_class = getattr(module, attr_name)
                        break
                
                if test_class:
                    try:
                        test_instance = test_class()
                        webnn = "✓" if hasattr(test_instance, "init_webnn") else "✗"
                        webgpu = "✓" if hasattr(test_instance, "init_webgpu") else "✗"
                        print(f"- {model} (WebNN: {webnn}, WebGPU: {webgpu})")
                    except Exception:
                        print(f"- {model} (initialization error)")
        
        return
    
    # Determine platforms to benchmark
    platforms = None
    if args.platform == "webnn":
        platforms = ["webnn"]
    elif args.platform == "webgpu":
        platforms = ["webgpu"]
    # else "both" is the default, and None will use both platforms
    
    # Run benchmark based on command line options
    if args.model:
        # Benchmark a specific model
        logger.info(f"Benchmarking model: {args.model}")
        
        suite = benchmark.run_benchmark_suite(
            models=[args.model],
            batch_sizes=args.batch_sizes,
            platforms=platforms,
            parallel=args.parallel,
            iterations=args.iterations,
            output_file=args.output
        )
        
    elif args.modality:
        # Benchmark models from a specific modality
        logger.info(f"Benchmarking {args.modality} models")
        
        # Get models for the specified modality
        modality_models = []
        for module_name in benchmark.skill_modules:
            if module_name.startswith("test_hf_"):
                model_name = module_name[8:]  # Remove "test_hf_" prefix
                if benchmark.detect_model_modality(model_name) == args.modality or args.modality == "all":
                    modality_models.append(model_name)
        
        # Limit to a reasonable number of models
        if len(modality_models) > 5:
            logger.info(f"Limiting to 5 models out of {len(modality_models)} available")
            modality_models = modality_models[:5]
        
        if not modality_models:
            logger.error(f"No models found for modality: {args.modality}")
            return
            
        suite = benchmark.run_benchmark_suite(
            models=modality_models,
            batch_sizes=args.batch_sizes,
            platforms=platforms,
            parallel=args.parallel,
            iterations=args.iterations,
            output_file=args.output
        )
        
    elif args.comparative:
        # Run comprehensive comparative benchmark
        logger.info("Running comparative benchmark")
        
        suite = benchmark.run_comparative_benchmark(
            modality=None,  # All modalities
            batch_sizes=args.batch_sizes,
            iterations=args.iterations,
            parallel=args.parallel
        )
        
    else:
        # Default: run a small comparative benchmark
        logger.info("Running default comparative benchmark")
        
        # Select representative models from each modality
        representative_models = [
            "bert",    # Text
            "vit",     # Vision
            "whisper", # Audio
            "clip"     # Multimodal
        ]
        
        # Filter to available models
        available_models = []
        for model in representative_models:
            module_name = f"test_hf_{model}"
            if module_name in benchmark.skill_modules:
                available_models.append(model)
        
        if not available_models:
            logger.error("No representative models found in the skills directory")
            return
            
        suite = benchmark.run_benchmark_suite(
            models=available_models,
            batch_sizes=args.batch_sizes,
            platforms=platforms,
            parallel=args.parallel,
            iterations=args.iterations,
            output_file=args.output
        )
    
    # Print summary and generate charts
    suite.print_summary()
    
    if not args.no_charts:
        logger.info(f"Generating comparison charts in {args.chart_dir}")
        suite.generate_comparison_chart(args.chart_dir)


if __name__ == "__main__":
    main()
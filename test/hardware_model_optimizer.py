#!/usr/bin/env python3
"""
Hardware-Model Optimizer

This script provides throughput and batch size optimization for models based on
hardware platform capabilities and validation results. It identifies optimal
configurations for running models on different hardware.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import our modules with graceful degradation
try:
    from model_validation_tracker import ModelValidationTracker
    HAS_TRACKER = True
except ImportError:
    logger.warning("ModelValidationTracker not available. Limited functionality.")
    HAS_TRACKER = False

try:
    from hardware_detection import detect_hardware_with_comprehensive_checks
    HAS_HARDWARE_DETECTION = True
except ImportError:
    logger.warning("Hardware detection module not available. Limited functionality.")
    HAS_HARDWARE_DETECTION = False

try:
    from resource_pool import get_global_resource_pool
    HAS_RESOURCE_POOL = True
except ImportError:
    logger.warning("ResourcePool not available. Limited functionality.")
    HAS_RESOURCE_POOL = False

# Define constants
DEFAULT_DB_PATH = "./model_validation.db"
DEFAULT_OUTPUT_DIR = "./optimization_results"
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_SEQUENCE_LENGTHS = [32, 64, 128, 256, 512]
DEFAULT_WARMUP_ITERATIONS = 3
DEFAULT_BENCHMARK_ITERATIONS = 10
DEFAULT_TIMEOUT = 300  # seconds

class HardwareModelOptimizer:
    """
    Optimizer for hardware-model combinations that provides batch size
    and throughput optimization based on hardware capabilities.
    """
    
    def __init__(self,
                 db_path: str = DEFAULT_DB_PATH,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 use_resource_pool: bool = True,
                 detect_hardware: bool = True):
        """
        Initialize the optimizer.
        
        Args:
            db_path: Path to the model validation database
            output_dir: Directory for optimization results
            use_resource_pool: Whether to use the ResourcePool for model caching
            detect_hardware: Whether to detect available hardware automatically
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.use_resource_pool = use_resource_pool
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize validation tracker if available
        self.tracker = None
        if HAS_TRACKER:
            try:
                self.tracker = ModelValidationTracker(db_path=db_path, create_if_missing=False)
                logger.info(f"Connected to validation database at {db_path}")
            except Exception as e:
                logger.warning(f"Could not connect to validation database: {e}")
        
        # Initialize resource pool if available and requested
        self.resource_pool = None
        if HAS_RESOURCE_POOL and use_resource_pool:
            try:
                self.resource_pool = get_global_resource_pool()
                logger.info("Initialized ResourcePool for model caching")
            except Exception as e:
                logger.warning(f"Could not initialize ResourcePool: {e}")
        
        # Detect available hardware if requested
        self.available_hardware = {}
        if detect_hardware and HAS_HARDWARE_DETECTION:
            self._detect_hardware()
    
    def _detect_hardware(self) -> Dict[str, bool]:
        """Detect available hardware platforms"""
        try:
            hardware_info = detect_hardware_with_comprehensive_checks()
            self.available_hardware = {k: v for k, v in hardware_info.items() 
                                      if k in ["cpu", "cuda", "mps", "rocm", "openvino", "webnn", "webgpu"]}
            
            logger.info(f"Detected hardware: {', '.join(k for k, v in self.available_hardware.items() if v)}")
            return self.available_hardware
        except Exception as e:
            logger.error(f"Error during hardware detection: {e}")
            # Fallback to basic detection
            self._basic_hardware_detection()
            return self.available_hardware
    
    def _basic_hardware_detection(self):
        """Basic hardware detection as fallback"""
        logger.info("Using basic hardware detection")
        
        # CPU is always available
        self.available_hardware["cpu"] = True
        
        # Check for PyTorch and CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.available_hardware["cuda"] = True
                logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.available_hardware["mps"] = True
                logger.info("MPS (Apple Silicon) available")
        except ImportError:
            logger.warning("PyTorch not available for hardware detection")
        
        # Check for OpenVINO
        try:
            import openvino
            self.available_hardware["openvino"] = True
            logger.info(f"OpenVINO available (version {openvino.__version__})")
        except ImportError:
            pass
    
    def get_model_hardware_recommendations(self, model_name: str) -> Dict[str, Any]:
        """
        Get hardware recommendations for a model from the validation tracker.
        
        Args:
            model_name: Model name
            
        Returns:
            Dictionary with hardware recommendations
        """
        if not self.tracker:
            logger.warning("Validation tracker not available for hardware recommendations")
            return {}
        
        try:
            return self.tracker.get_model_hardware_recommendations(model_name)
        except Exception as e:
            logger.error(f"Error getting hardware recommendations: {e}")
            return {}
    
    def find_optimal_batch_size(self, 
                              model_name: str,
                              family: str = None,
                              hardware_platform: str = None,
                              batch_sizes: List[int] = None,
                              sequence_lengths: List[int] = None,
                              warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
                              benchmark_iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
                              timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """
        Find the optimal batch size for a model on a specific hardware platform.
        
        Args:
            model_name: Model name
            family: Model family (embedding, text_generation, vision, audio, multimodal)
            hardware_platform: Hardware platform to optimize for
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test (for text models)
            warmup_iterations: Number of warmup iterations before timing
            benchmark_iterations: Number of iterations to measure
            timeout: Maximum time in seconds for a benchmark
            
        Returns:
            Dictionary with optimization results
        """
        # Determine model family if not provided
        if not family and self.tracker:
            try:
                # Query validation database for model family
                self.tracker.cursor.execute("SELECT family FROM models WHERE name = ?", (model_name,))
                result = self.tracker.cursor.fetchone()
                if result:
                    family = result["family"]
            except Exception as e:
                logger.warning(f"Error querying model family: {e}")
        
        # Try to infer family from model name if still not available
        if not family:
            model_lower = model_name.lower()
            if "bert" in model_lower or "roberta" in model_lower or "distilbert" in model_lower:
                family = "embedding"
            elif "gpt" in model_lower or "llama" in model_lower or "t5" in model_lower:
                family = "text_generation"
            elif "vit" in model_lower or "resnet" in model_lower or "convnext" in model_lower:
                family = "vision"
            elif "whisper" in model_lower or "wav2vec" in model_lower or "hubert" in model_lower:
                family = "audio"
            elif "clip" in model_lower or "llava" in model_lower or "blip" in model_lower:
                family = "multimodal"
            else:
                # Default to text generation if unclear
                family = "text_generation"
                logger.warning(f"Could not determine model family for {model_name}, defaulting to {family}")
        
        # Determine hardware platform if not provided
        if not hardware_platform:
            # Use hardware recommendations if available
            recommendations = self.get_model_hardware_recommendations(model_name)
            
            if recommendations and "recommendations" in recommendations:
                if "best_overall" in recommendations["recommendations"] and recommendations["recommendations"]["best_overall"]:
                    hardware_platform = recommendations["recommendations"]["best_overall"]["hardware"]
                    logger.info(f"Using recommended hardware platform: {hardware_platform}")
                elif "best_for_performance" in recommendations["recommendations"] and recommendations["recommendations"]["best_for_performance"]:
                    hardware_platform = recommendations["recommendations"]["best_for_performance"]["hardware"]
                    logger.info(f"Using recommended hardware platform for performance: {hardware_platform}")
            
            # Fallback to best available hardware if no recommendations
            if not hardware_platform:
                if "cuda" in self.available_hardware and self.available_hardware["cuda"]:
                    hardware_platform = "cuda"
                elif "mps" in self.available_hardware and self.available_hardware["mps"]:
                    hardware_platform = "mps"
                elif "rocm" in self.available_hardware and self.available_hardware["rocm"]:
                    hardware_platform = "rocm"
                elif "openvino" in self.available_hardware and self.available_hardware["openvino"]:
                    hardware_platform = "openvino"
                else:
                    hardware_platform = "cpu"
                
                logger.info(f"No hardware recommendations available, using best available: {hardware_platform}")
        
        # Set default batch sizes if not provided
        if not batch_sizes:
            batch_sizes = DEFAULT_BATCH_SIZES
        
        # Set default sequence lengths if not provided (for text models)
        if not sequence_lengths and family in ["embedding", "text_generation"]:
            sequence_lengths = DEFAULT_SEQUENCE_LENGTHS
        
        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run benchmarks with hardware_benchmark_runner.py
            logger.info(f"Running batch size optimization for {model_name} on {hardware_platform}...")
            
            # Build the command for hardware_benchmark_runner.py
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardware_benchmark_runner.py"),
                "--output-dir", temp_dir,
                "--model-families", family,
                "--hardware", hardware_platform,
                "--batch-sizes"
            ] + [str(b) for b in batch_sizes] + [
                "--warmup", str(warmup_iterations),
                "--iterations", str(benchmark_iterations),
                "--timeout", str(timeout)
            ]
            
            if self.use_resource_pool:
                cmd.append("--use-resource-pool")
            
            try:
                # Run the benchmark command
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Load benchmark results
                results_file = os.path.join(temp_dir, "benchmark_results.json")
                if not os.path.exists(results_file):
                    logger.error(f"Benchmark results file not found: {results_file}")
                    return {"status": "error", "message": "Benchmark results file not found"}
                
                with open(results_file, 'r') as f:
                    benchmark_data = json.load(f)
                    
                # Process benchmark results to find optimal batch size
                return self._analyze_batch_size_results(benchmark_data, model_name, family, hardware_platform)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running benchmark command: {e}")
                logger.error(f"Command output: {e.stdout}")
                logger.error(f"Command error: {e.stderr}")
                return {"status": "error", "message": f"Benchmark process failed: {e}"}
            
            except Exception as e:
                logger.error(f"Error during batch size optimization: {e}")
                return {"status": "error", "message": f"Optimization failed: {e}"}
    
    def _analyze_batch_size_results(self, benchmark_data: Dict, model_name: str, 
                                  family: str, hardware_platform: str) -> Dict[str, Any]:
        """Analyze benchmark results to find optimal batch size"""
        if not benchmark_data or "benchmarks" not in benchmark_data:
            return {"status": "error", "message": "Invalid benchmark data format"}
        
        try:
            # Extract benchmark results for the model
            model_results = {}
            if family in benchmark_data["benchmarks"]:
                for model, hw_results in benchmark_data["benchmarks"][family].items():
                    if model == model_name and hardware_platform in hw_results:
                        model_results = hw_results[hardware_platform]
                        break
            
            if not model_results or "benchmark_results" not in model_results:
                return {"status": "error", "message": "No benchmark results found for model"}
            
            # Analyze batch size performance
            batch_results = {}
            throughput_by_batch = {}
            latency_by_batch = {}
            memory_by_batch = {}
            
            for config, config_results in model_results["benchmark_results"].items():
                if config_results.get("status") != "completed":
                    continue
                
                # Extract batch size from config key (e.g., "batch_1_seq_32")
                if "batch_" in config:
                    parts = config.split("_")
                    batch_idx = parts.index("batch") + 1
                    seq_idx = parts.index("seq") + 1 if "seq" in parts else None
                    
                    if batch_idx < len(parts):
                        try:
                            batch_size = int(parts[batch_idx])
                            seq_length = int(parts[seq_idx]) if seq_idx and seq_idx < len(parts) else None
                            
                            # Initialize batch entry if needed
                            if batch_size not in batch_results:
                                batch_results[batch_size] = {}
                                throughput_by_batch[batch_size] = []
                                latency_by_batch[batch_size] = []
                                memory_by_batch[batch_size] = []
                            
                            # Add sequence length entry if applicable
                            if seq_length:
                                if seq_length not in batch_results[batch_size]:
                                    batch_results[batch_size][seq_length] = {}
                                
                                batch_results[batch_size][seq_length] = {
                                    "latency": config_results.get("avg_latency", 0) * 1000,  # ms
                                    "throughput": config_results.get("throughput", 0),
                                    "memory": config_results.get("max_memory_allocated", 0) if "max_memory_allocated" in config_results else None
                                }
                            
                            # Collect metrics for this batch size
                            throughput_by_batch[batch_size].append(config_results.get("throughput", 0))
                            latency_by_batch[batch_size].append(config_results.get("avg_latency", 0) * 1000)  # ms
                            
                            if "max_memory_allocated" in config_results:
                                memory_by_batch[batch_size].append(config_results.get("max_memory_allocated", 0))
                        except (ValueError, IndexError):
                            continue
            
            # Calculate average metrics for each batch size
            batch_metrics = {}
            for batch_size in batch_results.keys():
                if throughput_by_batch[batch_size]:
                    avg_throughput = sum(throughput_by_batch[batch_size]) / len(throughput_by_batch[batch_size])
                    batch_metrics[batch_size] = {
                        "avg_throughput": avg_throughput,
                        "avg_latency": sum(latency_by_batch[batch_size]) / len(latency_by_batch[batch_size]) if latency_by_batch[batch_size] else None,
                        "avg_memory": sum(memory_by_batch[batch_size]) / len(memory_by_batch[batch_size]) if memory_by_batch[batch_size] else None,
                        "throughput_per_batch": avg_throughput / batch_size  # Efficiency metric
                    }
            
            # Determine optimal batch size for throughput and latency
            optimal_throughput_batch = None
            max_throughput = -1
            
            optimal_efficiency_batch = None
            max_efficiency = -1
            
            optimal_latency_batch = None
            min_latency = float('inf')
            
            for batch_size, metrics in batch_metrics.items():
                # Optimal for throughput (highest items/sec)
                if metrics["avg_throughput"] > max_throughput:
                    max_throughput = metrics["avg_throughput"]
                    optimal_throughput_batch = batch_size
                
                # Optimal for efficiency (highest items/sec/batch)
                if metrics["throughput_per_batch"] > max_efficiency:
                    max_efficiency = metrics["throughput_per_batch"]
                    optimal_efficiency_batch = batch_size
                
                # Optimal for latency (lowest ms)
                if metrics["avg_latency"] and metrics["avg_latency"] < min_latency:
                    min_latency = metrics["avg_latency"]
                    optimal_latency_batch = batch_size
            
            # Create recommendation
            optimization_result = {
                "status": "success",
                "model_name": model_name,
                "family": family,
                "hardware_platform": hardware_platform,
                "optimal_batch_sizes": {
                    "throughput": {
                        "batch_size": optimal_throughput_batch,
                        "value": max_throughput,
                        "unit": "items/sec"
                    },
                    "efficiency": {
                        "batch_size": optimal_efficiency_batch,
                        "value": max_efficiency,
                        "unit": "items/sec/batch"
                    },
                    "latency": {
                        "batch_size": optimal_latency_batch,
                        "value": min_latency,
                        "unit": "ms"
                    }
                },
                "batch_metrics": batch_metrics,
                "detailed_results": batch_results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save optimization result
            self._save_optimization_result(optimization_result)
            
            return optimization_result
        
        except Exception as e:
            logger.error(f"Error analyzing batch size results: {e}")
            return {"status": "error", "message": f"Analysis failed: {e}"}
    
    def _save_optimization_result(self, result: Dict[str, Any]) -> str:
        """Save optimization result to file"""
        if "model_name" not in result or "hardware_platform" not in result:
            logger.warning("Invalid optimization result, missing model or hardware information")
            return None
        
        # Create unique filename
        model_name = result["model_name"].replace("/", "_")
        hardware = result["hardware_platform"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"optimization_{model_name}_{hardware}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved optimization result to {output_path}")
        return str(output_path)
    
    def generate_optimization_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all optimization results.
        
        Returns:
            Dictionary with optimization summary
        """
        # Find all optimization result files
        result_files = list(self.output_dir.glob("optimization_*.json"))
        if not result_files:
            logger.warning("No optimization result files found")
            return {"status": "warning", "message": "No optimization results found"}
        
        # Load all results
        optimization_results = []
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    optimization_results.append(result)
            except Exception as e:
                logger.error(f"Error loading optimization result from {file_path}: {e}")
        
        # Organize results by model family and hardware platform
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_results": len(optimization_results),
            "model_families": {},
            "hardware_platforms": {},
            "optimal_configurations": []
        }
        
        # Process results
        for result in optimization_results:
            if result.get("status") != "success":
                continue
            
            family = result.get("family", "unknown")
            hardware = result.get("hardware_platform", "unknown")
            model_name = result.get("model_name", "unknown")
            
            # Add to model family statistics
            if family not in summary["model_families"]:
                summary["model_families"][family] = {
                    "count": 0,
                    "hardware_platforms": {}
                }
            
            summary["model_families"][family]["count"] += 1
            
            if hardware not in summary["model_families"][family]["hardware_platforms"]:
                summary["model_families"][family]["hardware_platforms"][hardware] = {
                    "count": 0,
                    "avg_throughput": 0,
                    "avg_latency": 0,
                    "models": []
                }
            
            family_hw = summary["model_families"][family]["hardware_platforms"][hardware]
            family_hw["count"] += 1
            family_hw["models"].append(model_name)
            
            # Add throughput and latency if available
            if "optimal_batch_sizes" in result:
                if "throughput" in result["optimal_batch_sizes"] and "value" in result["optimal_batch_sizes"]["throughput"]:
                    throughput = result["optimal_batch_sizes"]["throughput"]["value"]
                    # Update running average
                    family_hw["avg_throughput"] = (family_hw["avg_throughput"] * (family_hw["count"] - 1) + throughput) / family_hw["count"]
                
                if "latency" in result["optimal_batch_sizes"] and "value" in result["optimal_batch_sizes"]["latency"]:
                    latency = result["optimal_batch_sizes"]["latency"]["value"]
                    # Update running average
                    family_hw["avg_latency"] = (family_hw["avg_latency"] * (family_hw["count"] - 1) + latency) / family_hw["count"]
            
            # Add to hardware platform statistics
            if hardware not in summary["hardware_platforms"]:
                summary["hardware_platforms"][hardware] = {
                    "count": 0,
                    "model_families": {}
                }
            
            summary["hardware_platforms"][hardware]["count"] += 1
            
            if family not in summary["hardware_platforms"][hardware]["model_families"]:
                summary["hardware_platforms"][hardware]["model_families"][family] = {
                    "count": 0,
                    "models": []
                }
            
            hw_family = summary["hardware_platforms"][hardware]["model_families"][family]
            hw_family["count"] += 1
            hw_family["models"].append(model_name)
            
            # Add to optimal configurations
            optimal_config = {
                "model_name": model_name,
                "family": family,
                "hardware_platform": hardware
            }
            
            if "optimal_batch_sizes" in result:
                # Add best batch size for throughput
                if "throughput" in result["optimal_batch_sizes"]:
                    optimal_config["throughput_batch_size"] = result["optimal_batch_sizes"]["throughput"].get("batch_size")
                    optimal_config["throughput"] = result["optimal_batch_sizes"]["throughput"].get("value")
                
                # Add best batch size for efficiency
                if "efficiency" in result["optimal_batch_sizes"]:
                    optimal_config["efficiency_batch_size"] = result["optimal_batch_sizes"]["efficiency"].get("batch_size")
                    optimal_config["efficiency"] = result["optimal_batch_sizes"]["efficiency"].get("value")
                
                # Add best batch size for latency
                if "latency" in result["optimal_batch_sizes"]:
                    optimal_config["latency_batch_size"] = result["optimal_batch_sizes"]["latency"].get("batch_size")
                    optimal_config["latency"] = result["optimal_batch_sizes"]["latency"].get("value")
            
            summary["optimal_configurations"].append(optimal_config)
        
        # Save summary to file
        summary_path = self.output_dir / "optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Generated optimization summary: {summary_path}")
        
        # Also generate a markdown report
        self._generate_markdown_report(summary)
        
        return summary
    
    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generate markdown report from optimization summary"""
        report_path = self.output_dir / "optimization_report.md"
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# Model Hardware Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Total optimized configurations: {summary.get('total_results', 0)}\n")
            f.write(f"- Model families: {len(summary.get('model_families', {}))}\n")
            f.write(f"- Hardware platforms: {len(summary.get('hardware_platforms', {}))}\n\n")
            
            # Hardware platform overview
            f.write("## Hardware Platform Summary\n\n")
            f.write("| Platform | Models Optimized | Top Model Family |\n")
            f.write("|----------|-----------------|------------------|\n")
            
            for hw_name, hw_data in summary.get("hardware_platforms", {}).items():
                # Find top model family for this hardware
                top_family = ""
                top_count = 0
                for family, family_data in hw_data.get("model_families", {}).items():
                    if family_data.get("count", 0) > top_count:
                        top_count = family_data.get("count", 0)
                        top_family = family
                
                f.write(f"| {hw_name} | {hw_data.get('count', 0)} | {top_family} |\n")
            
            f.write("\n")
            
            # Model family overview
            f.write("## Model Family Summary\n\n")
            f.write("| Family | Models Optimized | Best Hardware Platform | Avg Throughput | Avg Latency |\n")
            f.write("|--------|-----------------|------------------------|----------------|-------------|\n")
            
            for family, family_data in summary.get("model_families", {}).items():
                # Find best hardware platform for this family
                best_hw = ""
                best_throughput = 0
                avg_throughput = 0
                avg_latency = 0
                
                for hw_name, hw_data in family_data.get("hardware_platforms", {}).items():
                    if hw_data.get("avg_throughput", 0) > best_throughput:
                        best_throughput = hw_data.get("avg_throughput", 0)
                        best_hw = hw_name
                        avg_throughput = best_throughput
                        avg_latency = hw_data.get("avg_latency", 0)
                
                f.write(f"| {family} | {family_data.get('count', 0)} | {best_hw} | {avg_throughput:.2f} items/sec | {avg_latency:.2f} ms |\n")
            
            f.write("\n")
            
            # Optimal configurations
            f.write("## Optimal Configurations\n\n")
            f.write("| Model | Family | Hardware | Throughput Batch Size | Throughput (items/sec) | Latency Batch Size | Latency (ms) |\n")
            f.write("|-------|--------|----------|----------------------|------------------------|-------------------|-------------|\n")
            
            for config in summary.get("optimal_configurations", []):
                f.write(f"| {config.get('model_name', '')} | {config.get('family', '')} | {config.get('hardware_platform', '')} | ")
                f.write(f"{config.get('throughput_batch_size', '-')} | {config.get('throughput', 0):.2f} | ")
                f.write(f"{config.get('latency_batch_size', '-')} | {config.get('latency', 0):.2f} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Recommend batch size by family and hardware
            f.write("### Recommended Batch Sizes by Model Family and Hardware\n\n")
            f.write("| Family | Hardware | For Throughput | For Low Latency |\n")
            f.write("|--------|----------|----------------|----------------|\n")
            
            # Calculate average optimal batch sizes per family and hardware
            family_hw_batch_sizes = {}
            
            for config in summary.get("optimal_configurations", []):
                family = config.get("family", "")
                hardware = config.get("hardware_platform", "")
                throughput_batch = config.get("throughput_batch_size")
                latency_batch = config.get("latency_batch_size")
                
                if not family or not hardware:
                    continue
                
                if family not in family_hw_batch_sizes:
                    family_hw_batch_sizes[family] = {}
                
                if hardware not in family_hw_batch_sizes[family]:
                    family_hw_batch_sizes[family][hardware] = {
                        "throughput_batches": [],
                        "latency_batches": []
                    }
                
                if throughput_batch:
                    family_hw_batch_sizes[family][hardware]["throughput_batches"].append(throughput_batch)
                
                if latency_batch:
                    family_hw_batch_sizes[family][hardware]["latency_batches"].append(latency_batch)
            
            # Write recommendations
            for family, hw_data in family_hw_batch_sizes.items():
                for hw, batch_data in hw_data.items():
                    throughput_batches = batch_data["throughput_batches"]
                    latency_batches = batch_data["latency_batches"]
                    
                    if not throughput_batches and not latency_batches:
                        continue
                    
                    # Calculate mode (most common value)
                    throughput_rec = max(set(throughput_batches), key=throughput_batches.count) if throughput_batches else "-"
                    latency_rec = max(set(latency_batches), key=latency_batches.count) if latency_batches else "-"
                    
                    f.write(f"| {family} | {hw} | {throughput_rec} | {latency_rec} |\n")
            
            f.write("\n")
            
            # General recommendations
            f.write("### General Recommendations\n\n")
            f.write("1. **For embedding models (BERT, etc.):**\n")
            f.write("   - Use larger batch sizes (8-32) for higher throughput\n")
            f.write("   - CUDA GPUs provide the best performance for batch processing\n")
            f.write("   - CPU performance is adequate for small batches (1-4)\n\n")
            
            f.write("2. **For text generation models (LLMs):**\n") 
            f.write("   - Use smaller batch sizes (1-4) due to memory constraints\n")
            f.write("   - CUDA GPUs with high memory are recommended\n")
            f.write("   - Model quantization can enable larger batch sizes\n\n")
            
            f.write("3. **For vision models:**\n")
            f.write("   - Medium batch sizes (4-16) offer the best efficiency\n")
            f.write("   - OpenVINO provides good performance on Intel hardware\n")
            f.write("   - Image resolution significantly impacts throughput\n\n")
            
            f.write("4. **For audio models:**\n")
            f.write("   - Small batch sizes (1-4) recommended due to sequence length variability\n")
            f.write("   - CUDA provides the best performance for most audio models\n\n")
            
            f.write("5. **For multimodal models:**\n")
            f.write("   - Use small batch sizes (1-2) due to high memory requirements\n")
            f.write("   - High-memory GPUs are essential for good performance\n\n")
        
        logger.info(f"Generated markdown report: {report_path}")
        return str(report_path)
    
    def close(self):
        """Clean up resources"""
        if self.tracker:
            self.tracker.close()

def main():
    parser = argparse.ArgumentParser(description="Hardware-Model Optimizer")
    parser.add_argument("--model", help="Model name to optimize")
    parser.add_argument("--family", help="Model family (embedding, text_generation, vision, audio, multimodal)")
    parser.add_argument("--hardware", help="Hardware platform to optimize for")
    parser.add_argument("--batch-sizes", type=int, nargs="+", help="Batch sizes to test")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", help="Sequence lengths to test (for text models)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=DEFAULT_BENCHMARK_ITERATIONS, help="Number of benchmark iterations")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Maximum time in seconds for a benchmark")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the model validation database")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for optimization results")
    parser.add_argument("--no-resource-pool", action="store_true", help="Disable ResourcePool for model caching")
    parser.add_argument("--generate-summary", action="store_true", help="Generate summary from existing optimization results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Initialize optimizer
    optimizer = HardwareModelOptimizer(
        db_path=args.db_path,
        output_dir=args.output_dir,
        use_resource_pool=not args.no_resource_pool
    )
    
    try:
        # Generate summary if requested
        if args.generate_summary:
            summary = optimizer.generate_optimization_summary()
            
            if summary.get("status") == "success" or summary.get("total_results", 0) > 0:
                print(f"Generated optimization summary with {summary.get('total_results', 0)} configurations")
                print(f"See {args.output_dir}/optimization_report.md for details")
            else:
                print("No optimization results found")
        
        # Optimize a specific model if requested
        elif args.model:
            result = optimizer.find_optimal_batch_size(
                model_name=args.model,
                family=args.family,
                hardware_platform=args.hardware,
                batch_sizes=args.batch_sizes,
                sequence_lengths=args.sequence_lengths,
                warmup_iterations=args.warmup,
                benchmark_iterations=args.iterations,
                timeout=args.timeout
            )
            
            if result.get("status") == "success":
                # Print optimization results
                print(f"\nOptimization Results for {args.model} on {result['hardware_platform']}:\n")
                
                if "optimal_batch_sizes" in result:
                    if "throughput" in result["optimal_batch_sizes"]:
                        throughput = result["optimal_batch_sizes"]["throughput"]
                        print(f"Best for Throughput: Batch Size {throughput['batch_size']} ({throughput['value']:.2f} {throughput['unit']})")
                    
                    if "efficiency" in result["optimal_batch_sizes"]:
                        efficiency = result["optimal_batch_sizes"]["efficiency"]
                        print(f"Best for Efficiency: Batch Size {efficiency['batch_size']} ({efficiency['value']:.2f} {efficiency['unit']})")
                    
                    if "latency" in result["optimal_batch_sizes"]:
                        latency = result["optimal_batch_sizes"]["latency"]
                        print(f"Best for Latency: Batch Size {latency['batch_size']} ({latency['value']:.2f} {latency['unit']})")
                
                print(f"\nOptimization results saved to {args.output_dir}\n")
            else:
                print(f"Optimization failed: {result.get('message', 'Unknown error')}")
        
        else:
            # No action specified, print help
            parser.print_help()
    
    finally:
        # Clean up
        optimizer.close()

if __name__ == "__main__":
    main()
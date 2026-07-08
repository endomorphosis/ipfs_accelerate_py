#!/usr/bin/env python3
"""
OpenVINO Benchmark Tool

This script benchmarks models using the enhanced OpenVINO backend, 
supporting various precision levels (FP32, FP16, INT8) and optimum.intel integration.

Features:
- Comprehensive benchmarking of models with OpenVINO on Intel hardware
- Support for multiple precision formats (FP32, FP16, INT8)
- Integration with optimum.intel for HuggingFace models
- Direct integration with DuckDB for result storage
- Batch size variation for performance testing
- INT8 quantization with calibration data support
- Detailed performance metrics and analysis

Usage:
    python benchmark_openvino.py --model bert-base-uncased --precision FP32,FP16,INT8
    python benchmark_openvino.py --model-family text --device CPU
    python benchmark_openvino.py --all-models --batch-sizes 1,2,4,8
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("openvino_benchmark.log")
    ]
)
logger = logging.getLogger("openvino_benchmark")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the OpenVINO backend
try:
    from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
    HAS_BACKEND = True
except ImportError as e:
    logger.error(f"Failed to import OpenVINO backend: {e}")
    HAS_BACKEND = False

# Try to import the DB API
try:
    from benchmark_db_api import BenchmarkDBAPI
    HAS_DB_API = True
except ImportError:
    logger.warning("BenchmarkDBAPI not available. Results will not be stored in database.")
    HAS_DB_API = False

# Constants
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]
DEFAULT_PRECISIONS = ["FP32", "FP16", "INT8"]
OPENVINO_DEVICES = ["CPU", "GPU", "AUTO", "MYRIAD", "HETERO", "MULTI", "HDDL", "GNA"]

# Model families and examples
MODEL_FAMILIES = {
    "text": [
        "bert-base-uncased", 
        "bert-large-uncased", 
        "t5-small", 
        "distilbert-base-uncased",
        "roberta-base",
        "albert-base-v2"
    ],
    "vision": [
        "google/vit-base-patch16-224", 
        "facebook/detr-resnet-50", 
        "microsoft/resnet-50",
        "facebook/deit-base-patch16-224"
    ],
    "audio": [
        "openai/whisper-tiny", 
        "facebook/wav2vec2-base",
        "facebook/wav2vec2-large",
        "microsoft/wavlm-base"
    ],
    "multimodal": [
        "openai/clip-vit-base-patch32",
        "facebook/flava-full"
    ]
}

class OpenVINOBenchmark:
    """Benchmark tool for OpenVINO backend with database integration."""
    
    def __init__(self, args):
        """Initialize the benchmark tool with parsed arguments."""
        self.args = args
        
        # Get database path
        self.db_path = args.db_path
        if self.db_path is None:
            self.db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
            logger.info(f"Using database path from environment: {self.db_path}")
        
        # Create DB API client if available
        self.db_api = None
        if HAS_DB_API and not args.no_db:
            self.db_api = BenchmarkDBAPI(db_path=self.db_path)
            logger.info(f"Connected to benchmark database: {self.db_path}")
        
        # Initialize OpenVINO backend
        self.backend = None
        if HAS_BACKEND:
            self.backend = OpenVINOBackend()
            if not self.backend.is_available():
                logger.error("OpenVINO is not available. Please install OpenVINO Runtime.")
                sys.exit(1)
            logger.info("OpenVINO backend initialized successfully")
            
            # Get available devices
            self.available_devices = self.backend.get_all_devices()
            logger.info(f"Available OpenVINO devices: {len(self.available_devices)}")
            for i, device in enumerate(self.available_devices):
                logger.info(f"  - Device {i+1}: {device.get('device_name', 'Unknown')} ({device.get('full_name', 'Unknown')})")
        else:
            logger.error("OpenVINO backend not available. Please check your installation.")
            sys.exit(1)
        
        # Parse models to benchmark
        self.models = self._get_models_to_benchmark()
        logger.info(f"Models to benchmark: {len(self.models)}")
        for model in self.models[:10]:  # Limit to first 10 for logging
            logger.info(f"  - {model}")
        if len(self.models) > 10:
            logger.info(f"  - ... and {len(self.models) - 10} more")
        
        # Parse devices to benchmark
        self.devices = self._get_devices_to_benchmark()
        logger.info(f"Devices to benchmark: {self.devices}")
        
        # Parse precisions to benchmark
        self.precisions = self._get_precisions_to_benchmark()
        logger.info(f"Precisions to benchmark: {self.precisions}")
        
        # Parse batch sizes to benchmark
        self.batch_sizes = self._get_batch_sizes_to_benchmark()
        logger.info(f"Batch sizes to benchmark: {self.batch_sizes}")
        
        # Parse iterations for each benchmark
        self.iterations = args.iterations
        logger.info(f"Iterations per benchmark: {self.iterations}")
        
        # Other configuration options
        self.optimum_integration = not args.no_optimum
        self.simulate = args.simulate
        self.dry_run = args.dry_run
        self.calibration_samples = args.calibration_samples
        self.output_file = args.output_file
        self.report_format = args.report_format
        
        # Initialize results storage
        self.results = []
        
        # Initialize model type lookup
        self.model_types = self._initialize_model_types()
    
    def _initialize_model_types(self) -> Dict[str, str]:
        """Create a lookup dictionary for model types based on model names."""
        model_types = {}
        
        # Assign model types based on MODEL_FAMILIES
        for family, models in MODEL_FAMILIES.items():
            for model in models:
                model_types[model] = family
        
        # Try to infer model types from model names for any not already mapped
        for model in self.models:
            if model not in model_types:
                if any(substr in model.lower() for substr in ["bert", "gpt", "t5", "llama", "roberta", "albert"]):
                    model_types[model] = "text"
                elif any(substr in model.lower() for substr in ["vit", "resnet", "detr", "yolo", "deit"]):
                    model_types[model] = "vision"
                elif any(substr in model.lower() for substr in ["whisper", "wav2vec", "wavlm", "hubert"]):
                    model_types[model] = "audio"
                elif any(substr in model.lower() for substr in ["clip", "flava", "blip", "llava"]):
                    model_types[model] = "multimodal"
                else:
                    # Default to text if we can't determine
                    model_types[model] = "text"
        
        return model_types
    
    def _get_models_to_benchmark(self) -> List[str]:
        """Get the list of models to benchmark based on arguments."""
        models = []
        
        if self.args.all_models:
            # If we have DB access, get all models from database
            if self.db_api:
                try:
                    model_df = self.db_api.get_model_list()
                    models = model_df['model_name'].tolist()
                    logger.info(f"Loaded {len(models)} models from database")
                except Exception as e:
                    logger.error(f"Error getting models from database: {e}")
                    models = self._get_default_models()
            else:
                # No DB access, use default models
                models = self._get_default_models()
        elif self.args.model:
            # Use specifically provided models
            models = [model.strip() for model in self.args.model.split(',')]
        elif self.args.model_family:
            # Use models from specified families
            families = [family.strip() for family in self.args.model_family.split(',')]
            for family in families:
                if family in MODEL_FAMILIES:
                    models.extend(MODEL_FAMILIES[family])
                else:
                    logger.warning(f"Unknown model family: {family}. Valid options are: {list(MODEL_FAMILIES.keys())}")
        else:
            # Use a default subset of models if nothing specified
            models = self._get_default_models()
        
        return models
    
    def _get_default_models(self) -> List[str]:
        """Return a default list of models to benchmark with OpenVINO."""
        return [
            "bert-base-uncased",  # Text model
            "t5-small",           # Text-to-text model
            "distilbert-base-uncased",  # Small text model
            "google/vit-base-patch16-224",  # Vision model
            "openai/whisper-tiny"  # Audio model
        ]
    
    def _get_devices_to_benchmark(self) -> List[str]:
        """Get the list of OpenVINO devices to benchmark based on arguments."""
        devices = []
        
        if self.args.device:
            # Use specifically provided devices
            devices = [device.strip().upper() for device in self.args.device.split(',')]
        else:
            # Default to CPU if nothing specified
            devices = ["CPU"]
        
        # Check if devices are available
        available_device_names = [device.get('device_name', '').upper() for device in self.available_devices]
        validated_devices = []
        
        for device in devices:
            if device in available_device_names:
                validated_devices.append(device)
            elif device == "CPU" and "CPU" not in available_device_names:
                # Special case: CPU should be available even if not in the list
                validated_devices.append("CPU")
                logger.warning("CPU device not found in available devices, but will try to use it anyway")
            elif device == "AUTO" and "AUTO" not in available_device_names:
                # Special case: AUTO should usually be available
                validated_devices.append("AUTO")
                logger.warning("AUTO device not found in available devices, but will try to use it anyway")
            else:
                logger.warning(f"Device {device} not found in available devices, skipping")
        
        # If no valid devices, default to CPU
        if not validated_devices:
            logger.warning("No valid devices found, defaulting to CPU")
            validated_devices = ["CPU"]
        
        return validated_devices
    
    def _get_precisions_to_benchmark(self) -> List[str]:
        """Get the list of precision formats to benchmark based on arguments."""
        precisions = []
        
        if self.args.precision:
            # Use specifically provided precisions
            precisions = [precision.strip().upper() for precision in self.args.precision.split(',')]
        else:
            # Default to all precisions if nothing specified
            precisions = DEFAULT_PRECISIONS
        
        # Validate precisions
        validated_precisions = []
        for precision in precisions:
            if precision in DEFAULT_PRECISIONS:
                validated_precisions.append(precision)
            else:
                logger.warning(f"Unknown precision format: {precision}. Valid options are: {DEFAULT_PRECISIONS}")
        
        # If no valid precisions, default to FP32
        if not validated_precisions:
            logger.warning("No valid precision formats found, defaulting to FP32")
            validated_precisions = ["FP32"]
        
        return validated_precisions
    
    def _get_batch_sizes_to_benchmark(self) -> List[int]:
        """Get the list of batch sizes to benchmark based on arguments."""
        batch_sizes = []
        
        if self.args.batch_sizes:
            # Use specifically provided batch sizes
            try:
                batch_sizes = [int(batch_size.strip()) for batch_size in self.args.batch_sizes.split(',')]
            except ValueError:
                logger.warning(f"Invalid batch sizes provided, defaulting to {DEFAULT_BATCH_SIZES}")
                batch_sizes = DEFAULT_BATCH_SIZES
        else:
            # Default to standard batch sizes if nothing specified
            batch_sizes = DEFAULT_BATCH_SIZES
        
        return batch_sizes
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine the model type for a given model name."""
        return self.model_types.get(model_name, "text")
    
    def _store_result_in_db(self, result: Dict[str, Any]) -> bool:
        """Store benchmark result in the database."""
        if not self.db_api:
            logger.warning("Database API not available, result not stored")
            return False
        
        try:
            # Extract data from result
            model_name = result.get("model")
            hardware_type = "openvino"
            device = result.get("device")
            precision = result.get("precision")
            batch_size = result.get("batch_size", 1)
            
            # Performance metrics
            avg_latency = result.get("avg_latency_ms", 0)
            throughput = result.get("avg_throughput_items_per_sec", 0)
            memory_usage = result.get("avg_memory_usage_mb", 0)
            
            # Additional metadata
            is_simulated = self.simulate
            
            # Get model ID from database (or create if doesn't exist)
            model_id = self.db_api.get_or_create_model_id(model_name, self._get_model_type(model_name))
            
            # Get hardware ID from database (or create if doesn't exist)
            hardware_id = self.db_api.get_or_create_hardware_id(
                hardware_type, 
                f"openvino_{device}_{precision}",
                {"device": device, "precision": precision}
            )
            
            # Store performance result
            self.db_api.store_performance_result(
                model_id=model_id,
                hardware_id=hardware_id,
                batch_size=batch_size,
                average_latency_ms=avg_latency,
                throughput_items_per_second=throughput,
                memory_usage_mb=memory_usage,
                is_simulated=is_simulated,
                details=result
            )
            
            logger.info(f"Stored result in database for {model_name} on OpenVINO {device} with {precision}")
            return True
        
        except Exception as e:
            logger.error(f"Error storing result in database: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_results_to_file(self) -> bool:
        """Save all benchmark results to a file."""
        if not self.output_file:
            return False
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(self.output_file, 'w') as f:
                json.dump({
                    "benchmark_type": "openvino",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "models": self.models,
                    "devices": self.devices,
                    "precisions": self.precisions,
                    "batch_sizes": self.batch_sizes,
                    "iterations": self.iterations,
                    "results": self.results
                }, f, indent=2)
            
            logger.info(f"Saved benchmark results to {self.output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving results to file: {e}")
            return False
    
    def _generate_dummy_inputs(self, model_name: str, model_type: str, batch_size: int) -> Dict[str, Any]:
        """Generate dummy inputs for a model based on its type."""
        try:
            # For text models
            if model_type == "text":
                seq_length = 128  # Default sequence length
                return {
                    "input_ids": [[101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102] + [0] * (seq_length - 10)] * batch_size,
                    "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * (seq_length - 10)] * batch_size
                }
            
            # For vision models
            elif model_type == "vision":
                # Typical image size (batch, channels, height, width)
                return {
                    "pixel_values": [[[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]] * batch_size
                }
            
            # For audio models
            elif model_type == "audio":
                # Simplified audio features (batch, channels, time, freq)
                return {
                    "input_features": [[[[0.1 for _ in range(80)] for _ in range(128)] for _ in range(1)]] * batch_size
                }
            
            # For multimodal models (simplified)
            elif model_type == "multimodal":
                # Combine text and vision inputs
                return {
                    "input_ids": [[101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102] + [0] * (128 - 10)] * batch_size,
                    "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * (128 - 10)] * batch_size,
                    "pixel_values": [[[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]] * batch_size
                }
            
            # Default fallback
            else:
                return {
                    "input_ids": [[101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102] + [0] * (128 - 10)] * batch_size,
                    "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * (128 - 10)] * batch_size
                }
        
        except Exception as e:
            logger.error(f"Error generating dummy inputs: {e}")
            # Fallback to simple input
            return {
                "inputs": [[0.1 for _ in range(128)]] * batch_size
            }
    
    def _generate_calibration_data(self, model_name: str, model_type: str) -> List[Dict[str, Any]]:
        """Generate calibration data for INT8 quantization."""
        try:
            # Create multiple samples for calibration
            samples = []
            for i in range(self.calibration_samples):
                sample = self._generate_dummy_inputs(model_name, model_type, 1)
                samples.append(sample)
            
            logger.info(f"Generated {len(samples)} calibration samples for INT8 quantization")
            return samples
        
        except Exception as e:
            logger.error(f"Error generating calibration data: {e}")
            return []
    
    def run_benchmark(self, model_name: str, device: str, precision: str, batch_size: int) -> Dict[str, Any]:
        """
        Run benchmark for a specific model, device, precision, and batch size configuration.
        
        Args:
            model_name: Name of the model to benchmark
            device: OpenVINO device to use (CPU, GPU, etc.)
            precision: Precision format (FP32, FP16, INT8)
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking {model_name} on {device} with {precision} precision and batch size {batch_size}")
        
        if self.dry_run:
            logger.info("DRY RUN: Would run benchmark")
            return {
                "status": "skipped",
                "model": model_name,
                "device": device,
                "precision": precision,
                "batch_size": batch_size,
                "is_dry_run": True
            }
        
        model_type = self._get_model_type(model_name)
        logger.info(f"Model type: {model_type}")
        
        try:
            # Generate config for model loading
            config = {
                "device": device,
                "model_type": model_type,
                "precision": precision,
                "batch_size": batch_size,
                "use_optimum": self.optimum_integration
            }
            
            # For INT8 precision, add calibration data
            if precision == "INT8":
                calibration_data = self._generate_calibration_data(model_name, model_type)
                if calibration_data:
                    config["calibration_data"] = calibration_data
            
            # Start timing
            start_time = time.time()
            
            # Load model
            load_result = self.backend.load_model(model_name, config)
            
            if load_result.get("status") != "success":
                logger.error(f"Failed to load model: {load_result.get('message', 'Unknown error')}")
                return {
                    "status": "error",
                    "model": model_name,
                    "device": device,
                    "precision": precision,
                    "batch_size": batch_size,
                    "error": load_result.get("message", "Unknown error during loading")
                }
            
            # Measure load time
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Generate dummy inputs
            inputs = self._generate_dummy_inputs(model_name, model_type, batch_size)
            
            # Run warmup inference
            logger.info("Running warmup inference...")
            warmup_result = self.backend.run_inference(
                model_name,
                inputs,
                {
                    "device": device,
                    "model_type": model_type,
                    "batch_size": batch_size
                }
            )
            
            if warmup_result.get("status") != "success":
                logger.error(f"Warmup inference failed: {warmup_result.get('message', 'Unknown error')}")
                # Try to unload the model
                self.backend.unload_model(model_name, device)
                return {
                    "status": "error",
                    "model": model_name,
                    "device": device,
                    "precision": precision,
                    "batch_size": batch_size,
                    "error": warmup_result.get("message", "Unknown error during warmup")
                }
            
            # Collect benchmark metrics
            latencies = []
            throughputs = []
            memory_usages = []
            
            # Run benchmark iterations
            logger.info(f"Running {self.iterations} benchmark iterations...")
            
            for i in range(self.iterations):
                logger.info(f"Iteration {i+1}/{self.iterations}")
                
                inference_result = self.backend.run_inference(
                    model_name,
                    inputs,
                    {
                        "device": device,
                        "model_type": model_type,
                        "batch_size": batch_size
                    }
                )
                
                if inference_result.get("status") != "success":
                    logger.warning(f"Inference failed in iteration {i+1}: {inference_result.get('message', 'Unknown error')}")
                    continue
                
                # Collect metrics
                latencies.append(inference_result.get("latency_ms", 0))
                throughputs.append(inference_result.get("throughput_items_per_sec", 0))
                memory_usages.append(inference_result.get("memory_usage_mb", 0))
            
            # Unload the model
            self.backend.unload_model(model_name, device)
            
            # Calculate statistics if we have results
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                avg_throughput = sum(throughputs) / len(throughputs)
                min_throughput = min(throughputs)
                max_throughput = max(throughputs)
                
                avg_memory = sum(memory_usages) / len(memory_usages)
                
                # Calculate percentiles (50th, 95th, 99th)
                latencies.sort()
                p50_index = int(len(latencies) * 0.5)
                p95_index = int(len(latencies) * 0.95)
                p99_index = int(len(latencies) * 0.99)
                
                p50_latency = latencies[p50_index]
                p95_latency = latencies[p95_index]
                p99_latency = latencies[min(p99_index, len(latencies)-1)]
                
                # Log results
                logger.info("Benchmark Results:")
                logger.info(f"  Average Latency: {avg_latency:.2f} ms")
                logger.info(f"  P50 Latency: {p50_latency:.2f} ms")
                logger.info(f"  P95 Latency: {p95_latency:.2f} ms")
                logger.info(f"  P99 Latency: {p99_latency:.2f} ms")
                logger.info(f"  Min Latency: {min_latency:.2f} ms")
                logger.info(f"  Max Latency: {max_latency:.2f} ms")
                logger.info(f"  Average Throughput: {avg_throughput:.2f} items/sec")
                logger.info(f"  Average Memory Usage: {avg_memory:.2f} MB")
                
                # Return benchmark results
                result = {
                    "status": "success",
                    "model": model_name,
                    "model_type": model_type,
                    "device": device,
                    "precision": precision,
                    "batch_size": batch_size,
                    "iterations": len(latencies),
                    "load_time_sec": load_time,
                    "avg_latency_ms": avg_latency,
                    "p50_latency_ms": p50_latency,
                    "p95_latency_ms": p95_latency,
                    "p99_latency_ms": p99_latency,
                    "min_latency_ms": min_latency,
                    "max_latency_ms": max_latency,
                    "avg_throughput_items_per_sec": avg_throughput,
                    "min_throughput_items_per_sec": min_throughput,
                    "max_throughput_items_per_sec": max_throughput,
                    "avg_memory_usage_mb": avg_memory,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "optimum_integration": self.optimum_integration
                }
                
                return result
            else:
                logger.error("No valid benchmark results collected")
                return {
                    "status": "error",
                    "model": model_name,
                    "device": device,
                    "precision": precision,
                    "batch_size": batch_size,
                    "error": "No valid results collected"
                }
                
        except Exception as e:
            logger.error(f"Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to unload the model
            try:
                self.backend.unload_model(model_name, device)
            except:
                pass
                
            return {
                "status": "error",
                "model": model_name,
                "device": device,
                "precision": precision,
                "batch_size": batch_size,
                "error": str(e)
            }
    
    def generate_report(self, format="markdown") -> str:
        """
        Generate a report of benchmark results.
        
        Args:
            format: Output format (markdown, html, json)
            
        Returns:
            String containing the formatted report
        """
        if not self.results:
            return "No benchmark results available."
        
        # Filter for successful results only
        successful_results = [r for r in self.results if r.get("status") == "success"]
        
        if not successful_results:
            return "No successful benchmark results available."
        
        if format == "json":
            return json.dumps({
                "benchmark_type": "openvino",
                "timestamp": datetime.datetime.now().isoformat(),
                "result_count": len(successful_results),
                "results": successful_results
            }, indent=2)
        
        elif format == "html":
            # Generate HTML report
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>OpenVINO Benchmark Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #0071c5; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                </style>
            </head>
            <body>
                <h1>OpenVINO Benchmark Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Total benchmarks: {count}</p>
                
                <h2>Summary by Precision</h2>
                <table>
                    <tr>
                        <th>Precision</th>
                        <th>Avg Latency (ms)</th>
                        <th>Avg Throughput (items/sec)</th>
                        <th>Avg Memory (MB)</th>
                    </tr>
            """.format(
                timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                count=len(successful_results)
            )
            
            # Add precision summary
            precision_data = {}
            for result in successful_results:
                precision = result.get("precision")
                if precision not in precision_data:
                    precision_data[precision] = {
                        "latencies": [],
                        "throughputs": [],
                        "memories": []
                    }
                precision_data[precision]["latencies"].append(result.get("avg_latency_ms", 0))
                precision_data[precision]["throughputs"].append(result.get("avg_throughput_items_per_sec", 0))
                precision_data[precision]["memories"].append(result.get("avg_memory_usage_mb", 0))
            
            for precision, data in precision_data.items():
                avg_latency = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0
                avg_throughput = sum(data["throughputs"]) / len(data["throughputs"]) if data["throughputs"] else 0
                avg_memory = sum(data["memories"]) / len(data["memories"]) if data["memories"] else 0
                
                html += f"""
                <tr>
                    <td>{precision}</td>
                    <td>{avg_latency:.2f}</td>
                    <td>{avg_throughput:.2f}</td>
                    <td>{avg_memory:.2f}</td>
                </tr>
                """
            
            html += """
                </table>
                
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Device</th>
                        <th>Precision</th>
                        <th>Batch Size</th>
                        <th>Avg Latency (ms)</th>
                        <th>P95 Latency (ms)</th>
                        <th>Throughput (items/sec)</th>
                        <th>Memory (MB)</th>
                    </tr>
            """
            
            # Add detailed results
            for result in successful_results:
                html += f"""
                <tr>
                    <td>{result.get("model", "")}</td>
                    <td>{result.get("device", "")}</td>
                    <td>{result.get("precision", "")}</td>
                    <td>{result.get("batch_size", "")}</td>
                    <td>{result.get("avg_latency_ms", 0):.2f}</td>
                    <td>{result.get("p95_latency_ms", 0):.2f}</td>
                    <td>{result.get("avg_throughput_items_per_sec", 0):.2f}</td>
                    <td>{result.get("avg_memory_usage_mb", 0):.2f}</td>
                </tr>
                """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
        
        else:  # Default to markdown
            # Generate markdown report
            markdown = f"# OpenVINO Benchmark Report\n\n"
            markdown += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            markdown += f"Total benchmarks: {len(successful_results)}\n\n"
            
            # Summary by precision
            markdown += f"## Summary by Precision\n\n"
            markdown += f"| Precision | Avg Latency (ms) | Avg Throughput (items/sec) | Avg Memory (MB) |\n"
            markdown += f"|-----------|-----------------|----------------------------|----------------|\n"
            
            precision_data = {}
            for result in successful_results:
                precision = result.get("precision")
                if precision not in precision_data:
                    precision_data[precision] = {
                        "latencies": [],
                        "throughputs": [],
                        "memories": []
                    }
                precision_data[precision]["latencies"].append(result.get("avg_latency_ms", 0))
                precision_data[precision]["throughputs"].append(result.get("avg_throughput_items_per_sec", 0))
                precision_data[precision]["memories"].append(result.get("avg_memory_usage_mb", 0))
            
            for precision, data in precision_data.items():
                avg_latency = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0
                avg_throughput = sum(data["throughputs"]) / len(data["throughputs"]) if data["throughputs"] else 0
                avg_memory = sum(data["memories"]) / len(data["memories"]) if data["memories"] else 0
                
                markdown += f"| {precision} | {avg_latency:.2f} | {avg_throughput:.2f} | {avg_memory:.2f} |\n"
            
            # Detailed results
            markdown += f"\n## Detailed Results\n\n"
            markdown += f"| Model | Device | Precision | Batch Size | Avg Latency (ms) | P95 Latency (ms) | Throughput (items/sec) | Memory (MB) |\n"
            markdown += f"|-------|--------|-----------|------------|------------------|------------------|------------------------|------------|\n"
            
            for result in successful_results:
                markdown += f"| {result.get('model', '')} | {result.get('device', '')} | {result.get('precision', '')} | {result.get('batch_size', '')} | {result.get('avg_latency_ms', 0):.2f} | {result.get('p95_latency_ms', 0):.2f} | {result.get('avg_throughput_items_per_sec', 0):.2f} | {result.get('avg_memory_usage_mb', 0):.2f} |\n"
            
            return markdown
    
    def run(self) -> None:
        """Run all specified benchmarks."""
        logger.info("Starting OpenVINO benchmark run")
        
        # Track the number of benchmarks and successes
        total_benchmarks = len(self.models) * len(self.devices) * len(self.precisions) * len(self.batch_sizes)
        completed_benchmarks = 0
        successful_benchmarks = 0
        
        logger.info(f"Planning to run {total_benchmarks} benchmarks")
        
        try:
            # Run benchmarks for each combination
            for model in self.models:
                for device in self.devices:
                    for precision in self.precisions:
                        for batch_size in self.batch_sizes:
                            # Run the benchmark
                            result = self.run_benchmark(model, device, precision, batch_size)
                            
                            # Store the result
                            self.results.append(result)
                            
                            # Update counters
                            completed_benchmarks += 1
                            if result.get("status") == "success":
                                successful_benchmarks += 1
                                
                                # Store in database if available
                                if self.db_api and not self.dry_run:
                                    self._store_result_in_db(result)
                            
                            # Log progress
                            logger.info(f"Progress: {completed_benchmarks}/{total_benchmarks} benchmarks completed ({successful_benchmarks} successful)")
            
            logger.info(f"Benchmark run completed: {completed_benchmarks}/{total_benchmarks} benchmarks completed ({successful_benchmarks} successful)")
            
            # Save results to file if requested
            if self.output_file:
                self._save_results_to_file()
            
            # Generate and print report if requested
            if self.args.report:
                report = self.generate_report(format=self.report_format)
                
                # Save report to file if output file specified
                if self.args.report_file:
                    try:
                        with open(self.args.report_file, 'w') as f:
                            f.write(report)
                        logger.info(f"Saved report to {self.args.report_file}")
                    except Exception as e:
                        logger.error(f"Error saving report to file: {e}")
                
                # Print report to console
                print("\n" + report)
                
        except KeyboardInterrupt:
            logger.info("\nBenchmark run interrupted by user")
            logger.info(f"Completed {completed_benchmarks}/{total_benchmarks} benchmarks before interruption ({successful_benchmarks} successful)")
            
            # Save partial results to file if requested
            if self.output_file:
                self._save_results_to_file()
                
        except Exception as e:
            logger.error(f"Error during benchmark run: {e}")
            import traceback
            traceback.print_exc()
            
            # Save partial results to file if requested
            if self.output_file:
                self._save_results_to_file()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run OpenVINO benchmarks with different precisions and configurations")
    
    # Model selection options
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--all-models", action="store_true",
                           help="Benchmark all available models")
    model_group.add_argument("--model", type=str,
                           help="Comma-separated list of specific models to benchmark")
    model_group.add_argument("--model-family", type=str,
                           help="Comma-separated list of model families to benchmark (text, vision, audio, multimodal)")
    
    # Hardware options
    hardware_group = parser.add_argument_group("Hardware Options")
    hardware_group.add_argument("--device", type=str, default="CPU",
                               help="Comma-separated list of OpenVINO devices to use (CPU, GPU, AUTO, etc.)")
    
    # Precision options
    precision_group = parser.add_argument_group("Precision Options")
    precision_group.add_argument("--precision", type=str,
                                help=f"Comma-separated list of precision formats to test {DEFAULT_PRECISIONS}")
    precision_group.add_argument("--calibration-samples", type=int, default=10,
                                help="Number of calibration samples for INT8 quantization")
    
    # Benchmark configuration
    benchmark_group = parser.add_argument_group("Benchmark Configuration")
    benchmark_group.add_argument("--batch-sizes", type=str,
                               help="Comma-separated list of batch sizes to test")
    benchmark_group.add_argument("--iterations", type=int, default=10,
                               help="Number of iterations for each benchmark")
    benchmark_group.add_argument("--no-optimum", action="store_true",
                               help="Do not use optimum.intel integration for HuggingFace models")
    
    # Database options
    db_group = parser.add_argument_group("Database Options")
    db_group.add_argument("--db-path", type=str,
                        help="Path to the benchmark database (defaults to BENCHMARK_DB_PATH env var)")
    db_group.add_argument("--no-db", action="store_true",
                        help="Do not store results in database")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-file", type=str,
                             help="Path to save benchmark results as JSON")
    output_group.add_argument("--report", action="store_true",
                             help="Generate a report of benchmark results")
    output_group.add_argument("--report-file", type=str,
                             help="Path to save benchmark report")
    output_group.add_argument("--report-format", type=str, default="markdown",
                             choices=["markdown", "html", "json"],
                             help="Format for benchmark report")
    
    # Execution options
    execution_group = parser.add_argument_group("Execution Options")
    execution_group.add_argument("--simulate", action="store_true",
                               help="Simulate benchmarks instead of running real ones")
    execution_group.add_argument("--dry-run", action="store_true",
                               help="List benchmarks to run without actually running them")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Run OpenVINO benchmarks
    benchmark = OpenVINOBenchmark(args)
    benchmark.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
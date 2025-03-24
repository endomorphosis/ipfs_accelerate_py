#!/usr/bin/env python3
"""
Hardware benchmark runner for IPFS Accelerate Python framework.

This script provides standardized benchmarking for models across different hardware backends.
It measures latency, throughput, memory usage, and hardware-specific metrics.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import hardware detection module
    from hardware.hardware_detection import (
        detect_available_hardware,
        get_optimal_device,
        initialize_device,
        get_device_settings,
        is_device_compatible_with_model,
        get_model_hardware_recommendations
    )
    
    # Try to import torch for memory profiling
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        logger.warning("PyTorch not available, some memory metrics will be limited")
    
    # Try to import psutil for system memory monitoring
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        logger.warning("psutil not available, system memory monitoring will be limited")
        
    # Try to import numpy for statistics
    try:
        import numpy as np
        NUMPY_AVAILABLE = True
    except ImportError:
        NUMPY_AVAILABLE = False
        logger.warning("NumPy not available, statistical analysis will be limited")
        
    # Try to import transformers for model loading
    try:
        import transformers
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        logger.warning("Transformers not available, model loading will fail")
        
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please make sure you have installed the required dependencies")
    sys.exit(1)


class ModelBenchmark:
    """Model benchmarking class for standardized benchmarking."""
    
    def __init__(self, model_id: str, device: str = None, precision: str = "float32"):
        """
        Initialize the benchmark.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to benchmark on (default: None, will auto-select)
            precision: Precision mode (float32, float16, int8, etc.)
        """
        self.model_id = model_id
        self.precision = precision
        self.model_loaded = False
        self.start_memory = 0
        
        # Detect architecture type
        # This would normally import from architecture_detector.py
        # For now, use a simple placeholder
        self.architecture_type = self._detect_architecture_type(model_id)
        
        # Select device
        if device is None:
            # Get hardware recommendations for this architecture
            recommended_devices = get_model_hardware_recommendations(self.architecture_type)
            self.device = get_optimal_device(recommended_devices)
        else:
            self.device = device
            
        # Check compatibility
        is_compatible = is_device_compatible_with_model(self.device, self.architecture_type)
        if not is_compatible:
            logger.warning(f"Model {model_id} may not be compatible with {self.device}, performance may be suboptimal")
            
        # Initialize device
        self.device_info = initialize_device(self.device)
        if not self.device_info["success"] and self.device != "cpu":
            # Fallback to CPU if device initialization failed
            logger.warning(f"Failed to initialize {self.device}, falling back to CPU")
            self.device = "cpu"
            self.device_info = initialize_device(self.device)
            
        # Set task based on architecture type
        self.task = self._get_task_for_architecture(self.architecture_type)
            
        logger.info(f"Benchmarking {model_id} on {self.device} with {precision} precision")
        
    def _detect_architecture_type(self, model_id: str) -> str:
        """
        Detect architecture type from model ID.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            Architecture type string
        """
        # This would normally do more sophisticated detection
        # For now, use simple heuristics
        model_id_lower = model_id.lower()
        
        if any(x in model_id_lower for x in ["llama", "gpt", "falcon", "mistral", "phi", "mamba"]):
            return "decoder-only"
        elif any(x in model_id_lower for x in ["t5", "bart", "pegasus"]):
            return "encoder-decoder"
        elif any(x in model_id_lower for x in ["bert", "roberta", "albert", "electra"]):
            return "encoder-only"
        elif any(x in model_id_lower for x in ["vit", "swin", "deit", "convnext"]):
            return "vision"
        elif any(x in model_id_lower for x in ["clip", "blip", "git"]):
            return "vision-encoder-text-decoder"
        elif any(x in model_id_lower for x in ["whisper", "wav2vec", "hubert"]):
            return "speech"
        elif any(x in model_id_lower for x in ["flava", "llava", "imagebind"]):
            return "multimodal"
        elif any(x in model_id_lower for x in ["stable", "diffusion", "dalle"]):
            return "diffusion"
        elif any(x in model_id_lower for x in ["mixtral", "switch", "moe"]):
            return "mixture-of-experts"
        elif any(x in model_id_lower for x in ["mamba", "hyena", "rwkv"]):
            return "state-space"
        elif any(x in model_id_lower for x in ["rag"]):
            return "rag"
        else:
            # Default to encoder-only if uncertain
            return "encoder-only"
    
    def _get_task_for_architecture(self, architecture_type: str) -> str:
        """
        Get the default task for an architecture type.
        
        Args:
            architecture_type: Model architecture type
            
        Returns:
            Task name for HuggingFace
        """
        task_mapping = {
            "decoder-only": "text-generation",
            "encoder-decoder": "text2text-generation",
            "encoder-only": "fill-mask",
            "vision": "image-classification",
            "vision-encoder-text-decoder": "image-to-text",
            "speech": "automatic-speech-recognition",
            "multimodal": "multimodal-classification",
            "diffusion": "text-to-image",
            "mixture-of-experts": "text-generation",
            "state-space": "text-generation",
            "rag": "retrieval-augmented-generation"
        }
        
        return task_mapping.get(architecture_type, "text-classification")
    
    def _track_memory_usage(self, reset: bool = False) -> Dict[str, float]:
        """
        Track memory usage.
        
        Args:
            reset: Whether to reset the memory counter
            
        Returns:
            Dictionary with memory usage metrics
        """
        result = {"current_mb": 0, "peak_mb": 0, "system_mb": 0}
        
        if reset:
            self.start_memory = 0
            
        # Track PyTorch memory if available
        if TORCH_AVAILABLE and self.device in ["cuda", "mps"]:
            if self.device == "cuda":
                # CUDA memory tracking
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                if reset:
                    self.start_memory = current_memory
                    torch.cuda.reset_peak_memory_stats()
                    
                result["current_mb"] = current_memory - self.start_memory
                result["peak_mb"] = peak_memory - self.start_memory
                
            elif self.device == "mps":
                # MPS memory tracking if available
                if hasattr(torch.mps, 'current_allocated_memory'):
                    current_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
                    
                    if reset:
                        self.start_memory = current_memory
                    
                    result["current_mb"] = current_memory - self.start_memory
                    # MPS doesn't track peak memory like CUDA
                    result["peak_mb"] = result["current_mb"]
        
        # Track system memory with psutil if available
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            result["system_mb"] = memory_info.rss / (1024 * 1024)
            
        return result
        
    def load_model(self) -> Dict[str, Any]:
        """
        Load the model for benchmarking.
        
        Returns:
            Dictionary with model and tokenizer
        """
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers library not available"}
        
        try:
            # Reset memory tracking
            memory_before = self._track_memory_usage(reset=True)
            start_time = time.time()
            
            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            
            # Load the appropriate model class based on task
            if self.task == "text-generation":
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    device_map=self.device,
                    torch_dtype=self._get_torch_dtype()
                )
            elif self.task == "text2text-generation":
                model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id, 
                    device_map=self.device,
                    torch_dtype=self._get_torch_dtype()
                )
            elif self.task == "fill-mask":
                model = transformers.AutoModelForMaskedLM.from_pretrained(
                    self.model_id, 
                    device_map=self.device,
                    torch_dtype=self._get_torch_dtype()
                )
            elif self.task == "image-classification":
                processor = transformers.AutoFeatureExtractor.from_pretrained(self.model_id)
                model = transformers.AutoModelForImageClassification.from_pretrained(
                    self.model_id, 
                    device_map=self.device,
                    torch_dtype=self._get_torch_dtype()
                )
            else:
                # Fallback to AutoModel for other tasks
                model = transformers.AutoModel.from_pretrained(
                    self.model_id, 
                    device_map=self.device,
                    torch_dtype=self._get_torch_dtype()
                )
                processor = None
            
            end_time = time.time()
            memory_after = self._track_memory_usage()
            
            self.model_loaded = True
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "processor": processor if 'processor' in locals() else None,
                "load_time_seconds": end_time - start_time,
                "memory_usage_mb": memory_after["current_mb"],
                "peak_memory_mb": memory_after["peak_mb"],
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "model": None,
                "tokenizer": None,
                "processor": None,
                "load_time_seconds": 0,
                "memory_usage_mb": 0,
                "peak_memory_mb": 0,
                "success": False,
                "error": str(e)
            }
    
    def _get_torch_dtype(self):
        """
        Get torch dtype based on precision setting.
        
        Returns:
            torch dtype
        """
        if not TORCH_AVAILABLE:
            return None
            
        if self.precision == "float16":
            return torch.float16
        elif self.precision == "float32":
            return torch.float32
        elif self.precision == "bfloat16":
            return torch.bfloat16
        elif self.precision == "int8":
            # Not directly supported as a torch.dtype, handle in model loading
            return torch.float32
        else:
            return torch.float32
            
    def benchmark_inference(self, model_data: Dict[str, Any], 
                            batch_size: int = 1,
                            sequence_length: int = 128,
                            iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            model_data: Data from load_model
            batch_size: Batch size for inference
            sequence_length: Sequence length for text models
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        if not model_data["success"]:
            return {"success": False, "error": "Model failed to load"}
            
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        processor = model_data["processor"]
        
        results = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "iterations": iterations,
            "latency_ms": [],
            "first_token_latency_ms": None,
            "throughput_samples_per_sec": None,
            "memory_usage_mb": None,
            "peak_memory_mb": None,
            "success": False,
            "error": None
        }
        
        try:
            # Prepare input based on model type
            if self.architecture_type in ["encoder-only", "decoder-only", "encoder-decoder"]:
                # Text models
                text_input = "This is a test input " * (sequence_length // 5)  # Approximate tokens
                inputs = tokenizer([text_input] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
                
                # Move inputs to device
                if self.device in ["cuda", "mps"]:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            elif self.architecture_type == "vision":
                # For vision models, create a dummy image batch
                from PIL import Image
                import numpy as np
                
                # Create dummy RGB images
                images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(batch_size)]
                inputs = processor(images, return_tensors="pt")
                
                # Move inputs to device
                if self.device in ["cuda", "mps"]:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            else:
                # For other model types, return error for now
                return {
                    "success": False,
                    "error": f"Benchmark for {self.architecture_type} models not implemented yet"
                }
                
            # Reset memory tracking
            memory_before = self._track_memory_usage(reset=True)
            
            # Warmup run
            logger.info("Running warmup iteration...")
            with torch.no_grad():
                _ = model(**inputs)
            
            # Benchmark runs
            logger.info(f"Running {iterations} benchmark iterations...")
            latencies = []
            
            for i in range(iterations):
                # Reset CUDA cache if available
                if TORCH_AVAILABLE and self.device == "cuda":
                    torch.cuda.empty_cache()
                    
                # Start timer
                start_time = time.time()
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # End timer
                end_time = time.time()
                
                # Calculate latency
                latency = (end_time - start_time) * 1000  # ms
                latencies.append(latency)
                
                logger.info(f"Iteration {i+1}/{iterations}: {latency:.2f} ms")
                
                # Only capture memory usage on first iteration to avoid measurement overhead
                if i == 0:
                    memory_after = self._track_memory_usage()
                    results["memory_usage_mb"] = memory_after["current_mb"]
                    results["peak_memory_mb"] = memory_after["peak_mb"]
            
            # Calculate statistics
            if NUMPY_AVAILABLE:
                latencies_np = np.array(latencies)
                results["latency_ms"] = latencies
                results["latency_mean_ms"] = float(np.mean(latencies_np))
                results["latency_std_ms"] = float(np.std(latencies_np))
                results["latency_median_ms"] = float(np.median(latencies_np))
                results["latency_min_ms"] = float(np.min(latencies_np))
                results["latency_max_ms"] = float(np.max(latencies_np))
                # Filter out warmup effect by taking the 90th percentile
                results["latency_90p_ms"] = float(np.percentile(latencies_np, 90))
            else:
                # Calculate basic statistics without NumPy
                latencies.sort()
                results["latency_ms"] = latencies
                results["latency_mean_ms"] = sum(latencies) / len(latencies)
                results["latency_median_ms"] = latencies[len(latencies) // 2]
                results["latency_min_ms"] = latencies[0]
                results["latency_max_ms"] = latencies[-1]
            
            # Calculate throughput (samples/sec)
            mean_latency_sec = results["latency_mean_ms"] / 1000
            results["throughput_samples_per_sec"] = batch_size / mean_latency_sec
            
            # Set success flag
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            logger.error(traceback.format_exc())
            results["success"] = False
            results["error"] = str(e)
            
        return results
        
    def benchmark_model(self, 
                        batch_sizes: List[int] = [1, 2, 4, 8],
                        sequence_lengths: List[int] = [128, 512],
                        iterations: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive benchmarks for a model.
        
        Args:
            batch_sizes: List of batch sizes to benchmark
            sequence_lengths: List of sequence lengths to benchmark
            iterations: Number of iterations per benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "model_id": self.model_id,
            "device": self.device,
            "precision": self.precision,
            "architecture_type": self.architecture_type,
            "task": self.task,
            "device_info": self.device_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "load_results": None,
            "batch_results": {},
            "summary": {}
        }
        
        # Load model
        load_results = self.load_model()
        results["load_results"] = {
            "success": load_results["success"],
            "load_time_seconds": load_results.get("load_time_seconds", 0),
            "memory_usage_mb": load_results.get("memory_usage_mb", 0),
            "peak_memory_mb": load_results.get("peak_memory_mb", 0)
        }
        
        if not load_results["success"]:
            results["error"] = load_results.get("error", "Unknown error loading model")
            return results
        
        # Run benchmarks for each batch size and sequence length
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Skip large batch/sequence combinations that might OOM
                if batch_size * seq_len > 16384 and self.architecture_type in ["decoder-only", "mixture-of-experts"]:
                    logger.warning(f"Skipping batch_size={batch_size}, seq_len={seq_len} as it may cause OOM for {self.architecture_type} models")
                    continue
                    
                batch_key = f"b{batch_size}_s{seq_len}"
                logger.info(f"Benchmarking batch_size={batch_size}, seq_len={seq_len}")
                
                inference_results = self.benchmark_inference(
                    load_results,
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    iterations=iterations
                )
                
                results["batch_results"][batch_key] = inference_results
        
        # Compute summary metrics
        successful_results = [r for k, r in results["batch_results"].items() if r["success"]]
        
        if successful_results:
            # Average latency across all successful benchmarks
            avg_latency = sum(r["latency_mean_ms"] for r in successful_results) / len(successful_results)
            
            # Get best throughput across all configurations
            best_throughput = max(r["throughput_samples_per_sec"] for r in successful_results)
            best_throughput_config = [k for k, r in results["batch_results"].items() 
                                    if r["success"] and r["throughput_samples_per_sec"] == best_throughput][0]
            
            # Get best latency across all configurations
            best_latency = min(r["latency_mean_ms"] for r in successful_results)
            best_latency_config = [k for k, r in results["batch_results"].items() 
                                 if r["success"] and r["latency_mean_ms"] == best_latency][0]
            
            # Set summary metrics
            results["summary"] = {
                "average_latency_ms": avg_latency,
                "best_throughput_samples_per_sec": best_throughput,
                "best_throughput_config": best_throughput_config,
                "best_latency_ms": best_latency,
                "best_latency_config": best_latency_config,
                "success_percentage": len(successful_results) * 100 / len(results["batch_results"])
            }
        else:
            results["summary"] = {
                "error": "No successful benchmark runs"
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_id.replace("/", "_")
        filename = f"{model_name}_{self.device}_{self.precision}_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Write results to file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        return output_path


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Hardware benchmark for models")
    
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID")
    
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"],
                        help="Device to benchmark on (default: auto-select)")
    
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16", "int8"],
                        help="Precision for model")
    
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated list of batch sizes")
    
    parser.add_argument("--sequence-lengths", type=str, default="128,512",
                        help="Comma-separated list of sequence lengths")
    
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations per benchmark")
    
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    
    parser.add_argument("--save", action="store_true",
                        help="Save results to file")
    
    args = parser.parse_args()
    
    # Parse lists
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(",")]
    
    # Create benchmark
    benchmark = ModelBenchmark(args.model_id, args.device, args.precision)
    
    # Run benchmark
    results = benchmark.benchmark_model(
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        iterations=args.iterations
    )
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Model: {args.model_id}")
    print(f"Device: {benchmark.device}")
    print(f"Architecture: {benchmark.architecture_type}")
    
    if "load_results" in results and results["load_results"]["success"]:
        print(f"Load time: {results['load_results']['load_time_seconds']:.2f} seconds")
        print(f"Memory usage: {results['load_results']['memory_usage_mb']:.2f} MB")
    
    if "summary" in results and "average_latency_ms" in results["summary"]:
        print(f"Average latency: {results['summary']['average_latency_ms']:.2f} ms")
        print(f"Best throughput: {results['summary']['best_throughput_samples_per_sec']:.2f} samples/sec ({results['summary']['best_throughput_config']})")
        print(f"Best latency: {results['summary']['best_latency_ms']:.2f} ms ({results['summary']['best_latency_config']})")
    
    # Save results if requested
    if args.save:
        output_path = benchmark.save_results(results, args.output_dir)
        print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
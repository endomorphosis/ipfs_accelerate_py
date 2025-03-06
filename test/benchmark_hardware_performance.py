#!/usr/bin/env python
"""
Benchmark runner for local hardware performance testing with IPFS Accelerate.
This script runs comprehensive benchmarks on local hardware for different model families
and generates standardized performance reports.
"""

import os
import time
import json
import logging
import argparse
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import threading
import concurrent.futures

# Add DuckDB database support
try:
    from benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules with error handling
try:
    from resource_pool import get_global_resource_pool
except ImportError:
    logger.error("Failed to import ResourcePool. Make sure resource_pool.py is in the current directory.")
    raise

try:
    from hardware_detection import detect_hardware_with_comprehensive_checks
    HAS_HW_DETECTION = True
except ImportError:
    logger.warning("Hardware detection module not available. Using basic detection.")
    HAS_HW_DETECTION = False

try:
    from model_family_classifier import classify_model, ModelFamilyClassifier
    HAS_MODEL_CLASSIFIER = True
except ImportError:
    logger.warning("Model family classifier not available. Using model type as family.")
    HAS_MODEL_CLASSIFIER = False


class BenchmarkRunner:
    """
    Runs performance benchmarks on different hardware configurations for various model families.
    Generates standardized performance metrics and reports.
    """
    
    def __init__(self, 
                 output_dir: str = "./performance_results",
                 config_file: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize the benchmark runner
        
        Args:
            output_dir: Directory to save benchmark results
            config_file: Optional configuration file for benchmark settings
            debug: Enable debug logging
        """
        self.output_dir = output_dir
        self.debug = debug
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Get resource pool for efficient resource sharing
        self.pool = get_global_resource_pool()
        
        # Detect hardware capabilities
        self.hardware_info = self._detect_hardware()
        
        # Load benchmark config
        self.config = self._load_config(config_file)
        
        # Store benchmark results
        self.results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware": self.hardware_info,
            "benchmarks": []
        }
        
        # Running flag for stopping benchmarks
        self.running = True
        
        # Set up signal handling for graceful termination
        try:
            import signal
            def signal_handler(sig, frame):
                logger.info("Received termination signal. Stopping benchmarks gracefully...")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ImportError, AttributeError):
            logger.warning("Signal handling not available. Use Ctrl+C to stop benchmarks.")
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware using hardware_detection module if available"""
        if HAS_HW_DETECTION:
            return detect_hardware_with_comprehensive_checks()
        else:
            # Basic detection using PyTorch and system info
            hardware = {"cpu": True}
            
            # Try to detect CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    hardware["cuda"] = True
                    hardware["cuda_device_count"] = torch.cuda.device_count()
                    hardware["cuda_devices"] = []
                    
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        hardware["cuda_devices"].append({
                            "name": torch.cuda.get_device_name(i),
                            "total_memory": props.total_memory / (1024**3)  # Convert to GB
                        })
                else:
                    hardware["cuda"] = False
                
                # Try to detect MPS (Apple Silicon)
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    hardware["mps"] = True
                else:
                    hardware["mps"] = False
            except ImportError:
                hardware["cuda"] = False
                hardware["mps"] = False
            
            # Try to detect OpenVINO
            try:
                import openvino
                hardware["openvino"] = True
                hardware["openvino_version"] = openvino.__version__
            except ImportError:
                hardware["openvino"] = False
            
            return hardware
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load benchmark configuration
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Dictionary with benchmark configuration
        """
        default_config = {
            "batch_sizes": [1, 2, 4, 8],
            "sequence_lengths": [32, 64, 128, 256, 512],
            "warmup_iterations": 5,
            "benchmark_iterations": 10,
            "timeout_seconds": 300,  # 5 minutes per benchmark
            "mode": "inference",  # inference or training
            "hardware_platforms": ["cpu"],
            "model_families": {
                "embedding": {
                    "models": [
                        "prajjwal1/bert-tiny",
                        "bert-base-uncased"
                    ]
                },
                "text_generation": {
                    "models": [
                        "google/t5-efficient-tiny",
                        "gpt2" 
                    ]
                },
                "vision": {
                    "models": [
                        "google/vit-base-patch16-224"
                    ]
                },
                "audio": {
                    "models": [
                        "openai/whisper-tiny"
                    ]
                }
            }
        }
        
        # Set hardware platforms based on detected hardware
        if self.hardware_info.get("cuda", False):
            default_config["hardware_platforms"].append("cuda")
        if self.hardware_info.get("mps", False):
            default_config["hardware_platforms"].append("mps")
        if self.hardware_info.get("openvino", False):
            default_config["hardware_platforms"].append("openvino")
        
        # If config file is provided, load and merge with defaults
        if config_file and os.path.exists(config_file):
            try:
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
                    with open(config_file, 'r') as f:
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    user_config = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                            user_config = json.load(f)

                    
                    # Merge configs (user config takes precedence)
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                            # For nested dicts, merge recursively
                            default_config[key].update(value)
                        else:
                            # For simple values or new keys, just update
                            default_config[key] = value
                    
                    logger.info(f"Loaded benchmark configuration from {config_file}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Failed to load configuration from {config_file}: {str(e)}")
                    logger.info("Using default configuration")
            else:
                logger.info("Using default benchmark configuration")
            
            return default_config
        
        def import_model_dependencies(self) -> bool:
            """
            Import required dependencies for benchmarking
            
            Returns:
                True if all dependencies were successfully imported, False otherwise
            """
            try:
                # Import PyTorch and Transformers
                self.torch = self.pool.get_resource("torch", constructor=lambda: __import__("torch"))
                self.transformers = self.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
                
                # Check if imported successfully
                if not self.torch or not self.transformers:
                    logger.error("Failed to import required dependencies")
                    return False
                
                # Optionally import numpy for data processing
                self.numpy = self.pool.get_resource("numpy", constructor=lambda: __import__("numpy"))
                
                logger.info("Successfully imported all required dependencies")
                return True
            except Exception as e:
                logger.error(f"Error importing dependencies: {str(e)}")
                return False
        
        def run_benchmark(self, 
                          model_name: str, 
                          model_family: str, 
                          hardware_platform: str,
                          batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
            """
            Run benchmark for a specific model on a specific hardware platform
            
            Args:
                model_name: Name of the model to benchmark
                model_family: Family of the model (e.g., embedding, text_generation)
                hardware_platform: Hardware platform to run on (e.g., cpu, cuda, mps)
                batch_sizes: Optional list of batch sizes to test
                
            Returns:
                Dictionary with benchmark results
            """
            if not self.running:
                return {"status": "cancelled"}
            
            hardware_preferences = {"device": hardware_platform}
            
            logger.info(f"Benchmarking {model_name} ({model_family}) on {hardware_platform}")
            
            # Use batch sizes from config if not specified
            if batch_sizes is None:
                batch_sizes = self.config["batch_sizes"]
            
            # Create model constructor based on model family
            constructor = self._get_model_constructor(model_name, model_family)
            if not constructor:
                return {
                    "status": "error",
                    "error": f"Failed to create constructor for {model_name} ({model_family})"
                }
            
            # Load model through resource pool
            try:
                model = self.pool.get_model(
                    model_family,
                    model_name,
                    constructor=constructor,
                    hardware_preferences=hardware_preferences
                )
                
                if model is None:
                    return {
                        "status": "error",
                        "error": f"Failed to load model {model_name} on {hardware_platform}"
                    }
                
                # Determine device the model is on
                device = self._get_model_device(model)
                
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Error loading model: {str(e)}"
                }
            
            # Create tokenizer or processor based on model family
            tokenizer = None
            processor = None
            try:
                if model_family in ["embedding", "text_generation"]:
                    tokenizer = self.pool.get_tokenizer(
                        model_family,
                        model_name,
                        constructor=lambda: self.transformers.AutoTokenizer.from_pretrained(model_name)
                    )
                elif model_family == "vision":
                    processor = self.pool.get_resource(
                        f"processor:{model_name}",
                        constructor=lambda: self.transformers.AutoProcessor.from_pretrained(model_name)
                    )
                elif model_family == "audio":
                    processor = self.pool.get_resource(
                        f"processor:{model_name}",
                        constructor=lambda: self.transformers.AutoProcessor.from_pretrained(model_name)
                    )
            except Exception as e:
                logger.warning(f"Error loading tokenizer/processor: {str(e)}")
                # Continue without tokenizer/processor as we'll use random inputs
            
            # Run warmup iterations
            logger.info(f"Running {self.config['warmup_iterations']} warmup iterations")
            try:
                self._run_model_warmup(model, model_family, tokenizer, processor, device)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Error during warmup: {str(e)}"
                }
            
            # Initialize results structure
            benchmark_results = {
                "model_name": model_name,
                "model_family": model_family,
                "hardware_platform": hardware_platform,
                "device": str(device),
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "success",
                "batch_results": []
            }
            
            # Check if model has parameter count
            try:
                param_count = sum(p.numel() for p in model.parameters())
                benchmark_results["parameter_count"] = param_count
            except:
                pass
            
            # Estimate model size in memory
            try:
                model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                benchmark_results["model_size_mb"] = model_size_bytes / (1024 * 1024)
            except:
                pass
            
            # Run benchmark for each batch size
            for batch_size in batch_sizes:
                if not self.running:
                    benchmark_results["status"] = "cancelled"
                    break
                    
                logger.info(f"Benchmarking with batch size {batch_size}")
                
                # Run benchmarks with sequence lengths appropriate for the model family
                sequence_length_results = []
                seq_lengths = self._get_sequence_lengths(model_family)
                
                for seq_length in seq_lengths:
                    try:
                        # Set a timeout timer for this benchmark
                        timeout_event = threading.Event()
                        timer = threading.Timer(self.config["timeout_seconds"], lambda: timeout_event.set())
                        timer.start()
                        
                        # Start benchmark
                        start_time = time.time()
                        
                        # Run benchmark iterations
                        latencies = []
                        throughputs = []
                        memory_usages = []
                        
                        with self.torch.no_grad():  # Disable gradients for inference
                            for i in range(self.config["benchmark_iterations"]):
                                if timeout_event.is_set() or not self.running:
                                    # Benchmark timed out or was cancelled
                                    logger.warning(f"Benchmark timed out or was cancelled")
                                    break
                                
                                # Run a single iteration and measure performance
                                iter_result = self._run_single_iteration(
                                    model, 
                                    model_family, 
                                    batch_size, 
                                    seq_length, 
                                    tokenizer, 
                                    processor, 
                                    device
                                )
                                
                                latencies.append(iter_result["latency"])
                                throughputs.append(iter_result["throughput"])
                                memory_usages.append(iter_result["memory_usage"])
                                
                                if self.debug:
                                    logger.debug(f"Iteration {i+1}/{self.config['benchmark_iterations']}: "
                                                 f"Latency={iter_result['latency']:.4f}s, "
                                                 f"Throughput={iter_result['throughput']:.2f} items/s")
                        
                        # Clean up timer
                        timer.cancel()
                        
                        # Calculate statistics
                        if latencies:
                            avg_latency = sum(latencies) / len(latencies)
                            avg_throughput = sum(throughputs) / len(throughputs)
                            max_memory = max(memory_usages) if memory_usages else 0
                            
                            # Calculate standard deviation for latency
                            latency_std = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
                            
                            sequence_length_results.append({
                                "sequence_length": seq_length,
                                "average_latency_seconds": avg_latency,
                                "latency_std_dev": latency_std,
                                "average_throughput_items_per_second": avg_throughput,
                                "max_memory_usage_mb": max_memory,
                                "iterations_completed": len(latencies)
                            })
                        else:
                            sequence_length_results.append({
                                "sequence_length": seq_length,
                                "status": "failed",
                                "error": "No iterations completed"
                            })
                        
                    except Exception as e:
                        logger.error(f"Error benchmarking sequence length {seq_length}: {str(e)}")
                        sequence_length_results.append({
                            "sequence_length": seq_length,
                            "status": "error",
                            "error": str(e)
                        })
                
                # Add batch results
                benchmark_results["batch_results"].append({
                    "batch_size": batch_size,
                    "sequence_lengths": sequence_length_results
                })
            
            # Get final memory usage from resource pool
            stats = self.pool.get_stats()
            if "cuda_memory" in stats and hardware_platform == "cuda":
                benchmark_results["cuda_memory_stats"] = stats["cuda_memory"]
            
            return benchmark_results
        
        def run_training_benchmark(self, 
                              model_name: str, 
                              model_family: str, 
                              hardware_platform: str,
                              batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
            """
            Run training benchmark for a specific model on a specific hardware platform
            
            Args:
                model_name: Name of the model to benchmark
                model_family: Family of the model (e.g., embedding, text_generation)
                hardware_platform: Hardware platform to run on (e.g., cpu, cuda, mps)
                batch_sizes: Optional list of batch sizes to test
                
            Returns:
                Dictionary with benchmark results
            """
            if not self.running:
                return {"status": "cancelled"}
            
            hardware_preferences = {"device": hardware_platform}
            
            logger.info(f"Training benchmark for {model_name} ({model_family}) on {hardware_platform}")
            
            # Use batch sizes from config if not specified
            if batch_sizes is None:
                batch_sizes = self.config["batch_sizes"]
            
            # Create model constructor based on model family (for training)
            constructor = self._get_training_model_constructor(model_name, model_family)
            if not constructor:
                return {
                    "status": "error",
                    "error": f"Failed to create training constructor for {model_name} ({model_family})"
                }
            
            # Load model through resource pool
            try:
                model = self.pool.get_model(
                    model_family,
                    model_name,
                    constructor=constructor,
                    hardware_preferences=hardware_preferences
                )
                
                if model is None:
                    return {
                        "status": "error",
                        "error": f"Failed to load model {model_name} on {hardware_platform}"
                    }
                
                # Determine device the model is on
                device = self._get_model_device(model)
                
                # Set model to training mode
                model.train()
                
                logger.info(f"Training model loaded successfully on {device}")
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Error loading model for training: {str(e)}"
                }
            
            # Create tokenizer or processor based on model family
            tokenizer = None
            processor = None
            try:
                if model_family in ["embedding", "text_generation"]:
                    tokenizer = self.pool.get_tokenizer(
                        model_family,
                        model_name,
                        constructor=lambda: self.transformers.AutoTokenizer.from_pretrained(model_name)
                    )
                elif model_family == "vision":
                    processor = self.pool.get_resource(
                        f"processor:{model_name}",
                        constructor=lambda: self.transformers.AutoProcessor.from_pretrained(model_name)
                    )
                elif model_family == "audio":
                    processor = self.pool.get_resource(
                        f"processor:{model_name}",
                        constructor=lambda: self.transformers.AutoProcessor.from_pretrained(model_name)
                    )
            except Exception as e:
                logger.warning(f"Error loading tokenizer/processor for training: {str(e)}")
                # Continue without tokenizer/processor as we'll use random inputs
            
            # Create optimizer
            try:
                optimizer = self.torch.optim.AdamW(model.parameters(), lr=5e-5)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Error creating optimizer: {str(e)}"
                }
                
            # Run warmup iterations
            logger.info(f"Running {self.config['warmup_iterations']} training warmup iterations")
            try:
                self._run_training_warmup(model, optimizer, model_family, tokenizer, processor, device)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Error during training warmup: {str(e)}"
                }
            
            # Initialize results structure
            benchmark_results = {
                "model_name": model_name,
                "model_family": model_family,
                "hardware_platform": hardware_platform,
                "device": str(device),
                "mode": "training",
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "success",
                "batch_results": []
            }
            
            # Check if model has parameter count
            try:
                param_count = sum(p.numel() for p in model.parameters())
                benchmark_results["parameter_count"] = param_count
            except:
                pass
            
            # Estimate model size in memory
            try:
                model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                benchmark_results["model_size_mb"] = model_size_bytes / (1024 * 1024)
            except:
                pass
            
            # Run benchmark for each batch size
            for batch_size in batch_sizes:
                if not self.running:
                    benchmark_results["status"] = "cancelled"
                    break
                    
                logger.info(f"Training benchmark with batch size {batch_size}")
                
                # Run benchmarks with sequence lengths appropriate for the model family
                sequence_length_results = []
                seq_lengths = self._get_sequence_lengths(model_family)
                
                for seq_length in seq_lengths:
                    try:
                        # Set a timeout timer for this benchmark
                        timeout_event = threading.Event()
                        timer = threading.Timer(self.config["timeout_seconds"], lambda: timeout_event.set())
                        timer.start()
                        
                        # Start benchmark
                        start_time = time.time()
                        
                        # Run benchmark iterations
                        fwd_latencies = []
                        bwd_latencies = []
                        opt_latencies = []
                        total_latencies = []
                        throughputs = []
                        memory_usages = []
                        
                        for i in range(self.config["benchmark_iterations"]):
                            if timeout_event.is_set() or not self.running:
                                # Benchmark timed out or was cancelled
                                logger.warning(f"Training benchmark timed out or was cancelled")
                                break
                            
                            # Run a single training iteration and measure performance
                            iter_result = self._run_single_training_iteration(
                                model, 
                                optimizer,
                                model_family, 
                                batch_size, 
                                seq_length, 
                                tokenizer, 
                                processor, 
                                device
                            )
                            
                            fwd_latencies.append(iter_result["forward_latency"])
                            bwd_latencies.append(iter_result["backward_latency"])
                            opt_latencies.append(iter_result["optimizer_latency"])
                            total_latencies.append(iter_result["total_latency"])
                            throughputs.append(iter_result["throughput"])
                            memory_usages.append(iter_result["memory_usage"])
                            
                            if self.debug:
                                logger.debug(f"Training iteration {i+1}/{self.config['benchmark_iterations']}: "
                                             f"Total Latency={iter_result['total_latency']:.4f}s, "
                                             f"Throughput={iter_result['throughput']:.2f} items/s")
                        
                        # Clean up timer
                        timer.cancel()
                        
                        # Calculate statistics
                        if total_latencies:
                            avg_fwd_latency = sum(fwd_latencies) / len(fwd_latencies)
                            avg_bwd_latency = sum(bwd_latencies) / len(bwd_latencies)
                            avg_opt_latency = sum(opt_latencies) / len(opt_latencies) 
                            avg_total_latency = sum(total_latencies) / len(total_latencies)
                            avg_throughput = sum(throughputs) / len(throughputs)
                            max_memory = max(memory_usages) if memory_usages else 0
                            
                            # Calculate standard deviation for latency
                            latency_std = (sum((l - avg_total_latency) ** 2 for l in total_latencies) / len(total_latencies)) ** 0.5
                            
                            sequence_length_results.append({
                                "sequence_length": seq_length,
                                "average_forward_latency_seconds": avg_fwd_latency,
                                "average_backward_latency_seconds": avg_bwd_latency, 
                                "average_optimizer_latency_seconds": avg_opt_latency,
                                "average_total_latency_seconds": avg_total_latency,
                                "latency_std_dev": latency_std,
                                "average_throughput_items_per_second": avg_throughput,
                                "max_memory_usage_mb": max_memory,
                                "iterations_completed": len(total_latencies)
                            })
                        else:
                            sequence_length_results.append({
                                "sequence_length": seq_length,
                                "status": "failed",
                                "error": "No iterations completed"
                            })
                        
                    except Exception as e:
                        logger.error(f"Error benchmarking training sequence length {seq_length}: {str(e)}")
                        sequence_length_results.append({
                            "sequence_length": seq_length,
                            "status": "error",
                            "error": str(e)
                        })
                
                # Add batch results
                benchmark_results["batch_results"].append({
                    "batch_size": batch_size,
                    "sequence_lengths": sequence_length_results
                })
            
            # Get final memory usage from resource pool
            stats = self.pool.get_stats()
            if "cuda_memory" in stats and hardware_platform == "cuda":
                benchmark_results["cuda_memory_stats"] = stats["cuda_memory"]
            
            return benchmark_results
        
        def _get_model_constructor(self, model_name: str, model_family: str):
            """Get constructor function for the model"""
            model_class = None
            
            if model_family == "embedding":
                model_class = self.transformers.AutoModel
            elif model_family == "text_generation":
                # Check if model is T5 or GPT-style
                if "t5" in model_name.lower():
                    model_class = self.transformers.T5ForConditionalGeneration
                else:
                    model_class = self.transformers.AutoModelForCausalLM
            elif model_family == "vision":
                # Try to determine specific model class from name
                if "vit" in model_name.lower():
                    model_class = self.transformers.ViTForImageClassification
                else:
                    model_class = self.transformers.AutoModelForImageClassification
            elif model_family == "audio":
                if "whisper" in model_name.lower():
                    model_class = self.transformers.WhisperForConditionalGeneration
                elif "wav2vec" in model_name.lower():
                    model_class = self.transformers.Wav2Vec2ForCTC
                else:
                    model_class = self.transformers.AutoModelForAudioClassification
            
            if model_class:
                return lambda: model_class.from_pretrained(model_name)
            else:
                logger.error(f"Could not determine model class for {model_name} ({model_family})")
                return None
        
        def _get_model_device(self, model):
            """Get the device where the model is placed"""
            if hasattr(model, "device"):
                return model.device
            elif hasattr(model, "parameters"):
                try:
                    return next(model.parameters()).device
                except StopIteration:
                    return self.torch.device("cpu")
            else:
                return self.torch.device("cpu")
        
        def _run_model_warmup(self, model, model_family, tokenizer, processor, device):
            """Run warmup iterations"""
            with self.torch.no_grad():
                # Generate warmup inputs based on model family
                if model_family in ["embedding", "text_generation"]:
                    batch_size = 1
                    seq_length = 32
                    
                    if tokenizer:
                        # Use tokenizer if available
                        inputs = tokenizer("Warmup text for benchmarking", 
                                          return_tensors="pt", 
                                          padding="max_length", 
                                          max_length=seq_length)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    else:
                        # Generate random inputs
                        inputs = {
                            "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                            "attention_mask": self.torch.ones(batch_size, seq_length, device=device)
                        }
                    
                    # Run warmup iterations
                    for _ in range(self.config["warmup_iterations"]):
                        if model_family == "embedding":
                            model(**inputs)
                        else:  # text_generation
                            if "t5" in model.__class__.__name__.lower():
                                # Add decoder_input_ids for encoder-decoder models
                                decoder_inputs = {
                                    "decoder_input_ids": self.torch.zeros((batch_size, 1), dtype=self.torch.long, device=device)
                                }
                                model(**inputs, **decoder_inputs)
                            else:
                                model(**inputs)
                    
                elif model_family == "vision":
                    # Image input shape is typically [batch_size, channels, height, width]
                    batch_size = 1
                    
                    if processor:
                        # Use a dummy image if processor is available
                        try:
                            # Create simple dummy image
                            from PIL import Image
                            import numpy as np
                            dummy_image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
                            inputs = processor(images=dummy_image, return_tensors="pt")
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        except ImportError:
                            # Fall back to random tensors
                            inputs = {
                                "pixel_values": self.torch.rand(batch_size, 3, 224, 224, device=device)
                            }
                    else:
                        # Generate random image inputs
                        inputs = {
                            "pixel_values": self.torch.rand(batch_size, 3, 224, 224, device=device)
                        }
                    
                    # Run warmup iterations
                    for _ in range(self.config["warmup_iterations"]):
                        model(**inputs)
                        
                elif model_family == "audio":
                    batch_size = 1
                    seq_length = 16000  # 1 second of audio at 16kHz
                    
                    if processor:
                        try:
                            # Create a dummy audio array
                            import numpy as np
                            dummy_audio = np.random.randn(seq_length)
                            inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        except ImportError:
                            # Fall back to random tensors
                            inputs = {
                                "input_features": self.torch.rand(batch_size, 80, 3000, device=device)
                            }
                    else:
                        # Different audio models might have different input shapes
                        if "whisper" in model.__class__.__name__.lower():
                            # Whisper takes mel spectrograms of shape [batch_size, n_mels, seq_len]
                            inputs = {
                                "input_features": self.torch.rand(batch_size, 80, 3000, device=device)
                            }
                        elif "wav2vec" in model.__class__.__name__.lower():
                            # Wav2vec takes audio of shape [batch_size, seq_len]
                            inputs = {
                                "input_values": self.torch.rand(batch_size, seq_length, device=device)
                            }
                        else:
                            # Generic audio input
                            inputs = {
                                "input_values": self.torch.rand(batch_size, seq_length, device=device)
                            }
                    
                    # Run warmup iterations
                    for _ in range(self.config["warmup_iterations"]):
                        model(**inputs)
                
                else:
                    # Generic warmup for unknown model families
                    logger.warning(f"Unknown model family {model_family}. Using generic warmup.")
                    inputs = {
                        "input_ids": self.torch.randint(0, 1000, (1, 32), device=device),
                        "attention_mask": self.torch.ones(1, 32, device=device)
                    }
                    
                    for _ in range(self.config["warmup_iterations"]):
                        try:
                            model(**inputs)
                        except:
                            # Try alternative inputs if the first fails
                            try:
                                inputs = {
                                    "pixel_values": self.torch.rand(1, 3, 224, 224, device=device)
                                }
                                model(**inputs)
                            except:
                                logger.error("Failed to run warmup with generic inputs")
                                raise
        
        def _run_single_iteration(self, model, model_family, batch_size, seq_length, tokenizer, processor, device):
            """Run a single inference benchmark iteration"""
            # Clear CUDA cache before iteration if tracking memory
            if str(device).startswith("cuda"):
                self.torch.cuda.empty_cache()
                memory_before = self.torch.cuda.memory_allocated(device) / (1024 * 1024)
            else:
                memory_before = 0
            
            # Generate inputs based on model family
            if model_family in ["embedding", "text_generation"]:
                if tokenizer:
                    # Use tokenizer if available
                    text = "This is a sample text for benchmarking"
                    # Repeat the text for batch processing
                    texts = [text] * batch_size
                    inputs = tokenizer(texts, 
                                      return_tensors="pt", 
                                      padding="max_length", 
                                      max_length=seq_length)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    # Generate random inputs
                    inputs = {
                        "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                        "attention_mask": self.torch.ones(batch_size, seq_length, device=device)
                    }
                
                # Run the model
                start_time = time.time()
                
                if model_family == "embedding":
                    model(**inputs)
                else:  # text_generation
                    if "t5" in model.__class__.__name__.lower():
                        # Add decoder_input_ids for encoder-decoder models
                        decoder_inputs = {
                            "decoder_input_ids": self.torch.zeros((batch_size, 1), dtype=self.torch.long, device=device)
                        }
                        model(**inputs, **decoder_inputs)
                    else:
                        model(**inputs)
                        
                end_time = time.time()
                
            elif model_family == "vision":
                if processor:
                    try:
                        # Create batch of dummy images
                        from PIL import Image
                        import numpy as np
                        
                        # Create batch of images
                        dummy_images = []
                        for _ in range(batch_size):
                            dummy_image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
                            dummy_images.append(dummy_image)
                        
                        inputs = processor(images=dummy_images, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    except ImportError:
                        # Fall back to random tensors
                        inputs = {
                            "pixel_values": self.torch.rand(batch_size, 3, 224, 224, device=device)
                        }
                else:
                    # Generate random image inputs
                    inputs = {
                        "pixel_values": self.torch.rand(batch_size, 3, 224, 224, device=device)
                    }
                
                # Run the model
                start_time = time.time()
                model(**inputs)
                end_time = time.time()
                
            elif model_family == "audio":
                # For audio, we use an approximate "sequence length" that represents
                # duration in seconds × sample rate (e.g., 16000 samples per second)
                audio_length = seq_length * 16000
                
                if processor:
                    try:
                        # Create batch of dummy audio arrays
                        import numpy as np
                        
                        dummy_audios = []
                        for _ in range(batch_size):
                            dummy_audio = np.random.randn(audio_length)
                            dummy_audios.append(dummy_audio)
                        
                        inputs = processor(dummy_audios, sampling_rate=16000, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    except ImportError:
                        # Fall back to random tensors
                        if "whisper" in model.__class__.__name__.lower():
                            inputs = {
                                "input_features": self.torch.rand(batch_size, 80, seq_length * 100, device=device)
                            }
                        else:
                            inputs = {
                                "input_values": self.torch.rand(batch_size, audio_length, device=device)
                            }
                else:
                    # Different audio models might have different input shapes
                    if "whisper" in model.__class__.__name__.lower():
                        # Whisper takes mel spectrograms of shape [batch_size, n_mels, seq_len]
                        # Typically, seq_len is ~3000 for 30 seconds of audio
                        inputs = {
                            "input_features": self.torch.rand(batch_size, 80, seq_length * 100, device=device)
                        }
                    elif "wav2vec" in model.__class__.__name__.lower():
                        # Wav2vec takes audio of shape [batch_size, seq_len]
                        inputs = {
                            "input_values": self.torch.rand(batch_size, audio_length, device=device)
                        }
                    else:
                        # Generic audio input
                        inputs = {
                            "input_values": self.torch.rand(batch_size, audio_length, device=device)
                        }
                
                # Run the model
                start_time = time.time()
                model(**inputs)
                end_time = time.time()
                
            else:
                # Generic benchmark for unknown model families
                logger.warning(f"Unknown model family {model_family}. Using generic inputs.")
                inputs = {
                    "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                    "attention_mask": self.torch.ones(batch_size, seq_length, device=device)
                }
                
                # Try to run the model
                try:
                    start_time = time.time()
                    model(**inputs)
                    end_time = time.time()
                except:
                    # Try alternative inputs if the first fails
                    try:
                        inputs = {
                            "pixel_values": self.torch.rand(batch_size, 3, 224, 224, device=device)
                        }
                        start_time = time.time()
                        model(**inputs)
                        end_time = time.time()
                    except:
                        logger.error("Failed to run benchmark with generic inputs")
                        raise
            
            # Calculate latency and throughput
            latency = end_time - start_time
            throughput = batch_size / latency
            
            # Calculate memory usage
            if str(device).startswith("cuda"):
                memory_after = self.torch.cuda.memory_allocated(device) / (1024 * 1024)
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0
            
            return {
                "latency": latency,
                "throughput": throughput,
                "memory_usage": memory_usage
            }
        
        def _get_sequence_lengths(self, model_family: str) -> List[int]:
            """Get appropriate sequence lengths for the model family"""
            # Use sequence lengths from config
            default_lengths = self.config["sequence_lengths"]
            
            # Adjust based on model family
            if model_family == "text_generation":
                # Generation models typically need to test with longer sequences
                return default_lengths
            elif model_family == "embedding":
                # Embedding models typically use shorter sequences
                return [l for l in default_lengths if l <= 512]
            elif model_family == "vision":
                # Vision models typically use fixed input sizes
                return [1]  # Just use a single "sequence length" for images
            elif model_family == "audio":
                # For audio, sequence length represents seconds of audio
                return [1, 5, 10, 30]  # Test with 1, 5, 10, and 30 seconds of audio
            else:
                return default_lengths
        
        def run_all_benchmarks(self) -> Dict[str, Any]:
            """
            Run all benchmarks defined in the configuration
            
            Returns:
                Dictionary with all benchmark results
            """
            logger.info("Starting benchmark suite")
            
            # Import dependencies
            if not self.import_model_dependencies():
                return {"status": "error", "error": "Failed to import dependencies"}
            
            # Get hardware platforms from config
            hardware_platforms = self.config["hardware_platforms"]
            
            # Run benchmarks for all models on all hardware platforms
            for family_name, family_config in self.config["model_families"].items():
                for model_name in family_config["models"]:
                    for platform in hardware_platforms:
                        if not self.running:
                            logger.info("Benchmark run cancelled")
                            return {"status": "cancelled", "benchmarks": self.results["benchmarks"]}
                        
                        # Check if family-specific batch sizes are defined
                        batch_sizes = family_config.get("batch_sizes", self.config["batch_sizes"])
                        
                        try:
                            result = self.run_benchmark(
                                model_name=model_name,
                                model_family=family_name,
                                hardware_platform=platform,
                                batch_sizes=batch_sizes
                            )
                            
                            # Add result to overall results
                            self.results["benchmarks"].append(result)
                            
                            # Save intermediate results after each benchmark
                            self._save_results()
                            
                            # Generate model-specific report
                            self._generate_model_report(result)
                        except Exception as e:
                            logger.error(f"Error benchmarking {model_name} on {platform}: {str(e)}")
                            self.results["benchmarks"].append({
                                "model_name": model_name,
                                "model_family": family_name,
                                "hardware_platform": platform,
                                "status": "error",
                                "error": str(e)
                            })
            
            # Generate consolidated report
            self._generate_consolidated_report()
            
            logger.info("Benchmark suite completed successfully")
            return self.results
        
        def _save_results(self):
            """Save benchmark results to file"""
            # Get today's date for filename
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            output_file = os.path.join(self.output_dir, f"benchmark_results_{date_str}.json")
            
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

        
        logger.info(f"Saved benchmark results to {output_file}")
    
    def _generate_model_report(self, result: Dict[str, Any]):
        """Generate detailed report for a specific model benchmark"""
        if result.get("status") != "success":
            logger.warning(f"Skipping report generation for failed benchmark")
            return
        
        model_name = result["model_name"]
        model_family = result["model_family"]
        hardware_platform = result["hardware_platform"]
        
        # Create a short model name for filename
        short_name = model_name.split("/")[-1]
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        filename = f"benchmark_report_{short_name}_{hardware_platform}_{date_str}.md"
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w') as f:
            # Write report header
            f.write(f"# Performance Benchmark Report\n\n")
            f.write(f"- **Model:** {model_name}\n")
            f.write(f"- **Family:** {model_family}\n")
            f.write(f"- **Hardware Platform:** {hardware_platform}\n")
            f.write(f"- **Device:** {result.get('device', 'Unknown')}\n")
            f.write(f"- **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Write model details
            f.write(f"\n## Model Details\n\n")
            
            if "parameter_count" in result:
                f.write(f"- **Parameter Count:** {result['parameter_count']:,}\n")
            
            if "model_size_mb" in result:
                f.write(f"- **Model Size:** {result['model_size_mb']:.2f} MB\n")
            
            # Write benchmark results for each batch size
            f.write(f"\n## Benchmark Results\n\n")
            
            for batch_result in result["batch_results"]:
                batch_size = batch_result["batch_size"]
                f.write(f"### Batch Size: {batch_size}\n\n")
                
                # Create table of results
                f.write("| Sequence Length | Latency (s) | Throughput (items/s) | Memory Usage (MB) |\n")
                f.write("|----------------|------------|--------------------|-----------------|\n")
                
                for seq_result in batch_result["sequence_lengths"]:
                    if seq_result.get("status") == "error":
                        f.write(f"| {seq_result['sequence_length']} | Error: {seq_result.get('error', 'Unknown error')} | - | - |\n")
                    else:
                        latency = seq_result.get("average_latency_seconds", 0)
                        throughput = seq_result.get("average_throughput_items_per_second", 0)
                        memory = seq_result.get("max_memory_usage_mb", 0)
                        
                        f.write(f"| {seq_result['sequence_length']} | {latency:.4f} | {throughput:.2f} | {memory:.2f} |\n")
                
                f.write("\n")
            
            # Write CUDA memory info if available
            if "cuda_memory_stats" in result:
                f.write(f"\n## CUDA Memory Information\n\n")
                
                cuda_memory = result["cuda_memory_stats"]
                if "devices" in cuda_memory:
                    for device in cuda_memory["devices"]:
                        f.write(f"### Device {device.get('id', 0)}: {device.get('name', 'Unknown')}\n\n")
                        f.write(f"- **Total Memory:** {device.get('total_mb', 0):.2f} MB\n")
                        f.write(f"- **Allocated Memory:** {device.get('allocated_mb', 0):.2f} MB\n")
                        f.write(f"- **Free Memory:** {device.get('free_mb', 0):.2f} MB\n")
                        f.write(f"- **Utilization:** {device.get('percent_used', 0):.2f}%\n\n")
            
            # Write recommendations based on benchmark results
            f.write(f"\n## Performance Recommendations\n\n")
            
            # Find optimal batch size
            optimal_batch_size = self._find_optimal_batch_size(result)
            f.write(f"- **Recommended Batch Size:** {optimal_batch_size}\n")
            
            # Memory-related recommendations
            if hardware_platform == "cuda" and "cuda_memory_stats" in result:
                memory_util = 0
                for device in result["cuda_memory_stats"].get("devices", []):
                    memory_util = max(memory_util, device.get("percent_used", 0))
                
                if memory_util > 90:
                    f.write(f"- **Warning:** GPU memory utilization is very high ({memory_util:.1f}%). Consider using a smaller model or reducing batch size.\n")
                elif memory_util < 30:
                    f.write(f"- **Note:** GPU memory utilization is low ({memory_util:.1f}%). You may be able to increase batch size or use a larger model.\n")
            
            # Model family specific recommendations
            if model_family == "text_generation":
                f.write(f"- **Generation Models:** For optimal performance, consider using techniques like KV caching for inference.\n")
            elif model_family == "embedding":
                f.write(f"- **Embedding Models:** These models are well-suited for batch processing. Consider even larger batch sizes if memory allows.\n")
            elif model_family == "vision":
                f.write(f"- **Vision Models:** For higher throughput, consider preprocessing images to match the model's expected input size directly.\n")
            elif model_family == "audio":
                f.write(f"- **Audio Models:** Performance scales with audio length. Consider chunking long audio files for parallel processing.\n")
            
            # Write hardware-specific recommendations
            if hardware_platform == "cuda":
                f.write(f"- **CUDA Optimization:** Ensure you're using the latest CUDA drivers and libraries for optimal performance.\n")
            elif hardware_platform == "cpu":
                f.write(f"- **CPU Optimization:** Consider using quantized models for better CPU performance.\n")
            elif hardware_platform == "mps":
                f.write(f"- **MPS Optimization:** Apple Silicon performance is best with recent PyTorch versions (2.0+).\n")
            elif hardware_platform == "openvino":
                f.write(f"- **OpenVINO Optimization:** Consider using the OpenVINO Runtime API directly for further optimizations.\n")
        
        logger.info(f"Generated benchmark report: {report_path}")
    
    def _generate_consolidated_report(self):
        """Generate consolidated report for all benchmarks"""
        if not self.results.get("benchmarks"):
            logger.warning("No benchmark results to generate consolidated report")
            return
        
        # Create a filename with date
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        filename = f"consolidated_performance_summary_{date_str}.md"
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w') as f:
            # Write report header
            f.write(f"# Consolidated Performance Benchmark Report\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write system information
            f.write(f"## System Information\n\n")
            
            if "system" in self.hardware_info:
                system_info = self.hardware_info["system"]
                f.write(f"- **Platform:** {system_info.get('platform', 'Unknown')}\n")
                f.write(f"- **Architecture:** {system_info.get('architecture', 'Unknown')}\n")
                f.write(f"- **CPU Cores:** {system_info.get('cpu_count', 'Unknown')}\n")
                f.write(f"- **Python Version:** {system_info.get('python_version', 'Unknown')}\n")
                
                if "total_memory" in system_info:
                    f.write(f"- **Total Memory:** {system_info['total_memory']:.2f} GB\n")
            
            # Write available hardware
            f.write(f"\n## Available Hardware\n\n")
            
            # List detected hardware
            hardware_list = []
            for hw_type in ["cuda", "rocm", "mps", "openvino", "cpu"]:
                if self.hardware_info.get(hw_type, False):
                    hardware_list.append(hw_type.upper())
            
            f.write(f"- **Detected Hardware:** {', '.join(hardware_list)}\n")
            
            # Add CUDA details if available
            if self.hardware_info.get("cuda", False) and "cuda_devices" in self.hardware_info:
                f.write(f"\n### CUDA Devices\n\n")
                
                for i, device in enumerate(self.hardware_info["cuda_devices"]):
                    f.write(f"- **Device {i}:** {device.get('name', 'Unknown')}\n")
                    f.write(f"  - **Memory:** {device.get('total_memory', 0):.2f} GB\n")
            
            # Group benchmarks by model family
            family_results = {}
            for result in self.results["benchmarks"]:
                if result.get("status") != "success":
                    continue
                    
                family = result.get("model_family", "Unknown")
                if family not in family_results:
                    family_results[family] = []
                
                family_results[family].append(result)
            
            # Write summary tables by model family
            for family, results in family_results.items():
                f.write(f"\n## {family.capitalize()} Models\n\n")
                
                # Create performance comparison table
                f.write("| Model | Hardware | Batch Size | Sequence Length | Latency (s) | Throughput (items/s) | Memory (MB) |\n")
                f.write("|-------|----------|------------|----------------|-------------|---------------------|------------|\n")
                
                for result in results:
                    model_name = result["model_name"].split("/")[-1]
                    device = result["hardware_platform"]
                    
                    # Get best performing batch size
                    optimal_batch = self._find_optimal_batch_size(result)
                    
                    # Find batch result with optimal batch size
                    batch_result = next((b for b in result["batch_results"] if b["batch_size"] == optimal_batch), None)
                    
                    if not batch_result:
                        continue
                    
                    # Get median sequence length result
                    seq_results = batch_result["sequence_lengths"]
                    if not seq_results:
                        continue
                        
                    # Sort by sequence length
                    seq_results.sort(key=lambda x: x.get("sequence_length", 0))
                    
                    # Pick middle sequence length for comparison
                    middle_idx = len(seq_results) // 2
                    seq_result = seq_results[middle_idx]
                    
                    # Extract metrics
                    seq_len = seq_result.get("sequence_length", 0)
                    latency = seq_result.get("average_latency_seconds", 0)
                    throughput = seq_result.get("average_throughput_items_per_second", 0)
                    memory = seq_result.get("max_memory_usage_mb", 0)
                    
                    # Add row to table
                    f.write(f"| {model_name} | {device} | {optimal_batch} | {seq_len} | {latency:.4f} | {throughput:.2f} | {memory:.2f} |\n")
                
                f.write("\n")
            
            # Write performance insights
            f.write(f"\n## Performance Insights\n\n")
            
            # Add insights for each model family
            for family, results in family_results.items():
                f.write(f"### {family.capitalize()} Models\n\n")
                
                # Compare performance across hardware platforms
                if len(results) > 1:
                    # Group by model name
                    model_results = {}
                    for result in results:
                        model_name = result["model_name"]
                        if model_name not in model_results:
                            model_results[model_name] = []
                        model_results[model_name].append(result)
                    
                    # Compare hardware performance for each model
                    for model_name, model_data in model_results.items():
                        if len(model_data) <= 1:
                            continue
                            
                        short_name = model_name.split("/")[-1]
                        f.write(f"#### {short_name}\n\n")
                        
                        # Compare throughput across hardware
                        best_throughput = 0
                        best_platform = ""
                        
                        for result in model_data:
                            platform = result["hardware_platform"]
                            
                            # Find best throughput for this platform
                            best_batch_throughput = 0
                            for batch_result in result["batch_results"]:
                                for seq_result in batch_result["sequence_lengths"]:
                                    throughput = seq_result.get("average_throughput_items_per_second", 0)
                                    best_batch_throughput = max(best_batch_throughput, throughput)
                            
                            if best_batch_throughput > best_throughput:
                                best_throughput = best_batch_throughput
                                best_platform = platform
                                
                            f.write(f"- **{platform.upper()}:** Best throughput {best_batch_throughput:.2f} items/s\n")
                        
                        # Add summary insights
                        f.write(f"\nBest performance achieved on **{best_platform.upper()}** with {best_throughput:.2f} items/s throughput.\n\n")
                
                # Add family-specific insights
                if family == "text_generation":
                    f.write("\nText generation models show best performance with larger batch sizes on GPU hardware, but memory becomes a limiting factor. CPU performance is typically much lower due to the computational intensity of these models.\n\n")
                elif family == "embedding":
                    f.write("\nEmbedding models perform well across all hardware platforms and can utilize large batch sizes effectively. CPU performance is reasonable for production workloads with these models.\n\n")
                elif family == "vision":
                    f.write("\nVision models benefit significantly from GPU acceleration, with CUDA typically providing the best performance. OpenVINO offers competitive performance on CPU hardware.\n\n")
                elif family == "audio":
                    f.write("\nAudio model performance varies significantly based on input length. GPU acceleration provides substantial benefits for processing long audio sequences.\n\n")
            
            # Write overall recommendations
            f.write(f"\n## General Recommendations\n\n")
            
            f.write("1. **Hardware Selection:** Choose hardware based on model family and workload characteristics:\n")
            f.write("   - Text generation models perform best on CUDA GPUs with high memory capacity\n")
            f.write("   - Embedding models perform well on various hardware including CPU and MPS\n")
            f.write("   - Vision models benefit from both CUDA and specialized accelerators like OpenVINO\n")
            f.write("   - Audio models require CUDA GPUs for processing longer sequences efficiently\n\n")
            
            f.write("2. **Batch Size Optimization:** Select optimal batch sizes based on available memory and model characteristics:\n")
            f.write("   - Text generation: Smaller batch sizes (1-4) often optimal due to memory constraints\n")
            f.write("   - Embedding: Larger batch sizes (8-32) provide significant throughput improvements\n")
            f.write("   - Vision: Moderate batch sizes (4-16) balance throughput and memory usage\n")
            f.write("   - Audio: Batch size benefits diminish with longer audio sequences\n\n")
            
            f.write("3. **Memory Management:** Monitor memory usage carefully, especially for GPU deployments:\n")
            f.write("   - Reserve 10-20% GPU memory as buffer to avoid out-of-memory errors\n")
            f.write("   - Consider model quantization for memory-constrained environments\n")
            f.write("   - Use streaming inference for processing long inputs\n\n")
            
            f.write("4. **Platform-Specific Optimizations:**\n")
            f.write("   - CUDA: Ensure latest drivers and consider mixed precision for supported models\n")
            f.write("   - CPU: Enable multi-threading and consider quantized models\n")
            f.write("   - OpenVINO: Use model conversion tools for optimal performance\n")
            f.write("   - MPS (Apple Silicon): Use newer PyTorch versions (2.0+) for best performance\n\n")
        
        logger.info(f"Generated consolidated benchmark report: {report_path}")
    
    def _find_optimal_batch_size(self, result: Dict[str, Any]) -> int:
        """Find the optimal batch size based on throughput"""
        best_batch_size = 1
        best_throughput = 0
        
        for batch_result in result["batch_results"]:
            batch_size = batch_result["batch_size"]
            
            # Calculate average throughput across sequence lengths
            throughputs = []
            for seq_result in batch_result["sequence_lengths"]:
                throughput = seq_result.get("average_throughput_items_per_second", 0)
                if throughput > 0:
                    throughputs.append(throughput)
            
            if throughputs:
                avg_throughput = sum(throughputs) / len(throughputs)
                if avg_throughput > best_throughput:
                    best_throughput = avg_throughput
                    best_batch_size = batch_size
        
        return best_batch_size


def main():
    """Main function to run benchmarks"""
    parser = argparse.ArgumentParser(description="Run hardware benchmarks for various model families")
    parser.add_argument("--config", type=str, help="Path to benchmark configuration file")
    parser.add_argument("--output-dir", type=str, default="./performance_results", help="Directory to save benchmark results")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to benchmark, or 'all'")
    parser.add_argument("--family", type=str, help="Only benchmark models from this family (embedding, text_generation, vision, audio)")
    parser.add_argument("--hardware", type=str, help="Comma-separated list of hardware platforms to benchmark (cpu, cuda, mps, openvino)")
    parser.add_argument("--batch-size", type=int, help="Specific batch size to use for benchmarks")
    parser.add_argument("--batch-sizes", type=str, help="Comma-separated list of batch sizes to benchmark")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        config_file=args.config,
        debug=args.debug
    )
    
    # Apply command-line overrides to config
    if args.hardware:
        hardware_platforms = args.hardware.split(",")
        runner.config["hardware_platforms"] = hardware_platforms
        logger.info(f"Overriding hardware platforms: {hardware_platforms}")
    
    if args.batch_size:
        runner.config["batch_sizes"] = [args.batch_size]
        logger.info(f"Using single batch size: {args.batch_size}")
    elif args.batch_sizes:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        runner.config["batch_sizes"] = batch_sizes
        logger.info(f"Using custom batch sizes: {batch_sizes}")
    
    # Filter models if requested
    if args.models != "all":
        models = args.models.split(",")
        
        # Update models in config
        for family in runner.config["model_families"]:
            family_models = runner.config["model_families"][family]["models"]
            
            # Filter to only include requested models
            filtered_models = [m for m in family_models if m in models or m.split("/")[-1] in models]
            
            runner.config["model_families"][family]["models"] = filtered_models
            
        logger.info(f"Filtering to benchmark only models: {models}")
    
    # Filter by family if requested
    if args.family:
        family = args.family.lower()
        
        # Keep only the requested family
        families_to_remove = [f for f in runner.config["model_families"] if f != family]
        for f in families_to_remove:
            del runner.config["model_families"][f]
            
        if family not in runner.config["model_families"]:
            logger.error(f"Unknown model family: {family}")
            return 1
            
        logger.info(f"Filtering to benchmark only {family} models")
    
    # Run all benchmarks
    try:
        results = runner.run_all_benchmarks()
        
        if results.get("status") == "error":
            logger.error(f"Benchmark run failed: {results.get('error', 'Unknown error')}")
            return 1
        
        if results.get("status") == "cancelled":
            logger.warning("Benchmark run was cancelled")
            return 0
        
        logger.info("Benchmark run completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Benchmark run interrupted by user")
        # Save any results that have been collected
        runner._save_results()
        return 0
    except Exception as e:
        logger.error(f"Error during benchmark run: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
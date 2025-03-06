#!/usr/bin/env python
"""
Training Mode Benchmark for the IPFS Accelerate Python Framework.

This module implements training mode benchmarks as part of Phase 16 of the IPFS Accelerate Python framework 
project, focusing on:

1. Benchmarking model training performance across hardware platforms
2. Measuring forward pass, backward pass, and optimizer step metrics
3. Analyzing memory usage and throughput during training
4. Integrating with the benchmark database system

This is an extension to the benchmark_hardware_performance.py module, specifically focused on training 
rather than inference.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import threading
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    from model_family_classifier import classify_model
    HAS_MODEL_CLASSIFIER = True
except ImportError:
    logger.warning("Model family classifier not available. Using model type as family.")
    HAS_MODEL_CLASSIFIER = False


class TrainingBenchmarkRunner:
    """
    Runs training performance benchmarks on different hardware configurations for various model families.
    Measures forward pass, backward pass, and optimizer step performance, and generates standardized reports.
    """
    
    def __init__(self, 
                 output_dir: str = "./training_benchmark_results",
                 config_file: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize the training benchmark runner
        
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
        
        # Import required libraries
        try:
            import torch
            self.torch = torch
            HAS_TORCH = True
        except ImportError:
            logger.error("PyTorch is required for training benchmarks")
            HAS_TORCH = False
            self.torch = None
        
        try:
            import transformers
            self.transformers = transformers
            HAS_TRANSFORMERS = True
        except ImportError:
            logger.error("Transformers is required for training benchmarks")
            HAS_TRANSFORMERS = False
            self.transformers = None
        
        try:
            import numpy
            self.numpy = numpy
            HAS_NUMPY = True
        except ImportError:
            logger.warning("NumPy not available, using fallback random data generation")
            HAS_NUMPY = False
            self.numpy = None
        
        try:
            import PIL.Image
            self.PIL = PIL
            HAS_PIL = True
        except ImportError:
            logger.warning("PIL not available, using fallback image generation")
            HAS_PIL = False
            self.PIL = None
        
        # Load benchmark config
        self.config = self._load_config(config_file)
        
        # Store benchmark results
        self.results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware": self.hardware_info,
            "mode": "training",
            "benchmarks": []
        }
        
        # Running flag for stopping benchmarks
        self.running = True
        
        # Set up signal handling for graceful termination
        try:
            import signal
            def signal_handler(sig, frame):
                logger.info("Received termination signal. Stopping training benchmarks gracefully...")
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
            
            # Try to detect AMD ROCm
            try:
                if os.environ.get("ROCM_VERSION") or os.path.exists("/opt/rocm"):
                    hardware["rocm"] = True
                else:
                    hardware["rocm"] = False
            except:
                hardware["rocm"] = False
            
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
            "sequence_lengths": [32, 64, 128, 256],
            "warmup_iterations": 5,
            "benchmark_iterations": 10,
            "timeout_seconds": 600,  # 10 minutes per benchmark
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
            },
            "optimizers": {
                "default": {
                    "name": "adamw",
                    "lr": 5e-5,
                    "weight_decay": 0.01
                },
                "text_generation": {
                    "name": "adamw",
                    "lr": 1e-5,
                    "weight_decay": 0.01
                }
            }
        }
        
        # Set hardware platforms based on detected hardware
        if self.hardware_info.get("cuda", False):
            default_config["hardware_platforms"].append("cuda")
        if self.hardware_info.get("mps", False):
            default_config["hardware_platforms"].append("mps")
        if self.hardware_info.get("rocm", False):
            default_config["hardware_platforms"].append("rocm")
        if self.hardware_info.get("openvino", False):
            default_config["hardware_platforms"].append("openvino")
        
        # If configuration file provided, load it and merge with defaults
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict) and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration file: {str(e)}")
        
        return default_config
    
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
        
        # Get optimizer configuration
        optimizer_config = self.config["optimizers"].get(model_family, self.config["optimizers"]["default"])
        
        # Create optimizer
        try:
            if optimizer_config["name"].lower() == "adamw":
                optimizer = self.torch.optim.AdamW(
                    model.parameters(), 
                    lr=optimizer_config.get("lr", 5e-5),
                    weight_decay=optimizer_config.get("weight_decay", 0.01)
                )
            elif optimizer_config["name"].lower() == "adam":
                optimizer = self.torch.optim.Adam(
                    model.parameters(), 
                    lr=optimizer_config.get("lr", 5e-5)
                )
            elif optimizer_config["name"].lower() == "sgd":
                optimizer = self.torch.optim.SGD(
                    model.parameters(), 
                    lr=optimizer_config.get("lr", 0.01),
                    momentum=optimizer_config.get("momentum", 0.9)
                )
            else:
                # Default to AdamW
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
            "optimizer": optimizer_config["name"],
            "learning_rate": optimizer_config.get("lr", 5e-5),
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
                    losses = []
                    
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
                        if "loss" in iter_result:
                            losses.append(iter_result["loss"])
                        
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
                        
                        result = {
                            "sequence_length": seq_length,
                            "average_forward_latency_seconds": avg_fwd_latency,
                            "average_backward_latency_seconds": avg_bwd_latency, 
                            "average_optimizer_latency_seconds": avg_opt_latency,
                            "average_total_latency_seconds": avg_total_latency,
                            "latency_std_dev": latency_std,
                            "average_throughput_items_per_second": avg_throughput,
                            "max_memory_usage_mb": max_memory,
                            "iterations_completed": len(total_latencies)
                        }
                        
                        if losses:
                            result["average_loss"] = sum(losses) / len(losses)
                            
                        sequence_length_results.append(result)
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
    
    def _get_training_model_constructor(self, model_name: str, model_family: str):
        """Get constructor function for the model in training mode"""
        # Similar to _get_model_constructor but may have different configurations for training
        model_class = None
        
        if model_family == "embedding":
            model_class = self.transformers.AutoModelForMaskedLM
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
            logger.error(f"Could not determine training model class for {model_name} ({model_family})")
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
    
    def _get_sequence_lengths(self, model_family: str) -> List[int]:
        """Get appropriate sequence lengths for the model family"""
        if model_family in ["embedding", "text_generation"]:
            return self.config["sequence_lengths"]
        elif model_family == "vision":
            # Vision models don't typically use sequence length in the same way
            # Return just a single "sequence length" for benchmarking consistency
            return [224]  # Standard image size
        elif model_family == "audio":
            # Audio "sequence length" is in seconds of audio
            return [1, 2, 4]  # 1, 2, and 4 seconds of audio
        else:
            return [32]  # Default sequence length
            
    def _run_training_warmup(self, model, optimizer, model_family, tokenizer, processor, device):
        """Run warmup iterations for training benchmark"""
        batch_size = 1
        seq_length = 32
        
        # Run warmup iterations
        for _ in range(self.config["warmup_iterations"]):
            try:
                # Generate inputs based on model family
                if model_family in ["embedding", "text_generation"]:
                    if tokenizer:
                        # Use tokenizer to generate inputs
                        input_text = ["This is a test input for warmup"] * batch_size
                        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        if "labels" not in inputs:
                            inputs["labels"] = inputs["input_ids"].clone()
                    else:
                        # Generate random inputs
                        inputs = {
                            "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                            "attention_mask": self.torch.ones((batch_size, seq_length), device=device),
                            "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                        }
                elif model_family == "vision":
                    if processor:
                        try:
                            # Generate a random image
                            import numpy as np
                            from PIL import Image
                            
                            dummy_image = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
                            dummy_image = [Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)) for img in dummy_image]
                            inputs = processor(images=dummy_image, return_tensors="pt")
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            inputs["labels"] = self.torch.randint(0, 10, (batch_size,), device=device)
                        except ImportError:
                            # Fall back to random tensors
                            inputs = {
                                "pixel_values": self.torch.rand((batch_size, 3, 224, 224), device=device),
                                "labels": self.torch.randint(0, 10, (batch_size,), device=device)
                            }
                    else:
                        # Generate random inputs
                        inputs = {
                            "pixel_values": self.torch.rand((batch_size, 3, 224, 224), device=device),
                            "labels": self.torch.randint(0, 10, (batch_size,), device=device)
                        }
                elif model_family == "audio":
                    if processor:
                        try:
                            # Generate a random audio signal
                            import numpy as np
                            
                            sample_rate = 16000
                            dummy_audio = [np.random.randn(16000).astype(np.float32) for _ in range(batch_size)]
                            inputs = processor(audio=dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            if "labels" not in inputs:
                                inputs["labels"] = self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                        except ImportError:
                            # Fall back to random tensors
                            inputs = {
                                "input_values": self.torch.rand((batch_size, 16000), device=device),
                                "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                            }
                    else:
                        # Generate random inputs for audio models
                        inputs = {
                            "input_values": self.torch.rand((batch_size, 16000), device=device),
                            "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                        }
                else:
                    # Generate generic inputs for other model families
                    inputs = {
                        "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                        "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                    }
                
                # Run training step
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                logger.warning(f"Error during training warmup: {str(e)}")
                # Try with generic inputs if specific inputs failed
                try:
                    inputs = {
                        "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                        "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                    }
                    
                    optimizer.zero_grad()
                    outputs = model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                except:
                    logger.error("Failed to run training warmup with generic inputs")
                    raise
    
    def _run_single_training_iteration(self, model, optimizer, model_family, batch_size, seq_length, tokenizer, processor, device):
        """Run a single training benchmark iteration"""
        # Clear CUDA cache before iteration if tracking memory
        if str(device).startswith("cuda"):
            self.torch.cuda.empty_cache()
            memory_before = self.torch.cuda.memory_allocated(device) / (1024 * 1024)
        else:
            memory_before = 0
        
        # Generate inputs based on model family
        if model_family in ["embedding", "text_generation"]:
            if tokenizer:
                # Use tokenizer to generate inputs
                input_text = ["This is a test input for benchmarking"] * batch_size
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if "labels" not in inputs:
                    inputs["labels"] = inputs["input_ids"].clone()
            else:
                # Generate random inputs
                inputs = {
                    "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                    "attention_mask": self.torch.ones((batch_size, seq_length), device=device),
                    "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                }
        elif model_family == "vision":
            if processor:
                try:
                    # Generate a random image
                    import numpy as np
                    from PIL import Image
                    
                    dummy_image = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
                    dummy_image = [Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)) for img in dummy_image]
                    inputs = processor(images=dummy_image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    inputs["labels"] = self.torch.randint(0, 10, (batch_size,), device=device)
                except ImportError:
                    # Fall back to random tensors
                    inputs = {
                        "pixel_values": self.torch.rand((batch_size, 3, 224, 224), device=device),
                        "labels": self.torch.randint(0, 10, (batch_size,), device=device)
                    }
            else:
                # Generate random inputs
                inputs = {
                    "pixel_values": self.torch.rand((batch_size, 3, 224, 224), device=device),
                    "labels": self.torch.randint(0, 10, (batch_size,), device=device)
                }
        elif model_family == "audio":
            if processor:
                try:
                    # Generate a random audio signal
                    import numpy as np
                    
                    sample_rate = 16000
                    dummy_audio = [np.random.randn(sample_rate * seq_length).astype(np.float32) for _ in range(batch_size)]
                    inputs = processor(audio=dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    if "labels" not in inputs:
                        inputs["labels"] = self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                except ImportError:
                    # Fall back to random tensors
                    inputs = {
                        "input_values": self.torch.rand((batch_size, 16000 * seq_length), device=device),
                        "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                    }
            else:
                # Generate random inputs for audio models
                inputs = {
                    "input_values": self.torch.rand((batch_size, 16000 * seq_length), device=device),
                    "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
                }
        else:
            # Generate generic inputs for other model families
            inputs = {
                "input_ids": self.torch.randint(0, 1000, (batch_size, seq_length), device=device),
                "labels": self.torch.randint(0, 1000, (batch_size, seq_length), device=device)
            }
        
        # Forward pass timing
        start_time = time.time()
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        self.torch.cuda.synchronize() if str(device).startswith("cuda") else None
        forward_end_time = time.time()
        
        # Backward pass timing
        loss.backward()
        self.torch.cuda.synchronize() if str(device).startswith("cuda") else None
        backward_end_time = time.time()
        
        # Optimizer step timing
        optimizer.step()
        self.torch.cuda.synchronize() if str(device).startswith("cuda") else None
        optimizer_end_time = time.time()
        
        # Calculate memory usage
        if str(device).startswith("cuda"):
            memory_after = self.torch.cuda.memory_allocated(device) / (1024 * 1024)
            memory_usage = memory_after - memory_before
        else:
            memory_usage = 0
        
        # Calculate timing metrics
        forward_latency = forward_end_time - start_time
        backward_latency = backward_end_time - forward_end_time
        optimizer_latency = optimizer_end_time - backward_end_time
        total_latency = optimizer_end_time - start_time
        throughput = batch_size / total_latency
        
        # Get loss value
        loss_value = loss.item() if hasattr(loss, "item") else float(loss)
        
        return {
            "forward_latency": forward_latency,
            "backward_latency": backward_latency,
            "optimizer_latency": optimizer_latency,
            "total_latency": total_latency,
            "throughput": throughput,
            "memory_usage": memory_usage,
            "loss": loss_value
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all training benchmarks defined in the configuration"""
        logger.info("Starting training benchmark suite")
        
        # Run benchmarks for each model family and hardware platform
        for family_name, family_config in self.config["model_families"].items():
            logger.info(f"Benchmarking model family: {family_name}")
            
            for model_name in family_config["models"]:
                logger.info(f"Benchmarking model: {model_name}")
                
                for platform in self.config["hardware_platforms"]:
                    if not self.running:
                        logger.info("Benchmark run cancelled")
                        return {"status": "cancelled", "benchmarks": self.results["benchmarks"]}
                    
                    # Check if family-specific batch sizes are defined
                    batch_sizes = family_config.get("batch_sizes", self.config["batch_sizes"])
                    
                    try:
                        result = self.run_training_benchmark(
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
        
        logger.info("Training benchmark suite completed successfully")
        return self.results
    
    def _save_results(self):
        """Save benchmark results to file"""
        # Get today's date for filename
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(self.output_dir, f"training_benchmark_results_{date_str}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved training benchmark results to {output_file}")
    
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
        filename = f"training_benchmark_report_{short_name}_{hardware_platform}_{date_str}.md"
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w') as f:
            # Write report header
            f.write(f"# Training Performance Benchmark Report\n\n")
            f.write(f"- **Model:** {model_name}\n")
            f.write(f"- **Family:** {model_family}\n")
            f.write(f"- **Hardware Platform:** {hardware_platform}\n")
            f.write(f"- **Device:** {result.get('device', 'Unknown')}\n")
            f.write(f"- **Optimizer:** {result.get('optimizer', 'Unknown')}\n")
            f.write(f"- **Learning Rate:** {result.get('learning_rate', 'Unknown')}\n")
            f.write(f"- **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Write model details
            f.write(f"\n## Model Details\n\n")
            
            if "parameter_count" in result:
                f.write(f"- **Parameter Count:** {result['parameter_count']:,}\n")
            
            if "model_size_mb" in result:
                f.write(f"- **Model Size:** {result['model_size_mb']:.2f} MB\n")
            
            # Write benchmark results for each batch size
            f.write(f"\n## Training Benchmark Results\n\n")
            
            for batch_result in result["batch_results"]:
                batch_size = batch_result["batch_size"]
                f.write(f"### Batch Size: {batch_size}\n\n")
                
                # Create table of results
                f.write("| Sequence Length | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | Throughput (items/s) | Memory (MB) |\n")
                f.write("|----------------|------------|--------------|--------------|-----------|----------------------|-----------|\n")
                
                for seq_result in batch_result["sequence_lengths"]:
                    if seq_result.get("status") == "error":
                        f.write(f"| {seq_result['sequence_length']} | Error: {seq_result.get('error', 'Unknown error')} | - | - | - | - | - |\n")
                    else:
                        fwd = seq_result.get("average_forward_latency_seconds", 0)
                        bwd = seq_result.get("average_backward_latency_seconds", 0)
                        opt = seq_result.get("average_optimizer_latency_seconds", 0)
                        total = seq_result.get("average_total_latency_seconds", 0)
                        throughput = seq_result.get("average_throughput_items_per_second", 0)
                        memory = seq_result.get("max_memory_usage_mb", 0)
                        
                        f.write(f"| {seq_result['sequence_length']} | {fwd:.4f} | {bwd:.4f} | {opt:.4f} | {total:.4f} | {throughput:.2f} | {memory:.2f} |\n")
                
                f.write("\n")
                
                # Add loss information if available
                if any("average_loss" in seq_result for seq_result in batch_result["sequence_lengths"]):
                    f.write("| Sequence Length | Average Loss |\n")
                    f.write("|----------------|-------------|\n")
                    
                    for seq_result in batch_result["sequence_lengths"]:
                        if "average_loss" in seq_result:
                            f.write(f"| {seq_result['sequence_length']} | {seq_result['average_loss']:.6f} |\n")
                    
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
            
            # Write training-specific recommendations
            f.write(f"\n## Training Performance Recommendations\n\n")
            
            # Find optimal batch size
            optimal_batch_size = self._find_optimal_batch_size(result)
            f.write(f"- **Recommended Batch Size:** {optimal_batch_size}\n")
            
            # Memory-related recommendations
            if hardware_platform == "cuda" and "cuda_memory_stats" in result:
                memory_util = 0
                for device in result["cuda_memory_stats"].get("devices", []):
                    memory_util = max(memory_util, device.get("percent_used", 0))
                
                if memory_util > 90:
                    f.write(f"- **Warning:** GPU memory utilization is very high ({memory_util:.1f}%). Consider gradient accumulation or a smaller model.\n")
                elif memory_util < 30:
                    f.write(f"- **Note:** GPU memory utilization is low ({memory_util:.1f}%). You may be able to increase batch size for better throughput.\n")
            
            # Model family specific recommendations
            if model_family == "text_generation":
                f.write(f"- **Text Generation Models:** Training large language models benefits from techniques like gradient checkpointing to reduce memory usage.\n")
            elif model_family == "embedding":
                f.write(f"- **Embedding Models:** Consider using a higher learning rate and larger batch sizes for faster convergence.\n")
            elif model_family == "vision":
                f.write(f"- **Vision Models:** Mixed precision training (fp16) can significantly increase performance on supported hardware.\n")
            elif model_family == "audio":
                f.write(f"- **Audio Models:** Consider preprocessing audio features before training to reduce computational overhead.\n")
    
    def _generate_consolidated_report(self):
        """Generate consolidated report for all training benchmarks"""
        if not self.results.get("benchmarks"):
            logger.warning("No benchmark results to generate consolidated report")
            return
        
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        filename = f"training_benchmark_consolidated_report_{date_str}.md"
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w') as f:
            f.write("# Training Benchmark Consolidated Report\n\n")
            f.write(f"- **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Mode:** Training\n\n")
            
            # Write hardware information
            f.write("## Hardware Information\n\n")
            hw_info = self.hardware_info
            
            # CPU info
            f.write("### CPU\n\n")
            f.write("- Available: Yes\n")
            
            # CUDA info
            f.write("\n### CUDA\n\n")
            if hw_info.get("cuda", False):
                f.write("- Available: Yes\n")
                f.write(f"- Device Count: {hw_info.get('cuda_device_count', 'Unknown')}\n")
                
                if "cuda_devices" in hw_info:
                    for i, device in enumerate(hw_info["cuda_devices"]):
                        f.write(f"- Device {i}: {device.get('name', 'Unknown')} ({device.get('total_memory', 0):.2f} GB)\n")
            else:
                f.write("- Available: No\n")
            
            # MPS info
            f.write("\n### MPS (Apple Silicon)\n\n")
            if hw_info.get("mps", False):
                f.write("- Available: Yes\n")
            else:
                f.write("- Available: No\n")
            
            # AMD info
            f.write("\n### AMD ROCm\n\n")
            if hw_info.get("rocm", False):
                f.write("- Available: Yes\n")
            else:
                f.write("- Available: No\n")
            
            # OpenVINO info
            f.write("\n### OpenVINO\n\n")
            if hw_info.get("openvino", False):
                f.write("- Available: Yes\n")
                f.write(f"- Version: {hw_info.get('openvino_version', 'Unknown')}\n")
            else:
                f.write("- Available: No\n")
            
            # Write summary of benchmarks
            f.write("\n## Training Benchmark Summary\n\n")
            
            # Organize by model family
            by_family = {}
            for benchmark in self.results["benchmarks"]:
                if benchmark.get("status") != "success":
                    continue
                    
                family = benchmark.get("model_family", "unknown")
                if family not in by_family:
                    by_family[family] = []
                    
                by_family[family].append(benchmark)
            
            # Create a summary table for each family
            for family, benchmarks in by_family.items():
                f.write(f"### {family.capitalize()} Models\n\n")
                
                # Throughput comparison
                f.write("#### Training Throughput (items/second) by Batch Size\n\n")
                f.write("| Model | Hardware | Batch Size 1 | Batch Size 2 | Batch Size 4 | Batch Size 8 |\n")
                f.write("|-------|----------|--------------|--------------|--------------|-------------|\n")
                
                for benchmark in benchmarks:
                    model = benchmark["model_name"].split("/")[-1]
                    hardware = benchmark["hardware_platform"]
                    
                    # Extract throughput for each batch size
                    throughputs = {}
                    for batch_result in benchmark.get("batch_results", []):
                        batch_size = batch_result["batch_size"]
                        # Get average throughput across all sequence lengths
                        all_throughputs = [
                            seq_result.get("average_throughput_items_per_second", 0)
                            for seq_result in batch_result.get("sequence_lengths", [])
                            if "average_throughput_items_per_second" in seq_result
                        ]
                        
                        if all_throughputs:
                            throughputs[batch_size] = sum(all_throughputs) / len(all_throughputs)
                    
                    # Write row
                    row = f"| {model} | {hardware} "
                    for bs in [1, 2, 4, 8]:
                        if bs in throughputs:
                            row += f"| {throughputs[bs]:.2f} "
                        else:
                            row += "| - "
                    row += "|"
                    f.write(row + "\n")
                
                f.write("\n")
                
                # Forward/Backward/Optimizer breakdown
                f.write("#### Training Phase Breakdown (% of total time) with Batch Size 4\n\n")
                f.write("| Model | Hardware | Forward | Backward | Optimizer |\n")
                f.write("|-------|----------|---------|-----------|----------|\n")
                
                for benchmark in benchmarks:
                    model = benchmark["model_name"].split("/")[-1]
                    hardware = benchmark["hardware_platform"]
                    
                    # Find batch size 4 results or closest available
                    batch_4_results = None
                    for batch_result in benchmark.get("batch_results", []):
                        if batch_result["batch_size"] == 4:
                            batch_4_results = batch_result
                            break
                    
                    if not batch_4_results and benchmark.get("batch_results"):
                        # Get closest batch size
                        batch_sizes = [r["batch_size"] for r in benchmark["batch_results"]]
                        closest = min(batch_sizes, key=lambda x: abs(x - 4))
                        for batch_result in benchmark["batch_results"]:
                            if batch_result["batch_size"] == closest:
                                batch_4_results = batch_result
                                break
                    
                    if batch_4_results:
                        # Calculate average percentages across all sequence lengths
                        fwd_pct = []
                        bwd_pct = []
                        opt_pct = []
                        
                        for seq_result in batch_4_results.get("sequence_lengths", []):
                            if all(k in seq_result for k in ["average_forward_latency_seconds", 
                                                           "average_backward_latency_seconds",
                                                           "average_optimizer_latency_seconds",
                                                           "average_total_latency_seconds"]):
                                fwd = seq_result["average_forward_latency_seconds"]
                                bwd = seq_result["average_backward_latency_seconds"]
                                opt = seq_result["average_optimizer_latency_seconds"]
                                total = seq_result["average_total_latency_seconds"]
                                
                                fwd_pct.append(fwd / total * 100)
                                bwd_pct.append(bwd / total * 100)
                                opt_pct.append(opt / total * 100)
                        
                        if fwd_pct and bwd_pct and opt_pct:
                            avg_fwd_pct = sum(fwd_pct) / len(fwd_pct)
                            avg_bwd_pct = sum(bwd_pct) / len(bwd_pct)
                            avg_opt_pct = sum(opt_pct) / len(opt_pct)
                            
                            f.write(f"| {model} | {hardware} | {avg_fwd_pct:.1f}% | {avg_bwd_pct:.1f}% | {avg_opt_pct:.1f}% |\n")
                        else:
                            f.write(f"| {model} | {hardware} | - | - | - |\n")
                    else:
                        f.write(f"| {model} | {hardware} | - | - | - |\n")
                
                f.write("\n")
                
                # Memory Usage
                f.write("#### Peak Memory Usage (MB) by Batch Size\n\n")
                f.write("| Model | Hardware | Batch Size 1 | Batch Size 2 | Batch Size 4 | Batch Size 8 |\n")
                f.write("|-------|----------|--------------|--------------|--------------|-------------|\n")
                
                for benchmark in benchmarks:
                    model = benchmark["model_name"].split("/")[-1]
                    hardware = benchmark["hardware_platform"]
                    
                    # Extract memory usage for each batch size
                    memory_usage = {}
                    for batch_result in benchmark.get("batch_results", []):
                        batch_size = batch_result["batch_size"]
                        # Get max memory usage across all sequence lengths
                        all_memory = [
                            seq_result.get("max_memory_usage_mb", 0)
                            for seq_result in batch_result.get("sequence_lengths", [])
                            if "max_memory_usage_mb" in seq_result
                        ]
                        
                        if all_memory:
                            memory_usage[batch_size] = max(all_memory)
                    
                    # Write row
                    row = f"| {model} | {hardware} "
                    for bs in [1, 2, 4, 8]:
                        if bs in memory_usage:
                            row += f"| {memory_usage[bs]:.1f} "
                        else:
                            row += "| - "
                    row += "|"
                    f.write(row + "\n")
                
                f.write("\n")
            
            # Write cross-platform comparison
            f.write("\n## Cross-Platform Training Comparison\n\n")
            
            # Find models that have been benchmarked on multiple platforms
            multi_platform_models = {}
            for benchmark in self.results["benchmarks"]:
                if benchmark.get("status") != "success":
                    continue
                    
                model = benchmark["model_name"]
                platform = benchmark["hardware_platform"]
                
                if model not in multi_platform_models:
                    multi_platform_models[model] = set()
                    
                multi_platform_models[model].add(platform)
            
            # Filter to models with multiple platforms
            multi_platform_models = {
                model: platforms 
                for model, platforms in multi_platform_models.items() 
                if len(platforms) > 1
            }
            
            if multi_platform_models:
                # Create speedup table relative to CPU
                f.write("### Training Speedup Relative to CPU (Batch Size 4)\n\n")
                f.write("| Model | CUDA | ROCm | MPS | OpenVINO |\n")
                f.write("|-------|------|------|-----|----------|\n")
                
                for model in multi_platform_models:
                    model_short = model.split("/")[-1]
                    row = f"| {model_short} "
                    
                    # Find CPU benchmark for this model
                    cpu_throughput = None
                    cuda_throughput = None
                    rocm_throughput = None
                    mps_throughput = None
                    openvino_throughput = None
                    
                    for benchmark in self.results["benchmarks"]:
                        if benchmark.get("model_name") != model or benchmark.get("status") != "success":
                            continue
                            
                        platform = benchmark["hardware_platform"]
                        
                        # Get average throughput for batch size 4
                        for batch_result in benchmark.get("batch_results", []):
                            if batch_result["batch_size"] == 4:
                                # Average throughput across sequence lengths
                                throughputs = [
                                    seq_result.get("average_throughput_items_per_second", 0)
                                    for seq_result in batch_result.get("sequence_lengths", [])
                                    if "average_throughput_items_per_second" in seq_result
                                ]
                                
                                if throughputs:
                                    avg_throughput = sum(throughputs) / len(throughputs)
                                    
                                    if platform == "cpu":
                                        cpu_throughput = avg_throughput
                                    elif platform == "cuda":
                                        cuda_throughput = avg_throughput
                                    elif platform == "rocm":
                                        rocm_throughput = avg_throughput
                                    elif platform == "mps":
                                        mps_throughput = avg_throughput
                                    elif platform == "openvino":
                                        openvino_throughput = avg_throughput
                    
                    # Calculate speedups relative to CPU
                    if cpu_throughput:
                        # CUDA speedup
                        if cuda_throughput:
                            speedup = cuda_throughput / cpu_throughput
                            row += f"| {speedup:.1f}x "
                        else:
                            row += "| - "
                            
                        # ROCm speedup
                        if rocm_throughput:
                            speedup = rocm_throughput / cpu_throughput
                            row += f"| {speedup:.1f}x "
                        else:
                            row += "| - "
                            
                        # MPS speedup
                        if mps_throughput:
                            speedup = mps_throughput / cpu_throughput
                            row += f"| {speedup:.1f}x "
                        else:
                            row += "| - "
                            
                        # OpenVINO speedup
                        if openvino_throughput:
                            speedup = openvino_throughput / cpu_throughput
                            row += f"| {speedup:.1f}x "
                        else:
                            row += "| - "
                    else:
                        row += "| - | - | - | - "
                    
                    row += "|"
                    f.write(row + "\n")
            else:
                f.write("No models have been benchmarked on multiple platforms yet.\n")
            
            f.write("\n")
            
            # Write summary observations and recommendations
            f.write("\n## Overall Training Performance Observations\n\n")
            
            # Analyze hardware performance across models
            overall_speedups = {
                "cuda": [],
                "rocm": [],
                "mps": [],
                "openvino": []
            }
            
            for benchmark in self.results["benchmarks"]:
                if benchmark.get("status") != "success":
                    continue
                
                model = benchmark["model_name"]
                platform = benchmark["hardware_platform"]
                
                # Skip if this is a CPU benchmark
                if platform == "cpu":
                    continue
                
                # Find the corresponding CPU benchmark for this model
                cpu_benchmark = None
                for b in self.results["benchmarks"]:
                    if (b.get("model_name") == model and 
                        b.get("hardware_platform") == "cpu" and
                        b.get("status") == "success"):
                        cpu_benchmark = b
                        break
                
                if not cpu_benchmark:
                    continue
                
                # Calculate average speedup across batch sizes and sequence lengths
                speedups = []
                
                for batch_result in benchmark.get("batch_results", []):
                    batch_size = batch_result["batch_size"]
                    
                    # Find corresponding batch size in CPU benchmark
                    cpu_batch_result = None
                    for b in cpu_benchmark.get("batch_results", []):
                        if b["batch_size"] == batch_size:
                            cpu_batch_result = b
                            break
                    
                    if not cpu_batch_result:
                        continue
                    
                    # Calculate speedups for each sequence length
                    for seq_result in batch_result.get("sequence_lengths", []):
                        seq_length = seq_result.get("sequence_length")
                        if not seq_length or "average_throughput_items_per_second" not in seq_result:
                            continue
                        
                        # Find corresponding sequence length in CPU results
                        cpu_seq_result = None
                        for s in cpu_batch_result.get("sequence_lengths", []):
                            if s.get("sequence_length") == seq_length and "average_throughput_items_per_second" in s:
                                cpu_seq_result = s
                                break
                        
                        if not cpu_seq_result:
                            continue
                        
                        # Calculate speedup
                        gpu_throughput = seq_result["average_throughput_items_per_second"]
                        cpu_throughput = cpu_seq_result["average_throughput_items_per_second"]
                        
                        if cpu_throughput > 0:
                            speedup = gpu_throughput / cpu_throughput
                            speedups.append(speedup)
                
                if speedups:
                    avg_speedup = sum(speedups) / len(speedups)
                    overall_speedups[platform].append(avg_speedup)
            
            # Write observations based on the data
            for platform, speedups in overall_speedups.items():
                if speedups:
                    avg_speedup = sum(speedups) / len(speedups)
                    f.write(f"- **{platform.upper()}**: Average training speedup of {avg_speedup:.1f}x compared to CPU based on {len(speedups)} measurements.\n")
            
            f.write("\n")
            
            # Write general recommendations
            f.write("### General Training Recommendations\n\n")
            f.write("1. **Hardware Selection**: \n")
            f.write("   - For text generation models, CUDA GPUs provide the best training performance.\n")
            f.write("   - For embedding models, both CUDA and ROCm (AMD) GPUs offer good performance.\n")
            f.write("   - For vision models, CUDA GPUs generally offer better performance than other platforms.\n")
            f.write("   - For audio models, CUDA is the recommended hardware platform.\n\n")
            
            f.write("2. **Batch Size Optimization**: \n")
            f.write("   - Larger batch sizes generally improve throughput up to hardware memory limits.\n")
            f.write("   - For large models that exceed GPU memory, consider gradient accumulation or model parallelism.\n")
            f.write("   - Optimal batch size varies by model family and hardware platform - refer to model-specific reports.\n\n")
            
            f.write("3. **Training Configuration**: \n")
            f.write("   - Mixed precision training (FP16) can significantly improve performance on supported hardware.\n")
            f.write("   - Gradient checkpointing can reduce memory usage at the cost of recomputing activations.\n")
            f.write("   - Consider model-specific optimizations like attention caching for transformers.\n\n")
            
            f.write("4. **Memory Optimization**: \n")
            f.write("   - Monitor peak memory usage to avoid out-of-memory errors during training.\n")
            f.write("   - For very large models, consider model parallelism or sharded data parallelism.\n")
            f.write("   - Memory usage scales approximately linearly with batch size for most models.\n")
            
            # Write report location
            f.write(f"\n\nDetailed reports for each model benchmark can be found in the `{self.output_dir}` directory.\n")
    
    def _find_optimal_batch_size(self, result: Dict[str, Any]) -> int:
        """Find the optimal batch size based on throughput and memory constraints"""
        if result.get("status") != "success" or not result.get("batch_results"):
            return 1  # Default if no valid results
        
        # Get throughput for each batch size
        batch_throughputs = {}
        for batch_result in result["batch_results"]:
            batch_size = batch_result["batch_size"]
            
            # Average throughput across sequence lengths
            throughputs = []
            for seq_result in batch_result.get("sequence_lengths", []):
                if "average_throughput_items_per_second" in seq_result:
                    throughputs.append(seq_result["average_throughput_items_per_second"])
            
            if throughputs:
                batch_throughputs[batch_size] = sum(throughputs) / len(throughputs)
        
        if not batch_throughputs:
            return 1  # No valid throughput measurements
        
        # Find batch size with highest throughput
        best_batch_size = max(batch_throughputs.items(), key=lambda x: x[1])[0]
        
        # Check if memory usage is a constraint
        hardware_platform = result["hardware_platform"]
        if hardware_platform == "cuda" and "cuda_memory_stats" in result:
            memory_util = 0
            for device in result["cuda_memory_stats"].get("devices", []):
                memory_util = max(memory_util, device.get("percent_used", 0))
            
            # If memory utilization is very high, consider a smaller batch size
            if memory_util > 90 and best_batch_size > 1:
                # Find the next smaller batch size
                batch_sizes = sorted(batch_throughputs.keys())
                idx = batch_sizes.index(best_batch_size)
                if idx > 0:
                    return batch_sizes[idx - 1]
        
        return best_batch_size
        

def main():
    """Main entry point for the training benchmarking script"""
    parser = argparse.ArgumentParser(description='Run training benchmarks for IPFS Accelerate Python')
    parser.add_argument('--config', type=str, help='Path to benchmark configuration file')
    parser.add_argument('--output-dir', type=str, default='./training_benchmark_results',
                       help='Directory to store benchmark results')
    parser.add_argument('--model', type=str, help='Specific model to benchmark')
    parser.add_argument('--family', type=str, 
                       choices=['embedding', 'text_generation', 'vision', 'audio'],
                       help='Specific model family to benchmark')
    parser.add_argument('--hardware', type=str, 
                       choices=['cpu', 'cuda', 'mps', 'rocm', 'openvino', 'all'],
                       help='Hardware platform to benchmark on')
    parser.add_argument('--batch-sizes', type=str, default=None,
                       help='Comma-separated list of batch sizes to test')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = TrainingBenchmarkRunner(
        output_dir=args.output_dir,
        config_file=args.config,
        debug=args.debug
    )
    
    # Add hardware validation check
    if args.hardware:
        if args.hardware != 'all' and not runner.hardware_info.get(args.hardware.lower(), False):
            logger.error(f"Requested hardware platform {args.hardware} is not available on this system")
            return 1
    
    # Filter and customize benchmarks based on command line arguments
    
    # Filter by hardware if requested
    if args.hardware:
        if args.hardware != 'all':
            runner.config["hardware_platforms"] = [args.hardware]
        else:
            # Use all available hardware
            runner.config["hardware_platforms"] = [h for h, available in runner.hardware_info.items() if available and h != "cpu"]
            runner.config["hardware_platforms"].append("cpu")  # Always include CPU
    
    # Filter by model if requested
    if args.model:
        models = [args.model]
        
        # Filter all model families to only include the requested model
        for family in runner.config["model_families"].values():
            family_models = family.get("models", [])
            
            # Filter to only include requested models
            filtered_models = [m for m in family_models if m in models or m.split("/")[-1] in models]
            
            family["models"] = filtered_models
    
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
    
    # Parse batch sizes if provided
    if args.batch_sizes:
        try:
            batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
            runner.config["batch_sizes"] = batch_sizes
        except ValueError:
            logger.error(f"Invalid batch sizes: {args.batch_sizes}. Use comma-separated integers.")
            return 1
    
    try:
        # Run all benchmarks
        runner.run_all_benchmarks()
        
        # Save any results that have been collected
        runner._save_results()
        return 0
    except Exception as e:
        logger.error(f"Error during training benchmark run: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
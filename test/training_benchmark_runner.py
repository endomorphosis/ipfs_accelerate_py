"""
Training Benchmark Runner for the IPFS Accelerate Python Framework.

This module implements a comprehensive benchmarking system for measuring training
performance across different hardware platforms and models.
"""

import os
import json
import time
import datetime
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from benchmark_database import BenchmarkDatabase

# Add DuckDB database support
try:
    from benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("training_benchmark_runner")


class TrainingBenchmarkRunner:
    """A system for benchmarking model training performance across hardware platforms."""

    def __init__(self, 
                database_path: str = "./benchmark_results",
                models_path: str = "./models",
                config_path: Optional[str] = None):
        """
        Initialize the training benchmark runner.

        Args:
            database_path (str): Path to the directory where benchmark results are stored.
            models_path (str): Path to the directory where models are stored.
            config_path (Optional[str]): Path to the configuration file.
        """
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize benchmark database
        self.db = BenchmarkDatabase(database_path)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Detect available hardware
        self.available_hardware = self._detect_available_hardware()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path (Optional[str]): Path to the configuration file.
            
        Returns:
            Dict: Configuration dictionary.
        """
        default_config = {
            "batch_sizes": [1, 4, 16, 32, 64],
            "sequence_lengths": [32, 128, 256],
            "warmup_iterations": 5,
            "training_iterations": 50,
            "learning_rates": [1e-5, 5e-5, 1e-4],
            "optimizers": ["adam", "adamw"],
            "timeout": 600,
            "model_families": {
                "embedding": ["bert-base-uncased", "distilbert-base-uncased"],
                "text_generation": ["gpt2", "facebook/opt-125m"],
                "vision": ["google/vit-base-patch16-224", "microsoft/resnet-50"],
                "audio": ["openai/whisper-tiny", "facebook/wav2vec2-base"],
                "multimodal": ["openai/clip-vit-base-patch32"]
            },
            "hardware_types": ["cpu", "cuda", "mps", "rocm"],
            "use_resource_pool": True,
            "enable_profiling": True,
            "gradient_accumulation_steps": [1, 2, 4, 8],
            "mixed_precision": True,
            "distributed_configs": [
                {"world_size": 1, "backend": "nccl"},
                {"world_size": 2, "backend": "nccl"}
            ]
        }
        
        if config_path is None:
            return default_config
        
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} not found, using default configuration")
            return default_config
        
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
            with open(config_path, 'r') as f:
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    config = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                    config = json.load(f)

                
            return config
        
        def _detect_available_hardware(self) -> Dict[str, bool]:
            """
            Detect available hardware platforms.
            
            Returns:
                Dict[str, bool]: Dictionary mapping hardware types to availability.
            """
            available_hardware = {
                "cpu": True,  # CPU is always available
                "cuda": False,
                "mps": False,
                "rocm": False,
                "openvino": False,
                "webnn": False,
                "webgpu": False
            }
            
            # Check for CUDA (NVIDIA GPUs)
            try:
                import torch
                available_hardware["cuda"] = torch.cuda.is_available()
                if available_hardware["cuda"]:
                    logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
                    for i in range(torch.cuda.device_count()):
                        logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            except ImportError:
                logger.warning("PyTorch not available, CUDA detection skipped")
            
            # Check for MPS (Apple Metal Performance Shaders)
            try:
                import torch
                available_hardware["mps"] = (
                    hasattr(torch, 'backends') and 
                    hasattr(torch.backends, 'mps') and 
                    torch.backends.mps.is_available()
                )
                if available_hardware["mps"]:
                    logger.info("MPS available (Apple Silicon)")
            except:
                logger.warning("MPS detection failed")
            
            # Check for ROCm (AMD GPUs)
            try:
                import torch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0).lower()
                    if 'amd' in device_name or 'radeon' in device_name:
                        available_hardware["rocm"] = True
                        logger.info(f"ROCm available: {device_name}")
            except:
                logger.warning("ROCm detection failed")
            
            # Check for OpenVINO
            try:
                import openvino
                available_hardware["openvino"] = True
                logger.info("OpenVINO available")
            except ImportError:
                logger.warning("OpenVINO not available")
            
            # WebNN and WebGPU availability is determined by the environment
            # These are typically not used for training benchmarks
            
            return available_hardware
        
        def run_training_benchmark(self, 
                                 model_family: str,
                                 model_name: str,
                                 hardware_type: str,
                                 batch_sizes: Optional[List[int]] = None,
                                 sequence_lengths: Optional[List[int]] = None,
                                 learning_rates: Optional[List[float]] = None,
                                 warmup_iterations: Optional[int] = None,
                                 training_iterations: Optional[int] = None,
                                 optimizers: Optional[List[str]] = None,
                                 gradient_accumulation_steps: Optional[List[int]] = None,
                                 mixed_precision: Optional[bool] = None,
                                 distributed_config: Optional[Dict] = None,
                                 enable_profiling: Optional[bool] = None,
                                 timeout: Optional[int] = None) -> Dict:
            """
            Run training benchmark for a specific model on a specific hardware type.
            
            Args:
                model_family (str): Model family.
                model_name (str): Model name.
                hardware_type (str): Hardware type.
                batch_sizes (Optional[List[int]]): List of batch sizes to test.
                sequence_lengths (Optional[List[int]]): List of sequence lengths to test.
                learning_rates (Optional[List[float]]): List of learning rates to test.
                warmup_iterations (Optional[int]): Number of warmup iterations.
                training_iterations (Optional[int]): Number of training iterations.
                optimizers (Optional[List[str]]): List of optimizers to test.
                gradient_accumulation_steps (Optional[List[int]]): List of gradient accumulation steps to test.
                mixed_precision (Optional[bool]): Whether to use mixed precision.
                distributed_config (Optional[Dict]): Configuration for distributed training.
                enable_profiling (Optional[bool]): Whether to enable profiling.
                timeout (Optional[int]): Timeout in seconds.
                
            Returns:
                Dict: Benchmark results.
            """
            # Check if hardware is available
            if not self.available_hardware.get(hardware_type, False):
                logger.error(f"Hardware type {hardware_type} not available")
                return {
                    "status": "failed",
                    "error": f"Hardware type {hardware_type} not available"
                }
            
            # Use configuration defaults if not specified
            batch_sizes = batch_sizes or self.config["batch_sizes"]
            sequence_lengths = sequence_lengths or self.config["sequence_lengths"]
            learning_rates = learning_rates or self.config["learning_rates"]
            warmup_iterations = warmup_iterations or self.config["warmup_iterations"]
            training_iterations = training_iterations or self.config["training_iterations"]
            optimizers = optimizers or self.config["optimizers"]
            gradient_accumulation_steps = gradient_accumulation_steps or self.config["gradient_accumulation_steps"]
            mixed_precision = mixed_precision if mixed_precision is not None else self.config["mixed_precision"]
            enable_profiling = enable_profiling if enable_profiling is not None else self.config["enable_profiling"]
            timeout = timeout or self.config["timeout"]
            
            # Initialize benchmark results
            benchmark_results = {
                "status": "in_progress",
                "model_family": model_family,
                "model_name": model_name,
                "hardware_type": hardware_type,
                "start_time": datetime.datetime.now().isoformat(),
                "model_load_time": 0,
                "training_results": {},
                "performance_summary": {
                    "training_time": {},
                    "memory_usage": {},
                    "throughput": {},
                    "loss_convergence": {}
                },
                "profiling_results": {} if enable_profiling else None
            }
            
            # Set timeout handler
            start_time = time.time()
            def check_timeout():
                if time.time() - start_time > timeout:
                    logger.warning(f"Benchmark timed out after {timeout} seconds")
                    return True
                return False
            
            try:
                # Load model
                logger.info(f"Loading model {model_name} for training benchmark")
                model_load_start = time.time()
                model, tokenizer, optimizer_fn, loss_fn = self._load_model_for_training(
                    model_family, model_name, hardware_type
                )
                benchmark_results["model_load_time"] = time.time() - model_load_start
                
                # Check timeout after model loading
                if check_timeout():
                    benchmark_results["status"] = "timeout"
                    return benchmark_results
                
                # Run benchmarks for different configurations
                for batch_size in batch_sizes:
                    for seq_length in sequence_lengths:
                        for lr in learning_rates:
                            for optimizer_name in optimizers:
                                for grad_accum_steps in gradient_accumulation_steps:
                                    # Create configuration key
                                    config_key = f"batch_{batch_size}_seq_{seq_length}_lr_{lr}_opt_{optimizer_name}_accum_{grad_accum_steps}"
                                    
                                    # Log benchmark configuration
                                    logger.info(f"Running training benchmark for configuration: {config_key}")
                                    
                                    # Create synthetic dataset
                                    train_dataset = self._create_synthetic_dataset(
                                        model_family, model_name, tokenizer, batch_size, seq_length
                                    )
                                    
                                    # Initialize optimizer
                                    optimizer = optimizer_fn(model, lr, optimizer_name)
                                    
                                    # Setup mixed precision if enabled
                                    scaler = torch.cuda.amp.GradScaler() if mixed_precision and hardware_type in ["cuda", "rocm"] else None
                                    
                                    # Run benchmark
                                    result = self._run_training_iteration_benchmark(
                                        model=model,
                                        train_dataset=train_dataset,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        batch_size=batch_size,
                                        warmup_iterations=warmup_iterations,
                                        training_iterations=training_iterations,
                                        hardware_type=hardware_type,
                                        mixed_precision=mixed_precision,
                                        scaler=scaler,
                                        gradient_accumulation_steps=grad_accum_steps,
                                        enable_profiling=enable_profiling,
                                        distributed_config=distributed_config
                                    )
                                    
                                    # Store result
                                    benchmark_results["training_results"][config_key] = result
                                    
                                    # Check timeout
                                    if check_timeout():
                                        benchmark_results["status"] = "timeout"
                                        return benchmark_results
                
                # Compute performance summary
                benchmark_results["performance_summary"] = self._compute_performance_summary(
                    benchmark_results["training_results"]
                )
                
                # Mark benchmark as completed
                benchmark_results["status"] = "completed"
                benchmark_results["end_time"] = datetime.datetime.now().isoformat()
                
            except Exception as e:
                logger.exception(f"Error running training benchmark: {e}")
                benchmark_results["status"] = "failed"
                benchmark_results["error"] = str(e)
            
            return benchmark_results
        
        def _load_model_for_training(self, model_family: str, model_name: str, hardware_type: str) -> Tuple:
            """
            Load a model for training benchmark.
            
            Args:
                model_family (str): Model family.
                model_name (str): Model name.
                hardware_type (str): Hardware type.
                
            Returns:
                Tuple: (model, tokenizer, optimizer_fn, loss_fn)
            """
            device = self._get_device(hardware_type)
            
            # Load model based on family
            if model_family == "embedding":
                from transformers import AutoModel, AutoTokenizer
                model = AutoModel.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                loss_fn = torch.nn.MSELoss()
                
            elif model_family == "text_generation":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                loss_fn = None  # Model returns loss directly
                
            elif model_family == "vision":
                from transformers import AutoModelForImageClassification, AutoFeatureExtractor
                model = AutoModelForImageClassification.from_pretrained(model_name)
                tokenizer = AutoFeatureExtractor.from_pretrained(model_name)
                loss_fn = torch.nn.CrossEntropyLoss()
                
            elif model_family == "audio":
                # For audio models like Whisper or Wav2Vec2
                if "whisper" in model_name.lower():
                    from transformers import WhisperForConditionalGeneration, WhisperProcessor
                    model = WhisperForConditionalGeneration.from_pretrained(model_name)
                    tokenizer = WhisperProcessor.from_pretrained(model_name)
                    loss_fn = None  # Model returns loss directly
                else:
                    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                    model = Wav2Vec2ForCTC.from_pretrained(model_name)
                    tokenizer = Wav2Vec2Processor.from_pretrained(model_name)
                    loss_fn = None  # Model returns loss directly
                    
            elif model_family == "multimodal":
                # For multimodal models like CLIP
                if "clip" in model_name.lower():
                    from transformers import CLIPModel, CLIPProcessor
                    model = CLIPModel.from_pretrained(model_name)
                    tokenizer = CLIPProcessor.from_pretrained(model_name)
                    loss_fn = None  # Custom loss function needed
                else:
                    raise ValueError(f"Unsupported multimodal model: {model_name}")
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
            
            # Move model to device
            model = model.to(device)
            
            # Define optimizer function
            def optimizer_fn(model, learning_rate, optimizer_name):
                if optimizer_name.lower() == "adam":
                    return torch.optim.Adam(model.parameters(), lr=learning_rate)
                elif optimizer_name.lower() == "adamw":
                    return torch.optim.AdamW(model.parameters(), lr=learning_rate)
                elif optimizer_name.lower() == "sgd":
                    return torch.optim.SGD(model.parameters(), lr=learning_rate)
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
            return model, tokenizer, optimizer_fn, loss_fn
        
        def _create_synthetic_dataset(self, 
                                    model_family: str, 
                                    model_name: str, 
                                    tokenizer, 
                                    batch_size: int, 
                                    seq_length: int) -> Dict:
            """
            Create a synthetic dataset for training benchmark.
            
            Args:
                model_family (str): Model family.
                model_name (str): Model name.
                tokenizer: Model tokenizer or processor.
                batch_size (int): Batch size.
                seq_length (int): Sequence length.
                
            Returns:
                Dict: Synthetic dataset.
            """
            if model_family == "embedding" or model_family == "text_generation":
                # Create random input IDs for text models
                input_ids = torch.randint(
                    low=0, 
                    high=tokenizer.vocab_size, 
                    size=(batch_size, seq_length)
                )
                attention_mask = torch.ones_like(input_ids)
                
                # For text generation, also create labels
                if model_family == "text_generation":
                    labels = input_ids.clone()
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                else:
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }
                    
            elif model_family == "vision":
                # Create random pixel values for vision models
                pixel_values = torch.rand(batch_size, 3, 224, 224)
                labels = torch.randint(0, 1000, (batch_size,))
                return {
                    "pixel_values": pixel_values,
                    "labels": labels
                }
                
            elif model_family == "audio":
                # Create random audio features
                if "whisper" in model_name.lower():
                    # For Whisper, create inputs similar to melspectrogram
                    input_features = torch.rand(batch_size, 80, 3000)
                    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
                    return {
                        "input_features": input_features,
                        "labels": input_ids
                    }
                else:
                    # For Wav2Vec2, create inputs similar to waveform
                    input_values = torch.rand(batch_size, 16000)
                    labels = torch.randint(0, 100, (batch_size, seq_length))
                    return {
                        "input_values": input_values,
                        "labels": labels
                    }
                    
            elif model_family == "multimodal":
                # For multimodal models like CLIP
                if "clip" in model_name.lower():
                    pixel_values = torch.rand(batch_size, 3, 224, 224)
                    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
                    attention_mask = torch.ones_like(input_ids)
                    return {
                        "pixel_values": pixel_values,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "return_loss": True
                    }
                else:
                    raise ValueError(f"Unsupported multimodal model: {model_name}")
                    
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
        
        def _run_training_iteration_benchmark(self,
                                            model,
                                            train_dataset: Dict,
                                            optimizer,
                                            loss_fn,
                                            batch_size: int,
                                            warmup_iterations: int,
                                            training_iterations: int,
                                            hardware_type: str,
                                            mixed_precision: bool,
                                            scaler,
                                            gradient_accumulation_steps: int,
                                            enable_profiling: bool,
                                            distributed_config: Optional[Dict]) -> Dict:
            """
            Run training iteration benchmark.
            
            Args:
                model: Model to benchmark.
                train_dataset (Dict): Training dataset.
                optimizer: Optimizer to use.
                loss_fn: Loss function to use.
                batch_size (int): Batch size.
                warmup_iterations (int): Number of warmup iterations.
                training_iterations (int): Number of training iterations.
                hardware_type (str): Hardware type.
                mixed_precision (bool): Whether to use mixed precision.
                scaler: Gradient scaler for mixed precision.
                gradient_accumulation_steps (int): Number of gradient accumulation steps.
                enable_profiling (bool): Whether to enable profiling.
                distributed_config (Optional[Dict]): Configuration for distributed training.
                
            Returns:
                Dict: Benchmark results.
            """
            device = self._get_device(hardware_type)
            
            # Initialize distributed training if configured
            is_distributed = False
            if distributed_config is not None and hardware_type in ["cuda", "rocm"]:
                is_distributed = self._setup_distributed_training(distributed_config)
            
            # Move dataset to device
            for key, value in train_dataset.items():
                if isinstance(value, torch.Tensor):
                    train_dataset[key] = value.to(device)
            
            # Initialize profiler if enabled
            profiler = None
            if enable_profiling:
                activities = []
                
                if hardware_type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                    
                activities.append(torch.profiler.ProfilerActivity.CPU)
                    
                profiler = torch.profiler.profile(
                    activities=activities,
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=3
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling"),
                    record_shapes=True,
                    with_stack=True
                )
            
            # Initialize counters and timers
            times = []
            loss_values = []
            memory_usage = []
            
            # Perform warmup iterations
            model.train()
            for i in range(warmup_iterations):
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                if mixed_precision and hardware_type in ["cuda", "rocm"]:
                    with torch.cuda.amp.autocast():
                        if loss_fn is None:
                            # Model computes loss internally
                            outputs = model(**train_dataset)
                            loss = outputs.loss
                        else:
                            # Custom loss function
                            outputs = model(**{k: v for k, v in train_dataset.items() if k != "labels"})
                            labels = train_dataset.get("labels")
                            loss = loss_fn(outputs.logits if hasattr(outputs, "logits") else outputs, 
                                         labels)
                else:
                    if loss_fn is None:
                        # Model computes loss internally
                        outputs = model(**train_dataset)
                        loss = outputs.loss
                    else:
                        # Custom loss function
                        outputs = model(**{k: v for k, v in train_dataset.items() if k != "labels"})
                        labels = train_dataset.get("labels")
                        loss = loss_fn(outputs.logits if hasattr(outputs, "logits") else outputs, 
                                     labels)
                
                # Backward pass
                if is_distributed:
                    loss = loss.mean()  # Mean across distributed processes
                    
                # Apply gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                if mixed_precision and hardware_type in ["cuda", "rocm"]:
                    scaler.scale(loss).backward()
                    if (i + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (i + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
            
            # Start profiler if enabled
            if profiler:
                profiler.start()
            
            # Perform benchmark iterations
            model.train()
            for i in range(training_iterations):
                # Record start time
                start_time = time.time()
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                if mixed_precision and hardware_type in ["cuda", "rocm"]:
                    with torch.cuda.amp.autocast():
                        if loss_fn is None:
                            # Model computes loss internally
                            outputs = model(**train_dataset)
                            loss = outputs.loss
                        else:
                            # Custom loss function
                            outputs = model(**{k: v for k, v in train_dataset.items() if k != "labels"})
                            labels = train_dataset.get("labels")
                            loss = loss_fn(outputs.logits if hasattr(outputs, "logits") else outputs, 
                                         labels)
                else:
                    if loss_fn is None:
                        # Model computes loss internally
                        outputs = model(**train_dataset)
                        loss = outputs.loss
                    else:
                        # Custom loss function
                        outputs = model(**{k: v for k, v in train_dataset.items() if k != "labels"})
                        labels = train_dataset.get("labels")
                        loss = loss_fn(outputs.logits if hasattr(outputs, "logits") else outputs, 
                                     labels)
                
                # Backward pass
                if is_distributed:
                    loss = loss.mean()  # Mean across distributed processes
                    
                # Apply gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                if mixed_precision and hardware_type in ["cuda", "rocm"]:
                    scaler.scale(loss).backward()
                    if (i + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (i + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Record end time and loss
                end_time = time.time()
                times.append(end_time - start_time)
                loss_values.append(loss.item() * gradient_accumulation_steps)  # Multiply by grad_accum to get true loss
                
                # Record memory usage
                if hardware_type == "cuda":
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024 / 1024)  # Convert to MB
                    torch.cuda.reset_peak_memory_stats()
                else:
                    # For CPU, use psutil
                    import psutil
                    memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # Convert to MB
                
                # Step profiler if enabled
                if profiler:
                    profiler.step()
            
            # Stop profiler if enabled
            if profiler:
                profiler.stop()
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            avg_loss = np.mean(loss_values)
            min_loss = np.min(loss_values)
            max_loss = np.max(loss_values)
            std_loss = np.std(loss_values)
            
            avg_memory = np.mean(memory_usage)
            max_memory = np.max(memory_usage)
            
            # Calculate throughput (samples per second)
            throughput = batch_size / avg_time
            
            # Calculate loss convergence rate (slope of loss curve)
            loss_convergence = 0
            if len(loss_values) > 1:
                x = np.arange(len(loss_values))
                loss_convergence = np.polyfit(x, loss_values, 1)[0]
            
            # Prepare benchmark results
            benchmark_result = {
                "status": "completed",
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "training_time": {
                    "min": min_time,
                    "max": max_time,
                    "mean": avg_time,
                    "std": std_time,
                    "per_iteration": times
                },
                "loss": {
                    "min": min_loss,
                    "max": max_loss,
                    "mean": avg_loss,
                    "std": std_loss,
                    "per_iteration": loss_values
                },
                "memory_usage": {
                    "mean": avg_memory,
                    "max": max_memory,
                    "per_iteration": memory_usage
                },
                "throughput": throughput,
                "loss_convergence_rate": loss_convergence,
                "mixed_precision": mixed_precision,
                "distributed": is_distributed
            }
            
            # Add profiling results if enabled
            if profiler:
                benchmark_result["profiling_data"] = "Profiling data stored in ./profiling"
            
            return benchmark_result
        
        def _compute_performance_summary(self, training_results: Dict) -> Dict:
            """
            Compute performance summary from training results.
            
            Args:
                training_results (Dict): Training benchmark results.
                
            Returns:
                Dict: Performance summary.
            """
            # Initialize summary
            summary = {
                "training_time": {
                    "min": float('inf'),
                    "max": 0,
                    "mean": 0,
                    "by_batch_size": {},
                    "by_optimizer": {},
                    "by_mixed_precision": {}
                },
                "memory_usage": {
                    "min": float('inf'),
                    "max": 0,
                    "mean": 0,
                    "by_batch_size": {},
                    "by_optimizer": {}
                },
                "throughput": {
                    "min": float('inf'),
                    "max": 0,
                    "mean": 0,
                    "by_batch_size": {},
                    "by_optimizer": {},
                    "by_mixed_precision": {}
                },
                "loss_convergence": {
                    "min": float('inf'),
                    "max": 0,
                    "mean": 0,
                    "by_batch_size": {},
                    "by_learning_rate": {},
                    "by_optimizer": {}
                }
            }
            
            # Collect all results
            all_times = []
            all_memories = []
            all_throughputs = []
            all_convergences = []
            
            # Process each configuration
            for config_key, result in training_results.items():
                if result["status"] != "completed":
                    continue
                
                # Parse configuration key
                parts = config_key.split("_")
                batch_size = int(parts[1])
                learning_rate = float(parts[5])
                optimizer = parts[7]
                mixed_precision = result.get("mixed_precision", False)
                
                # Extract metrics
                avg_time = result["training_time"]["mean"]
                avg_memory = result["memory_usage"]["mean"]
                throughput = result["throughput"]
                convergence = result["loss_convergence_rate"]
                
                # Update overall min/max
                summary["training_time"]["min"] = min(summary["training_time"]["min"], avg_time)
                summary["training_time"]["max"] = max(summary["training_time"]["max"], avg_time)
                
                summary["memory_usage"]["min"] = min(summary["memory_usage"]["min"], avg_memory)
                summary["memory_usage"]["max"] = max(summary["memory_usage"]["max"], avg_memory)
                
                summary["throughput"]["min"] = min(summary["throughput"]["min"], throughput)
                summary["throughput"]["max"] = max(summary["throughput"]["max"], throughput)
                
                summary["loss_convergence"]["min"] = min(summary["loss_convergence"]["min"], convergence)
                summary["loss_convergence"]["max"] = max(summary["loss_convergence"]["max"], convergence)
                
                # Collect for computing means
                all_times.append(avg_time)
                all_memories.append(avg_memory)
                all_throughputs.append(throughput)
                all_convergences.append(convergence)
                
                # Group by batch size
                if batch_size not in summary["training_time"]["by_batch_size"]:
                    summary["training_time"]["by_batch_size"][batch_size] = []
                    summary["memory_usage"]["by_batch_size"][batch_size] = []
                    summary["throughput"]["by_batch_size"][batch_size] = []
                    summary["loss_convergence"]["by_batch_size"][batch_size] = []
                
                summary["training_time"]["by_batch_size"][batch_size].append(avg_time)
                summary["memory_usage"]["by_batch_size"][batch_size].append(avg_memory)
                summary["throughput"]["by_batch_size"][batch_size].append(throughput)
                summary["loss_convergence"]["by_batch_size"][batch_size].append(convergence)
                
                # Group by optimizer
                if optimizer not in summary["training_time"]["by_optimizer"]:
                    summary["training_time"]["by_optimizer"][optimizer] = []
                    summary["memory_usage"]["by_optimizer"][optimizer] = []
                    summary["throughput"]["by_optimizer"][optimizer] = []
                    summary["loss_convergence"]["by_optimizer"][optimizer] = []
                
                summary["training_time"]["by_optimizer"][optimizer].append(avg_time)
                summary["memory_usage"]["by_optimizer"][optimizer].append(avg_memory)
                summary["throughput"]["by_optimizer"][optimizer].append(throughput)
                summary["loss_convergence"]["by_optimizer"][optimizer].append(convergence)
                
                # Group by mixed precision
                mp_key = "mixed_precision" if mixed_precision else "full_precision"
                if mp_key not in summary["training_time"]["by_mixed_precision"]:
                    summary["training_time"]["by_mixed_precision"][mp_key] = []
                    summary["throughput"]["by_mixed_precision"][mp_key] = []
                
                summary["training_time"]["by_mixed_precision"][mp_key].append(avg_time)
                summary["throughput"]["by_mixed_precision"][mp_key].append(throughput)
                
                # Group by learning rate for convergence
                if learning_rate not in summary["loss_convergence"]["by_learning_rate"]:
                    summary["loss_convergence"]["by_learning_rate"][learning_rate] = []
                
                summary["loss_convergence"]["by_learning_rate"][learning_rate].append(convergence)
            
            # Compute means
            if all_times:
                summary["training_time"]["mean"] = np.mean(all_times)
            else:
                summary["training_time"]["min"] = 0
                
            if all_memories:
                summary["memory_usage"]["mean"] = np.mean(all_memories)
            else:
                summary["memory_usage"]["min"] = 0
                
            if all_throughputs:
                summary["throughput"]["mean"] = np.mean(all_throughputs)
            else:
                summary["throughput"]["min"] = 0
                
            if all_convergences:
                summary["loss_convergence"]["mean"] = np.mean(all_convergences)
            else:
                summary["loss_convergence"]["min"] = 0
            
            # Compute means for grouped data
            for metric in ["training_time", "memory_usage", "throughput", "loss_convergence"]:
                for group_key in ["by_batch_size", "by_optimizer"]:
                    if group_key in summary[metric]:
                        for key, values in summary[metric][group_key].items():
                            summary[metric][group_key][key] = np.mean(values)
                
                if metric in ["training_time", "throughput"]:
                    for key, values in summary[metric]["by_mixed_precision"].items():
                        summary[metric]["by_mixed_precision"][key] = np.mean(values)
                        
                if metric == "loss_convergence":
                    for key, values in summary[metric]["by_learning_rate"].items():
                        summary[metric]["by_learning_rate"][key] = np.mean(values)
            
            return summary
        
        def _get_device(self, hardware_type: str) -> torch.device:
            """
            Get the appropriate device for the hardware type.
            
            Args:
                hardware_type (str): Hardware type.
                
            Returns:
                torch.device: Device to use.
            """
            if hardware_type == "cuda":
                return torch.device("cuda")
            elif hardware_type == "mps":
                return torch.device("mps")
            elif hardware_type == "rocm":
                return torch.device("cuda")  # ROCm uses CUDA device in PyTorch
            else:
                return torch.device("cpu")
        
        def _setup_distributed_training(self, distributed_config: Dict) -> bool:
            """
            Setup distributed training.
            
            Args:
                distributed_config (Dict): Configuration for distributed training.
                
            Returns:
                bool: Whether distributed training is enabled.
            """
            try:
                import torch.distributed as dist
                
                world_size = distributed_config.get("world_size", 1)
                rank = distributed_config.get("rank", 0)
                backend = distributed_config.get("backend", "nccl")
                
                # Check if distributed is already initialized
                if dist.is_initialized():
                    return True
                
                # Skip if world_size is 1 (single process)
                if world_size <= 1:
                    return False
                
                # Initialize process group
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    world_size=world_size,
                    rank=rank
                )
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to setup distributed training: {e}")
                return False
        
        def run_benchmarks(self, 
                          model_families: Optional[List[str]] = None, 
                          hardware_types: Optional[List[str]] = None,
                          output_path: Optional[str] = None) -> Dict:
            """
            Run benchmarks for multiple models on multiple hardware types.
            
            Args:
                model_families (Optional[List[str]]): List of model families to benchmark.
                hardware_types (Optional[List[str]]): List of hardware types to benchmark.
                output_path (Optional[str]): Path to save benchmark results.
                
            Returns:
                Dict: Benchmark results.
            """
            # Use configuration defaults if not specified
            model_families = model_families or list(self.config["model_families"].keys())
            hardware_types = hardware_types or self.config["hardware_types"]
            
            # Filter to available hardware
            hardware_types = [hw for hw in hardware_types if self.available_hardware.get(hw, False)]
            
            # Initialize results
            all_results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "system_info": self._get_system_info(),
                "benchmarks": {}
            }
            
            # Run benchmarks for each model family and hardware type
            for model_family in model_families:
                logger.info(f"Running benchmarks for model family: {model_family}")
                
                # Get models for this family
                models = self.config["model_families"].get(model_family, [])
                if not models:
                    logger.warning(f"No models configured for family: {model_family}")
                    continue
                
                all_results["benchmarks"][model_family] = {}
                
                for model_name in models:
                    logger.info(f"Running benchmarks for model: {model_name}")
                    
                    all_results["benchmarks"][model_family][model_name] = {}
                    
                    for hardware_type in hardware_types:
                        logger.info(f"Running benchmark on hardware: {hardware_type}")
                        
                        # Run benchmark
                        result = self.run_training_benchmark(
                            model_family=model_family,
                            model_name=model_name,
                            hardware_type=hardware_type
                        )
                        
                        # Store result
                        all_results["benchmarks"][model_family][model_name][hardware_type] = result
            
            # Save results
            if output_path:
                output_path = Path(output_path)
                output_path.mkdir(exist_ok=True, parents=True)
                output_file = output_path / f"training_benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

            
            logger.info(f"Benchmark results saved to: {output_file}")
        
        # Store in database
        run_id = self.db.store_benchmark_results(all_results)
        logger.info(f"Benchmark results stored in database with run ID: {run_id}")
        
        return all_results
    
    def _get_system_info(self) -> Dict:
        """
        Get system information.
        
        Returns:
            Dict: System information.
        """
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
            "available_hardware": self.available_hardware
        }
        
        # Add CUDA information if available
        if self.available_hardware.get("cuda", False):
            import torch
            system_info["cuda_version"] = torch.version.cuda
            system_info["cuda_device_count"] = torch.cuda.device_count()
            system_info["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                }
                for i in range(torch.cuda.device_count())
            ]
        
        return system_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Benchmark Runner")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Directory to store benchmark results")
    parser.add_argument("--model-families", type=str, nargs="+", choices=["embedding", "text_generation", "vision", "audio", "multimodal"], help="Model families to benchmark")
    parser.add_argument("--hardware", type=str, nargs="+", choices=["cpu", "cuda", "mps", "rocm", "openvino"], help="Hardware types to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", help="Batch sizes to test")
    parser.add_argument("--warmup", type=int, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, help="Number of benchmark iterations")
    parser.add_argument("--learning-rates", type=float, nargs="+", help="Learning rates to test")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--profiling", action="store_true", help="Enable profiling")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger("training_benchmark_runner").setLevel(logging.DEBUG)
    
    # Create benchmark runner
    runner = TrainingBenchmarkRunner(
        database_path=args.output_dir,
        config_path=args.config
    )
    
    # Override configuration with command line arguments
    config_overrides = {}
    
    if args.batch_sizes:
        config_overrides["batch_sizes"] = args.batch_sizes
        
    if args.warmup:
        config_overrides["warmup_iterations"] = args.warmup
        
    if args.iterations:
        config_overrides["training_iterations"] = args.iterations
        
    if args.learning_rates:
        config_overrides["learning_rates"] = args.learning_rates
        
    if args.mixed_precision:
        config_overrides["mixed_precision"] = True
        
    if args.profiling:
        config_overrides["enable_profiling"] = True
    
    # Run benchmarks
    results = runner.run_benchmarks(
        model_families=args.model_families,
        hardware_types=args.hardware,
        output_path=args.output_dir
    )
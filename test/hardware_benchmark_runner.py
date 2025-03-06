#!/usr/bin/env python
# Automated hardware benchmark runner for IPFS Accelerate framework

import os
import sys
import time
import json
import argparse
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import concurrent.futures
import platform

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

# Try to import framework components with graceful degradation
try:
    from hardware_detection import detect_hardware_with_comprehensive_checks, CPU, CUDA, MPS, ROCM, OPENVINO, WEBNN, WEBGPU
    from model_family_classifier import classify_model, ModelFamilyClassifier
    from resource_pool import get_global_resource_pool
    HAS_ALL_COMPONENTS = True
except ImportError as e:
    logger.warning(f"Could not import all components: {e}. Some functionality may be limited.")
    HAS_ALL_COMPONENTS = False

# Standard benchmark model definitions
BENCHMARK_MODELS = {
    "embedding": [
        {"name": "bert-base-uncased", "size": "base", "class": "BertModel"},
        {"name": "distilbert-base-uncased", "size": "small", "class": "DistilBertModel"},
        {"name": "roberta-base", "size": "base", "class": "RobertaModel"}
    ],
    "text_generation": [
        {"name": "gpt2", "size": "base", "class": "GPT2LMHeadModel"},
        {"name": "t5-small", "size": "small", "class": "T5ForConditionalGeneration"},
        {"name": "google/flan-t5-small", "size": "small", "class": "T5ForConditionalGeneration"}
    ],
    "vision": [
        {"name": "google/vit-base-patch16-224", "size": "base", "class": "ViTForImageClassification"},
        {"name": "microsoft/resnet-50", "size": "base", "class": "ResNetForImageClassification"},
        {"name": "facebook/convnext-tiny-224", "size": "small", "class": "ConvNextForImageClassification"}
    ],
    "audio": [
        {"name": "openai/whisper-tiny", "size": "small", "class": "WhisperForConditionalGeneration"},
        {"name": "facebook/wav2vec2-base", "size": "base", "class": "Wav2Vec2ForCTC"}
    ],
    "multimodal": [
        {"name": "openai/clip-vit-base-patch32", "size": "base", "class": "CLIPModel"}
    ]
}

# Benchmark parameters
DEFAULT_BATCH_SIZES = [1, 4, 8]
DEFAULT_SEQUENCE_LENGTHS = [32, 128, 512]
DEFAULT_IMAGE_SIZES = [(224, 224), (384, 384)]
DEFAULT_AUDIO_LENGTHS = [5, 10, 30]  # seconds
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_BENCHMARK_ITERATIONS = 20
DEFAULT_TIMEOUT = 600  # seconds

class HardwareBenchmarkRunner:
    """
    Automated benchmark runner for hardware performance testing across different models and hardware.
    
    This class provides comprehensive benchmarking capabilities for testing model performance
    across different hardware platforms, with detailed metrics collection and analysis.
    """
    
    def __init__(self, 
                 output_dir: str = "./benchmark_results",
                 models_config: Optional[Dict[str, List[Dict[str, str]]]] = None,
                 detect_hardware: bool = True,
                 include_web_platforms: bool = False,
                 use_resource_pool: bool = True):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            models_config: Optional custom model configuration (defaults to BENCHMARK_MODELS)
            detect_hardware: Whether to detect available hardware automatically
            include_web_platforms: Whether to include WebNN and WebGPU in benchmarks
            use_resource_pool: Whether to use the ResourcePool for model caching
        """
        self.output_dir = output_dir
        self.models_config = models_config or BENCHMARK_MODELS
        self.use_resource_pool = use_resource_pool
        self.include_web_platforms = include_web_platforms
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect available hardware if requested
        self.available_hardware = {}
        if detect_hardware:
            self._detect_hardware()
        
        # Initialize ResourcePool if available and requested
        self.resource_pool = None
        if use_resource_pool and 'resource_pool' in sys.modules:
            self.resource_pool = get_global_resource_pool()
        
        # Track benchmark results
        self.benchmark_results = {}
    
    def _detect_hardware(self) -> Dict[str, bool]:
        """Detect available hardware platforms"""
        try:
            if 'hardware_detection' in sys.modules:
                # Use comprehensive detection if available
                hardware_info = detect_hardware_with_comprehensive_checks()
                
                # Extract available hardware platforms
                for hw_type in [CPU, CUDA, MPS, ROCM, OPENVINO]:
                    self.available_hardware[hw_type] = hardware_info.get(hw_type, False)
                
                # Include web platforms if requested
                if self.include_web_platforms:
                    for web_hw in [WEBNN, WEBGPU]:
                        self.available_hardware[web_hw] = hardware_info.get(web_hw, False)
                
                # Get detailed hardware information
                self.hardware_details = hardware_info
                logger.info(f"Detected hardware: {', '.join(k for k, v in self.available_hardware.items() if v)}")
                
                return self.available_hardware
            else:
                # Fallback to basic detection
                self._basic_hardware_detection()
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
    
    def run_benchmarks(self, 
                       model_families: Optional[List[str]] = None,
                       hardware_types: Optional[List[str]] = None,
                       batch_sizes: Optional[List[int]] = None,
                       sequence_lengths: Optional[List[int]] = None,
                       warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
                       benchmark_iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
                       timeout: int = DEFAULT_TIMEOUT,
                       parallel: bool = False) -> Dict[str, Any]:
        """
        Run benchmarks across specified models and hardware platforms.
        
        Args:
            model_families: List of model families to benchmark (e.g., "embedding", "text_generation")
            hardware_types: List of hardware types to benchmark (e.g., "cuda", "cpu")
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test for text models
            warmup_iterations: Number of warmup iterations before timing
            benchmark_iterations: Number of iterations to time
            timeout: Maximum time in seconds for a benchmark
            parallel: Whether to run benchmarks in parallel across hardware types
            
        Returns:
            Dictionary with benchmark results
        """
        # Use defaults if not specified
        model_families = model_families or list(self.models_config.keys())
        hardware_types = hardware_types or [k for k, v in self.available_hardware.items() if v]
        batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
        sequence_lengths = sequence_lengths or DEFAULT_SEQUENCE_LENGTHS
        
        logger.info(f"Starting benchmarks for model families: {model_families}")
        logger.info(f"Hardware platforms: {hardware_types}")
        logger.info(f"Batch sizes: {batch_sizes}")
        
        # Initialize results dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.benchmark_results = {
            "timestamp": timestamp,
            "system_info": self._get_system_info(),
            "hardware_info": self.available_hardware,
            "benchmarks": {}
        }
        
        # Add hardware details if available
        if hasattr(self, 'hardware_details'):
            self.benchmark_results["hardware_details"] = self.hardware_details
        
        # Run benchmarks for each model family, hardware type, and model
        for family in model_families:
            if family not in self.models_config:
                logger.warning(f"Model family {family} not found in configuration")
                continue
                
            self.benchmark_results["benchmarks"][family] = {}
            family_models = self.models_config[family]
            
            for model_config in family_models:
                model_name = model_config["name"]
                model_class = model_config.get("class")
                
                self.benchmark_results["benchmarks"][family][model_name] = {}
                
                # Run benchmark on each hardware type
                if parallel and len(hardware_types) > 1:
                    # Parallel execution across hardware types
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = {}
                        for hw_type in hardware_types:
                            if not self.available_hardware.get(hw_type, False):
                                logger.warning(f"Hardware {hw_type} not available, skipping")
                                continue
                                
                            # Submit benchmark task
                            future = executor.submit(
                                self._benchmark_model_on_hardware,
                                family, model_name, model_class, hw_type,
                                batch_sizes, sequence_lengths, 
                                warmup_iterations, benchmark_iterations, timeout
                            )
                            futures[future] = hw_type
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(futures):
                            hw_type = futures[future]
                            try:
                                result = future.result()
                                self.benchmark_results["benchmarks"][family][model_name][hw_type] = result
                            except Exception as e:
                                logger.error(f"Error benchmarking {model_name} on {hw_type}: {e}")
                                self.benchmark_results["benchmarks"][family][model_name][hw_type] = {
                                    "error": str(e),
                                    "status": "failed"
                                }
                else:
                    # Sequential execution
                    for hw_type in hardware_types:
                        if not self.available_hardware.get(hw_type, False):
                            logger.warning(f"Hardware {hw_type} not available, skipping")
                            continue
                        
                        try:
                            result = self._benchmark_model_on_hardware(
                                family, model_name, model_class, hw_type,
                                batch_sizes, sequence_lengths, 
                                warmup_iterations, benchmark_iterations, timeout
                            )
                            self.benchmark_results["benchmarks"][family][model_name][hw_type] = result
                        except Exception as e:
                            logger.error(f"Error benchmarking {model_name} on {hw_type}: {e}")
                            self.benchmark_results["benchmarks"][family][model_name][hw_type] = {
                                "error": str(e),
                                "status": "failed"
                            }
        
        # Save results
        self._save_results(timestamp)
        
        # Update compatibility matrix if possible
        self._update_compatibility_matrix()
        
        return self.benchmark_results
    
    def _benchmark_model_on_hardware(self, 
                                    family: str, 
                                    model_name: str, 
                                    model_class: Optional[str],
                                    hardware_type: str,
                                    batch_sizes: List[int],
                                    sequence_lengths: List[int],
                                    warmup_iterations: int,
                                    benchmark_iterations: int,
                                    timeout: int) -> Dict[str, Any]:
        """Run benchmark for a specific model on a specific hardware type"""
        logger.info(f"Benchmarking {model_name} on {hardware_type}")
        
        # Verify hardware is available
        if not self.available_hardware.get(hardware_type, False):
            raise ValueError(f"Hardware {hardware_type} not available")
        
        # Prepare result dictionary
        result = {
            "status": "pending",
            "model_name": model_name,
            "model_class": model_class,
            "family": family,
            "hardware": hardware_type,
            "benchmark_results": {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Load model using ResourcePool if available
        model = None
        model_load_time_start = time.time()
        try:
            model = self._load_model(family, model_name, model_class, hardware_type)
            model_load_time = time.time() - model_load_time_start
            result["model_load_time"] = model_load_time
            result["status"] = "model_loaded"
            logger.info(f"Model {model_name} loaded in {model_load_time:.2f}s on {hardware_type}")
        except Exception as e:
            logger.error(f"Error loading model {model_name} on {hardware_type}: {e}")
            result["status"] = "load_failed"
            result["error"] = str(e)
            return result
        
        if model is None:
            result["status"] = "load_failed"
            result["error"] = "Model loading returned None"
            return result
        
        # Run benchmarks for different configurations based on model family
        try:
            if family in ["embedding", "text_generation"]:
                # Text-based models
                result["benchmark_results"] = self._benchmark_text_model(
                    model, family, hardware_type, batch_sizes, sequence_lengths,
                    warmup_iterations, benchmark_iterations, timeout
                )
            elif family == "vision":
                # Vision models
                result["benchmark_results"] = self._benchmark_vision_model(
                    model, hardware_type, batch_sizes, DEFAULT_IMAGE_SIZES,
                    warmup_iterations, benchmark_iterations, timeout
                )
            elif family == "audio":
                # Audio models
                result["benchmark_results"] = self._benchmark_audio_model(
                    model, hardware_type, batch_sizes, DEFAULT_AUDIO_LENGTHS,
                    warmup_iterations, benchmark_iterations, timeout
                )
            elif family == "multimodal":
                # Multimodal models
                result["benchmark_results"] = self._benchmark_multimodal_model(
                    model, hardware_type, batch_sizes,
                    warmup_iterations, benchmark_iterations, timeout
                )
            else:
                logger.warning(f"Benchmark method not implemented for family {family}")
                result["status"] = "not_implemented"
                return result
                
            result["status"] = "completed"
        except Exception as e:
            logger.error(f"Error during benchmark of {model_name} on {hardware_type}: {e}")
            result["status"] = "benchmark_failed"
            result["error"] = str(e)
        
        # Calculate derived metrics and statistics
        if result["status"] == "completed" and "benchmark_results" in result:
            result["performance_summary"] = self._calculate_performance_metrics(result["benchmark_results"])
        
        # Clean up
        self._cleanup_model(model, hardware_type)
        
        return result
    
    def _load_model(self, family: str, model_name: str, model_class: Optional[str], hardware_type: str):
        """Load model with appropriate hardware configuration"""
        # Use ResourcePool if available and requested
        if self.resource_pool and self.use_resource_pool:
            logger.info(f"Loading {model_name} using ResourcePool")
            
            # Define model constructor
            def create_model():
                try:
                    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
                    from transformers import AutoModelForImageClassification, AutoModelForAudioClassification
                    
                    if family == "embedding":
                        return AutoModel.from_pretrained(model_name)
                    elif family == "text_generation":
                        if "t5" in model_name.lower():
                            return AutoModelForSeq2SeqLM.from_pretrained(model_name)
                        else:
                            return AutoModelForCausalLM.from_pretrained(model_name)
                    elif family == "vision":
                        return AutoModelForImageClassification.from_pretrained(model_name)
                    elif family == "audio":
                        if "whisper" in model_name.lower():
                            from transformers import WhisperForConditionalGeneration
                            return WhisperForConditionalGeneration.from_pretrained(model_name)
                        return AutoModelForAudioClassification.from_pretrained(model_name)
                    elif family == "multimodal":
                        if "clip" in model_name.lower():
                            from transformers import CLIPModel
                            return CLIPModel.from_pretrained(model_name)
                        else:
                            # Default to AutoModel for other multimodal models
                            return AutoModel.from_pretrained(model_name)
                    else:
                        logger.warning(f"Unsupported model family: {family}")
                        return None
                except Exception as e:
                    logger.error(f"Error in model constructor for {model_name}: {e}")
                    raise
            
            # Define hardware preferences
            hardware_preferences = {"device": hardware_type}
            
            # Load model using ResourcePool
            return self.resource_pool.get_model(
                model_type=family,
                model_name=model_name,
                constructor=create_model,
                hardware_preferences=hardware_preferences
            )
        else:
            # Manual loading
            logger.info(f"Loading {model_name} manually (ResourcePool not used)")
            
            try:
                import torch
                from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
                from transformers import AutoModelForImageClassification, AutoModelForAudioClassification
                
                # Map hardware_type to torch device
                device = "cpu"
                if hardware_type == "cuda":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                elif hardware_type == "mps":
                    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
                
                # Load appropriate model type
                if family == "embedding":
                    model = AutoModel.from_pretrained(model_name)
                elif family == "text_generation":
                    if "t5" in model_name.lower():
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_name)
                elif family == "vision":
                    model = AutoModelForImageClassification.from_pretrained(model_name)
                elif family == "audio":
                    if "whisper" in model_name.lower():
                        from transformers import WhisperForConditionalGeneration
                        model = WhisperForConditionalGeneration.from_pretrained(model_name)
                    else:
                        model = AutoModelForAudioClassification.from_pretrained(model_name)
                elif family == "multimodal":
                    if "clip" in model_name.lower():
                        from transformers import CLIPModel
                        model = CLIPModel.from_pretrained(model_name)
                    else:
                        # Default to AutoModel for other multimodal models
                        model = AutoModel.from_pretrained(model_name)
                else:
                    logger.warning(f"Unsupported model family: {family}")
                    return None
                
                # Move model to device
                model = model.to(device)
                return model
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                raise
    
    def _cleanup_model(self, model, hardware_type):
        """Clean up model resources"""
        if not self.use_resource_pool:
            # Only manual cleanup needed when ResourcePool not used
            try:
                import torch
                
                # Move model to CPU to free GPU memory
                if hardware_type in ["cuda", "mps"] and hasattr(model, "to"):
                    model.to("cpu")
                
                # Clear CUDA cache if applicable
                if hardware_type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
            except Exception as e:
                logger.error(f"Error during model cleanup: {e}")
    
    def _benchmark_text_model(self, model, family, hardware_type, batch_sizes, sequence_lengths, 
                             warmup_iterations, benchmark_iterations, timeout):
        """Benchmark text-based models (embedding, text generation)"""
        import torch
        
        # Get model device
        device = next(model.parameters()).device
        
        # Create tokenizer for input generation
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        except Exception as e:
            logger.error(f"Error creating tokenizer: {e}")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Fallback
        
        benchmark_results = {}
        
        # Test different combinations of batch size and sequence length
        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                key = f"batch_{batch_size}_seq_{seq_length}"
                benchmark_results[key] = {"status": "pending"}
                
                # Create input data
                try:
                    # Generate different input text based on model family
                    if family == "embedding":
                        # For embedding models, use random text
                        text = ["Hello world! This is a benchmark test."] * batch_size
                        inputs = tokenizer(text, padding="max_length", truncation=True, 
                                         max_length=seq_length, return_tensors="pt")
                    else:
                        # For text generation models, use appropriate prompts
                        text = ["Translate the following to French: Hello world!"] * batch_size
                        inputs = tokenizer(text, padding="max_length", truncation=True, 
                                         max_length=seq_length, return_tensors="pt")
                        
                        # Add generation-specific parameters for text generation models
                        if hasattr(model, "generate"):
                            generate_params = {
                                "max_length": seq_length + 20,
                                "min_length": seq_length + 5,
                                "num_beams": 2
                            }
                except Exception as e:
                    logger.error(f"Error creating input data: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Input creation failed: {str(e)}"
                    continue
                
                # Move inputs to the appropriate device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Warmup
                try:
                    logger.info(f"Warming up model with {warmup_iterations} iterations")
                    model.eval()
                    
                    with torch.no_grad():
                        for _ in range(warmup_iterations):
                            if family == "embedding":
                                _ = model(**inputs)
                            elif hasattr(model, "generate"):
                                _ = model.generate(inputs.input_ids, **generate_params)
                            else:
                                _ = model(**inputs)
                except Exception as e:
                    logger.error(f"Error during warmup: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Warmup failed: {str(e)}"
                    continue
                
                # Benchmark
                try:
                    logger.info(f"Benchmarking with batch_size={batch_size}, seq_length={seq_length}")
                    model.eval()
                    
                    # Record timing information
                    latencies = []
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for i in range(benchmark_iterations):
                            iter_start = time.time()
                            
                            if family == "embedding":
                                outputs = model(**inputs)
                            elif hasattr(model, "generate"):
                                outputs = model.generate(inputs.input_ids, **generate_params)
                            else:
                                outputs = model(**inputs)
                            
                            # Ensure operation is complete, especially important for GPU
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            
                            iter_end = time.time()
                            latencies.append(iter_end - iter_start)
                            
                            # Check timeout
                            if time.time() - start_time > timeout:
                                logger.warning(f"Timeout after {i+1} iterations")
                                benchmark_results[key]["timeout"] = True
                                benchmark_results[key]["completed_iterations"] = i+1
                                break
                    
                    # Calculate statistics
                    total_time = time.time() - start_time
                    avg_latency = sum(latencies) / len(latencies)
                    throughput = batch_size * len(latencies) / total_time
                    
                    # For text generation, calculate tokens per second
                    if family == "text_generation" and hasattr(model, "generate"):
                        # Approximate tokens per second based on input and output lengths
                        avg_output_length = sum(len(output) for output in outputs) / len(outputs)
                        tokens_per_second = throughput * (seq_length + avg_output_length)
                        benchmark_results[key]["tokens_per_second"] = tokens_per_second
                        benchmark_results[key]["avg_output_length"] = float(avg_output_length)
                    
                    # Save results
                    benchmark_results[key]["status"] = "completed"
                    benchmark_results[key]["latencies"] = latencies
                    benchmark_results[key]["avg_latency"] = avg_latency
                    benchmark_results[key]["throughput"] = throughput
                    benchmark_results[key]["total_time"] = total_time
                    benchmark_results[key]["iterations"] = len(latencies)
                    
                    # Additional memory stats if available
                    if device.type == "cuda" and torch.cuda.is_available():
                        benchmark_results[key]["max_memory_allocated"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
                        benchmark_results[key]["max_memory_reserved"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)  # MB
                        torch.cuda.reset_peak_memory_stats(device)
                
                except Exception as e:
                    logger.error(f"Error during benchmark: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Benchmark failed: {str(e)}"
        
        return benchmark_results
    
    def _benchmark_vision_model(self, model, hardware_type, batch_sizes, image_sizes, 
                              warmup_iterations, benchmark_iterations, timeout):
        """Benchmark vision models"""
        import torch
        import numpy as np
        
        # Get model device
        device = next(model.parameters()).device
        
        # Initialize vision processor
        try:
            from transformers import AutoFeatureExtractor, AutoImageProcessor
            
            # Try different processor types based on what's available
            try:
                processor = AutoImageProcessor.from_pretrained(model.config._name_or_path)
            except:
                try:
                    processor = AutoFeatureExtractor.from_pretrained(model.config._name_or_path)
                except:
                    # Fallback to a known working processor
                    processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        except Exception as e:
            logger.error(f"Error creating vision processor: {e}")
            return {"status": "failed", "error": f"Processor creation failed: {str(e)}"}
        
        benchmark_results = {}
        
        # Test different combinations of batch size and image size
        for batch_size in batch_sizes:
            for img_size in image_sizes:
                key = f"batch_{batch_size}_img_{img_size[0]}x{img_size[1]}"
                benchmark_results[key] = {"status": "pending"}
                
                # Create input data
                try:
                    # Create random images
                    images = []
                    for _ in range(batch_size):
                        # Create a random RGB image
                        img = np.random.randint(0, 256, (img_size[0], img_size[1], 3), dtype=np.uint8)
                        images.append(img)
                    
                    # Process images
                    inputs = processor(images=images, return_tensors="pt")
                except Exception as e:
                    logger.error(f"Error creating input data: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Input creation failed: {str(e)}"
                    continue
                
                # Move inputs to the appropriate device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Warmup
                try:
                    logger.info(f"Warming up model with {warmup_iterations} iterations")
                    model.eval()
                    
                    with torch.no_grad():
                        for _ in range(warmup_iterations):
                            _ = model(**inputs)
                except Exception as e:
                    logger.error(f"Error during warmup: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Warmup failed: {str(e)}"
                    continue
                
                # Benchmark
                try:
                    logger.info(f"Benchmarking with batch_size={batch_size}, image_size={img_size}")
                    model.eval()
                    
                    # Record timing information
                    latencies = []
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for i in range(benchmark_iterations):
                            iter_start = time.time()
                            
                            outputs = model(**inputs)
                            
                            # Ensure operation is complete, especially important for GPU
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            
                            iter_end = time.time()
                            latencies.append(iter_end - iter_start)
                            
                            # Check timeout
                            if time.time() - start_time > timeout:
                                logger.warning(f"Timeout after {i+1} iterations")
                                benchmark_results[key]["timeout"] = True
                                benchmark_results[key]["completed_iterations"] = i+1
                                break
                    
                    # Calculate statistics
                    total_time = time.time() - start_time
                    avg_latency = sum(latencies) / len(latencies)
                    throughput = batch_size * len(latencies) / total_time
                    
                    # Save results
                    benchmark_results[key]["status"] = "completed"
                    benchmark_results[key]["latencies"] = latencies
                    benchmark_results[key]["avg_latency"] = avg_latency
                    benchmark_results[key]["throughput"] = throughput
                    benchmark_results[key]["total_time"] = total_time
                    benchmark_results[key]["iterations"] = len(latencies)
                    
                    # Additional memory stats if available
                    if device.type == "cuda" and torch.cuda.is_available():
                        benchmark_results[key]["max_memory_allocated"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
                        benchmark_results[key]["max_memory_reserved"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)  # MB
                        torch.cuda.reset_peak_memory_stats(device)
                
                except Exception as e:
                    logger.error(f"Error during benchmark: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Benchmark failed: {str(e)}"
        
        return benchmark_results
    
    def _benchmark_audio_model(self, model, hardware_type, batch_sizes, audio_lengths, 
                              warmup_iterations, benchmark_iterations, timeout):
        """Benchmark audio models"""
        import torch
        import numpy as np
        
        # Get model device
        device = next(model.parameters()).device
        
        # Check model type for processor selection
        model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else "unknown"
        is_whisper = "whisper" in model_name.lower()
        is_wav2vec = "wav2vec" in model_name.lower()
        
        # Initialize audio processor
        try:
            if is_whisper:
                from transformers import WhisperProcessor
                processor = WhisperProcessor.from_pretrained(model_name)
            elif is_wav2vec:
                from transformers import Wav2Vec2Processor
                processor = Wav2Vec2Processor.from_pretrained(model_name)
            else:
                # Generic audio processor fallback
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error creating audio processor: {e}")
            return {"status": "failed", "error": f"Processor creation failed: {str(e)}"}
        
        benchmark_results = {}
        
        # Test different combinations of batch size and audio length
        for batch_size in batch_sizes:
            for audio_length in audio_lengths:
                key = f"batch_{batch_size}_audio_{audio_length}s"
                benchmark_results[key] = {"status": "pending"}
                
                # Create input data
                try:
                    # Create random audio samples (16 kHz is standard for many models)
                    sample_rate = 16000
                    num_samples = sample_rate * audio_length
                    
                    # Generate batch of random audio
                    audio_samples = []
                    for _ in range(batch_size):
                        # Create random audio with values between -1 and 1
                        audio = np.random.uniform(-0.5, 0.5, num_samples).astype(np.float32)
                        audio_samples.append(audio)
                    
                    # Process audio
                    if is_whisper:
                        inputs = processor(audio_samples, sampling_rate=sample_rate, return_tensors="pt")
                    elif is_wav2vec:
                        inputs = processor(audio_samples, sampling_rate=sample_rate, return_tensors="pt")
                    else:
                        # Generic processing
                        inputs = processor(audio_samples, sampling_rate=sample_rate, return_tensors="pt")
                except Exception as e:
                    logger.error(f"Error creating input data: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Input creation failed: {str(e)}"
                    continue
                
                # Move inputs to the appropriate device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Warmup
                try:
                    logger.info(f"Warming up model with {warmup_iterations} iterations")
                    model.eval()
                    
                    with torch.no_grad():
                        for _ in range(warmup_iterations):
                            if is_whisper and hasattr(model, "generate"):
                                # Whisper often uses generation
                                _ = model.generate(inputs.input_features)
                            else:
                                # Standard forward pass
                                _ = model(**inputs)
                except Exception as e:
                    logger.error(f"Error during warmup: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Warmup failed: {str(e)}"
                    continue
                
                # Benchmark
                try:
                    logger.info(f"Benchmarking with batch_size={batch_size}, audio_length={audio_length}s")
                    model.eval()
                    
                    # Record timing information
                    latencies = []
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for i in range(benchmark_iterations):
                            iter_start = time.time()
                            
                            if is_whisper and hasattr(model, "generate"):
                                # Whisper often uses generation
                                outputs = model.generate(inputs.input_features)
                            else:
                                # Standard forward pass
                                outputs = model(**inputs)
                            
                            # Ensure operation is complete, especially important for GPU
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            
                            iter_end = time.time()
                            latencies.append(iter_end - iter_start)
                            
                            # Check timeout
                            if time.time() - start_time > timeout:
                                logger.warning(f"Timeout after {i+1} iterations")
                                benchmark_results[key]["timeout"] = True
                                benchmark_results[key]["completed_iterations"] = i+1
                                break
                    
                    # Calculate statistics
                    total_time = time.time() - start_time
                    avg_latency = sum(latencies) / len(latencies)
                    throughput = batch_size * len(latencies) / total_time
                    
                    # Calculate audio processing speed (real-time factor)
                    real_time_factor = (audio_length * batch_size * len(latencies)) / total_time
                    benchmark_results[key]["real_time_factor"] = real_time_factor
                    
                    # Save results
                    benchmark_results[key]["status"] = "completed"
                    benchmark_results[key]["latencies"] = latencies
                    benchmark_results[key]["avg_latency"] = avg_latency
                    benchmark_results[key]["throughput"] = throughput
                    benchmark_results[key]["total_time"] = total_time
                    benchmark_results[key]["iterations"] = len(latencies)
                    
                    # Additional memory stats if available
                    if device.type == "cuda" and torch.cuda.is_available():
                        benchmark_results[key]["max_memory_allocated"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
                        benchmark_results[key]["max_memory_reserved"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)  # MB
                        torch.cuda.reset_peak_memory_stats(device)
                
                except Exception as e:
                    logger.error(f"Error during benchmark: {e}")
                    benchmark_results[key]["status"] = "failed"
                    benchmark_results[key]["error"] = f"Benchmark failed: {str(e)}"
        
        return benchmark_results
    
    def _benchmark_multimodal_model(self, model, hardware_type, batch_sizes,
                                  warmup_iterations, benchmark_iterations, timeout):
        """Benchmark multimodal models (e.g., CLIP)"""
        import torch
        import numpy as np
        
        # Get model device
        device = next(model.parameters()).device
        
        # Check model type for processor selection
        model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else "unknown"
        is_clip = "clip" in model_name.lower()
        
        # Initialize processor
        try:
            if is_clip:
                from transformers import CLIPProcessor
                processor = CLIPProcessor.from_pretrained(model_name)
            else:
                # Generic processor fallback
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error creating multimodal processor: {e}")
            return {"status": "failed", "error": f"Processor creation failed: {str(e)}"}
        
        benchmark_results = {}
        
        # Test different batch sizes
        for batch_size in batch_sizes:
            key = f"batch_{batch_size}"
            benchmark_results[key] = {"status": "pending"}
            
            # Create input data
            try:
                # For CLIP, create both text and image inputs
                if is_clip:
                    # Create random images (224x224 is standard for CLIP)
                    images = []
                    for _ in range(batch_size):
                        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                        images.append(img)
                    
                    # Create random text
                    texts = ["A photo of a cat"] * batch_size
                    
                    # Process inputs
                    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
                else:
                    # Generic input creation - may need modification for specific models
                    logger.warning(f"Using generic input creation for {model_name}, may not be optimal")
                    
                    # Create random images
                    images = []
                    for _ in range(batch_size):
                        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                        images.append(img)
                    
                    # Create random text
                    texts = ["Input text for multimodal model"] * batch_size
                    
                    # Try different input formats based on what works
                    try:
                        inputs = processor(text=texts, images=images, return_tensors="pt")
                    except:
                        try:
                            inputs = processor(images=images, return_tensors="pt")
                        except:
                            inputs = processor(text=texts, return_tensors="pt")
            except Exception as e:
                logger.error(f"Error creating input data: {e}")
                benchmark_results[key]["status"] = "failed"
                benchmark_results[key]["error"] = f"Input creation failed: {str(e)}"
                continue
            
            # Move inputs to the appropriate device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Warmup
            try:
                logger.info(f"Warming up model with {warmup_iterations} iterations")
                model.eval()
                
                with torch.no_grad():
                    for _ in range(warmup_iterations):
                        _ = model(**inputs)
            except Exception as e:
                logger.error(f"Error during warmup: {e}")
                benchmark_results[key]["status"] = "failed"
                benchmark_results[key]["error"] = f"Warmup failed: {str(e)}"
                continue
            
            # Benchmark
            try:
                logger.info(f"Benchmarking with batch_size={batch_size}")
                model.eval()
                
                # Record timing information
                latencies = []
                start_time = time.time()
                
                with torch.no_grad():
                    for i in range(benchmark_iterations):
                        iter_start = time.time()
                        
                        outputs = model(**inputs)
                        
                        # Ensure operation is complete, especially important for GPU
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        
                        iter_end = time.time()
                        latencies.append(iter_end - iter_start)
                        
                        # Check timeout
                        if time.time() - start_time > timeout:
                            logger.warning(f"Timeout after {i+1} iterations")
                            benchmark_results[key]["timeout"] = True
                            benchmark_results[key]["completed_iterations"] = i+1
                            break
                
                # Calculate statistics
                total_time = time.time() - start_time
                avg_latency = sum(latencies) / len(latencies)
                throughput = batch_size * len(latencies) / total_time
                
                # Save results
                benchmark_results[key]["status"] = "completed"
                benchmark_results[key]["latencies"] = latencies
                benchmark_results[key]["avg_latency"] = avg_latency
                benchmark_results[key]["throughput"] = throughput
                benchmark_results[key]["total_time"] = total_time
                benchmark_results[key]["iterations"] = len(latencies)
                
                # Additional memory stats if available
                if device.type == "cuda" and torch.cuda.is_available():
                    benchmark_results[key]["max_memory_allocated"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
                    benchmark_results[key]["max_memory_reserved"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)  # MB
                    torch.cuda.reset_peak_memory_stats(device)
            
            except Exception as e:
                logger.error(f"Error during benchmark: {e}")
                benchmark_results[key]["status"] = "failed"
                benchmark_results[key]["error"] = f"Benchmark failed: {str(e)}"
        
        return benchmark_results
    
    def _calculate_performance_metrics(self, benchmark_results):
        """Calculate derived metrics and statistics from benchmark results"""
        summary = {
            "latency": {},
            "throughput": {},
            "memory": {}
        }
        
        # Skip if benchmark results are invalid
        if not benchmark_results or not isinstance(benchmark_results, dict):
            return summary
        
        # Collect metrics across different configurations
        latencies = []
        throughputs = []
        memory_allocated = []
        
        for key, results in benchmark_results.items():
            if results.get("status") != "completed":
                continue
                
            # Extract metrics
            latencies.append(results.get("avg_latency", 0))
            throughputs.append(results.get("throughput", 0))
            
            # Add memory metrics if available
            if "max_memory_allocated" in results:
                memory_allocated.append(results.get("max_memory_allocated", 0))
        
        # Calculate summary statistics for latency
        if latencies:
            summary["latency"]["min"] = min(latencies)
            summary["latency"]["max"] = max(latencies)
            summary["latency"]["mean"] = sum(latencies) / len(latencies)
            summary["latency"]["median"] = sorted(latencies)[len(latencies) // 2]
        
        # Calculate summary statistics for throughput
        if throughputs:
            summary["throughput"]["min"] = min(throughputs)
            summary["throughput"]["max"] = max(throughputs)
            summary["throughput"]["mean"] = sum(throughputs) / len(throughputs)
            summary["throughput"]["median"] = sorted(throughputs)[len(throughputs) // 2]
        
        # Calculate summary statistics for memory usage
        if memory_allocated:
            summary["memory"]["min_allocated"] = min(memory_allocated)
            summary["memory"]["max_allocated"] = max(memory_allocated)
            summary["memory"]["mean_allocated"] = sum(memory_allocated) / len(memory_allocated)
        
        return summary
    
    def _get_system_info(self):
        """Get detailed system information"""
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add CPU info
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=False)
            logical_cpu_count = psutil.cpu_count(logical=True)
            
            system_info["cpu_count_physical"] = cpu_count
            system_info["cpu_count_logical"] = logical_cpu_count
            
            # Add memory information
            mem = psutil.virtual_memory()
            system_info["total_memory_gb"] = mem.total / (1024 ** 3)
            system_info["available_memory_gb"] = mem.available / (1024 ** 3)
            system_info["memory_percent_used"] = mem.percent
        except ImportError:
            # psutil not available, add minimal CPU info
            import multiprocessing
            system_info["cpu_count_logical"] = multiprocessing.cpu_count()
        
        # Add GPU info if available
        try:
            import torch
            system_info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                system_info["cuda_version"] = torch.version.cuda
                system_info["cuda_device_count"] = torch.cuda.device_count()
                system_info["cuda_devices"] = []
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    system_info["cuda_devices"].append({
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024 ** 3),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count
                    })
        except ImportError:
            system_info["cuda_available"] = False
        
        # Add Apple Silicon info if available
        try:
            import torch
            system_info["mps_available"] = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        except ImportError:
            system_info["mps_available"] = False
        
        # Add OpenVINO info if available
        try:
            import openvino
            system_info["openvino_available"] = True
            system_info["openvino_version"] = openvino.__version__
        except ImportError:
            system_info["openvino_available"] = False
        
        return system_info
    
    def _save_results(self, timestamp):
        """Save benchmark results to output directory"""
        # Create results directory if it doesn't exist
        results_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        # Check if database storage is available and should be used
        if BENCHMARK_DB_AVAILABLE:
            # Store results in database
            try:
                db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
                db_api = BenchmarkDBAPI(db_path=db_path)
                
                # Save all benchmark results to database
                db_api.save_benchmark_results(self.benchmark_results)
                logger.info(f"Benchmark results saved to database: {db_path}")
                
                # If JSON output is deprecated, don't save to files
                if DEPRECATE_JSON_OUTPUT:
                    logger.info("JSON output is deprecated. Results stored only in database.")
                    return results_dir
                    
            except Exception as e:
                logger.error(f"Error saving to database: {e}")
                logger.warning("Falling back to JSON file storage")
        elif DEPRECATE_JSON_OUTPUT:
            logger.warning("Database API not available but JSON output is deprecated. "
                           "Install required packages to use database storage.")
        
        # If not deprecated or database save failed, save to JSON file
        if not DEPRECATE_JSON_OUTPUT:
            # Save complete benchmark results
            results_file = os.path.join(results_dir, "benchmark_results.json")
            with open(results_file, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2)
            logger.info(f"Benchmark results saved to {results_file}")
            
            # Also save a summary report
            summary = self._generate_summary_report()
            summary_file = os.path.join(results_dir, "benchmark_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Benchmark summary saved to {summary_file}")
            
            # Create a human-readable report
            report_file = os.path.join(results_dir, "benchmark_report.md")
            with open(report_file, 'w') as f:
                f.write(self._generate_markdown_report())
            logger.info(f"Benchmark report saved to {report_file}")
        
        return results_dir
        
    def _generate_summary_report(self):
        """Generate a summary of benchmark results"""
        summary = {
            "timestamp": self.benchmark_results["timestamp"],
            "system_info": {
                "platform": self.benchmark_results["system_info"]["platform"],
                "cpu": self.benchmark_results["system_info"]["processor"],
                "cuda_available": self.benchmark_results["system_info"].get("cuda_available", False),
                "mps_available": self.benchmark_results["system_info"].get("mps_available", False),
                "openvino_available": self.benchmark_results["system_info"].get("openvino_available", False)
            },
            "hardware_types": list(self.available_hardware.keys()),
            "model_families": {},
            "performance_comparison": {}
        }
        
        # Summarize results by model family
        for family, models in self.benchmark_results["benchmarks"].items():
            summary["model_families"][family] = {
                "models_tested": list(models.keys()),
                "hardware_results": {}
            }
            
            # Collect hardware-specific performance data
            for model_name, hw_results in models.items():
                for hw_type, results in hw_results.items():
                    # Skip failed benchmarks
                    if results.get("status") != "completed":
                        continue
                    
                    # Initialize hardware entry if needed
                    if hw_type not in summary["model_families"][family]["hardware_results"]:
                        summary["model_families"][family]["hardware_results"][hw_type] = {
                            "latency_ms": [],
                            "throughput": []
                        }
                    
                    # Add performance summary if available
                    if "performance_summary" in results:
                        perf_summary = results["performance_summary"]
                        
                        # Add latency (convert to ms)
                        if "latency" in perf_summary and "mean" in perf_summary["latency"]:
                            summary["model_families"][family]["hardware_results"][hw_type]["latency_ms"].append(
                                perf_summary["latency"]["mean"] * 1000  # convert to ms
                            )
                        
                        # Add throughput
                        if "throughput" in perf_summary and "mean" in perf_summary["throughput"]:
                            summary["model_families"][family]["hardware_results"][hw_type]["throughput"].append(
                                perf_summary["throughput"]["mean"]
                            )
        
        # Generate performance comparison across hardware types
        hardware_types = list(self.available_hardware.keys())
        for family in summary["model_families"]:
            summary["performance_comparison"][family] = {}
            
            family_data = summary["model_families"][family]["hardware_results"]
            available_hw = [hw for hw in hardware_types if hw in family_data]
            
            # Calculate relative performance for latency (lower is better)
            if len(available_hw) >= 2:
                # Use CPU as baseline for latency comparison
                if "cpu" in available_hw and family_data["cpu"]["latency_ms"]:
                    base_latency = sum(family_data["cpu"]["latency_ms"]) / len(family_data["cpu"]["latency_ms"])
                    
                    for hw in available_hw:
                        if hw != "cpu" and family_data[hw]["latency_ms"]:
                            hw_latency = sum(family_data[hw]["latency_ms"]) / len(family_data[hw]["latency_ms"])
                            relative_speedup = base_latency / hw_latency
                            
                            if "relative_latency_speedup" not in summary["performance_comparison"][family]:
                                summary["performance_comparison"][family]["relative_latency_speedup"] = {}
                            
                            summary["performance_comparison"][family]["relative_latency_speedup"][hw] = relative_speedup
        
        return summary
        
    def _generate_markdown_report(self):
        """Generate a human-readable Markdown report of benchmark results"""
        report = []
        
        # Add report header
        timestamp = self.benchmark_results["timestamp"]
        report.append(f"# Hardware Benchmark Report - {timestamp}")
        report.append("")
        
        # Add system information
        report.append("## System Information")
        report.append("")
        sys_info = self.benchmark_results["system_info"]
        report.append(f"- **Platform**: {sys_info['platform']}")
        report.append(f"- **Processor**: {sys_info['processor']}")
        report.append(f"- **Python Version**: {sys_info['python_version']}")
        
        # Add CPU details
        if "cpu_count_physical" in sys_info and "cpu_count_logical" in sys_info:
            report.append(f"- **CPU Cores**: {sys_info['cpu_count_physical']} physical cores, {sys_info['cpu_count_logical']} logical cores")
        
        # Add memory details
        if "total_memory_gb" in sys_info:
            report.append(f"- **System Memory**: {sys_info['total_memory_gb']:.2f} GB total")
            if "available_memory_gb" in sys_info:
                report.append(f"- **Available Memory**: {sys_info['available_memory_gb']:.2f} GB")
        
        # Add GPU details
        if "cuda_available" in sys_info and sys_info["cuda_available"]:
            report.append(f"- **CUDA Version**: {sys_info.get('cuda_version', 'Unknown')}")
            report.append(f"- **GPU Count**: {sys_info.get('cuda_device_count', 0)}")
            
            # Add details for each GPU
            if "cuda_devices" in sys_info:
                report.append("- **GPUs**:")
                for i, gpu in enumerate(sys_info["cuda_devices"]):
                    report.append(f"  - GPU {i}: {gpu['name']}, {gpu['total_memory_gb']:.2f} GB, Compute Capability {gpu['compute_capability']}")
        else:
            report.append("- **CUDA**: Not available")
        
        # Add MPS details
        if "mps_available" in sys_info:
            report.append(f"- **Apple Silicon MPS**: {'Available' if sys_info['mps_available'] else 'Not available'}")
        
        # Add OpenVINO details
        if "openvino_available" in sys_info and sys_info["openvino_available"]:
            report.append(f"- **OpenVINO Version**: {sys_info.get('openvino_version', 'Unknown')}")
        else:
            report.append("- **OpenVINO**: Not available")
        
        # Line break
        report.append("")
        report.append("---")
        
        # Add hardware detection summary
        report.append("## Available Hardware")
        report.append("")
        for hw_type, available in self.available_hardware.items():
            status = "✅ Available" if available else "❌ Not available"
            report.append(f"- **{hw_type}**: {status}")
        report.append("")
        
        # Add benchmark results by model family
        report.append("## Benchmark Results")
        report.append("")
        
        for family, models in self.benchmark_results["benchmarks"].items():
            # Add section for model family
            report.append(f"### {family.title()} Models")
            report.append("")
            
            # Add table for latency comparison across hardware types
            hardware_types = [hw for hw in self.available_hardware.keys() if self.available_hardware[hw]]
            
            # Prepare header row for latency table
            report.append("#### Latency Comparison (ms)")
            report.append("")
            header = "| Model |"
            separator = "|---|"
            for hw in hardware_types:
                header += f" {hw.upper()} |"
                separator += "---|"
            report.append(header)
            report.append(separator)
            
            # Add model rows
            for model_name, hw_results in models.items():
                row = f"| {model_name} |"
                
                for hw in hardware_types:
                    if hw in hw_results and hw_results[hw].get("status") == "completed":
                        # Extract median latency from benchmark results
                        median_latency = None
                        if "performance_summary" in hw_results[hw]:
                            perf = hw_results[hw]["performance_summary"]
                            if "latency" in perf and "median" in perf["latency"]:
                                # Convert to milliseconds
                                median_latency = perf["latency"]["median"] * 1000
                        
                        if median_latency is not None:
                            row += f" {median_latency:.2f} |"
                        else:
                            row += " - |"
                    else:
                        row += " - |"
                
                report.append(row)
            
            report.append("")
            
            # Add table for throughput comparison
            report.append("#### Throughput Comparison (items/sec)")
            report.append("")
            header = "| Model |"
            separator = "|---|"
            for hw in hardware_types:
                header += f" {hw.upper()} |"
                separator += "---|"
            report.append(header)
            report.append(separator)
            
            # Add model rows
            for model_name, hw_results in models.items():
                row = f"| {model_name} |"
                
                for hw in hardware_types:
                    if hw in hw_results and hw_results[hw].get("status") == "completed":
                        # Extract median throughput from benchmark results
                        median_throughput = None
                        if "performance_summary" in hw_results[hw]:
                            perf = hw_results[hw]["performance_summary"]
                            if "throughput" in perf and "median" in perf["throughput"]:
                                median_throughput = perf["throughput"]["median"]
                        
                        if median_throughput is not None:
                            row += f" {median_throughput:.2f} |"
                        else:
                            row += " - |"
                    else:
                        row += " - |"
                
                report.append(row)
                
            # Add detailed results for each model
            for model_name, hw_results in models.items():
                report.append("")
                report.append(f"#### {model_name} - Detailed Results")
                report.append("")
                
                for hw_type, results in hw_results.items():
                    report.append(f"##### {hw_type.upper()}")
                    
                    if results.get("status") != "completed":
                        report.append(f"Status: {results.get('status')}")
                        if "error" in results:
                            report.append(f"Error: {results['error']}")
                        report.append("")
                        continue
                    
                    report.append("")
                    report.append(f"- **Model Load Time**: {results.get('model_load_time', 0):.2f} seconds")
                    
                    # Add performance summary
                    if "performance_summary" in results:
                        perf = results["performance_summary"]
                        
                        # Add latency information
                        if "latency" in perf:
                            report.append("- **Latency (seconds)**:")
                            report.append(f"  - Min: {perf['latency'].get('min', 0):.4f}")
                            report.append(f"  - Max: {perf['latency'].get('max', 0):.4f}")
                            report.append(f"  - Mean: {perf['latency'].get('mean', 0):.4f}")
                            report.append(f"  - Median: {perf['latency'].get('median', 0):.4f}")
                        
                        # Add throughput information
                        if "throughput" in perf:
                            report.append("- **Throughput (items/sec)**:")
                            report.append(f"  - Min: {perf['throughput'].get('min', 0):.2f}")
                            report.append(f"  - Max: {perf['throughput'].get('max', 0):.2f}")
                            report.append(f"  - Mean: {perf['throughput'].get('mean', 0):.2f}")
                            report.append(f"  - Median: {perf['throughput'].get('median', 0):.2f}")
                        
                        # Add memory information
                        if "memory" in perf:
                            report.append("- **Memory Usage (MB)**:")
                            report.append(f"  - Min Allocated: {perf['memory'].get('min_allocated', 0):.2f}")
                            report.append(f"  - Max Allocated: {perf['memory'].get('max_allocated', 0):.2f}")
                            report.append(f"  - Mean Allocated: {perf['memory'].get('mean_allocated', 0):.2f}")
                    
                    # Add detailed benchmark configurations
                    report.append("- **Benchmark Configurations**:")
                    
                    for config, config_results in results.get("benchmark_results", {}).items():
                        if config_results.get("status") != "completed":
                            continue
                            
                        report.append(f"  - **{config}**:")
                        report.append(f"    - Avg Latency: {config_results.get('avg_latency', 0):.4f} seconds")
                        report.append(f"    - Throughput: {config_results.get('throughput', 0):.2f} items/second")
                        
                        # Add additional metrics specific to model types
                        if "tokens_per_second" in config_results:
                            report.append(f"    - Tokens per Second: {config_results.get('tokens_per_second', 0):.2f}")
                        
                        if "real_time_factor" in config_results:
                            report.append(f"    - Real-time Factor: {config_results.get('real_time_factor', 0):.2f}x")
                    
                    report.append("")
            
            # Add relative performance comparison if applicable
            summary = self._generate_summary_report()
            if family in summary["performance_comparison"] and "relative_latency_speedup" in summary["performance_comparison"][family]:
                report.append("#### Relative Performance (vs CPU)")
                report.append("")
                report.append("| Hardware | Speedup Factor |")
                report.append("|---|---|")
                
                speedups = summary["performance_comparison"][family]["relative_latency_speedup"]
                for hw, speedup in speedups.items():
                    report.append(f"| {hw.upper()} | {speedup:.2f}x |")
                
                report.append("")
            
            # Add separator between model families
            report.append("---")
            report.append("")
        
        return "\n".join(report)
            if "model_family_classifier" not in sys.modules:
                logger.warning("Model family classifier not available, skipping compatibility matrix update")
                return
            
            # Initialize compatibility matrix
            compatibility_matrix = {}
            
            # Try to load from database first
            db_loaded = False
            if BENCHMARK_DB_AVAILABLE and not DEPRECATE_JSON_OUTPUT:
                try:
                    db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
                    db_api = BenchmarkDBAPI(db_path=db_path)
                    compatibility_matrix = db_api.get_compatibility_matrix()
                    logger.info("Successfully loaded compatibility matrix from database")
                    db_loaded = True
                except Exception as e:
                    logger.warning(f"Error reading from database: {e}")
                    logger.warning("Falling back to JSON file")
            
            # If database not available or load failed, try from file
            compatibility_file = os.path.join(self.output_dir, "hardware_compatibility_matrix.json")
            if not db_loaded and os.path.exists(compatibility_file) and not DEPRECATE_JSON_OUTPUT:
                try:
                    with open(compatibility_file, 'r') as f:
                        compatibility_matrix = json.load(f)
                    logger.info(f"Loaded compatibility matrix from {compatibility_file}")
                except Exception as e:
                    logger.warning(f"Error reading compatibility matrix from file: {e}")
                    # Initialize a new one
                    compatibility_matrix = {}
            
            # Initialize compatibility matrix if needed
            if "model_families" not in compatibility_matrix:
                compatibility_matrix["model_families"] = {}
            
            if "hardware_types" not in compatibility_matrix:
                compatibility_matrix["hardware_types"] = list(self.available_hardware.keys())
            
            # Update compatibility matrix based on benchmark results
            for family, models in self.benchmark_results["benchmarks"].items():
                # Initialize family entry if needed
                if family not in compatibility_matrix["model_families"]:
                    compatibility_matrix["model_families"][family] = {
                        "hardware_compatibility": {}
                    }
                
                # Update hardware compatibility for each family
                for model_name, hw_results in models.items():
                    for hw_type, results in hw_results.items():
                        # Initialize hardware entry if needed
                        if hw_type not in compatibility_matrix["model_families"][family]["hardware_compatibility"]:
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type] = {
                                "compatible": False,
                                "performance_rating": None,
                                "benchmark_results": []
                            }
                        
                        # Update compatibility and performance based on benchmark results
                        if results.get("status") == "completed":
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["compatible"] = True
                            
                            # Add benchmark result
                            benchmark_summary = {}
                            
                            if "performance_summary" in results:
                                perf = results["performance_summary"]
                                
                                if "latency" in perf and "mean" in perf["latency"]:
                                    benchmark_summary["mean_latency"] = perf["latency"]["mean"]
                                
                                if "throughput" in perf and "mean" in perf["throughput"]:
                                    benchmark_summary["mean_throughput"] = perf["throughput"]["mean"]
                                
                                if "memory" in perf and "max_allocated" in perf["memory"]:
                                    benchmark_summary["max_memory"] = perf["memory"]["max_allocated"]
                            
                            # Add benchmark result with timestamp
                            benchmark_summary["timestamp"] = datetime.datetime.now().isoformat()
                            benchmark_summary["model_name"] = model_name
                            
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["benchmark_results"].append(benchmark_summary)
                        elif results.get("status") == "load_failed" or results.get("status") == "benchmark_failed":
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["compatible"] = False
                            
                            # Add error information
                            if "error" in results:
                                compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["error"] = results["error"]
                        
                        # Calculate performance rating based on all benchmark results
                        benchmark_results = compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["benchmark_results"]
                        if benchmark_results:
                            # Calculate average throughput across all benchmark results
                            throughputs = [res.get("mean_throughput", 0) for res in benchmark_results if "mean_throughput" in res]
                            if throughputs:
                                avg_throughput = sum(throughputs) / len(throughputs)
                                
                                # Assign performance rating based on throughput
                                if avg_throughput > 0:
                                    if hw_type == "cpu":
                                        # CPU is baseline
                                        rating = "medium"
                                    elif avg_throughput > 5:
                                        rating = "high"
                                    elif avg_throughput > 1:
                                        rating = "medium"
                                    else:
                                        rating = "low"
                                        
                                    compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["performance_rating"] = rating
            
            # Save updated compatibility matrix to database first if available
            if BENCHMARK_DB_AVAILABLE:
                try:
                    db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
                    db_api = BenchmarkDBAPI(db_path=db_path)
                    db_api.save_compatibility_matrix(compatibility_matrix)
                    logger.info(f"Compatibility matrix saved to database: {db_path}")
                    
                    # If JSON is deprecated, don't save to file
                    if DEPRECATE_JSON_OUTPUT:
                        return
                except Exception as e:
                    logger.error(f"Error saving compatibility matrix to database: {e}")
                    logger.warning("Falling back to JSON file storage")
            
            # Save to JSON file if not deprecated or database save failed
            if not DEPRECATE_JSON_OUTPUT:
                with open(compatibility_file, 'w') as f:
                    json.dump(compatibility_matrix, f, indent=2)
                logger.info(f"Hardware compatibility matrix updated at {compatibility_file}")
            
        except Exception as e:
            logger.error(f"Error updating compatibility matrix: {e}")

def main():
    """Main function for running benchmarks from command line"""
    parser = argparse.ArgumentParser(description="Hardware benchmark runner for model performance testing")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--model-families", type=str, nargs="+", help="Model families to benchmark (embedding, text_generation, vision, audio, multimodal)")
    parser.add_argument("--hardware", type=str, nargs="+", help="Hardware types to benchmark (cpu, cuda, mps, openvino)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES, help="Batch sizes to test")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=DEFAULT_BENCHMARK_ITERATIONS, help="Number of benchmark iterations")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Maximum time in seconds for a benchmark")
    parser.add_argument("--parallel", action="store_true", help="Run benchmarks in parallel across hardware types")
    parser.add_argument("--use-resource-pool", action="store_true", default=True, help="Use ResourcePool for model caching")
    parser.add_argument("--include-web-platforms", action="store_true", help="Include WebNN and WebGPU in benchmarks")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Create benchmark runner
    runner = HardwareBenchmarkRunner(
        output_dir=args.output_dir,
        use_resource_pool=args.use_resource_pool,
        include_web_platforms=args.include_web_platforms
    )

    # Run benchmarks
    runner.run_benchmarks(
        model_families=args.model_families,
        hardware_types=args.hardware,
        batch_sizes=args.batch_sizes,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        timeout=args.timeout,
        parallel=args.parallel
    )

if __name__ == "__main__":
    main()
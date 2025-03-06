#!/usr/bin/env python
# Continuous Hardware Benchmarking System for ResourcePool

import os
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

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

# Try to import hardware detection
try:
    from hardware_detection import detect_hardware_with_comprehensive_checks
    HAS_HARDWARE_DETECTION = True
except ImportError:
    logger.warning("hardware_detection module not available. Using basic detection.")
    HAS_HARDWARE_DETECTION = False

# Try to import model family classifier
try:
    from model_family_classifier import classify_model, ModelFamilyClassifier
    HAS_MODEL_CLASSIFIER = True
except ImportError:
    logger.warning("model_family_classifier module not available. Using basic model classification.")
    HAS_MODEL_CLASSIFIER = False

# Required imports
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not available. Some benchmarks will be skipped.")
    HAS_TORCH = False

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("Transformers library not available. Model benchmarks will be skipped.")
    HAS_TRANSFORMERS = False

# Try to import resource pool
try:
    from resource_pool import get_global_resource_pool
    HAS_RESOURCE_POOL = True
except ImportError:
    logger.warning("ResourcePool not available. Using standalone benchmarking.")
    HAS_RESOURCE_POOL = False


class HardwareBenchmark:
    """
    Comprehensive hardware benchmarking system for model performance testing.
    Measures performance across different hardware platforms and model types.
    
    Features:
    - Automated benchmark execution for hardware-model combinations
    - Standardized benchmarks for different model families
    - Historical performance tracking
    - Hardware compatibility matrix generation
    - Performance visualization and reporting
    """
    
    def __init__(self, 
                 output_dir: str = "./benchmark_results", 
                 database_path: str = "./benchmark_database.json",
                 use_resource_pool: bool = True):
        """
        Initialize the benchmark system
        
        Args:
            output_dir: Directory to store benchmark results
            database_path: Path to the benchmark database file
            use_resource_pool: Whether to use ResourcePool for model loading
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.database_path = Path(database_path)
        self.database = self._load_database()
        
        # Hardware information
        self.hardware_info = self._detect_hardware()
        
        # ResourcePool integration
        self.use_resource_pool = use_resource_pool and HAS_RESOURCE_POOL
        if self.use_resource_pool:
            self.resource_pool = get_global_resource_pool()
            logger.info("Using ResourcePool for model caching and hardware selection")
        else:
            self.resource_pool = None
            logger.info("ResourcePool not used - models will be loaded directly")
        
        # Initialize model registry
        self.test_models = self._get_test_models()
        
        # Statistics tracking
        self.current_results = {}
        
        logger.info(f"HardwareBenchmark initialized with output directory: {output_dir}")
        logger.info(f"Detected hardware: {list(self.hardware_info.keys())}")

    def _load_database(self) -> Dict:
        """Load the benchmark database or create a new one if it doesn't exist"""
        if self.database_path.exists():
            try:
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
                    with open(self.database_path, 'r') as f:
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    database = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                            database = json.load(f)

                    logger.info(f"Loaded benchmark database with {len(database.get('results', []))} entries")
                    return database
                except Exception as e:
                    logger.warning(f"Error loading database: {e}. Creating new one.")
            
            # Create new database structure
            database = {
                "schema_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "results": [],
                "compatibility_matrix": {},
                "hardware_profiles": {},
                "model_profiles": {}
            }
            
            return database
        
        def _save_database(self):
            """Save the benchmark database to disk"""
            # Update last modified timestamp
            self.database["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.database_path, 'w') as f:
                json.dump(self.database, f, indent=2)
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

        
        logger.info(f"Benchmark database saved to {self.database_path}")
    
    def _detect_hardware(self) -> Dict:
        """Detect available hardware platforms using comprehensive detection"""
        hardware_info = {}
        
        # Try using hardware_detection module first
        if HAS_HARDWARE_DETECTION:
            try:
                hardware_info = detect_hardware_with_comprehensive_checks()
                logger.info("Hardware detection completed using hardware_detection module")
                return hardware_info
            except Exception as e:
                logger.warning(f"Error using hardware_detection module: {e}. Falling back to basic detection.")
        
        # Basic hardware detection as fallback
        hardware_info = {"cpu": True}  # CPU is always available
        
        # Check for CUDA
        if HAS_TORCH:
            if torch.cuda.is_available():
                # Get CUDA device information
                device_count = torch.cuda.device_count()
                devices = []
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "name": props.name,
                        "capability": f"{props.major}.{props.minor}",
                        "total_memory": props.total_memory,
                        "index": i
                    })
                
                hardware_info["cuda"] = True
                hardware_info["cuda_device_count"] = device_count
                hardware_info["cuda_devices"] = devices
            else:
                hardware_info["cuda"] = False
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                hardware_info["mps"] = True
            else:
                hardware_info["mps"] = False
                
        # Check for other hardware platforms
        try:
            import openvino
            hardware_info["openvino"] = True
        except ImportError:
            hardware_info["openvino"] = False
        
        # Check for ROCm by looking for AMD GPUs through torch if CUDA is not available
        if HAS_TORCH and not hardware_info.get("cuda", False):
            try:
                # Simple check - might need refinement
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    hardware_info["rocm"] = True
                else:
                    hardware_info["rocm"] = False
            except:
                hardware_info["rocm"] = False
        
        # WebNN and WebGPU would be checked in browser environments
        hardware_info["webnn"] = False
        hardware_info["webgpu"] = False
        
        return hardware_info
    
    def _get_test_models(self) -> Dict:
        """
        Get dictionary of test models for each model family and hardware platform.
        These models will be used for standardized benchmarking.
        """
        test_models = {
            "embedding": [
                {
                    "name": "prajjwal1/bert-tiny",
                    "class": "BertModel",
                    "test_inputs": {"input_shape": [1, 128]},
                    "description": "Tiny BERT model (good for all hardware)"
                },
                {
                    "name": "bert-base-uncased",
                    "class": "BertModel",
                    "test_inputs": {"input_shape": [1, 128]},
                    "description": "Standard BERT base model"
                }
            ],
            "text_generation": [
                {
                    "name": "google/t5-efficient-tiny",
                    "class": "T5ForConditionalGeneration",
                    "test_inputs": {"input_shape": [1, 128]},
                    "description": "Tiny T5 model (good for all hardware)"
                },
                {
                    "name": "gpt2",
                    "class": "GPT2LMHeadModel",
                    "test_inputs": {"input_shape": [1, 128]},
                    "description": "GPT-2 small model"
                }
            ],
            "vision": [
                {
                    "name": "google/vit-base-patch16-224",
                    "class": "ViTForImageClassification",
                    "test_inputs": {"input_shape": [1, 3, 224, 224]},
                    "description": "ViT base model"
                }
            ],
            "audio": [
                {
                    "name": "openai/whisper-tiny",
                    "class": "WhisperForConditionalGeneration",
                    "test_inputs": {"input_shape": [1, 80, 3000]},
                    "description": "Tiny Whisper model"
                }
            ],
            "multimodal": [
                {
                    "name": "openai/clip-vit-base-patch32",
                    "class": "CLIPModel",
                    "test_inputs": {
                        "vision_shape": [1, 3, 224, 224],
                        "text_shape": [1, 77]
                    },
                    "description": "CLIP base model"
                }
            ]
        }
        
        # Add hardware compatibility flags
        for family in test_models:
            for model in test_models[family]:
                # Default compatibility - CPU always works
                model["hardware_compatibility"] = {"cpu": True}
                
                # Add compatibility for other hardware based on family and size
                if family == "embedding":
                    # Embedding models generally work on all hardware
                    model["hardware_compatibility"].update({
                        "cuda": True,
                        "mps": True,
                        "rocm": True,
                        "openvino": True,
                        "webnn": True,
                        "webgpu": True
                    })
                
                elif family == "text_generation":
                    # Text generation models need more memory
                    if "tiny" in model["name"].lower():
                        model["hardware_compatibility"].update({
                            "cuda": True,
                            "mps": True,
                            "rocm": True,
                            "openvino": True,
                            "webnn": True if "t5" in model["name"].lower() else False,
                            "webgpu": False
                        })
                    else:
                        model["hardware_compatibility"].update({
                            "cuda": True,
                            "mps": True,
                            "rocm": True,
                            "openvino": False,
                            "webnn": False,
                            "webgpu": False
                        })
                
                elif family == "vision":
                    # Vision models generally work well on most hardware
                    model["hardware_compatibility"].update({
                        "cuda": True,
                        "mps": True,
                        "rocm": True,
                        "openvino": True,
                        "webnn": True,
                        "webgpu": True
                    })
                
                elif family == "audio":
                    # Audio models often need GPU
                    model["hardware_compatibility"].update({
                        "cuda": True,
                        "mps": "tiny" in model["name"].lower(),
                        "rocm": "tiny" in model["name"].lower(),
                        "openvino": "tiny" in model["name"].lower(),
                        "webnn": False,
                        "webgpu": False
                    })
                
                elif family == "multimodal":
                    # Multimodal models like CLIP work on most platforms
                    if "clip" in model["name"].lower():
                        model["hardware_compatibility"].update({
                            "cuda": True,
                            "mps": True,
                            "rocm": True,
                            "openvino": True,
                            "webnn": False,
                            "webgpu": True
                        })
                    else:
                        # Other multimodal models often need CUDA
                        model["hardware_compatibility"].update({
                            "cuda": True,
                            "mps": False,
                            "rocm": False,
                            "openvino": False,
                            "webnn": False,
                            "webgpu": False
                        })
        
        return test_models
    
    def _create_dummy_input(self, model_info: Dict, device: str) -> Dict:
        """Create dummy input tensors for benchmarking based on model type"""
        inputs = {}
        
        if not HAS_TORCH:
            logger.error("PyTorch not available - cannot create dummy inputs")
            return inputs
        
        # Get input information
        test_inputs = model_info.get("test_inputs", {})
        
        # Handle different model types
        if "input_shape" in test_inputs:
            # Basic input for most models
            shape = test_inputs["input_shape"]
            inputs["input_ids"] = torch.randint(0, 1000, shape).to(device)
            
            # Add attention mask for transformer models
            inputs["attention_mask"] = torch.ones(shape).to(device)
        
        elif "vision_shape" in test_inputs and "text_shape" in test_inputs:
            # Multimodal models like CLIP
            vision_shape = test_inputs["vision_shape"]
            text_shape = test_inputs["text_shape"]
            
            # Create vision input
            inputs["pixel_values"] = torch.rand(vision_shape).to(device)
            
            # Create text input
            inputs["input_ids"] = torch.randint(0, 1000, text_shape).to(device)
            inputs["attention_mask"] = torch.ones(text_shape).to(device)
        
        else:
            # Fallback to model class specific inputs
            model_class = model_info.get("class", "").lower()
            
            if "vit" in model_class or "vision" in model_class:
                # Vision models
                inputs["pixel_values"] = torch.rand(1, 3, 224, 224).to(device)
            
            elif "whisper" in model_class or "audio" in model_class:
                # Audio models
                inputs["input_features"] = torch.rand(1, 80, 3000).to(device)
            
            else:
                # Default text input
                inputs["input_ids"] = torch.randint(0, 1000, (1, 128)).to(device)
                inputs["attention_mask"] = torch.ones(1, 128).to(device)
        
        return inputs
    
    def load_model(self, model_info: Dict, device: str = "cpu") -> Tuple[Any, float]:
        """
        Load a model for benchmarking
        
        Args:
            model_info: Dictionary with model information
            device: Target device
            
        Returns:
            Tuple of (model, load_time_seconds)
        """
        if not HAS_TRANSFORMERS:
            logger.error("Transformers library not available - cannot load models")
            return None, 0
        
        model_name = model_info["name"]
        model_class = model_info.get("class")
        
        start_time = time.time()
        
        try:
            # Use resource pool if available
            if self.use_resource_pool:
                def model_constructor():
                    if model_class:
                        # Get model class from transformers
                        class_obj = getattr(transformers, model_class)
                        model = class_obj.from_pretrained(model_name)
                        if device != "cpu":
                            model = model.to(device)
                        return model
                    else:
                        # Fall back to auto model
                        model = transformers.AutoModel.from_pretrained(model_name)
                        if device != "cpu":
                            model = model.to(device)
                        return model
                
                # Use resource pool to get model
                logger.info(f"Loading model {model_name} via ResourcePool on {device}")
                model = self.resource_pool.get_model(
                    model_type=model_info.get("family", "unknown"),
                    model_name=model_name,
                    constructor=model_constructor,
                    hardware_preferences={"device": device}
                )
            else:
                # Load model directly
                logger.info(f"Loading model {model_name} directly on {device}")
                if model_class:
                    # Get model class from transformers
                    class_obj = getattr(transformers, model_class)
                    model = class_obj.from_pretrained(model_name)
                else:
                    # Fall back to auto model
                    model = transformers.AutoModel.from_pretrained(model_name)
                
                # Move to device
                if device != "cpu":
                    model = model.to(device)
        
        except Exception as e:
            logger.error(f"Error loading model {model_name} on {device}: {e}")
            return None, 0
        
        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
        
        return model, load_time
    
    def run_inference_benchmark(self, model: Any, model_info: Dict, 
                               device: str, iterations: int = 10, 
                               warmup: int = 3, batch_sizes: List[int] = None) -> Dict:
        """
        Run inference benchmark on a loaded model
        
        Args:
            model: Loaded model
            model_info: Dictionary with model information
            device: Target device
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if not HAS_TORCH or model is None:
            logger.error("Cannot run benchmark - PyTorch not available or model not loaded")
            return {}
        
        # Default batch sizes if not specified
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        
        # Results structure
        results = {
            "model_name": model_info["name"],
            "model_class": model_info.get("class", ""),
            "device": device,
            "timestamp": datetime.now().isoformat(),
            "latency": {},
            "throughput": {},
            "memory_usage": {},
            "batch_results": {}
        }
        
        # For memory tracking
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create test inputs and run benchmarks for each batch size
        for batch_size in batch_sizes:
            # Skip large batch sizes on memory-limited devices
            if device == "mps" and batch_size > 8:
                logger.info(f"Skipping batch size {batch_size} on MPS (memory limitations)")
                continue
                
            # Skip large batch sizes for large models
            model_size = model_info.get("size", "small")
            if model_size == "large" and batch_size > 2:
                logger.info(f"Skipping batch size {batch_size} for large model (memory limitations)")
                continue
            
            # Adjust test inputs for this batch size
            try:
                # Create batch-specific inputs by modifying the dummy input
                single_inputs = self._create_dummy_input(model_info, device)
                batch_inputs = {}
                
                for key, tensor in single_inputs.items():
                    # Adjust batch dimension (assume it's always the first dimension)
                    current_shape = list(tensor.shape)
                    current_shape[0] = batch_size
                    
                    # Create new tensor with adjusted batch size
                    batch_inputs[key] = torch.zeros(current_shape, dtype=tensor.dtype, device=tensor.device)
                
                logger.info(f"Running benchmark for batch size {batch_size}")
                
                # Warmup runs
                model.eval()
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(**batch_inputs)
                
                # Benchmark runs
                latencies = []
                start_time = time.time()
                
                # Track peak memory before benchmarking
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = torch.cuda.memory_allocated()
                
                # Run benchmark iterations
                with torch.no_grad():
                    for i in range(iterations):
                        iter_start = time.time()
                        _ = model(**batch_inputs)
                        latencies.append((time.time() - iter_start) * 1000)  # ms
                
                # Calculate total time and stats
                total_time = time.time() - start_time
                
                # Get peak memory usage
                peak_memory_mb = 0
                if device.startswith("cuda") and torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated()
                    peak_memory_mb = (peak_memory - start_memory) / (1024 * 1024)
                
                # Store batch-specific results
                batch_results = {
                    "latency_ms_mean": float(np.mean(latencies)),
                    "latency_ms_median": float(np.median(latencies)),
                    "latency_ms_std": float(np.std(latencies)),
                    "latency_ms_min": float(np.min(latencies)),
                    "latency_ms_max": float(np.max(latencies)),
                    "latency_ms_p95": float(np.percentile(latencies, 95)),
                    "throughput_items_per_sec": float(batch_size * iterations / total_time),
                    "inference_time_seconds": float(total_time),
                    "peak_memory_mb": float(peak_memory_mb)
                }
                
                # Add to overall results
                results["batch_results"][str(batch_size)] = batch_results
                
                # Update summary metrics
                results["latency"][str(batch_size)] = float(np.mean(latencies))
                results["throughput"][str(batch_size)] = float(batch_size * iterations / total_time)
                results["memory_usage"][str(batch_size)] = float(peak_memory_mb)
                
                logger.info(f"Batch {batch_size}: Avg latency {batch_results['latency_ms_mean']:.2f}ms, "
                           f"Throughput {batch_results['throughput_items_per_sec']:.2f} items/s, "
                           f"Memory {batch_results['peak_memory_mb']:.2f}MB")
                
            except Exception as e:
                logger.error(f"Error benchmarking batch size {batch_size}: {e}")
                continue
        
        return results
    
    def run_model_family_benchmark(self, family: str, device: str, iterations: int = 10, 
                                  batch_sizes: List[int] = None) -> List[Dict]:
        """
        Run benchmarks for all models in a specific family
        
        Args:
            family: Model family name (embedding, text_generation, etc.)
            device: Target device
            iterations: Number of benchmark iterations
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of benchmark result dictionaries
        """
        results = []
        
        if family not in self.test_models:
            logger.error(f"Unknown model family: {family}")
            return results
        
        logger.info(f"Running benchmarks for {family} models on {device}")
        
        # For each model in the family
        for model_info in self.test_models[family]:
            # Check hardware compatibility first
            hardware_compatibility = model_info.get("hardware_compatibility", {})
            hardware_type = device.split(":")[0]  # Extract base device type (cuda, mps, etc.)
            
            if not hardware_compatibility.get(hardware_type, False):
                logger.warning(f"Model {model_info['name']} not compatible with {hardware_type}. Skipping.")
                continue
            
            # Add family info to model_info for resource pool
            model_info["family"] = family
            
            try:
                # Load the model
                model, load_time = self.load_model(model_info, device)
                
                if model is None:
                    logger.error(f"Failed to load model {model_info['name']}. Skipping.")
                    continue
                
                # Run benchmark
                benchmark_result = self.run_inference_benchmark(
                    model=model,
                    model_info=model_info,
                    device=device,
                    iterations=iterations,
                    batch_sizes=batch_sizes
                )
                
                # Add load time and model family to results
                benchmark_result["load_time_seconds"] = load_time
                benchmark_result["model_family"] = family
                
                # Add to results list
                results.append(benchmark_result)
                
                # Clean up to free memory
                if not self.use_resource_pool:
                    del model
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error benchmarking model {model_info['name']}: {e}")
                continue
        
        return results
    
    def run_comprehensive_benchmark(self, device: str = "all", families: List[str] = None,
                                   iterations: int = 10, batch_sizes: List[int] = None) -> Dict:
        """
        Run comprehensive benchmarks across hardware platforms and model families
        
        Args:
            device: Target device or "all" for all available devices
            families: List of model families to benchmark or None for all
            iterations: Number of benchmark iterations
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results organized by device and family
        """
        # Initialize results
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "hardware_info": self.hardware_info,
            "results_by_device": {},
            "results_by_family": {}
        }
        
        # Determine which devices to benchmark
        devices_to_test = []
        if device == "all":
            # Test all available devices
            if self.hardware_info.get("cuda", False):
                devices_to_test.append("cuda")
            if self.hardware_info.get("mps", False):
                devices_to_test.append("mps")
            if self.hardware_info.get("rocm", False):
                devices_to_test.append("rocm")
            # Always include CPU
            devices_to_test.append("cpu")
        else:
            # Test specified device
            devices_to_test.append(device)
        
        # Determine which families to benchmark
        families_to_test = families or list(self.test_models.keys())
        
        # Run benchmarks for each device and family
        for device in devices_to_test:
            logger.info(f"Running benchmarks on {device}")
            device_results = []
            
            for family in families_to_test:
                logger.info(f"Benchmarking {family} models on {device}")
                
                family_results = self.run_model_family_benchmark(
                    family=family,
                    device=device,
                    iterations=iterations,
                    batch_sizes=batch_sizes
                )
                
                # Add to device results
                device_results.extend(family_results)
                
                # Add to results by family
                if family not in all_results["results_by_family"]:
                    all_results["results_by_family"][family] = {}
                
                all_results["results_by_family"][family][device] = family_results
            
            # Add to results by device
            all_results["results_by_device"][device] = device_results
        
        # Store results for later use
        self.current_results = all_results
        
        # Save results to database
        self._update_database_with_results(all_results)
        
        return all_results
    
    def _update_database_with_results(self, results: Dict):
        """Update the benchmark database with new results"""
        # Extract results into flat structure for database
        flat_results = []
        
        for device, device_results in results["results_by_device"].items():
            for result in device_results:
                # Create flat entry for database
                entry = {
                    "timestamp": result["timestamp"],
                    "model_name": result["model_name"],
                    "model_class": result["model_class"],
                    "model_family": result["model_family"],
                    "device": result["device"],
                    "load_time_seconds": result.get("load_time_seconds", 0),
                    "latency": result["latency"],
                    "throughput": result["throughput"],
                    "memory_usage": result["memory_usage"]
                }
                
                flat_results.append(entry)
        
        # Add to database
        self.database["results"].extend(flat_results)
        
        # Update hardware profiles
        self.database["hardware_profiles"][datetime.now().isoformat()] = self.hardware_info
        
        # Update compatibility matrix
        self._update_compatibility_matrix(results)
        
        # Save updated database
        self._save_database()
    
    def _update_compatibility_matrix(self, results: Dict):
        """Update the hardware compatibility matrix based on benchmark results"""
        # Initialize new matrix structure if it doesn't exist
        if "compatibility_matrix" not in self.database:
            self.database["compatibility_matrix"] = {}
        
        compatibility_matrix = self.database["compatibility_matrix"]
        
        # Process the results to update the matrix
        for family, family_results in results["results_by_family"].items():
            if family not in compatibility_matrix:
                compatibility_matrix[family] = {}
            
            for device, device_results in family_results.items():
                if device not in compatibility_matrix[family]:
                    compatibility_matrix[family][device] = {
                        "compatible": False,
                        "performance_rating": 0,
                        "models_tested": [],
                        "model_status": {}
                    }
                
                # Clear old model list for latest update
                compatibility_matrix[family][device]["models_tested"] = []
                
                # Process each model's results
                for result in device_results:
                    model_name = result["model_name"]
                    
                    # Add to tested models
                    if model_name not in compatibility_matrix[family][device]["models_tested"]:
                        compatibility_matrix[family][device]["models_tested"].append(model_name)
                    
                    # Update model status
                    compatibility_matrix[family][device]["model_status"][model_name] = {
                        "compatible": True,  # If we got results, it's compatible
                        "last_tested": result["timestamp"],
                        "latency_ms": result["latency"].get("1", 0),  # Batch size 1 latency
                        "throughput": result["throughput"].get("1", 0),  # Batch size 1 throughput
                        "memory_mb": result["memory_usage"].get("1", 0)  # Batch size 1 memory
                    }
                
                # Update overall compatibility status
                compatibility_matrix[family][device]["compatible"] = len(device_results) > 0
                
                # Calculate performance rating (1-10 scale)
                if device_results:
                    # Average throughput across models (batch size 1)
                    avg_throughput = sum(r["throughput"].get("1", 0) for r in device_results) / len(device_results)
                    
                    # Average latency across models (batch size 1)
                    avg_latency = sum(r["latency"].get("1", 0) for r in device_results) / len(device_results)
                    
                    # Performance rating formula (higher throughput, lower latency is better)
                    # Scale to 1-10 range based on empirical observations
                    throughput_score = min(10, max(1, avg_throughput / 10))
                    latency_score = min(10, max(1, 1000 / avg_latency))
                    
                    # Combined score (weighted)
                    performance_rating = (throughput_score * 0.7 + latency_score * 0.3)
                    
                    compatibility_matrix[family][device]["performance_rating"] = round(performance_rating, 1)
                    compatibility_matrix[family][device]["last_updated"] = datetime.now().isoformat()
    
    def save_results(self, results: Dict, filename: str = None) -> str:
        """
        Save benchmark results to a file
        
        Args:
            results: Benchmark results to save
            filename: Optional filename, if None a timestamped name will be generated
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Make sure it's a full path
        filepath = self.output_dir / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
        return str(filepath)
    
    def generate_report(self, results: Dict = None) -> str:
        """
        Generate a markdown report from benchmark results
        
        Args:
            results: Benchmark results or None to use most recent results
            
        Returns:
            Path to the generated report file
        """
        # Use provided results or latest results
        report_data = results or self.current_results
        
        if not report_data:
            logger.error("No benchmark results available for report generation")
            return ""
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"benchmark_report_{timestamp}.md"
        report_path = self.output_dir / report_filename
        
        # Create report content
        report_content = [
            "# Hardware Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Hardware Information",
            ""
        ]
        
        # Add hardware information
        hw_info = report_data.get("hardware_info", {})
        for platform, available in hw_info.items():
            if isinstance(available, bool):
                report_content.append(f"- **{platform}**: {'Available' if available else 'Not Available'}")
            elif platform == "cuda_devices" and isinstance(available, list):
                for i, device in enumerate(available):
                    report_content.append(f"  - GPU {i}: {device.get('name', 'Unknown')} ({device.get('total_memory', 0) / (1024**3):.1f} GB)")
        
        report_content.append("")
        
        # Add summary table
        report_content.extend([
            "## Performance Summary",
            "",
            "### Latency by Model Family and Device (ms, batch size=1)",
            "",
            "| Model Family | " + " | ".join(report_data.get("results_by_device", {}).keys()) + " |",
            "| --- | " + " | ".join(["---" for _ in report_data.get("results_by_device", {})]) + " |"
        ])
        
        # Add rows for each family
        for family in report_data.get("results_by_family", {}):
            row = f"| {family} | "
            
            for device in report_data.get("results_by_device", {}):
                # Get average latency for this family/device
                if device in report_data.get("results_by_family", {}).get(family, {}):
                    device_results = report_data["results_by_family"][family][device]
                    if device_results:
                        # Calculate average latency (batch size 1)
                        latencies = [r["latency"].get("1", 0) for r in device_results if "1" in r["latency"]]
                        if latencies:
                            avg_latency = sum(latencies) / len(latencies)
                            row += f"{avg_latency:.2f} | "
                        else:
                            row += "N/A | "
                    else:
                        row += "N/A | "
                else:
                    row += "N/A | "
            
            report_content.append(row)
        
        report_content.append("")
        
        # Add throughput table
        report_content.extend([
            "### Throughput by Model Family and Device (items/sec, batch size=1)",
            "",
            "| Model Family | " + " | ".join(report_data.get("results_by_device", {}).keys()) + " |",
            "| --- | " + " | ".join(["---" for _ in report_data.get("results_by_device", {})]) + " |"
        ])
        
        # Add rows for each family
        for family in report_data.get("results_by_family", {}):
            row = f"| {family} | "
            
            for device in report_data.get("results_by_device", {}):
                # Get average throughput for this family/device
                if device in report_data.get("results_by_family", {}).get(family, {}):
                    device_results = report_data["results_by_family"][family][device]
                    if device_results:
                        # Calculate average throughput (batch size 1)
                        throughputs = [r["throughput"].get("1", 0) for r in device_results if "1" in r["throughput"]]
                        if throughputs:
                            avg_throughput = sum(throughputs) / len(throughputs)
                            row += f"{avg_throughput:.2f} | "
                        else:
                            row += "N/A | "
                    else:
                        row += "N/A | "
                else:
                    row += "N/A | "
            
            report_content.append(row)
        
        report_content.append("")
        
        # Add detailed results for each device/model
        for device, device_results in report_data.get("results_by_device", {}).items():
            report_content.extend([
                f"## Detailed Results: {device}",
                ""
            ])
            
            for result in device_results:
                model_name = result["model_name"]
                model_family = result.get("model_family", "unknown")
                
                report_content.extend([
                    f"### {model_name} ({model_family})",
                    "",
                    f"- **Load Time**: {result.get('load_time_seconds', 0):.2f} seconds",
                    "",
                    "#### Performance by Batch Size",
                    "",
                    "| Batch Size | Latency (ms) | Throughput (items/sec) | Memory (MB) |",
                    "| --- | --- | --- | --- |"
                ])
                
                # Add row for each batch size
                for batch_size in sorted([int(bs) for bs in result["latency"].keys()]):
                    batch_str = str(batch_size)
                    latency = result["latency"].get(batch_str, 0)
                    throughput = result["throughput"].get(batch_str, 0)
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
                        memory = result["memory_usage"].get(batch_str, 0)
                        
                        report_content.append(f"| {batch_size} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |")
                    
                    report_content.append("")
            
            # Write report to file
            with open(report_path, 'w') as f:
                f.write("\n".join(report_content))
            
            logger.info(f"Benchmark report generated: {report_path}")
            return str(report_path)
        
        def get_compatibility_matrix(self) -> Dict:
            """
            Get the current hardware compatibility matrix
            
            Returns:
                Dictionary with compatibility information by model family and hardware
            """
            return self.database.get("compatibility_matrix", {})
        
        def export_compatibility_matrix(self, filename: str = None) -> str:
            """
            Export the hardware compatibility matrix to a file
            
            Args:
                filename: Optional filename, if None a timestamped name will be generated
                
            Returns:
                Path to the saved file
            """
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hardware_compatibility_matrix_{timestamp}.json"
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

        
        # Make sure it's a full path
        filepath = self.output_dir / filename
        
        # Get compatibility matrix
        matrix = self.get_compatibility_matrix()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(matrix, f, indent=2)
        
        logger.info(f"Compatibility matrix exported to {filepath}")
        return str(filepath)
    
    def generate_compatibility_report(self, filename: str = None) -> str:
        """
        Generate a markdown report of the hardware compatibility matrix
        
        Args:
            filename: Optional filename, if None a timestamped name will be generated
            
        Returns:
            Path to the generated report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hardware_compatibility_report_{timestamp}.md"
        
        # Make sure it's a full path
        filepath = self.output_dir / filename
        
        # Get compatibility matrix
        matrix = self.get_compatibility_matrix()
        
        # Create report content
        report_content = [
            "# Hardware Compatibility Matrix",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This report shows the compatibility and performance of different model families across hardware platforms.",
            "",
            "## Compatibility Summary",
            "",
            "Legend:",
            "- ✅ Compatible with good performance (rating 7-10)",
            "- ⚠️ Compatible with moderate performance (rating 4-6)",
            "- ⚡ Compatible with low performance (rating 1-3)",
            "- ❌ Not compatible",
            "",
            "| Model Family | " + " | ".join(sorted(set(platform for family in matrix.values() for platform in family.keys()))) + " |",
            "| --- | " + " | ".join(["---" for _ in set(platform for family in matrix.values() for platform in family.keys())]) + " |"
        ]
        
        # Get unique device list
        devices = sorted(set(platform for family in matrix.values() for platform in family.keys()))
        
        # Add rows for each family
        for family in sorted(matrix.keys()):
            row = f"| {family} | "
            
            for device in devices:
                if device in matrix[family]:
                    info = matrix[family][device]
                    compatible = info.get("compatible", False)
                    rating = info.get("performance_rating", 0)
                    
                    if compatible:
                        if rating >= 7:
                            row += "✅ | "
                        elif rating >= 4:
                            row += "⚠️ | "
                        else:
                            row += "⚡ | "
                    else:
                        row += "❌ | "
                else:
                    row += "❌ | "
            
            report_content.append(row)
        
        report_content.append("")
        
        # Add performance ratings
        report_content.extend([
            "## Performance Ratings (1-10 scale)",
            "",
            "| Model Family | " + " | ".join(devices) + " |",
            "| --- | " + " | ".join(["---" for _ in devices]) + " |"
        ])
        
        for family in sorted(matrix.keys()):
            row = f"| {family} | "
            
            for device in devices:
                if device in matrix[family]:
                    info = matrix[family][device]
                    compatible = info.get("compatible", False)
                    rating = info.get("performance_rating", 0)
                    
                    if compatible:
                        row += f"{rating} | "
                    else:
                        row += "N/A | "
                else:
                    row += "N/A | "
            
            report_content.append(row)
        
        report_content.append("")
        
        # Add detailed notes for each family
        for family in sorted(matrix.keys()):
            report_content.extend([
                f"## {family} Models",
                ""
            ])
            
            for device in devices:
                if device in matrix[family]:
                    info = matrix[family][device]
                    compatible = info.get("compatible", False)
                    
                    if compatible:
                        report_content.extend([
                            f"### {device}",
                            f"- **Performance Rating**: {info.get('performance_rating', 0)}",
                            f"- **Last Updated**: {info.get('last_updated', 'Unknown')}",
                            "",
                            "#### Tested Models",
                            ""
                        ])
                        
                        # Add models
                        for model in info.get("models_tested", []):
                            model_status = info.get("model_status", {}).get(model, {})
                            latency = model_status.get("latency_ms", 0)
                            throughput = model_status.get("throughput", 0)
                            memory = model_status.get("memory_mb", 0)
                            
                            report_content.append(f"- **{model}**:")
                            report_content.append(f"  - Latency: {latency:.2f} ms")
                            report_content.append(f"  - Throughput: {throughput:.2f} items/sec")
                            report_content.append(f"  - Memory: {memory:.2f} MB")
                            report_content.append("")
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write("\n".join(report_content))
        
        logger.info(f"Compatibility report generated: {filepath}")
        return str(filepath)
    
    def get_performance_trends(self, model_family: str = None, device: str = None) -> Dict:
        """
        Analyze performance trends from historical data
        
        Args:
            model_family: Optional filter by model family
            device: Optional filter by device
            
        Returns:
            Dictionary with trend analysis
        """
        results = self.database.get("results", [])
        
        if not results:
            logger.warning("No historical data available for trend analysis")
            return {}
        
        # Filter results if needed
        if model_family:
            results = [r for r in results if r.get("model_family") == model_family]
        
        if device:
            results = [r for r in results if r.get("device") == device]
        
        # Group by model and device
        grouped_results = {}
        for result in results:
            model_name = result["model_name"]
            device_name = result["device"]
            timestamp = result["timestamp"]
            
            key = f"{model_name}:{device_name}"
            if key not in grouped_results:
                grouped_results[key] = []
            
            # Add to grouped results
            grouped_results[key].append({
                "timestamp": timestamp,
                "latency": result["latency"].get("1", 0),
                "throughput": result["throughput"].get("1", 0),
                "memory_usage": result["memory_usage"].get("1", 0)
            })
        
        # Analyze trends
        trends = {
            "by_model_device": {},
            "by_family": {},
            "by_device": {}
        }
        
        # Analyze trends for each model/device
        for key, data in grouped_results.items():
            model_name, device_name = key.split(":")
            
            # Sort by timestamp
            data.sort(key=lambda x: x["timestamp"])
            
            # Skip if less than 2 data points
            if len(data) < 2:
                continue
            
            # Calculate changes
            first = data[0]
            last = data[-1]
            
            latency_change = (last["latency"] - first["latency"]) / first["latency"] * 100
            throughput_change = (last["throughput"] - first["throughput"]) / first["throughput"] * 100
            
            trends["by_model_device"][key] = {
                "first_timestamp": first["timestamp"],
                "last_timestamp": last["timestamp"],
                "latency_change_percent": latency_change,
                "throughput_change_percent": throughput_change,
                "data_points": len(data)
            }
        
        # Aggregate by model family
        family_data = {}
        for result in results:
            family = result.get("model_family")
            device_name = result["device"]
            
            if not family:
                continue
            
            key = f"{family}:{device_name}"
            if key not in family_data:
                family_data[key] = []
            
            # Add data point
            family_data[key].append({
                "timestamp": result["timestamp"],
                "latency": result["latency"].get("1", 0),
                "throughput": result["throughput"].get("1", 0)
            })
        
        # Calculate family trends
        for key, data in family_data.items():
            family, device_name = key.split(":")
            
            # Sort by timestamp
            data.sort(key=lambda x: x["timestamp"])
            
            # Skip if less than 2 data points
            if len(data) < 2:
                continue
            
            # Group by timestamp (daily)
            by_date = {}
            for point in data:
                date = point["timestamp"].split("T")[0]
                if date not in by_date:
                    by_date[date] = []
                by_date[date].append(point)
            
            # Calculate daily averages
            daily_averages = []
            for date, points in by_date.items():
                avg_latency = sum(p["latency"] for p in points) / len(points)
                avg_throughput = sum(p["throughput"] for p in points) / len(points)
                
                daily_averages.append({
                    "date": date,
                    "avg_latency": avg_latency,
                    "avg_throughput": avg_throughput,
                    "count": len(points)
                })
            
            # Sort by date
            daily_averages.sort(key=lambda x: x["date"])
            
            # Calculate trend
            if len(daily_averages) >= 2:
                first = daily_averages[0]
                last = daily_averages[-1]
                
                latency_change = (last["avg_latency"] - first["avg_latency"]) / first["avg_latency"] * 100
                throughput_change = (last["avg_throughput"] - first["avg_throughput"]) / first["avg_throughput"] * 100
                
                trends["by_family"][key] = {
                    "first_date": first["date"],
                    "last_date": last["date"],
                    "latency_change_percent": latency_change,
                    "throughput_change_percent": throughput_change,
                    "data_points": sum(day["count"] for day in daily_averages)
                }
        
        return trends
    
    def update_resource_pool_with_benchmarks(self) -> bool:
        """
        Update ResourcePool with benchmark results for hardware-aware model selection
        
        Returns:
            True if successful, False otherwise
        """
        if not self.use_resource_pool:
            logger.warning("ResourcePool not available - cannot update")
            return False
        
        try:
            # Get compatibility matrix
            matrix = self.get_compatibility_matrix()
            
            # Access resource pool's hardware selection system
            if not hasattr(self.resource_pool, "_optimal_hardware_matrix"):
                setattr(self.resource_pool, "_optimal_hardware_matrix", {})
            
            # Update resource pool's hardware matrix
            for family, family_data in matrix.items():
                # Create optimized hardware priority list
                platforms = []
                
                for platform, data in family_data.items():
                    if data.get("compatible", False):
                        # Add to platforms with rating
                        platforms.append((platform, data.get("performance_rating", 0)))
                
                # Sort by performance rating (descending)
                platforms.sort(key=lambda x: x[1], reverse=True)
                
                # Create priority list
                priority_list = [p[0] for p in platforms]
                
                # Add CPU as fallback if not already in list
                if "cpu" not in priority_list:
                    priority_list.append("cpu")
                
                # Store in resource pool
                self.resource_pool._optimal_hardware_matrix[family] = priority_list
            
            # Update resource pool timestamp
            setattr(self.resource_pool, "_hardware_matrix_updated", datetime.now().isoformat())
            
            logger.info(f"ResourcePool updated with benchmark data for {len(matrix)} model families")
            return True
            
        except Exception as e:
            logger.error(f"Error updating ResourcePool with benchmark data: {e}")
            return False

# Function to run benchmarks from command line
def main():
    parser = argparse.ArgumentParser(description='Run hardware benchmarks for different model families')
    parser.add_argument('--device', type=str, default='all', help='Device to benchmark (cpu, cuda, mps, all)')
    parser.add_argument('--family', type=str, default=None, help='Model family to benchmark (embedding, text_generation, vision, audio, multimodal)')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4,8', help='Comma-separated list of batch sizes to test')
    parser.add_argument('--iterations', type=int, default=10, help='Number of benchmark iterations')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results', help='Output directory for results')
    parser.add_argument('--report', action='store_true', help='Generate benchmark report')
    parser.add_argument('--matrix', action='store_true', help='Generate compatibility matrix report')
    parser.add_argument('--update_pool', action='store_true', help='Update ResourcePool with benchmark results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse batch sizes
    batch_sizes = [int(s) for s in args.batch_sizes.split(',')]
    
    # Parse model families
    families = [args.family] if args.family else None
    
    # Create benchmark system
    benchmark = HardwareBenchmark(output_dir=args.output_dir)
    
    # Run benchmarks
    results = benchmark.run_comprehensive_benchmark(
        device=args.device,
        families=families,
        iterations=args.iterations,
        batch_sizes=batch_sizes
    )
    
    # Save results
    benchmark.save_results(results)
    
    # Generate report if requested
    if args.report:
        report_path = benchmark.generate_report(results)
        print(f"Report generated: {report_path}")
    
    # Generate compatibility matrix if requested
    if args.matrix:
        matrix_path = benchmark.generate_compatibility_report()
        print(f"Compatibility matrix report generated: {matrix_path}")
    
    # Update ResourcePool if requested
    if args.update_pool:
        success = benchmark.update_resource_pool_with_benchmarks()
        if success:
            print("ResourcePool updated with benchmark results")
        else:
            print("Failed to update ResourcePool")

if __name__ == "__main__":
    main()
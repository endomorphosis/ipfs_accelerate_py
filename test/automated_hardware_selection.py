#!/usr/bin/env python
"""
Automated Hardware Selection System for the IPFS Accelerate Framework.

This script provides a comprehensive system for automatically selecting optimal hardware
for various models and tasks based on benchmarking data, model characteristics, and
available hardware. It integrates the hardware_selector.py, hardware_model_predictor.py,
and model_performance_predictor.py modules to provide accurate hardware recommendations.

Part of Phase 16 of the IPFS Accelerate project.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
try:
    from hardware_selector import HardwareSelector
    HARDWARE_SELECTOR_AVAILABLE = True
    logger.info("Hardware selector module available")
except ImportError:
    HARDWARE_SELECTOR_AVAILABLE = False
    logger.warning("Hardware selector module not available")

try:
    from hardware_model_predictor import HardwareModelPredictor
    PREDICTOR_AVAILABLE = True
    logger.info("Hardware model predictor module available")
except ImportError:
    PREDICTOR_AVAILABLE = False
    logger.warning("Hardware model predictor module not available")

# Try to import database modules
try:
    import duckdb
    import pandas as pd
    DUCKDB_AVAILABLE = True
    logger.info("DuckDB available for database integration")
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available, database integration will be limited")

class AutomatedHardwareSelection:
    """Main class for automated hardware selection."""
    
    def __init__(self, 
                 database_path: Optional[str] = None,
                 benchmark_dir: str = "./benchmark_results",
                 config_path: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize the automated hardware selection system.
        
        Args:
            database_path: Path to the benchmark database
            benchmark_dir: Directory with benchmark results
            config_path: Path to configuration file
            debug: Enable debug logging
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.config_path = config_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
            
        # Set database path
        if database_path:
            self.database_path = database_path
        elif DUCKDB_AVAILABLE:
            # Check for default database locations
            default_db = self.benchmark_dir / "benchmark_db.duckdb"
            if default_db.exists():
                self.database_path = str(default_db)
            else:
                self.database_path = "./benchmark_database.duckdb"
                logger.info(f"Using default database path: {self.database_path}")
        else:
            self.database_path = None
            
        # Initialize components
        self.hardware_selector = self._initialize_hardware_selector()
        self.predictor = self._initialize_predictor()
        
        # Detect available hardware
        self.available_hardware = self._detect_available_hardware()
        logger.info(f"Detected hardware: {', '.join([hw for hw, available in self.available_hardware.items() if available])}")
        
        # Load compatibility matrix
        self.compatibility_matrix = self._load_compatibility_matrix()
        
    def _initialize_hardware_selector(self) -> Optional[Any]:
        """Initialize the hardware selector component."""
        if HARDWARE_SELECTOR_AVAILABLE:
            try:
                selector = HardwareSelector(
                    database_path=str(self.benchmark_dir),
                    config_path=self.config_path
                )
                logger.info("Hardware selector initialized successfully")
                return selector
            except Exception as e:
                logger.warning(f"Failed to initialize hardware selector: {e}")
        return None
    
    def _initialize_predictor(self) -> Optional[Any]:
        """Initialize the hardware model predictor component."""
        if PREDICTOR_AVAILABLE:
            try:
                predictor = HardwareModelPredictor(
                    benchmark_dir=str(self.benchmark_dir),
                    database_path=self.database_path,
                    config_path=self.config_path
                )
                logger.info("Hardware model predictor initialized successfully")
                return predictor
            except Exception as e:
                logger.warning(f"Failed to initialize hardware model predictor: {e}")
        return None
    
    def _detect_available_hardware(self) -> Dict[str, bool]:
        """Detect available hardware."""
        if self.predictor:
            return self.predictor.available_hardware
        
        # Basic detection if predictor not available
        available_hw = {
            "cpu": True,  # CPU is always available
            "cuda": False,
            "rocm": False,
            "mps": False,
            "openvino": False,
            "webnn": False,
            "webgpu": False
        }
        
        # Try to detect CUDA
        try:
            import torch
            available_hw["cuda"] = torch.cuda.is_available()
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps"):
                available_hw["mps"] = torch.backends.mps.is_available()
        except ImportError:
            pass
        
        # Try to detect ROCm through PyTorch
        try:
            import torch
            if torch.cuda.is_available() and "rocm" in torch.__version__.lower():
                available_hw["rocm"] = True
        except (ImportError, AttributeError):
            pass
        
        # Try to detect OpenVINO
        try:
            import openvino
            available_hw["openvino"] = True
        except ImportError:
            pass
        
        return available_hw
    
    def _load_compatibility_matrix(self) -> Dict[str, Any]:
        """Load the hardware compatibility matrix."""
        if self.hardware_selector:
            return self.hardware_selector.compatibility_matrix
        
        # Basic compatibility matrix if hardware selector not available
        matrix_file = self.benchmark_dir / "hardware_compatibility_matrix.json"
        if matrix_file.exists():
            try:
                with open(matrix_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load compatibility matrix from {matrix_file}: {e}")
                
        # Default matrix
        return {
            "timestamp": str(datetime.datetime.now().isoformat()),
            "hardware_types": ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"],
            "model_families": {
                "embedding": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "medium"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "high"},
                        "mps": {"compatible": True, "performance_rating": "high"},
                        "openvino": {"compatible": True, "performance_rating": "medium"},
                        "webnn": {"compatible": True, "performance_rating": "high"},
                        "webgpu": {"compatible": True, "performance_rating": "medium"}
                    }
                },
                "text_generation": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "low"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "medium"},
                        "mps": {"compatible": True, "performance_rating": "medium"},
                        "openvino": {"compatible": True, "performance_rating": "low"},
                        "webnn": {"compatible": False, "performance_rating": "unknown"},
                        "webgpu": {"compatible": True, "performance_rating": "low"}
                    }
                },
                "vision": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "medium"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "medium"},
                        "mps": {"compatible": True, "performance_rating": "high"},
                        "openvino": {"compatible": True, "performance_rating": "high"},
                        "webnn": {"compatible": True, "performance_rating": "medium"},
                        "webgpu": {"compatible": True, "performance_rating": "medium"}
                    }
                },
                "audio": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "medium"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "medium"},
                        "mps": {"compatible": True, "performance_rating": "medium"},
                        "openvino": {"compatible": True, "performance_rating": "medium"},
                        "webnn": {"compatible": False, "performance_rating": "unknown"},
                        "webgpu": {"compatible": False, "performance_rating": "unknown"}
                    }
                },
                "multimodal": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "low"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": False, "performance_rating": "unknown"},
                        "mps": {"compatible": False, "performance_rating": "unknown"},
                        "openvino": {"compatible": False, "performance_rating": "unknown"},
                        "webnn": {"compatible": False, "performance_rating": "unknown"},
                        "webgpu": {"compatible": False, "performance_rating": "unknown"}
                    }
                }
            }
        }
        
    def select_hardware(self, 
                       model_name: str,
                       model_family: Optional[str] = None,
                       batch_size: int = 1,
                       sequence_length: int = 128,
                       mode: str = "inference",
                       precision: str = "fp32",
                       available_hardware: Optional[List[str]] = None,
                       task_type: Optional[str] = None,
                       distributed: bool = False,
                       gpu_count: int = 1) -> Dict[str, Any]:
        """
        Select optimal hardware for a given model and configuration.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family (if not provided, will be inferred)
            batch_size: Batch size to use
            sequence_length: Sequence length for the model
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            available_hardware: List of available hardware platforms
            task_type: Specific task type
            distributed: Whether to consider distributed training
            gpu_count: Number of GPUs for distributed training
            
        Returns:
            Dict with hardware selection results
        """
        # Use detected available hardware if not specified
        if available_hardware is None:
            available_hardware = [hw for hw, available in self.available_hardware.items() if available]
            
        # Determine model family if not provided
        if model_family is None:
            model_family = self._determine_model_family(model_name)
            logger.info(f"Inferred model family: {model_family}")
        
        # Try predictor first if available
        if self.predictor:
            try:
                # Use task-specific selection if task provided
                if task_type:
                    return self.predictor.predict_optimal_hardware(
                        model_name=model_name,
                        model_family=model_family,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        mode=mode,
                        precision=precision,
                        available_hardware=available_hardware
                    )
                else:
                    return self.predictor.predict_optimal_hardware(
                        model_name=model_name,
                        model_family=model_family,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        mode=mode,
                        precision=precision,
                        available_hardware=available_hardware
                    )
            except Exception as e:
                logger.warning(f"Hardware model predictor selection failed: {e}, falling back to hardware selector")
        
        # Try hardware selector if predictor failed or not available
        if self.hardware_selector:
            try:
                if task_type and distributed and mode == "training":
                    # Use task-specific selection with distributed training
                    training_config = None
                    if precision != "fp32":
                        training_config = {"mixed_precision": True}
                        
                    return self.hardware_selector.select_hardware_for_task(
                        model_family=model_family,
                        model_name=model_name,
                        task_type=task_type,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        available_hardware=available_hardware,
                        distributed=True,
                        gpu_count=gpu_count,
                        training_config=training_config
                    )
                elif task_type:
                    # Use task-specific selection without distributed training
                    return self.hardware_selector.select_hardware_for_task(
                        model_family=model_family,
                        model_name=model_name,
                        task_type=task_type,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        available_hardware=available_hardware
                    )
                else:
                    # Use standard selection
                    return self.hardware_selector.select_hardware(
                        model_family=model_family,
                        model_name=model_name,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        mode=mode,
                        available_hardware=available_hardware,
                        precision=precision
                    )
            except Exception as e:
                logger.warning(f"Hardware selector selection failed: {e}, falling back to basic selection")
                
        # Fallback to basic selection
        return self._basic_hardware_selection(
            model_name=model_name,
            model_family=model_family,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mode=mode,
            precision=precision,
            available_hardware=available_hardware
        )
    
    def _basic_hardware_selection(self,
                                model_name: str,
                                model_family: str,
                                batch_size: int,
                                sequence_length: int,
                                mode: str,
                                precision: str,
                                available_hardware: List[str]) -> Dict[str, Any]:
        """Basic hardware selection as fallback."""
        # Determine model size
        model_size = self._estimate_model_size(model_name)
        model_size_category = "small" if model_size < 100000000 else "medium" if model_size < 1000000000 else "large"
        
        # Simple hardware preference lists by model family
        preferences = {
            "embedding": ["cuda", "mps", "rocm", "openvino", "cpu"],
            "text_generation": ["cuda", "rocm", "mps", "cpu"],
            "vision": ["cuda", "openvino", "rocm", "mps", "cpu"],
            "audio": ["cuda", "cpu", "mps", "rocm"],
            "multimodal": ["cuda", "cpu"]
        }
        
        # Get preferences for this family
        family_preferences = preferences.get(model_family, ["cuda", "cpu"])
        
        # Filter by available hardware
        compatible_hw = [hw for hw in family_preferences if hw in available_hardware]
        
        # Default to CPU if nothing else is available
        if not compatible_hw:
            compatible_hw = ["cpu"]
            
        # Check compatibility from matrix if available
        try:
            matrix_compatible = []
            for hw in compatible_hw:
                hw_compat = self.compatibility_matrix["model_families"][model_family]["hardware_compatibility"].get(hw, {})
                if hw_compat.get("compatible", False):
                    matrix_compatible.append(hw)
            
            if matrix_compatible:
                compatible_hw = matrix_compatible
        except (KeyError, TypeError):
            pass
            
        # Create recommendation
        result = {
            "model_family": model_family,
            "model_name": model_name,
            "model_size": model_size,
            "model_size_category": model_size_category,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "precision": precision,
            "mode": mode,
            "primary_recommendation": compatible_hw[0],
            "fallback_options": compatible_hw[1:],
            "compatible_hardware": compatible_hw,
            "explanation": f"Basic selection based on model family preferences and compatibility matrix",
            "prediction_source": "basic_selection"
        }
        
        return result
    
    def _determine_model_family(self, model_name: str) -> str:
        """Determine model family from model name."""
        model_name_lower = model_name.lower()
        
        if any(term in model_name_lower for term in ["bert", "roberta", "distilbert", "electra", "albert", "mpnet"]):
            return "embedding"
        elif any(term in model_name_lower for term in ["gpt", "llama", "llm", "opt", "bloom", "falcon", "mistral", "phi", "t5", "mt5"]):
            return "text_generation"
        elif any(term in model_name_lower for term in ["vit", "resnet", "convnext", "deit", "beit", "swin"]):
            return "vision"
        elif any(term in model_name_lower for term in ["whisper", "wav2vec", "hubert", "audio"]):
            return "audio"
        elif any(term in model_name_lower for term in ["clip", "llava", "blip", "flava", "multimodal"]):
            return "multimodal"
        else:
            return "embedding"  # Default to embedding for unknown models
    
    def _estimate_model_size(self, model_name: str) -> int:
        """Estimate model size based on model name."""
        model_name_lower = model_name.lower()
        
        # Look for size indicators in the model name
        if "tiny" in model_name_lower:
            return 10000000  # 10M parameters
        elif "small" in model_name_lower:
            return 50000000  # 50M parameters
        elif "base" in model_name_lower:
            return 100000000  # 100M parameters
        elif "large" in model_name_lower:
            return 300000000  # 300M parameters
        elif "xl" in model_name_lower or "huge" in model_name_lower:
            return 1000000000  # 1B parameters
        
        # Check for specific models
        if "bert" in model_name_lower:
            if "tiny" in model_name_lower:
                return 4000000  # 4M parameters
            elif "mini" in model_name_lower:
                return 11000000  # 11M parameters
            elif "small" in model_name_lower:
                return 29000000  # 29M parameters
            elif "base" in model_name_lower:
                return 110000000  # 110M parameters
            elif "large" in model_name_lower:
                return 340000000  # 340M parameters
            else:
                return 110000000  # Default to base size
        elif "t5" in model_name_lower:
            if "small" in model_name_lower:
                return 60000000  # 60M parameters
            elif "base" in model_name_lower:
                return 220000000  # 220M parameters
            elif "large" in model_name_lower:
                return 770000000  # 770M parameters
            elif "3b" in model_name_lower:
                return 3000000000  # 3B parameters
            elif "11b" in model_name_lower:
                return 11000000000  # 11B parameters
            else:
                return 220000000  # Default to base size
        elif "gpt2" in model_name_lower:
            if "small" in model_name_lower or "sm" in model_name_lower:
                return 124000000  # 124M parameters
            elif "medium" in model_name_lower or "med" in model_name_lower:
                return 355000000  # 355M parameters
            elif "large" in model_name_lower or "lg" in model_name_lower:
                return 774000000  # 774M parameters
            elif "xl" in model_name_lower:
                return 1500000000  # 1.5B parameters
            else:
                return 124000000  # Default to small size
        
        # Default size if not recognized
        return 100000000  # 100M parameters
    
    def predict_performance(self,
                          model_name: str,
                          hardware: Union[str, List[str]],
                          model_family: Optional[str] = None,
                          batch_size: int = 1,
                          sequence_length: int = 128,
                          mode: str = "inference",
                          precision: str = "fp32") -> Dict[str, Any]:
        """
        Predict performance metrics for a model on specified hardware.
        
        Args:
            model_name: Name of the model
            hardware: Hardware type or list of hardware types
            model_family: Optional model family (if not provided, will be inferred)
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use
            
        Returns:
            Dict with performance predictions
        """
        # Determine model family if not provided
        if model_family is None:
            model_family = self._determine_model_family(model_name)
            
        # Convert single hardware to list
        if isinstance(hardware, str):
            hardware_list = [hardware]
        else:
            hardware_list = hardware
            
        # Try predictor first if available
        if self.predictor:
            try:
                return self.predictor.predict_performance(
                    model_name=model_name,
                    model_family=model_family,
                    hardware=hardware_list,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    mode=mode,
                    precision=precision
                )
            except Exception as e:
                logger.warning(f"Performance prediction failed: {e}, falling back to basic prediction")
                
        # Fallback to basic prediction
        model_size = self._estimate_model_size(model_name)
        
        result = {
            "model_name": model_name,
            "model_family": model_family,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision,
            "predictions": {}
        }
        
        for hw in hardware_list:
            # Base values depend on hardware type
            if hw == "cuda":
                base_throughput = 100
                base_latency = 10
            elif hw == "rocm":
                base_throughput = 80
                base_latency = 12
            elif hw == "mps":
                base_throughput = 60
                base_latency = 15
            elif hw == "openvino":
                base_throughput = 50
                base_latency = 18
            else:
                base_throughput = 20
                base_latency = 30
            
            # Adjust for batch size
            throughput = base_throughput * (batch_size / (1 + (batch_size / 32)))
            latency = base_latency * (1 + (batch_size / 16))
            
            # Adjust for model size
            size_factor = 1.0
            if model_size > 1000000000:  # > 1B params
                size_factor = 5.0
            elif model_size > 100000000:  # > 100M params
                size_factor = 2.0
            
            throughput /= size_factor
            latency *= size_factor
            
            # Adjust for precision
            if precision == "fp16":
                throughput *= 1.3
                latency /= 1.3
            elif precision == "int8":
                throughput *= 1.6
                latency /= 1.6
            
            result["predictions"][hw] = {
                "throughput": throughput,
                "latency": latency,
                "memory_usage": model_size * 0.004 * batch_size,  # Rough estimate based on model size
                "source": "basic_heuristic"
            }
        
        return result
    
    def get_distributed_training_config(self,
                                      model_name: str,
                                      model_family: Optional[str] = None,
                                      gpu_count: int = 8,
                                      batch_size: int = 8,
                                      max_memory_gb: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a distributed training configuration for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family
            gpu_count: Number of GPUs
            batch_size: Per-GPU batch size
            max_memory_gb: Maximum GPU memory in GB
            
        Returns:
            Dict with distributed training configuration
        """
        # Determine model family if not provided
        if model_family is None:
            model_family = self._determine_model_family(model_name)
            
        # Use hardware selector if available
        if self.hardware_selector:
            try:
                return self.hardware_selector.get_distributed_training_config(
                    model_family=model_family,
                    model_name=model_name,
                    gpu_count=gpu_count,
                    batch_size=batch_size,
                    max_memory_gb=max_memory_gb
                )
            except Exception as e:
                logger.warning(f"Failed to get distributed training config from hardware selector: {e}")
                
        # Basic fallback implementation
        model_size = self._estimate_model_size(model_name)
        model_size_gb = model_size * 4 / (1024 * 1024 * 1024)  # Approximate size in GB (4 bytes per parameter)
        
        # Determine appropriate strategy
        if gpu_count <= 2:
            strategy = "DDP"
        elif gpu_count <= 8:
            if model_size_gb > 10:  # For models larger than 10GB memory footprint
                strategy = "FSDP"
            else:
                strategy = "DDP"
        else:  # More than 8 GPUs
            if model_size_gb > 20:  # For very large models
                strategy = "DeepSpeed"
            elif model_size_gb > 10:
                strategy = "FSDP"
            else:
                strategy = "DDP"
        
        # Base configuration
        config = {
            "model_family": model_family,
            "model_name": model_name,
            "distributed_strategy": strategy,
            "gpu_count": gpu_count,
            "per_gpu_batch_size": batch_size,
            "global_batch_size": batch_size * gpu_count,
            "mixed_precision": True,
            "gradient_accumulation_steps": 1
        }
        
        # Calculate memory requirements
        params_memory_gb = model_size_gb
        activations_memory_gb = model_size_gb * 0.5 * batch_size  # Rough estimate for activations
        optimizer_memory_gb = model_size_gb * 2  # Adam optimizer states

        total_memory_gb = params_memory_gb + activations_memory_gb + optimizer_memory_gb
        memory_per_gpu_gb = total_memory_gb / gpu_count

        # Add memory estimates
        config["estimated_memory"] = {
            "parameters_gb": params_memory_gb,
            "activations_gb": activations_memory_gb,
            "optimizer_gb": optimizer_memory_gb,
            "total_gb": total_memory_gb,
            "per_gpu_gb": memory_per_gpu_gb
        }
        
        # Apply memory optimizations if needed
        if max_memory_gb is not None and memory_per_gpu_gb > max_memory_gb:
            optimizations = []
            
            # 1. Gradient accumulation
            grad_accum_steps = max(1, int(memory_per_gpu_gb / max_memory_gb) + 1)
            config["gradient_accumulation_steps"] = grad_accum_steps
            config["global_batch_size"] = batch_size * gpu_count * grad_accum_steps
            optimizations.append(f"Gradient accumulation (x{grad_accum_steps})")
            memory_per_gpu_gb = (params_memory_gb + (activations_memory_gb / grad_accum_steps) + optimizer_memory_gb) / gpu_count
            
            # 2. Gradient checkpointing
            if memory_per_gpu_gb > max_memory_gb:
                config["gradient_checkpointing"] = True
                memory_per_gpu_gb = (params_memory_gb + (activations_memory_gb / (grad_accum_steps * 3)) + optimizer_memory_gb) / gpu_count
                optimizations.append("Gradient checkpointing")
            
            # 3. Strategy-specific optimizations
            if memory_per_gpu_gb > max_memory_gb:
                if strategy == "DeepSpeed":
                    config["zero_stage"] = 3
                    optimizations.append("ZeRO Stage 3")
                elif strategy == "FSDP":
                    config["cpu_offload"] = True
                    optimizations.append("FSDP CPU Offloading")
            
            config["memory_optimizations"] = optimizations
            config["estimated_memory"]["optimized_per_gpu_gb"] = memory_per_gpu_gb
            
            if memory_per_gpu_gb > max_memory_gb:
                config["memory_warning"] = "Even with optimizations, memory requirements exceed available GPU memory."
        
        return config
    
    def create_hardware_map(self, 
                         model_families: Optional[List[str]] = None,
                         batch_sizes: Optional[List[int]] = None,
                         hardware_platforms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive hardware selection map for different model families, sizes, and batch sizes.
        
        Args:
            model_families: List of model families to include
            batch_sizes: List of batch sizes to test
            hardware_platforms: List of hardware platforms to test
            
        Returns:
            Dict with hardware selection map
        """
        # Use all model families if not specified
        if model_families is None:
            model_families = ["embedding", "text_generation", "vision", "audio", "multimodal"]
        
        # Use hardware selector if available
        if self.hardware_selector:
            try:
                return self.hardware_selector.create_hardware_selection_map(model_families)
            except Exception as e:
                logger.warning(f"Failed to create hardware map with selector: {e}")
                
        # If hardware selector not available or failed, create basic map
        # Define model sizes and batch sizes to test
        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 32, 64]
            
        if hardware_platforms is None:
            hardware_platforms = [hw for hw, available in self.available_hardware.items() if available]
            
        model_sizes = {
            "small": "small",  # Example model name suffix
            "medium": "base",
            "large": "large"
        }
        
        # Create selection map
        selection_map = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_families": {}
        }
        
        for model_family in model_families:
            selection_map["model_families"][model_family] = {
                "model_sizes": {},
                "inference": {
                    "batch_sizes": {}
                },
                "training": {
                    "batch_sizes": {}
                }
            }
            
            # Test different model sizes with default batch size
            for size_category, size_suffix in model_sizes.items():
                model_name = f"{model_family}-{size_suffix}"
                
                # Select hardware for inference and training
                try:
                    inference_result = self.select_hardware(
                        model_name=model_name,
                        model_family=model_family,
                        batch_size=1,
                        mode="inference"
                    )
                    
                    training_result = self.select_hardware(
                        model_name=model_name,
                        model_family=model_family,
                        batch_size=16,
                        mode="training"
                    )
                    
                    # Store results
                    selection_map["model_families"][model_family]["model_sizes"][size_category] = {
                        "inference": {
                            "primary": inference_result["primary_recommendation"],
                            "fallbacks": inference_result["fallback_options"]
                        },
                        "training": {
                            "primary": training_result["primary_recommendation"],
                            "fallbacks": training_result["fallback_options"]
                        }
                    }
                except Exception as e:
                    logger.warning(f"Error testing model size {size_category} for {model_family}: {e}")
            
            # Test different batch sizes with medium-sized model
            model_name = f"{model_family}-base"
            
            for batch_size in batch_sizes:
                try:
                    # Select hardware for inference and training
                    inference_result = self.select_hardware(
                        model_name=model_name,
                        model_family=model_family,
                        batch_size=batch_size,
                        mode="inference"
                    )
                    
                    training_result = self.select_hardware(
                        model_name=model_name,
                        model_family=model_family,
                        batch_size=batch_size,
                        mode="training"
                    )
                    
                    # Store results
                    selection_map["model_families"][model_family]["inference"]["batch_sizes"][str(batch_size)] = {
                        "primary": inference_result["primary_recommendation"],
                        "fallbacks": inference_result["fallback_options"]
                    }
                    
                    selection_map["model_families"][model_family]["training"]["batch_sizes"][str(batch_size)] = {
                        "primary": training_result["primary_recommendation"],
                        "fallbacks": training_result["fallback_options"]
                    }
                except Exception as e:
                    logger.warning(f"Error testing batch size {batch_size} for {model_family}: {e}")
        
        return selection_map
    
    def save_hardware_map(self, output_file: str = "hardware_selection_map.json"):
        """
        Create and save a hardware selection map.
        
        Args:
            output_file: Output file to save the map
        """
        selection_map = self.create_hardware_map()
        
        with open(output_file, 'w') as f:
            json.dump(selection_map, f, indent=2)
        
        logger.info(f"Hardware selection map saved to {output_file}")
    
    def select_optimal_hardware_for_model_list(self, 
                                            models: List[Dict[str, str]],
                                            batch_size: int = 1,
                                            mode: str = "inference") -> Dict[str, Dict[str, str]]:
        """
        Select optimal hardware for multiple models in one go.
        
        Args:
            models: List of model dictionaries with 'name' and 'family' keys
            batch_size: Batch size to use
            mode: "inference" or "training"
            
        Returns:
            Dict mapping model names to hardware recommendations
        """
        results = {}
        
        for model in models:
            model_name = model["name"]
            model_family = model.get("family")
            
            try:
                result = self.select_hardware(
                    model_name=model_name,
                    model_family=model_family,
                    batch_size=batch_size,
                    mode=mode
                )
                
                results[model_name] = {
                    "primary": result["primary_recommendation"],
                    "fallbacks": result["fallback_options"],
                    "explanation": result["explanation"]
                }
            except Exception as e:
                logger.warning(f"Error selecting hardware for {model_name}: {e}")
                results[model_name] = {
                    "primary": "cpu",
                    "fallbacks": [],
                    "error": str(e)
                }
        
        return results
    
    def analyze_model_performance_across_hardware(self, 
                                               model_name: str,
                                               model_family: Optional[str] = None,
                                               batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze model performance across all available hardware for a specific model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dict with performance analysis
        """
        # Determine model family if not provided
        if model_family is None:
            model_family = self._determine_model_family(model_name)
            
        # Set default batch sizes if not provided
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
            
        # Get available hardware
        hardware_platforms = [hw for hw, available in self.available_hardware.items() if available]
        
        # Create analysis structure
        analysis = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware_platforms": hardware_platforms,
            "batch_sizes": batch_sizes,
            "timestamp": datetime.datetime.now().isoformat(),
            "inference": {
                "performance": {},
                "recommendations": {}
            },
            "training": {
                "performance": {},
                "recommendations": {}
            }
        }
        
        # Analyze inference performance
        for batch_size in batch_sizes:
            # Get recommendation
            inference_result = self.select_hardware(
                model_name=model_name,
                model_family=model_family,
                batch_size=batch_size,
                mode="inference"
            )
            
            # Get performance predictions
            performance = self.predict_performance(
                model_name=model_name,
                model_family=model_family,
                hardware=hardware_platforms,
                batch_size=batch_size,
                mode="inference"
            )
            
            # Store results
            analysis["inference"]["recommendations"][str(batch_size)] = {
                "primary": inference_result["primary_recommendation"],
                "fallbacks": inference_result["fallback_options"]
            }
            
            analysis["inference"]["performance"][str(batch_size)] = {}
            for hw, pred in performance["predictions"].items():
                analysis["inference"]["performance"][str(batch_size)][hw] = {
                    "throughput": pred.get("throughput"),
                    "latency": pred.get("latency"),
                    "memory_usage": pred.get("memory_usage")
                }
        
        # Analyze training performance
        for batch_size in batch_sizes:
            # Get recommendation
            training_result = self.select_hardware(
                model_name=model_name,
                model_family=model_family,
                batch_size=batch_size,
                mode="training"
            )
            
            # Get performance predictions
            performance = self.predict_performance(
                model_name=model_name,
                model_family=model_family,
                hardware=hardware_platforms,
                batch_size=batch_size,
                mode="training"
            )
            
            # Store results
            analysis["training"]["recommendations"][str(batch_size)] = {
                "primary": training_result["primary_recommendation"],
                "fallbacks": training_result["fallback_options"]
            }
            
            analysis["training"]["performance"][str(batch_size)] = {}
            for hw, pred in performance["predictions"].items():
                analysis["training"]["performance"][str(batch_size)][hw] = {
                    "throughput": pred.get("throughput"),
                    "latency": pred.get("latency"),
                    "memory_usage": pred.get("memory_usage")
                }
        
        return analysis
    
    def save_model_analysis(self, model_name: str, model_family: Optional[str] = None, output_file: Optional[str] = None):
        """
        Analyze and save performance analysis for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family
            output_file: Output file to save the analysis
        """
        # Perform analysis
        analysis = self.analyze_model_performance_across_hardware(model_name, model_family)
        
        # Determine output file if not provided
        if output_file is None:
            output_file = f"{model_name.replace('/', '_')}_hardware_analysis.json"
            
        # Save analysis
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        logger.info(f"Model analysis saved to {output_file}")
        
        return output_file

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated Hardware Selection System")
    
    # Required parameters
    parser.add_argument("--model", type=str, help="Model name to analyze")
    
    # Optional parameters
    parser.add_argument("--family", type=str, help="Model family/category")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--mode", type=str, choices=["inference", "training"], default="inference", help="Mode")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    parser.add_argument("--hardware", type=str, nargs="+", help="Hardware platforms to consider")
    parser.add_argument("--task", type=str, help="Specific task type")
    parser.add_argument("--distributed", action="store_true", help="Consider distributed training")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs for distributed training")
    
    # File paths
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_results", help="Benchmark results directory")
    parser.add_argument("--database", type=str, help="Path to benchmark database")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Output file path")
    
    # Actions
    parser.add_argument("--create-map", action="store_true", help="Create hardware selection map")
    parser.add_argument("--analyze", action="store_true", help="Analyze model across hardware")
    parser.add_argument("--detect-hardware", action="store_true", help="Detect available hardware")
    parser.add_argument("--distributed-config", action="store_true", help="Generate distributed training configuration")
    parser.add_argument("--max-memory-gb", type=int, help="Maximum GPU memory in GB for distributed training")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="store_true", help="Show version information")
    
    args = parser.parse_args()
    
    # Show version
    if args.version:
        print("Automated Hardware Selection System (Phase 16)")
        print("Version: 1.0.0 (March 2025)")
        print("Part of IPFS Accelerate Python Framework")
        return
    
    # Create hardware selection system
    selector = AutomatedHardwareSelection(
        database_path=args.database,
        benchmark_dir=args.benchmark_dir,
        config_path=args.config,
        debug=args.debug
    )
    
    # Detect hardware
    if args.detect_hardware:
        print("Detected Hardware:")
        for hw_type, available in selector.available_hardware.items():
            status = "✅ Available" if available else "❌ Not available"
            print(f"  - {hw_type}: {status}")
        return
    
    # Create hardware selection map
    if args.create_map:
        output_file = args.output or "hardware_selection_map.json"
        selector.save_hardware_map(output_file)
        print(f"Hardware selection map created and saved to {output_file}")
        return
    
    # Analyze model across hardware
    if args.analyze and args.model:
        output_file = args.output or f"{args.model.replace('/', '_')}_hardware_analysis.json"
        analysis_file = selector.save_model_analysis(args.model, args.family, output_file)
        print(f"Model hardware analysis saved to {analysis_file}")
        return
    
    # Generate distributed training configuration
    if args.distributed_config and args.model:
        if not args.family:
            args.family = selector._determine_model_family(args.model)
            
        config = selector.get_distributed_training_config(
            model_name=args.model,
            model_family=args.family,
            gpu_count=args.gpu_count,
            batch_size=args.batch_size,
            max_memory_gb=args.max_memory_gb
        )
        
        # Print results
        print(f"\nDistributed Training Configuration for {args.model}:")
        print(f"  Model family: {config['model_family']}")
        print(f"  Strategy: {config['distributed_strategy']}")
        print(f"  GPU count: {config['gpu_count']}")
        print(f"  Per-GPU batch size: {config['per_gpu_batch_size']}")
        print(f"  Global batch size: {config['global_batch_size']}")
        print(f"  Mixed precision: {config['mixed_precision']}")
        
        if "gradient_accumulation_steps" in config and config["gradient_accumulation_steps"] > 1:
            print(f"  Gradient accumulation steps: {config['gradient_accumulation_steps']}")
        
        if "gradient_checkpointing" in config and config["gradient_checkpointing"]:
            print("  Gradient checkpointing: Enabled")
        
        if "memory_optimizations" in config and config["memory_optimizations"]:
            print(f"  Memory optimizations: {', '.join(config['memory_optimizations'])}")
        
        print("\nMemory estimates:")
        memory_info = config.get("estimated_memory", {})
        print(f"  Parameters: {memory_info.get('parameters_gb', 0):.2f} GB")
        print(f"  Activations: {memory_info.get('activations_gb', 0):.2f} GB")
        print(f"  Optimizer states: {memory_info.get('optimizer_gb', 0):.2f} GB")
        print(f"  Total memory: {memory_info.get('total_gb', 0):.2f} GB")
        print(f"  Per-GPU memory: {memory_info.get('per_gpu_gb', 0):.2f} GB")
        
        if "optimized_per_gpu_gb" in memory_info:
            print(f"  Optimized per-GPU memory: {memory_info['optimized_per_gpu_gb']:.2f} GB")
        
        if "memory_warning" in config:
            print(f"\nWARNING: {config['memory_warning']}")
            
        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\nConfiguration saved to {args.output}")
            
        return
    
    # Select hardware for model
    if args.model:
        # Get hardware recommendations
        recommendation = selector.select_hardware(
            model_name=args.model,
            model_family=args.family,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            mode=args.mode,
            precision=args.precision,
            available_hardware=args.hardware,
            task_type=args.task,
            distributed=args.distributed,
            gpu_count=args.gpu_count
        )
        
        # Get performance prediction for recommended hardware
        performance = selector.predict_performance(
            model_name=args.model,
            model_family=recommendation["model_family"],
            hardware=recommendation["primary_recommendation"],
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            mode=args.mode,
            precision=args.precision
        )
        
        # Print results
        print(f"\nHardware Recommendation for {args.model}:")
        print(f"  Primary Recommendation: {recommendation['primary_recommendation']}")
        print(f"  Fallback Options: {', '.join(recommendation['fallback_options'])}")
        print(f"  Compatible Hardware: {', '.join(recommendation['compatible_hardware'])}")
        print(f"  Model Family: {recommendation['model_family']}")
        print(f"  Model Size: {recommendation['model_size_category']} ({recommendation['model_size']} parameters)")
        print(f"  Explanation: {recommendation['explanation']}")
        
        # Print performance predictions
        hw = recommendation["primary_recommendation"]
        if hw in performance["predictions"]:
            pred = performance["predictions"][hw]
            print("\nPerformance Prediction:")
            print(f"  Throughput: {pred.get('throughput', 'N/A'):.2f} items/sec")
            print(f"  Latency: {pred.get('latency', 'N/A'):.2f} ms")
            print(f"  Memory Usage: {pred.get('memory_usage', 'N/A'):.2f} MB")
            print(f"  Prediction Source: {pred.get('source', 'N/A')}")
            
        # Save results if output file specified
        if args.output:
            output = {
                "recommendation": recommendation,
                "performance": performance
            }
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
        return
    
    # If no specific action, print help
    parser.print_help()

if __name__ == "__main__":
    main()
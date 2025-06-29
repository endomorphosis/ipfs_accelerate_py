#!/usr/bin/env python
# Advanced Model Compression and Optimization System

import os
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

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

# Import resource pool if available
try:
    from resource_pool import get_global_resource_pool
    HAS_RESOURCE_POOL = True
except ImportError:
    logger.warning("ResourcePool not available. Using standalone optimization.")
    HAS_RESOURCE_POOL = False

# Required imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not available. Some optimizations will be skipped.")
    HAS_TORCH = False

# Try to import transformers
try:
    import transformers
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("Transformers library not available. Model optimizations will be limited.")
    HAS_TRANSFORMERS = False

# Try to import optimum for quantization
try:
    import optimum
    from optimum.intel import OVQuantizer
    HAS_OPTIMUM = True
except ImportError:
    logger.warning("Optimum library not available. Advanced quantization will be skipped.")
    HAS_OPTIMUM = False

# Try to import ONNX & ONNX Runtime
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    logger.warning("ONNX/ONNX Runtime not available. ONNX optimizations will be skipped.")
    HAS_ONNX = False


class ModelCompressor:
    """
    Advanced model compression and optimization system.
    
    Features:
    - Comprehensive model quantization with hardware-specific optimizations
    - Pruning for model size reduction
    - Knowledge distillation for compact models
    - Layer fusion and graph optimizations
    - ONNX conversion and optimization
    - Hardware-specific compression strategies
    - Accuracy validation
    - Integration with ResourcePool
    """
    
    def __init__(self, 
                 output_dir: str = "./optimized_models",
                 use_resource_pool: bool = True,
                 validation_dataset: str = None):
        """
        Initialize the model compressor
        
        Args:
            output_dir: Directory to store optimized models
            use_resource_pool: Whether to use ResourcePool for model loading
            validation_dataset: Optional path to validation dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
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
        
        # Validation dataset
        self.validation_dataset = validation_dataset
        
        # Compatible optimizations by hardware
        self.hardware_optimizations = self._get_hardware_optimizations()
        
        # Results tracking
        self.optimization_results = {}
        
        logger.info(f"ModelCompressor initialized with output directory: {output_dir}")
        logger.info(f"Detected hardware: {list(self.hardware_info.keys())}")
        logger.info(f"Available optimizations: {self._list_available_optimizations()}")

    def _detect_hardware(self) -> Dict:
        """Detect available hardware platforms for optimization"""
        if HAS_HARDWARE_DETECTION:
            try:
                hardware_info = detect_hardware_with_comprehensive_checks()
                logger.info("Hardware detection completed using hardware_detection module")
                return hardware_info
            except Exception as e:
                logger.warning(f"Error using hardware_detection module: {e}. Falling back to basic detection.")
        
        # Basic detection
        hardware_info = {"cpu": True}
        
        if HAS_TORCH:
            # Check CUDA
            if torch.cuda.is_available():
                hardware_info["cuda"] = True
                hardware_info["cuda_device_count"] = torch.cuda.device_count()
                
                # Get CUDA compute capability
                if torch.cuda.device_count() > 0:
                    device_props = torch.cuda.get_device_properties(0)
                    hardware_info["cuda_compute_capability"] = f"{device_props.major}.{device_props.minor}"
            else:
                hardware_info["cuda"] = False
            
            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                hardware_info["mps"] = True
            else:
                hardware_info["mps"] = False
        
        # Check for OpenVINO
        try:
            import openvino
            hardware_info["openvino"] = True
            
            try:
                # Get available OpenVINO devices
                from openvino.runtime import Core
                core = Core()
                hardware_info["openvino_devices"] = core.available_devices
            except Exception as e:
                logger.debug(f"Could not get OpenVINO devices: {e}")
                hardware_info["openvino_devices"] = ["CPU"]
        except ImportError:
            hardware_info["openvino"] = False
        
        # Check for TensorRT
        try:
            import tensorrt
            hardware_info["tensorrt"] = True
        except ImportError:
            hardware_info["tensorrt"] = False
        
        return hardware_info
    
    def _get_hardware_optimizations(self) -> Dict:
        """Get compatible optimizations for each detected hardware"""
        optimizations = {}
        
        # CPU optimizations
        optimizations["cpu"] = [
            "int8_quantization",
            "fp16_conversion",
            "dynamic_quantization",
            "pruning",
            "onnx_conversion",
            "knowledge_distillation",
            "graph_optimization"
        ]
        
        # CUDA optimizations
        if self.hardware_info.get("cuda", False):
            optimizations["cuda"] = [
                "int8_quantization", 
                "fp16_conversion", 
                "pruning",
                "onnx_conversion",
                "tensorrt_conversion" if self.hardware_info.get("tensorrt", False) else None,
                "graph_optimization",
                "knowledge_distillation"
            ]
            optimizations["cuda"] = [opt for opt in optimizations["cuda"] if opt is not None]
            
            # Check compute capability for specific optimizations
            compute_capability = self.hardware_info.get("cuda_compute_capability", "0.0")
            if compute_capability >= "7.0":  # Volta or newer
                optimizations["cuda"].append("tensor_cores_optimization")
        
        # MPS optimizations (Apple Silicon)
        if self.hardware_info.get("mps", False):
            optimizations["mps"] = [
                "fp16_conversion",
                "dynamic_quantization",
                "pruning",
                "onnx_conversion",
                "coreml_conversion"
            ]
        
        # OpenVINO optimizations
        if self.hardware_info.get("openvino", False):
            optimizations["openvino"] = [
                "int8_quantization",
                "fp16_conversion",
                "pruning",
                "graph_optimization",
                "openvino_optimization"
            ]
        
        return optimizations
    
    def _list_available_optimizations(self) -> List[str]:
        """List all available optimization techniques"""
        all_optimizations = set()
        for hardware, opts in self.hardware_optimizations.items():
            all_optimizations.update(opts)
        return sorted(list(all_optimizations))
    
    def load_model(self, model_name_or_path: str, model_type: str = None, device: str = "cpu") -> Any:
        """
        Load a model for optimization
        
        Args:
            model_name_or_path: HuggingFace model name or path to model
            model_type: Type of model (embedding, text_generation, etc.)
            device: Target device for model
            
        Returns:
            Loaded model or None if loading failed
        """
        if not HAS_TRANSFORMERS:
            logger.error("Transformers library not available - cannot load models")
            return None
        
        # Try to infer model family if not provided
        if model_type is None and HAS_MODEL_CLASSIFIER:
            try:
                model_info = classify_model(model_name=model_name_or_path)
                model_type = model_info.get("family", "unknown")
                logger.info(f"Model {model_name_or_path} classified as {model_type}")
            except Exception as e:
                logger.warning(f"Could not classify model: {e}")
        
        # Use resource pool if available
        if self.use_resource_pool:
            try:
                def model_constructor():
                    model = transformers.AutoModel.from_pretrained(model_name_or_path)
                    model.to(device)
                    return model
                
                logger.info(f"Loading model {model_name_or_path} via ResourcePool")
                model = self.resource_pool.get_model(
                    model_type=model_type or "unknown",
                    model_name=model_name_or_path,
                    constructor=model_constructor,
                    hardware_preferences={"device": device}
                )
                return model
            except Exception as e:
                logger.error(f"Error loading model via ResourcePool: {e}")
                # Fall back to direct loading
        
        # Direct loading
        try:
            logger.info(f"Loading model {model_name_or_path} directly")
            model = transformers.AutoModel.from_pretrained(model_name_or_path)
            model.to(device)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def validate_model(self, model: Any, tokenizer: Any, reference_model: Any = None) -> Dict:
        """
        Validate model performance on sample inputs
        
        Args:
            model: Model to validate
            tokenizer: Tokenizer for the model
            reference_model: Optional reference model for comparison
            
        Returns:
            Dictionary with validation results
        """
        if not HAS_TORCH:
            logger.error("PyTorch not available - cannot validate model")
            return {"error": "PyTorch not available"}
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "model_type": str(type(model).__name__),
            "metrics": {}
        }
        
        # Create sample inputs
        try:
            # Generate basic text inputs
            texts = [
                "This is a sample sentence to validate the model.",
                "Another example that helps ensure the model is working correctly.",
                "The quick brown fox jumps over the lazy dog."
            ]
            
            # Tokenize inputs
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to the same device as model
            if hasattr(model, "device"):
                device = model.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run inference with optimized model
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                optimized_outputs = model(**inputs)
                optimized_time = time.time() - start_time
            
            # Run inference with reference model if provided
            if reference_model is not None:
                reference_model.eval()
                with torch.no_grad():
                    start_time = time.time()
                    reference_outputs = reference_model(**inputs)
                    reference_time = time.time() - start_time
                
                # Compare outputs
                if hasattr(optimized_outputs, "last_hidden_state") and hasattr(reference_outputs, "last_hidden_state"):
                    # Compute cosine similarity between embeddings
                    opt_embeds = optimized_outputs.last_hidden_state
                    ref_embeds = reference_outputs.last_hidden_state
                    
                    # Normalize embeddings
                    opt_norm = torch.nn.functional.normalize(opt_embeds, p=2, dim=2)
                    ref_norm = torch.nn.functional.normalize(ref_embeds, p=2, dim=2)
                    
                    # Compute cosine similarity
                    similarities = torch.bmm(
                        opt_norm.view(opt_norm.size(0) * opt_norm.size(1), 1, opt_norm.size(2)),
                        ref_norm.view(ref_norm.size(0) * ref_norm.size(1), ref_norm.size(2), 1)
                    ).squeeze()
                    
                    # Reshape and compute average similarity
                    similarities = similarities.view(opt_norm.size(0), opt_norm.size(1))
                    avg_similarity = torch.mean(similarities).item()
                    
                    validation_results["metrics"]["embedding_similarity"] = avg_similarity
                    validation_results["metrics"]["similarity_threshold"] = 0.97  # Expected threshold for good results
                    validation_results["metrics"]["passes_similarity"] = avg_similarity >= 0.97
                
                # Compare speed
                speedup = reference_time / max(optimized_time, 1e-10)
                validation_results["metrics"]["speedup"] = speedup
                validation_results["metrics"]["reference_time_ms"] = reference_time * 1000
            
            # Record timing information
            validation_results["metrics"]["inference_time_ms"] = optimized_time * 1000
            
            # Record memory usage if on CUDA
            if hasattr(model, "device") and str(model.device).startswith("cuda") and torch.cuda.is_available():
                memory_bytes = torch.cuda.max_memory_allocated()
                validation_results["metrics"]["memory_usage_mb"] = memory_bytes / (1024 * 1024)
            
            logger.info(f"Model validation completed. Inference time: {optimized_time*1000:.2f}ms")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            validation_results["error"] = str(e)
            return validation_results
    
    def quantize_model(self, model: Any, model_name: str, bits: int = 8, 
                      device: str = "cpu", dynamic: bool = True) -> Tuple[Any, Dict]:
        """
        Quantize model to reduce size and improve performance
        
        Args:
            model: Model to quantize
            model_name: Name of the model
            bits: Bit precision (8 for int8, 16 for float16)
            device: Target device for quantized model
            dynamic: Whether to use dynamic quantization
            
        Returns:
            Tuple of (quantized_model, quantization_info)
        """
        if not HAS_TORCH:
            logger.error("PyTorch not available - cannot quantize model")
            return None, {"error": "PyTorch not available"}
        
        quantization_info = {
            "original_model": model_name,
            "bits": bits,
            "method": "dynamic" if dynamic else "static",
            "target_device": device,
            "timestamp": datetime.now().isoformat()
        }
        
        if bits == 8:
            return self._quantize_int8(model, model_name, dynamic, device, quantization_info)
        elif bits == 16:
            return self._quantize_fp16(model, model_name, device, quantization_info)
        else:
            logger.error(f"Unsupported quantization bits: {bits}")
            quantization_info["error"] = f"Unsupported quantization: {bits} bits"
            return None, quantization_info
    
    def _quantize_int8(self, model: Any, model_name: str, dynamic: bool, 
                      device: str, info: Dict) -> Tuple[Any, Dict]:
        """Perform INT8 quantization"""
        try:
            start_time = time.time()
            
            if dynamic:
                # Dynamic INT8 quantization
                logger.info("Applying dynamic INT8 quantization")
                
                # Get list of modules eligible for quantization
                qconfig_mapping = torch.quantization.QConfig(
                    activation=torch.quantization.default_dynamic_quant_observer,
                    weight=torch.quantization.default_weight_observer
                )
                
                # Prepare model for quantization
                model.cpu()  # Dynamic quantization only works on CPU
                model.eval()
                
                # Get example inputs for tracing
                example_inputs = self._get_example_inputs(model, model_name)
                
                # Create quantized model
                try:
                    # Try to use new quantization API
                    from torch.quantization.quantize_fx import prepare_fx, convert_fx
                    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
                    quantized_model = convert_fx(prepared_model)
                except (ImportError, AttributeError, RuntimeError):
                    # Fall back to torch.quantization.quantize_dynamic
                    logger.info("Using legacy dynamic quantization API")
                    quantized_model = torch.quantization.quantize_dynamic(
                        model, 
                        {torch.nn.Linear}, 
                        dtype=torch.qint8
                    )
            else:
                # Static INT8 quantization requires calibration data
                logger.info("Static INT8 quantization requires calibration data")
                logger.info("Falling back to dynamic quantization")
                return self._quantize_int8(model, model_name, True, device, info)
            
            # Move back to original device if needed
            if device != "cpu":
                try:
                    quantized_model = quantized_model.to(device)
                except Exception as e:
                    logger.warning(f"Could not move quantized model to {device}: {e}")
            
            # Update quantization info
            info["success"] = True
            info["duration_seconds"] = time.time() - start_time
            
            # Calculate model size
            original_size_mb = self._get_model_size_mb(model)
            quantized_size_mb = self._get_model_size_mb(quantized_model)
            info["original_size_mb"] = original_size_mb
            info["quantized_size_mb"] = quantized_size_mb
            info["size_reduction_percent"] = (1 - quantized_size_mb / max(original_size_mb, 1e-10)) * 100
            
            logger.info(f"INT8 quantization completed. Size reduction: {info['size_reduction_percent']:.2f}%")
            return quantized_model, info
            
        except Exception as e:
            logger.error(f"Error during INT8 quantization: {e}")
            info["error"] = str(e)
            info["success"] = False
            return None, info
    
    def _quantize_fp16(self, model: Any, model_name: str, device: str, info: Dict) -> Tuple[Any, Dict]:
        """Convert model to FP16 precision"""
        try:
            start_time = time.time()
            
            logger.info("Converting model to FP16 precision")
            
            # Create FP16 model
            model.eval()  # Ensure model is in eval mode
            
            # For CUDA, we can use torch.cuda.amp
            if device.startswith("cuda") and torch.cuda.is_available():
                quantized_model = model.to(device).half()  # Convert to half precision
            else:
                # For CPU, MPS, etc.
                quantized_model = model.to(device)
                # Convert parameters to half precision
                for param in quantized_model.parameters():
                    param.data = param.data.half()
            
            # Update quantization info
            info["success"] = True
            info["duration_seconds"] = time.time() - start_time
            
            # Calculate model size
            original_size_mb = self._get_model_size_mb(model)
            quantized_size_mb = self._get_model_size_mb(quantized_model)
            info["original_size_mb"] = original_size_mb
            info["quantized_size_mb"] = quantized_size_mb
            info["size_reduction_percent"] = (1 - quantized_size_mb / max(original_size_mb, 1e-10)) * 100
            
            logger.info(f"FP16 conversion completed. Size reduction: {info['size_reduction_percent']:.2f}%")
            return quantized_model, info
            
        except Exception as e:
            logger.error(f"Error during FP16 conversion: {e}")
            info["error"] = str(e)
            info["success"] = False
            return None, info
    
    def _get_model_size_mb(self, model: Any) -> float:
        """Calculate model size in MB"""
        if not hasattr(model, "parameters"):
            return 0.0
        
        size_bytes = 0
        for param in model.parameters():
            size_bytes += param.nelement() * param.element_size()
        
        return size_bytes / (1024 * 1024)
    
    def _get_example_inputs(self, model: Any, model_name: str) -> Dict:
        """Create example inputs for model tracing/quantization"""
        # Try to get transformer tokenizer
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Create simple text input
            text = "This is an example input for model quantization"
            encodings = tokenizer(text, return_tensors="pt")
            
            # If model is a sequence classification model, we need to add a label
            if isinstance(model, transformers.PreTrainedModel) and model.config.architectures:
                arch = model.config.architectures[0]
                if "ForSequenceClassification" in arch:
                    encodings["labels"] = torch.zeros(1, dtype=torch.long)
            
            return encodings
        except Exception as e:
            logger.debug(f"Could not create tokenizer-based inputs: {e}")
        
        # Fallback to simple tensor inputs based on model type
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
            return {
                "input_ids": torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long)
            }
        
        # Very basic fallback
        return {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long)
        }
    
    def prune_model(self, model: Any, model_name: str, sparsity: float = 0.3,
                  method: str = "magnitude", device: str = "cpu") -> Tuple[Any, Dict]:
        """
        Prune model to reduce size by removing weights
        
        Args:
            model: Model to prune
            model_name: Name of the model
            sparsity: Fraction of weights to prune (0.0-1.0)
            method: Pruning method ('magnitude', 'random', etc.)
            device: Target device for pruned model
            
        Returns:
            Tuple of (pruned_model, pruning_info)
        """
        if not HAS_TORCH:
            logger.error("PyTorch not available - cannot prune model")
            return None, {"error": "PyTorch not available"}
        
        # Import torch pruning module
        try:
            import torch.nn.utils.prune as prune
        except (ImportError, AttributeError):
            logger.error("PyTorch pruning module not available")
            return None, {"error": "PyTorch pruning module not available"}
        
        pruning_info = {
            "original_model": model_name,
            "sparsity": sparsity,
            "method": method,
            "target_device": device,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            start_time = time.time()
            
            logger.info(f"Pruning model with {sparsity:.1%} sparsity using {method} method")
            
            # Create a copy of the model
            pruned_model = type(model)(model.config) if hasattr(model, "config") else model
            pruned_model.load_state_dict(model.state_dict())
            pruned_model.eval()
            
            # Count parameters before pruning
            n_params_before = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
            
            # Apply pruning to all Linear layers
            for module_name, module in pruned_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if method == "magnitude":
                        prune.l1_unstructured(module, name="weight", amount=sparsity)
                    elif method == "random":
                        prune.random_unstructured(module, name="weight", amount=sparsity)
                    else:
                        logger.warning(f"Unknown pruning method: {method}, falling back to magnitude")
                        prune.l1_unstructured(module, name="weight", amount=sparsity)
                    
                    # Make pruning permanent
                    prune.remove(module, "weight")
            
            # Count parameters after pruning
            n_zeros = 0
            n_params_after = 0
            for param in pruned_model.parameters():
                if param.requires_grad:
                    n_zeros += torch.sum(param == 0).item()
                    n_params_after += param.numel()
            
            actual_sparsity = n_zeros / max(n_params_after, 1)
            
            # Move to target device
            pruned_model = pruned_model.to(device)
            
            # Update pruning info
            pruning_info["success"] = True
            pruning_info["duration_seconds"] = time.time() - start_time
            pruning_info["actual_sparsity"] = actual_sparsity
            pruning_info["n_params_before"] = n_params_before
            pruning_info["n_params_after"] = n_params_after
            pruning_info["n_zeros"] = n_zeros
            
            # Calculate model size
            original_size_mb = self._get_model_size_mb(model)
            pruned_size_mb = self._get_model_size_mb(pruned_model)
            pruning_info["original_size_mb"] = original_size_mb
            pruning_info["pruned_size_mb"] = pruned_size_mb
            pruning_info["size_reduction_percent"] = (1 - pruned_size_mb / max(original_size_mb, 1e-10)) * 100
            
            logger.info(f"Pruning completed. Actual sparsity: {actual_sparsity:.2%}, "
                      f"Size reduction: {pruning_info['size_reduction_percent']:.2f}%")
            
            return pruned_model, pruning_info
            
        except Exception as e:
            logger.error(f"Error during pruning: {e}")
            pruning_info["error"] = str(e)
            pruning_info["success"] = False
            return None, pruning_info
    
    def convert_to_onnx(self, model: Any, model_name: str, device: str = "cpu") -> Tuple[str, Dict]:
        """
        Convert model to ONNX format for inference optimization
        
        Args:
            model: Model to convert
            model_name: Name of the model
            device: Target device for ONNX model
            
        Returns:
            Tuple of (onnx_path, conversion_info)
        """
        if not HAS_ONNX or not HAS_TORCH:
            logger.error("ONNX or PyTorch not available - cannot convert model")
            return "", {"error": "ONNX or PyTorch not available"}
        
        conversion_info = {
            "original_model": model_name,
            "format": "onnx",
            "target_device": device,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            start_time = time.time()
            
            logger.info(f"Converting model to ONNX format")
            
            # Create output directory for ONNX models
            onnx_dir = self.output_dir / "onnx"
            onnx_dir.mkdir(exist_ok=True, parents=True)
            
            # Create safe model name for file
            safe_model_name = model_name.replace("/", "_")
            onnx_path = str(onnx_dir / f"{safe_model_name}.onnx")
            
            # Prepare model for export
            model.eval()
            model.to("cpu")  # ONNX export requires CPU tensors
            
            # Get example inputs for tracing
            dummy_inputs = self._get_example_inputs(model, model_name)
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_inputs,),
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}}
                )
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Optimize ONNX model with ONNX Runtime
            try:
                import onnxruntime.transformers.optimizer as ort_optimizer
                optimized_model_path = str(onnx_dir / f"{safe_model_name}_optimized.onnx")
                
                # Apply optimizations
                optimizer = ort_optimizer.get_optimizer('bert')
                optimization_options = optimizer.get_optimization_options()
                onnx_model = optimizer.optimize(onnx_model)
                
                # Save optimized model
                onnx.save(onnx_model, optimized_model_path)
                logger.info(f"ONNX model optimized and saved to {optimized_model_path}")
                conversion_info["optimized_model_path"] = optimized_model_path
            except (ImportError, Exception) as e:
                logger.warning(f"Could not optimize ONNX model: {e}")
            
            # Create ONNX Runtime inference session
            try:
                if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                
                # Save list of available providers
                conversion_info["available_providers"] = ort.get_available_providers()
                conversion_info["selected_providers"] = providers
                
                # Create session
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
                conversion_info["session_created"] = True
            except Exception as e:
                logger.warning(f"Could not create ONNX Runtime session: {e}")
                conversion_info["session_created"] = False
                conversion_info["session_error"] = str(e)
            
            # Update conversion info
            conversion_info["success"] = True
            conversion_info["duration_seconds"] = time.time() - start_time
            conversion_info["onnx_path"] = onnx_path
            
            # Calculate size information
            original_size_mb = self._get_model_size_mb(model)
            onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            conversion_info["original_size_mb"] = original_size_mb
            conversion_info["onnx_size_mb"] = onnx_size_mb
            
            logger.info(f"ONNX conversion completed. Model saved to {onnx_path}")
            return onnx_path, conversion_info
            
        except Exception as e:
            logger.error(f"Error during ONNX conversion: {e}")
            conversion_info["error"] = str(e)
            conversion_info["success"] = False
            return "", conversion_info
    
    def convert_to_openvino(self, model: Any, model_name: str) -> Tuple[str, Dict]:
        """
        Convert model to OpenVINO format for CPU optimization
        
        Args:
            model: Model to convert
            model_name: Name of the model
            
        Returns:
            Tuple of (openvino_path, conversion_info)
        """
        conversion_info = {
            "original_model": model_name,
            "format": "openvino",
            "target_device": "cpu",  # OpenVINO is primarily for CPU
            "timestamp": datetime.now().isoformat()
        }
        
        # Check for OpenVINO
        if not self.hardware_info.get("openvino", False):
            logger.error("OpenVINO not available - cannot convert model")
            conversion_info["error"] = "OpenVINO not available"
            return "", conversion_info
        
        try:
            # Try to import OpenVINO
            from openvino.runtime import Core, serialize
            
            start_time = time.time()
            
            logger.info(f"Converting model to OpenVINO format")
            
            # Create output directory for OpenVINO models
            openvino_dir = self.output_dir / "openvino"
            openvino_dir.mkdir(exist_ok=True, parents=True)
            
            # Create safe model name for file
            safe_model_name = model_name.replace("/", "_")
            
            # First convert to ONNX as intermediate format
            onnx_path, onnx_info = self.convert_to_onnx(model, model_name)
            
            if not onnx_info.get("success", False):
                logger.error("ONNX conversion failed - cannot convert to OpenVINO")
                conversion_info["error"] = "ONNX conversion failed"
                conversion_info["onnx_error"] = onnx_info.get("error", "Unknown error")
                return "", conversion_info
            
            # Convert ONNX to OpenVINO
            openvino_path = str(openvino_dir / f"{safe_model_name}")
            
            core = Core()
            ov_model = core.read_model(onnx_path)
            
            # Optimize for CPU
            ov_model.reshape({name: shape for name, shape in ov_model.inputs[0].get_partial_shape().items()})
            ov_compiled = core.compile_model(ov_model, "CPU")
            
            # Save OpenVINO model
            serialize(ov_model, f"{openvino_path}.xml", f"{openvino_path}.bin")
            
            # Update conversion info
            conversion_info["success"] = True
            conversion_info["duration_seconds"] = time.time() - start_time
            conversion_info["openvino_path"] = openvino_path
            conversion_info["onnx_path"] = onnx_path
            
            # Calculate size information
            original_size_mb = self._get_model_size_mb(model)
            xml_size_mb = os.path.getsize(f"{openvino_path}.xml") / (1024 * 1024)
            bin_size_mb = os.path.getsize(f"{openvino_path}.bin") / (1024 * 1024)
            conversion_info["original_size_mb"] = original_size_mb
            conversion_info["openvino_size_mb"] = xml_size_mb + bin_size_mb
            
            logger.info(f"OpenVINO conversion completed. Model saved to {openvino_path}")
            return openvino_path, conversion_info
            
        except Exception as e:
            logger.error(f"Error during OpenVINO conversion: {e}")
            conversion_info["error"] = str(e)
            conversion_info["success"] = False
            return "", conversion_info
    
    def run_optimization_pipeline(self, model_name_or_path: str, optimization_types: List[str] = None,
                                target_device: str = "cpu") -> Dict:
        """
        Run comprehensive optimization pipeline on a model
        
        Args:
            model_name_or_path: HuggingFace model name or path to model
            optimization_types: List of optimization techniques to apply
            target_device: Target device for optimized model
            
        Returns:
            Dictionary with optimization results
        """
        # Default optimizations if none specified
        if optimization_types is None:
            # Use device-specific defaults
            device_type = target_device.split(":")[0]  # Extract base device type
            optimization_types = self.hardware_optimizations.get(device_type, ["int8_quantization", "pruning"])
        
        # Prepare results structure
        results = {
            "model_name": model_name_or_path,
            "target_device": target_device,
            "optimizations": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Load the original model
        try:
            original_model = self.load_model(model_name_or_path, device="cpu")  # Start on CPU for compatibility
            
            if original_model is None:
                logger.error(f"Failed to load model {model_name_or_path}")
                results["error"] = "Failed to load model"
                return results
            
            # Get tokenizer for validation
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Validate original model
            original_validation = self.validate_model(original_model, tokenizer)
            results["original_validation"] = original_validation
            
            # Record original model size
            original_size_mb = self._get_model_size_mb(original_model)
            results["original_size_mb"] = original_size_mb
            
            # Keep track of current model for optimization chain
            current_model = original_model
            
            # Apply optimizations in sequence
            for opt_type in optimization_types:
                logger.info(f"Applying optimization: {opt_type}")
                
                if opt_type == "int8_quantization":
                    quantized_model, quant_info = self.quantize_model(
                        current_model, model_name_or_path, bits=8, device=target_device
                    )
                    
                    results["optimizations"]["int8_quantization"] = quant_info
                    
                    if quant_info.get("success", False):
                        # Validate the quantized model
                        quant_validation = self.validate_model(quantized_model, tokenizer, original_model)
                        results["optimizations"]["int8_quantization"]["validation"] = quant_validation
                        
                        # Update current model if successful
                        current_model = quantized_model
                    
                elif opt_type == "fp16_conversion":
                    fp16_model, fp16_info = self.quantize_model(
                        current_model, model_name_or_path, bits=16, device=target_device
                    )
                    
                    results["optimizations"]["fp16_conversion"] = fp16_info
                    
                    if fp16_info.get("success", False):
                        # Validate the FP16 model
                        fp16_validation = self.validate_model(fp16_model, tokenizer, original_model)
                        results["optimizations"]["fp16_conversion"]["validation"] = fp16_validation
                        
                        # Update current model if successful
                        current_model = fp16_model
                
                elif opt_type == "pruning":
                    # Determine appropriate sparsity based on model type
                    sparsity = 0.3  # Default value
                    
                    if HAS_MODEL_CLASSIFIER:
                        try:
                            model_info = classify_model(model_name=model_name_or_path)
                            family = model_info.get("family", "unknown")
                            
                            # Adjust sparsity based on model family
                            if family == "text_generation":
                                sparsity = 0.2  # Less pruning for text generation
                            elif family == "embedding":
                                sparsity = 0.4  # More pruning for embeddings
                        except Exception:
                            pass
                    
                    pruned_model, pruning_info = self.prune_model(
                        current_model, model_name_or_path, sparsity=sparsity, device=target_device
                    )
                    
                    results["optimizations"]["pruning"] = pruning_info
                    
                    if pruning_info.get("success", False):
                        # Validate the pruned model
                        pruning_validation = self.validate_model(pruned_model, tokenizer, original_model)
                        results["optimizations"]["pruning"]["validation"] = pruning_validation
                        
                        # Update current model if successful
                        current_model = pruned_model
                
                elif opt_type == "onnx_conversion":
                    onnx_path, onnx_info = self.convert_to_onnx(
                        current_model, model_name_or_path, device=target_device
                    )
                    
                    results["optimizations"]["onnx_conversion"] = onnx_info
                    
                    # We don't update current_model for format conversions
                
                elif opt_type == "openvino_optimization":
                    if self.hardware_info.get("openvino", False):
                        openvino_path, openvino_info = self.convert_to_openvino(
                            current_model, model_name_or_path
                        )
                        
                        results["optimizations"]["openvino_optimization"] = openvino_info
                    else:
                        logger.warning("OpenVINO not available - skipping optimization")
                        results["optimizations"]["openvino_optimization"] = {
                            "success": False,
                            "error": "OpenVINO not available"
                        }
                
                elif opt_type == "tensorrt_conversion":
                    # TensorRT conversion is more involved and requires ONNX first
                    if not self.hardware_info.get("tensorrt", False):
                        logger.warning("TensorRT not available - skipping optimization")
                        results["optimizations"]["tensorrt_conversion"] = {
                            "success": False,
                            "error": "TensorRT not available"
                        }
                        continue
                    
                    # First convert to ONNX
                    onnx_path, onnx_info = self.convert_to_onnx(
                        current_model, model_name_or_path, device="cuda"
                    )
                    
                    if not onnx_info.get("success", False):
                        logger.error("ONNX conversion failed - cannot convert to TensorRT")
                        results["optimizations"]["tensorrt_conversion"] = {
                            "success": False,
                            "error": "ONNX conversion failed",
                            "onnx_error": onnx_info.get("error", "Unknown error")
                        }
                        continue
                    
                    # TensorRT conversion would go here
                    # This is a placeholder - actual implementation would require TensorRT
                    results["optimizations"]["tensorrt_conversion"] = {
                        "success": False,
                        "error": "TensorRT conversion not fully implemented"
                    }
                
                else:
                    logger.warning(f"Unknown optimization type: {opt_type}")
            
            # Calculate overall stats
            optimized_size_mb = self._get_model_size_mb(current_model)
            results["optimized_size_mb"] = optimized_size_mb
            results["size_reduction_percent"] = (1 - optimized_size_mb / max(original_size_mb, 1e-10)) * 100
            
            # Save optimized model
            if hasattr(current_model, "save_pretrained"):
                try:
                    # Create output directory
                    model_dir = self.output_dir / f"{model_name_or_path.replace('/', '_')}_optimized"
                    model_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Save model
                    current_model.save_pretrained(str(model_dir))
                    
                    # Save tokenizer
                    if tokenizer is not None:
                        tokenizer.save_pretrained(str(model_dir))
                    
                    results["optimized_model_path"] = str(model_dir)
                    logger.info(f"Optimized model saved to {model_dir}")
                except Exception as e:
                    logger.error(f"Error saving optimized model: {e}")
                    results["save_error"] = str(e)
            
            # Save results
            self.optimization_results[model_name_or_path] = results
            
            logger.info(f"Optimization pipeline completed for {model_name_or_path}. "
                       f"Size reduction: {results['size_reduction_percent']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in optimization pipeline: {e}")
            results["error"] = str(e)
            return results
    
    def suggest_optimizations(self, model_name_or_path: str, target_device: str = "cpu") -> Dict:
        """
        Suggest optimal optimization techniques for a given model and device
        
        Args:
            model_name_or_path: HuggingFace model name or path to model
            target_device: Target device for optimized model
            
        Returns:
            Dictionary with optimization suggestions
        """
        suggestions = {
            "model_name": model_name_or_path,
            "target_device": target_device,
            "suggested_optimizations": [],
            "estimated_benefits": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine device type
        device_type = target_device.split(":")[0]  # Extract base device type
        
        # Get available optimizations for this device
        available_optimizations = self.hardware_optimizations.get(device_type, [])
        
        # Classify model if possible
        model_family = "unknown"
        if HAS_MODEL_CLASSIFIER:
            try:
                model_info = classify_model(model_name=model_name_or_path)
                model_family = model_info.get("family", "unknown")
                suggestions["model_family"] = model_family
                logger.info(f"Model {model_name_or_path} classified as {model_family}")
            except Exception as e:
                logger.warning(f"Could not classify model: {e}")
        
        # Device-specific recommendations
        if device_type == "cpu":
            # For CPU, prefer int8 quantization and ONNX/OpenVINO
            primary_suggestions = ["int8_quantization", "onnx_conversion"]
            
            # Add OpenVINO if available
            if self.hardware_info.get("openvino", False):
                primary_suggestions.append("openvino_optimization")
            
            # Model family specific adjustments
            if model_family == "embedding":
                # Embeddings usually tolerate more pruning
                primary_suggestions.append("pruning")
            
            # Filter for available optimizations
            suggestions["suggested_optimizations"] = [opt for opt in primary_suggestions if opt in available_optimizations]
            
            # Estimate benefits
            suggestions["estimated_benefits"] = {
                "int8_quantization": {"size_reduction": "65-75%", "speed_improvement": "minimal"},
                "onnx_conversion": {"size_reduction": "0-10%", "speed_improvement": "20-40%"},
                "openvino_optimization": {"size_reduction": "0-5%", "speed_improvement": "30-60%"}
            }
        
        elif device_type == "cuda":
            # For CUDA, prefer fp16 and pruning
            primary_suggestions = ["fp16_conversion", "pruning"]
            
            # Add TensorRT if available
            if self.hardware_info.get("tensorrt", False):
                primary_suggestions.append("tensorrt_conversion")
            
            # Model family specific adjustments
            if model_family == "text_generation":
                # Text generation benefits from fp16
                suggestions["priority"] = "fp16_conversion"
            
            # Filter for available optimizations
            suggestions["suggested_optimizations"] = [opt for opt in primary_suggestions if opt in available_optimizations]
            
            # Estimate benefits
            suggestions["estimated_benefits"] = {
                "fp16_conversion": {"size_reduction": "40-50%", "speed_improvement": "40-60%"},
                "pruning": {"size_reduction": "20-30%", "speed_improvement": "10-20%"},
                "tensorrt_conversion": {"size_reduction": "0-5%", "speed_improvement": "50-100%"}
            }
        
        elif device_type == "mps":
            # For Apple Silicon (MPS), prefer fp16
            primary_suggestions = ["fp16_conversion", "pruning"]
            
            # Filter for available optimizations
            suggestions["suggested_optimizations"] = [opt for opt in primary_suggestions if opt in available_optimizations]
            
            # Estimate benefits
            suggestions["estimated_benefits"] = {
                "fp16_conversion": {"size_reduction": "40-50%", "speed_improvement": "30-50%"},
                "pruning": {"size_reduction": "20-30%", "speed_improvement": "10-20%"}
            }
        
        else:
            # Default suggestions
            suggestions["suggested_optimizations"] = available_optimizations
            suggestions["estimated_benefits"] = {
                "int8_quantization": {"size_reduction": "65-75%", "speed_improvement": "varies by device"},
                "fp16_conversion": {"size_reduction": "40-50%", "speed_improvement": "varies by device"}
            }
        
        # Always add general recommendations
        suggestions["general_recommendations"] = [
            "Consider smaller model variants if available",
            "Use batching where possible to maximize throughput",
            "Enable caching for repetitive inference"
        ]
        
        logger.info(f"Generated optimization suggestions for {model_name_or_path} on {target_device}")
        return suggestions
    
    def save_results(self, results: Dict, filename: str = None) -> str:
        """
        Save optimization results to file
        
        Args:
            results: Optimization results to save
            filename: Optional filename, if None a timestamped name will be generated
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results.get("model_name", "unknown").replace("/", "_")
            filename = f"optimization_results_{model_name}_{timestamp}.json"
        
        # Make sure it's a full path
        filepath = self.output_dir / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")
        return str(filepath)
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a markdown report from optimization results
        
        Args:
            results: Optimization results
            
        Returns:
            Path to the generated report file
        """
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = results.get("model_name", "unknown").replace("/", "_")
        report_filename = f"optimization_report_{model_name}_{timestamp}.md"
        report_path = self.output_dir / report_filename
        
        # Create report content
        report_content = [
            "# Model Optimization Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Model: {results.get('model_name', 'Unknown')}",
            f"Target Device: {results.get('target_device', 'Unknown')}",
            "",
            "## Summary",
            ""
        ]
        
        # Add summary section
        original_size = results.get("original_size_mb", 0)
        optimized_size = results.get("optimized_size_mb", 0)
        size_reduction = results.get("size_reduction_percent", 0)
        
        report_content.extend([
            f"- Original Model Size: {original_size:.2f} MB",
            f"- Optimized Model Size: {optimized_size:.2f} MB",
            f"- Size Reduction: {size_reduction:.2f}%",
            f"- Optimized Model Path: {results.get('optimized_model_path', 'N/A')}",
            "",
            "## Applied Optimizations",
            ""
        ])
        
        # Add details for each optimization
        optimizations = results.get("optimizations", {})
        for opt_name, opt_results in optimizations.items():
            report_content.extend([
                f"### {opt_name}",
                f"- Success: {opt_results.get('success', False)}",
                f"- Duration: {opt_results.get('duration_seconds', 0):.2f} seconds"
            ])
            
            # Add optimization-specific details
            if opt_name == "int8_quantization" or opt_name == "fp16_conversion":
                report_content.extend([
                    f"- Original Size: {opt_results.get('original_size_mb', 0):.2f} MB",
                    f"- Optimized Size: {opt_results.get('quantized_size_mb', 0):.2f} MB",
                    f"- Size Reduction: {opt_results.get('size_reduction_percent', 0):.2f}%"
                ])
            
            elif opt_name == "pruning":
                report_content.extend([
                    f"- Sparsity: {opt_results.get('sparsity', 0):.2f}",
                    f"- Actual Sparsity: {opt_results.get('actual_sparsity', 0):.2f}",
                    f"- Original Size: {opt_results.get('original_size_mb', 0):.2f} MB",
                    f"- Pruned Size: {opt_results.get('pruned_size_mb', 0):.2f} MB",
                    f"- Size Reduction: {opt_results.get('size_reduction_percent', 0):.2f}%"
                ])
            
            elif opt_name == "onnx_conversion":
                report_content.extend([
                    f"- ONNX Path: {opt_results.get('onnx_path', 'N/A')}",
                    f"- ONNX Size: {opt_results.get('onnx_size_mb', 0):.2f} MB"
                ])
            
            elif opt_name == "openvino_optimization":
                report_content.extend([
                    f"- OpenVINO Path: {opt_results.get('openvino_path', 'N/A')}",
                    f"- OpenVINO Size: {opt_results.get('openvino_size_mb', 0):.2f} MB"
                ])
            
            # Add validation results if available
            if "validation" in opt_results:
                validation = opt_results["validation"]
                
                report_content.append("")
                report_content.append("#### Validation Results")
                
                if "error" in validation:
                    report_content.append(f"- Error: {validation['error']}")
                else:
                    metrics = validation.get("metrics", {})
                    
                    # Add key metrics
                    if "embedding_similarity" in metrics:
                        similarity = metrics["embedding_similarity"]
                        threshold = metrics.get("similarity_threshold", 0.97)
                        passes = metrics.get("passes_similarity", similarity >= threshold)
                        
                        report_content.extend([
                            f"- Embedding Similarity: {similarity:.4f}",
                            f"- Similarity Threshold: {threshold:.4f}",
                            f"- Passes Similarity Test: {passes}"
                        ])
                    
                    if "inference_time_ms" in metrics:
                        inference_time = metrics["inference_time_ms"]
                        report_content.append(f"- Inference Time: {inference_time:.2f} ms")
                    
                    if "speedup" in metrics:
                        speedup = metrics["speedup"]
                        report_content.append(f"- Speedup: {speedup:.2f}x")
                    
                    if "memory_usage_mb" in metrics:
                        memory_usage = metrics["memory_usage_mb"]
                        report_content.append(f"- Memory Usage: {memory_usage:.2f} MB")
            
            # Add error if there was one
            if "error" in opt_results:
                report_content.append(f"- Error: {opt_results['error']}")
            
            report_content.append("")
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write("\n".join(report_content))
        
        logger.info(f"Optimization report generated: {report_path}")
        return str(report_path)


# Function to run model compression from command line
def main():
    parser = argparse.ArgumentParser(description='Compress and optimize models for different hardware platforms')
    parser.add_argument('--model', type=str, required=True, help='Model name or path to optimize')
    parser.add_argument('--device', type=str, default='cpu', help='Target device (cpu, cuda, mps)')
    parser.add_argument('--optimizations', type=str, default=None, 
                        help='Comma-separated list of optimizations to apply (int8_quantization, fp16_conversion, pruning, etc.)')
    parser.add_argument('--output_dir', type=str, default='./optimized_models', help='Output directory')
    parser.add_argument('--suggest', action='store_true', help='Only suggest optimizations without applying them')
    parser.add_argument('--report', action='store_true', help='Generate optimization report')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse optimizations
    optimizations = args.optimizations.split(',') if args.optimizations else None
    
    # Create model compressor
    compressor = ModelCompressor(output_dir=args.output_dir)
    
    # Only suggest optimizations if requested
    if args.suggest:
        suggestions = compressor.suggest_optimizations(args.model, args.device)
        print(f"\nSuggested optimizations for {args.model} on {args.device}:")
        for opt in suggestions["suggested_optimizations"]:
            benefits = suggestions["estimated_benefits"].get(opt, {})
            size_red = benefits.get("size_reduction", "unknown")
            speed_imp = benefits.get("speed_improvement", "unknown")
            print(f"- {opt}:")
            print(f"  Size reduction: {size_red}")
            print(f"  Speed improvement: {speed_imp}")
        
        print("\nGeneral recommendations:")
        for rec in suggestions["general_recommendations"]:
            print(f"- {rec}")
    else:
        # Run optimization pipeline
        results = compressor.run_optimization_pipeline(
            args.model,
            optimization_types=optimizations,
            target_device=args.device
        )
        
        # Save results
        compressor.save_results(results)
        
        # Generate report if requested
        if args.report:
            report_path = compressor.generate_report(results)
            print(f"Optimization report generated: {report_path}")
        
        # Print summary
        print(f"\nOptimization summary for {args.model}:")
        print(f"- Original size: {results.get('original_size_mb', 0):.2f} MB")
        print(f"- Optimized size: {results.get('optimized_size_mb', 0):.2f} MB")
        print(f"- Size reduction: {results.get('size_reduction_percent', 0):.2f}%")
        
        if "optimized_model_path" in results:
            print(f"- Optimized model saved to: {results['optimized_model_path']}")

if __name__ == "__main__":
    main()
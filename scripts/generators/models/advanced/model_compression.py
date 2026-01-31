#!/usr/bin/env python
# Advanced model compression and optimization for IPFS Accelerate framework

import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import framework components with graceful degradation
try:
    from scripts.generators.hardware.hardware_detection import detect_hardware_with_comprehensive_checks
    from model_family_classifier import classify_model, ModelFamilyClassifier
    from scripts.generators.utils.resource_pool import get_global_resource_pool
    HAS_ALL_COMPONENTS = True
except ImportError as e:
    logger.warning(f"Could not import all components: {e}. Some functionality may be limited.")
    HAS_ALL_COMPONENTS = False

# Try to import compression dependencies
try:
    import torch
    from torch.quantization import quantize_dynamic, quantize_qat
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not available. Quantization functionality will be limited.")
    HAS_TORCH = False

# Try to import Hugging Face transformers and optimum
try:
    import transformers
    HAS_TRANSFORMERS = True
    
    try:
        import optimum
        from optimum.onnxruntime import ORTQuantizer
        HAS_OPTIMUM = True
    except ImportError:
        logger.warning("Hugging Face Optimum not available. Advanced quantization will be limited.")
        HAS_OPTIMUM = False
except ImportError:
    logger.warning("Transformers not available. Model compression functionality will be limited.")
    HAS_TRANSFORMERS = False
    HAS_OPTIMUM = False

# Try to import ONNX
try:
    import onnx
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    logger.warning("ONNX Runtime not available. ONNX-based optimizations will be limited.")
    HAS_ONNX = False

# Default supported compression methods
COMPRESSION_METHODS = {
    "quantization": {
        "dynamic": "PyTorch dynamic quantization (post-training, reduces model size, works for CPU inference)",
        "static": "PyTorch static quantization (requires calibration data, better performance than dynamic)",
        "qat": "Quantization-aware training (requires training data and fine-tuning, best accuracy)",
        "onnx": "ONNX quantization (hardware-specific quantization with ONNX Runtime)",
        "int8": "Int8 precision (8-bit weights and activations)",
        "int4": "Int4 precision (4-bit weights, more aggressive compression)",
        "fp16": "Mixed precision (16-bit floating point, good for GPU acceleration)",
        "bf16": "BFloat16 precision (brain floating point format, good for newer GPUs and TPUs)"
    },
    "pruning": {
        "magnitude": "Magnitude-based weight pruning (removes weights below threshold)",
        "structured": "Structured pruning (removes entire channels/neurons)",
        "progressive": "Progressive pruning (gradual pruning during training)"
    },
    "distillation": {
        "standard": "Standard knowledge distillation (train smaller model using larger model)",
        "self": "Self-distillation (improve model using its own predictions)",
        "token": "Token-level distillation (for sequence models)"
    },
    "graph_optimization": {
        "fusion": "Operator fusion (fuse multiple operations for better performance)",
        "constant_folding": "Constant folding (pre-compute constant expressions)",
        "onnx_graph": "ONNX graph optimizations (comprehensive graph-level optimizations)"
    }
}

class ModelCompressor:
    """
    Advanced model compression and optimization toolkit.
    
    This class provides comprehensive tools for compressing and optimizing models,
    with support for quantization, pruning, knowledge distillation, and graph optimization.
    """
    
    def __init__(self, 
                 output_dir: str = "./compressed_models",
                 cache_dir: Optional[str] = None,
                 use_resource_pool: bool = True,
                 detect_hardware: bool = True):
        """
        Initialize the model compressor.
        
        Args:
            output_dir: Directory to save compressed models
            cache_dir: Directory for model caching
            use_resource_pool: Whether to use the ResourcePool for model loading
            detect_hardware: Whether to detect hardware capabilities automatically
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.use_resource_pool = use_resource_pool
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize resource pool if available and requested
        self.resource_pool = None
        if use_resource_pool and 'resource_pool' in sys.modules:
            self.resource_pool = get_global_resource_pool()
        
        # Detect hardware if requested
        self.hardware_info = None
        if detect_hardware:
            self._detect_hardware()
        
        # Set up runtime attributes for tracking
        self.original_model = None
        self.compressed_model = None
        self.model_family = None
        self.model_type = None
        self.compression_stats = {}
        self.validation_results = {}
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware capabilities"""
        try:
            if 'hardware_detection' in sys.modules:
                # Use comprehensive hardware detection if available
                self.hardware_info = detect_hardware_with_comprehensive_checks()
                logger.info(f"Detected hardware capabilities using hardware_detection module")
                
                # Log detected hardware types
                hardware_types = [hw for hw, available in self.hardware_info.items() 
                                 if available and hw in ['cpu', 'cuda', 'mps', 'rocm', 'openvino']]
                logger.info(f"Available hardware: {', '.join(hardware_types)}")
                
                return self.hardware_info
            else:
                # Use basic hardware detection as fallback
                self._basic_hardware_detection()
                return self.hardware_info
        except Exception as e:
            logger.error(f"Error detecting hardware: {e}")
            # Fallback to basic detection
            self._basic_hardware_detection()
            return self.hardware_info
    
    def _basic_hardware_detection(self):
        """Basic hardware detection as fallback"""
        self.hardware_info = {"cpu": True}
        
        try:
            if HAS_TORCH:
                # Check for CUDA
                if torch.cuda.is_available():
                    self.hardware_info["cuda"] = True
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                    logger.info(f"CUDA available: {gpu_name}")
                else:
                    self.hardware_info["cuda"] = False
                
                # Check for MPS (Apple Silicon)
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.hardware_info["mps"] = True
                    logger.info("MPS (Apple Silicon) available")
                else:
                    self.hardware_info["mps"] = False
                
                # Check for ROCm (AMD)
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    self.hardware_info["rocm"] = True
                    logger.info("ROCm (AMD) available")
                else:
                    self.hardware_info["rocm"] = False
            
            # Check for ONNX Runtime and providers
            if HAS_ONNX:
                providers = onnxruntime.get_available_providers()
                logger.info(f"ONNX Runtime providers: {providers}")
                
                # Check for OpenVINO provider
                if 'OpenVINOExecutionProvider' in providers:
                    self.hardware_info["openvino"] = True
                    logger.info("OpenVINO available through ONNX Runtime")
                else:
                    self.hardware_info["openvino"] = False
            
            logger.info(f"Basic hardware detection completed")
        except Exception as e:
            logger.error(f"Error in basic hardware detection: {e}")
    
    def load_model(self, model_name: str, model_type: Optional[str] = None) -> Any:
        """
        Load a model for compression.
        
        Args:
            model_name: The name or path of the model to load
            model_type: Optional model type for classification (auto-detected if None)
            
        Returns:
            The loaded model
        """
        self.model_name = model_name
        
        # Classify model type if not provided
        if model_type is None and 'model_family_classifier' in sys.modules:
            try:
                classification = classify_model(model_name=model_name)
                model_type = classification.get("family")
                self.model_family = classification
                logger.info(f"Model classified as {model_type} (confidence: {classification.get('confidence', 0):.2f})")
            except Exception as e:
                logger.warning(f"Error classifying model: {e}")
                model_type = "unknown"
        
        self.model_type = model_type or "unknown"
        
        # Load model using resource pool if available
        if self.resource_pool and self.use_resource_pool:
            logger.info(f"Loading model {model_name} using ResourcePool")
            
            try:
                # Define model constructor based on model type
                def create_model():
                    if not HAS_TRANSFORMERS:
                        raise ImportError("Transformers library is required for model loading")
                    
                    if self.model_type == "embedding":
                        return transformers.AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                    elif self.model_type == "text_generation":
                        if "t5" in model_name.lower():
                            return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.cache_dir)
                        else:
                            return transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.cache_dir)
                    elif self.model_type == "vision":
                        return transformers.AutoModelForImageClassification.from_pretrained(model_name, cache_dir=self.cache_dir)
                    elif self.model_type == "audio":
                        if "whisper" in model_name.lower():
                            return transformers.WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=self.cache_dir)
                        else:
                            return transformers.AutoModelForAudioClassification.from_pretrained(model_name, cache_dir=self.cache_dir)
                    elif self.model_type == "multimodal":
                        if "clip" in model_name.lower():
                            return transformers.CLIPModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                        else:
                            return transformers.AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                    else:
                        logger.warning(f"Unknown model type: {self.model_type}, using AutoModel")
                        return transformers.AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                
                # Use CPU for initial model loading to enable quantization
                hardware_preferences = {"device": "cpu"}
                
                # Load model
                model = self.resource_pool.get_model(
                    model_type=self.model_type,
                    model_name=model_name,
                    constructor=create_model,
                    hardware_preferences=hardware_preferences
                )
                
                self.original_model = model
                return model
            except Exception as e:
                logger.error(f"Error loading model with ResourcePool: {e}")
                raise
        else:
            # Manual model loading
            logger.info(f"Loading model {model_name} directly")
            
            if not HAS_TRANSFORMERS:
                raise ImportError("Transformers library is required for model loading")
            
            try:
                # Load model based on type
                if self.model_type == "embedding":
                    model = transformers.AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                elif self.model_type == "text_generation":
                    if "t5" in model_name.lower():
                        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.cache_dir)
                    else:
                        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.cache_dir)
                elif self.model_type == "vision":
                    model = transformers.AutoModelForImageClassification.from_pretrained(model_name, cache_dir=self.cache_dir)
                elif self.model_type == "audio":
                    if "whisper" in model_name.lower():
                        model = transformers.WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=self.cache_dir)
                    else:
                        model = transformers.AutoModelForAudioClassification.from_pretrained(model_name, cache_dir=self.cache_dir)
                elif self.model_type == "multimodal":
                    if "clip" in model_name.lower():
                        model = transformers.CLIPModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                    else:
                        model = transformers.AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                else:
                    logger.warning(f"Unknown model type: {self.model_type}, using AutoModel")
                    model = transformers.AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                
                self.original_model = model
                return model
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def get_recommended_compression(self, model_type: Optional[str] = None, target_hardware: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recommended compression methods for the given model type and target hardware.
        
        Args:
            model_type: The model type to get recommendations for (uses loaded model type if None)
            target_hardware: Target hardware for optimized deployment
            
        Returns:
            Dictionary with recommended compression methods and parameters
        """
        model_type = model_type or self.model_type
        if not model_type:
            logger.warning("No model type specified or detected. Using generic recommendations.")
            model_type = "unknown"
        
        # Use current model type if not specified
        if not target_hardware and self.hardware_info:
            # Find best available hardware
            for hw in ["cuda", "rocm", "mps", "openvino", "cpu"]:
                if hw in self.hardware_info and self.hardware_info[hw]:
                    target_hardware = hw
                    break
            target_hardware = target_hardware or "cpu"
        else:
            target_hardware = target_hardware or "cpu"
        
        logger.info(f"Getting compression recommendations for {model_type} model targeting {target_hardware}")
        
        # Basic recommendations based on model type and target hardware
        recommendations = {
            "recommended_methods": [],
            "target_hardware": target_hardware,
            "model_type": model_type,
            "parameters": {}
        }
        
        # General recommendations by hardware type
        if target_hardware == "cuda":
            recommendations["recommended_methods"].append("quantization:fp16")  # Half precision for CUDA
            recommendations["parameters"]["fp16"] = {"enabled": True}
            
            # For newer GPUs, also recommend BF16
            if self.hardware_info and self.hardware_info.get("cuda_capabilities", []) and any(c >= 8.0 for c in self.hardware_info.get("cuda_capabilities", [])):
                recommendations["recommended_methods"].append("quantization:bf16")
                recommendations["parameters"]["bf16"] = {"enabled": True}
        
        elif target_hardware == "cpu":
            # Int8 quantization for CPU
            recommendations["recommended_methods"].append("quantization:dynamic")
            recommendations["parameters"]["dynamic"] = {"dtype": "qint8"}
            
            # ONNX optimizations for CPU
            if HAS_ONNX:
                recommendations["recommended_methods"].append("graph_optimization:onnx_graph")
                recommendations["parameters"]["onnx_graph"] = {"target": "cpu"}
        
        elif target_hardware == "openvino":
            # OpenVINO optimizations
            recommendations["recommended_methods"].append("graph_optimization:onnx_graph")
            recommendations["parameters"]["onnx_graph"] = {"target": "openvino"}
            recommendations["recommended_methods"].append("quantization:int8")
            recommendations["parameters"]["int8"] = {"calibration_required": True}
        
        # Model-specific recommendations
        if model_type == "embedding":
            # Embedding models work well with all quantization approaches
            recommendations["recommended_methods"].append("pruning:magnitude")
            recommendations["parameters"]["magnitude"] = {"sparsity": 0.3}
        
        elif model_type == "text_generation":
            # For text generation, add distillation if it's a large model
            if "llama" in self.model_name.lower() or "gpt" in self.model_name.lower():
                recommendations["recommended_methods"].append("distillation:standard")
                recommendations["parameters"]["standard"] = {"temperature": 2.0}
            
            # Add sequence length optimization
            recommendations["recommended_methods"].append("graph_optimization:fusion")
            recommendations["parameters"]["fusion"] = {"enabled": True}
        
        elif model_type == "vision":
            # Vision models work well with structured pruning
            recommendations["recommended_methods"].append("pruning:structured")
            recommendations["parameters"]["structured"] = {"sparsity": 0.2}
            
            # Add quantization-aware training for vision models
            recommendations["recommended_methods"].append("quantization:qat")
            recommendations["parameters"]["qat"] = {"epochs": 3}
        
        elif model_type == "audio":
            # Audio models often benefit from mixed precision
            if target_hardware in ["cuda", "rocm"]:
                recommendations["recommended_methods"].append("quantization:fp16")
                recommendations["parameters"]["fp16"] = {"enabled": True}
        
        elif model_type == "multimodal":
            # Multimodal models need component-specific optimizations
            recommendations["recommended_methods"].append("graph_optimization:fusion")
            recommendations["parameters"]["fusion"] = {"enabled": True}
            
            # For CLIP-like models with separate encoders
            if "clip" in self.model_name.lower():
                recommendations["recommended_methods"].append("distillation:token")
                recommendations["parameters"]["token"] = {"enabled": True}
        
        logger.info(f"Recommended compression methods: {recommendations['recommended_methods']}")
        return recommendations
    
    def apply_compression(self, methods: List[str], 
                         parameters: Optional[Dict[str, Any]] = None,
                         model: Optional[Any] = None,
                         validation_data: Optional[Any] = None) -> Any:
        """
        Apply compression methods to the model.
        
        Args:
            methods: List of compression methods to apply (format: "category:method")
            parameters: Parameters for each compression method
            model: Model to compress (uses self.original_model if None)
            validation_data: Optional validation data for calibration and accuracy checking
            
        Returns:
            Compressed model
        """
        if not methods:
            logger.warning("No compression methods specified")
            return model or self.original_model
        
        # Use provided model or fall back to original model
        if model is None:
            if self.original_model is None:
                raise ValueError("No model provided and no original model loaded")
            model = self.original_model
        
        parameters = parameters or {}
        compressed_model = model
        
        # Track compression steps and results
        compression_steps = []
        
        # Apply each compression method in sequence
        for method_spec in methods:
            # Parse method specification (category:method)
            parts = method_spec.split(":")
            if len(parts) != 2:
                logger.warning(f"Invalid method specification: {method_spec}. Should be 'category:method'")
                continue
                
            category, method = parts
            
            # Get method parameters
            method_params = parameters.get(method, {})
            
            logger.info(f"Applying {category}:{method} compression")
            
            try:
                # Apply appropriate compression method
                if category == "quantization":
                    compressed_model = self._apply_quantization(compressed_model, method, method_params, validation_data)
                elif category == "pruning":
                    compressed_model = self._apply_pruning(compressed_model, method, method_params, validation_data)
                elif category == "distillation":
                    compressed_model = self._apply_distillation(compressed_model, method, method_params, validation_data)
                elif category == "graph_optimization":
                    compressed_model = self._apply_graph_optimization(compressed_model, method, method_params)
                else:
                    logger.warning(f"Unknown compression category: {category}")
                    continue
                
                # Record compression step
                compression_steps.append({
                    "method": method_spec,
                    "parameters": method_params,
                    "success": compressed_model is not None
                })
                
                # Update compression_stats if the step was successful
                if compressed_model is not None:
                    # Calculate model size if possible
                    original_size = self._get_model_size(model)
                    compressed_size = self._get_model_size(compressed_model)
                    
                    compression_steps[-1]["original_size"] = original_size
                    compression_steps[-1]["compressed_size"] = compressed_size
                    
                    if original_size and compressed_size:
                        compression_ratio = original_size / max(compressed_size, 1)
                        compression_steps[-1]["compression_ratio"] = compression_ratio
                        logger.info(f"  Compression ratio: {compression_ratio:.2f}x (from {original_size/1e6:.2f}MB to {compressed_size/1e6:.2f}MB)")
                
            except Exception as e:
                logger.error(f"Error applying {category}:{method} compression: {e}")
                compression_steps.append({
                    "method": method_spec,
                    "parameters": method_params,
                    "success": False,
                    "error": str(e)
                })
        
        # Store compression stats
        self.compression_stats = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "compression_steps": compression_steps,
            "methods_applied": [step["method"] for step in compression_steps if step["success"]],
            "methods_failed": [step["method"] for step in compression_steps if not step["success"]]
        }
        
        # Calculate overall compression ratio if possible
        original_size = self._get_model_size(model)
        final_size = self._get_model_size(compressed_model)
        
        if original_size and final_size:
            self.compression_stats["original_size"] = original_size
            self.compression_stats["compressed_size"] = final_size
            self.compression_stats["overall_compression_ratio"] = original_size / max(final_size, 1)
            
            logger.info(f"Overall compression ratio: {self.compression_stats['overall_compression_ratio']:.2f}x")
        
        # Store the compressed model
        self.compressed_model = compressed_model
        return compressed_model
    
    def _apply_quantization(self, model, method, params, validation_data=None):
        """Apply quantization to the model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for quantization")
        
        # Make sure model is on CPU for quantization
        if hasattr(model, "to"):
            model = model.to("cpu")
        
        if method == "dynamic":
            logger.info("Applying dynamic quantization")
            
            # Get quantization parameters
            dtype = params.get("dtype", "qint8")
            if dtype == "qint8":
                dtype = torch.qint8
            elif dtype == "quint8":
                dtype = torch.quint8
            
            # Prepare model for dynamic quantization
            try:
                # For transformers models, we need to prepare specific modules
                if hasattr(model, "prepare_for_dynamic_quantization"):
                    model.prepare_for_dynamic_quantization()
                
                # Apply dynamic quantization
                q_modules = ["Linear"]  # Typically linear layers are quantized
                if "modules" in params:
                    q_modules = params["modules"]
                
                # This is a generalized approach - may need model-specific adjustments
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},  # Default to linear layers
                    dtype=dtype
                )
                
                return quantized_model
            except Exception as e:
                logger.error(f"Dynamic quantization failed: {e}")
                # Try model-specific quantization
                try:
                    # For HuggingFace Transformers models
                    from transformers import AutoModelForSequenceClassification
                    
                    # Use optimum if available
                    if HAS_OPTIMUM:
                        from optimum.quantization import PostTrainingQuantConfig, QuantizationConfig
                        
                        config = PostTrainingQuantConfig(
                            approach="dynamic",
                            target_device="cpu"
                        )
                        
                        from optimum.onnxruntime import ORTQuantizer
                        quantizer = ORTQuantizer.from_pretrained(model)
                        quantized_model = quantizer.quantize(
                            quantization_config=config
                        )
                        return quantized_model
                except Exception as sub_e:
                    logger.error(f"Alternative quantization approach also failed: {sub_e}")
                    raise
        
        elif method == "static":
            logger.info("Applying static quantization")
            
            if validation_data is None:
                logger.warning("Static quantization requires calibration data. Falling back to dynamic quantization.")
                return self._apply_quantization(model, "dynamic", params)
            
            try:
                if HAS_OPTIMUM:
                    from optimum.quantization import PostTrainingQuantConfig
                    
                    config = PostTrainingQuantConfig(
                        approach="static",
                        target_device="cpu"
                    )
                    
                    from optimum.onnxruntime import ORTQuantizer
                    quantizer = ORTQuantizer.from_pretrained(model)
                    
                    # Use validation data for calibration
                    quantized_model = quantizer.quantize(
                        quantization_config=config,
                        calibration_dataset=validation_data
                    )
                    return quantized_model
                else:
                    logger.warning("Static quantization requires Optimum library. Falling back to dynamic quantization.")
                    return self._apply_quantization(model, "dynamic", params)
            except Exception as e:
                logger.error(f"Static quantization failed: {e}")
                raise
        
        elif method == "qat":
            logger.info("Applying quantization-aware training")
            
            if validation_data is None:
                logger.warning("Quantization-aware training requires training data. Falling back to dynamic quantization.")
                return self._apply_quantization(model, "dynamic", params)
            
            try:
                # Simple QAT approach using PyTorch's built-in functionality
                # This is just a skeleton - real QAT would need more code
                # Typically this would involve:
                # 1. Preparing the model for QAT (adding fake quantization nodes)
                # 2. Fine-tuning for a few epochs
                # 3. Converting to a quantized model
                
                if HAS_OPTIMUM:
                    # For a more complete implementation, would need to set up actual training
                    logger.warning("Quantization-aware training requires full training setup. Falling back to static quantization.")
                    return self._apply_quantization(model, "static", params, validation_data)
                else:
                    logger.warning("QAT requires Optimum library. Falling back to dynamic quantization.")
                    return self._apply_quantization(model, "dynamic", params)
            except Exception as e:
                logger.error(f"QAT failed: {e}")
                raise
        
        elif method == "onnx":
            logger.info("Applying ONNX quantization")
            
            if not HAS_ONNX:
                logger.warning("ONNX Runtime not available. Falling back to dynamic quantization.")
                return self._apply_quantization(model, "dynamic", params)
            
            try:
                # Convert model to ONNX format
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Export model to ONNX
                    export_path = os.path.join(tmpdirname, "model.onnx")
                    
                    if HAS_TRANSFORMERS:
                        # Use transformers-specific ONNX export
                        from transformers.onnx import export
                        from pathlib import Path
                        
                        # Get appropriate configuration
                        if hasattr(model, "config"):
                            model_kind = model.config.model_type
                            
                            # Use optimum if available for better export
                            if HAS_OPTIMUM:
                                from optimum.onnxruntime import ORTModelForSequenceClassification
                                
                                # Just an example - would need model-specific approach
                                ort_model = ORTModelForSequenceClassification.from_pretrained(
                                    self.model_name, export=True
                                )
                                
                                # Apply quantization
                                quantizer = ORTQuantizer.from_pretrained(ort_model)
                                
                                # Create quantization configuration
                                qconfig = PostTrainingQuantConfig(
                                    approach="dynamic", 
                                    target_device="cpu"
                                )
                                
                                # Apply quantization
                                quantized_model = quantizer.quantize(qconfig)
                                
                                return quantized_model
                            else:
                                logger.warning("Optimum not available for ONNX export. Falling back to dynamic quantization.")
                                return self._apply_quantization(model, "dynamic", params)
                    else:
                        logger.warning("Transformers not available for ONNX export. Falling back to dynamic quantization.")
                        return self._apply_quantization(model, "dynamic", params)
            except Exception as e:
                logger.error(f"ONNX quantization failed: {e}")
                raise
        
        elif method == "fp16":
            logger.info("Applying FP16 (half precision) quantization")
            
            try:
                # For PyTorch models, this is simply converting to half precision
                if not hasattr(model, "half"):
                    logger.warning("Model doesn't support half precision. Returning original model.")
                    return model
                
                # Convert to half precision
                model_fp16 = model.half()
                
                # For CUDA, also verify half precision support
                if self.hardware_info and self.hardware_info.get("cuda", False):
                    # Check if CUDA supports efficient half precision (compute capability >= 7.0)
                    cuda_capable = True
                    if "cuda_capabilities" in self.hardware_info:
                        cuda_capable = any(cap >= 7.0 for cap in self.hardware_info["cuda_capabilities"])
                    
                    if not cuda_capable:
                        logger.warning("GPU may not support efficient FP16 operations. Performance might be impacted.")
                
                return model_fp16
            except Exception as e:
                logger.error(f"FP16 conversion failed: {e}")
                raise
        
        elif method == "bf16":
            logger.info("Applying BF16 (bfloat16) quantization")
            
            try:
                # Check if bfloat16 is supported by the hardware
                if self.hardware_info:
                    bf16_supported = False
                    
                    # CUDA 11+ with compute capability >= 8.0 (Ampere+)
                    if self.hardware_info.get("cuda", False):
                        if "cuda_capabilities" in self.hardware_info:
                            bf16_supported = any(cap >= 8.0 for cap in self.hardware_info["cuda_capabilities"])
                    
                    if not bf16_supported:
                        logger.warning("Hardware may not support BF16. Falling back to FP16.")
                        return self._apply_quantization(model, "fp16", params)
                
                # For PyTorch models, check if bfloat16 is available
                if not hasattr(torch, "bfloat16"):
                    logger.warning("PyTorch doesn't support bfloat16 on this platform. Falling back to FP16.")
                    return self._apply_quantization(model, "fp16", params)
                
                # Convert to bfloat16
                if hasattr(model, "to") and callable(model.to):
                    model_bf16 = model.to(torch.bfloat16)
                    return model_bf16
                else:
                    logger.warning("Model doesn't support bfloat16 conversion. Returning original model.")
                    return model
            except Exception as e:
                logger.error(f"BF16 conversion failed: {e}")
                raise
        
        elif method in ["int8", "int4"]:
            logger.info(f"Applying {method} quantization")
            
            # This is a simplified implementation - real int4/int8 quantization would be more complex
            if HAS_OPTIMUM:
                try:
                    # Use Optimum's quantization capabilities
                    from optimum.quantization import PostTrainingQuantConfig
                    
                    bits = 8 if method == "int8" else 4
                    config = PostTrainingQuantConfig(
                        bits=bits,
                        approach="dynamic",
                        target_device="cpu"
                    )
                    
                    # Create quantizer
                    from optimum.onnxruntime import ORTQuantizer
                    
                    # Export to ONNX first if needed
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        onnx_path = os.path.join(tmpdirname, "model.onnx")
                        
                        # Try to use optimum's from_pretrained if model is a string
                        try:
                            # This works best for HuggingFace model IDs
                            quantizer = ORTQuantizer.from_pretrained(model)
                        except:
                            # For custom models, we need more setup
                            logger.warning(f"{method} quantization requires specialized setup for this model type")
                            logger.warning(f"Falling back to dynamic quantization")
                            return self._apply_quantization(model, "dynamic", params)
                        
                        # Apply quantization
                        quantized_model = quantizer.quantize(
                            quantization_config=config,
                            calibration_dataset=validation_data
                        )
                        
                        return quantized_model
                except Exception as e:
                    logger.error(f"{method} quantization with Optimum failed: {e}")
                    logger.warning(f"Falling back to dynamic quantization")
                    return self._apply_quantization(model, "dynamic", params)
            else:
                logger.warning(f"{method} quantization requires the Optimum library. Falling back to dynamic quantization.")
                return self._apply_quantization(model, "dynamic", params)
        
        else:
            logger.warning(f"Unknown quantization method: {method}")
            return model
    
    def _apply_pruning(self, model, method, params, validation_data=None):
        """Apply pruning to the model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for pruning")
        
        if method == "magnitude":
            logger.info("Applying magnitude-based pruning")
            
            # Get pruning parameters
            sparsity = params.get("sparsity", 0.3)  # Default 30% sparsity
            
            try:
                # Import pruning modules
                import torch.nn.utils.prune as prune
                
                # Make a copy of the model
                pruned_model = model
                
                # Apply pruning to linear layers
                for name, module in pruned_model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=sparsity)
                        prune.remove(module, 'weight')  # Make pruning permanent
                
                return pruned_model
            except Exception as e:
                logger.error(f"Magnitude-based pruning failed: {e}")
                raise
        
        elif method == "structured":
            logger.info("Applying structured pruning")
            
            # Get pruning parameters
            sparsity = params.get("sparsity", 0.2)  # Default 20% sparsity
            
            try:
                # Import pruning modules
                import torch.nn.utils.prune as prune
                
                # Make a copy of the model
                pruned_model = model
                
                # Apply structured pruning to linear layers
                for name, module in pruned_model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)  # Prune rows (output channels)
                        prune.remove(module, 'weight')  # Make pruning permanent
                
                return pruned_model
            except Exception as e:
                logger.error(f"Structured pruning failed: {e}")
                raise
        
        elif method == "progressive":
            logger.info("Applying progressive pruning")
            
            if validation_data is None:
                logger.warning("Progressive pruning requires training data. Falling back to magnitude pruning.")
                return self._apply_pruning(model, "magnitude", params)
            
            # This would be a more complex implementation involving training
            # For simplicity, we fall back to magnitude pruning
            logger.warning("Progressive pruning requires full training setup. Falling back to magnitude pruning.")
            return self._apply_pruning(model, "magnitude", params)
        
        else:
            logger.warning(f"Unknown pruning method: {method}")
            return model
    
    def _apply_distillation(self, model, method, params, validation_data=None):
        """Apply knowledge distillation to the model"""
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError("PyTorch and Transformers are required for distillation")
        
        if validation_data is None:
            logger.warning("Distillation requires training data. Skipping distillation.")
            return model
        
        if method == "standard":
            logger.info("Applying standard knowledge distillation")
            
            # Get distillation parameters
            temperature = params.get("temperature", 2.0)
            
            # This would be a complex implementation involving training a student model
            # For simplicity, we just log the process
            logger.warning("Standard distillation requires training a student model. Not implemented in this simplified version.")
            return model
        
        elif method == "self":
            logger.info("Applying self-distillation")
            
            # This would involve training the model using its own predictions
            logger.warning("Self-distillation requires training setup. Not implemented in this simplified version.")
            return model
        
        elif method == "token":
            logger.info("Applying token-level distillation")
            
            # This would be even more complex for sequence models
            logger.warning("Token-level distillation requires specialized training. Not implemented in this simplified version.")
            return model
        
        else:
            logger.warning(f"Unknown distillation method: {method}")
            return model
    
    def _apply_graph_optimization(self, model, method, params):
        """Apply graph-level optimizations to the model"""
        if method == "fusion":
            logger.info("Applying operator fusion")
            
            # Operator fusion is typically done during ONNX export or with specialized tools
            logger.warning("Operator fusion is best applied during ONNX export. No direct PyTorch implementation.")
            return model
        
        elif method == "constant_folding":
            logger.info("Applying constant folding")
            
            # Constant folding is typically handled by ONNX runtime or specialized tools
            logger.warning("Constant folding is best applied during ONNX export. No direct PyTorch implementation.")
            return model
        
        elif method == "onnx_graph":
            logger.info("Applying ONNX graph optimizations")
            
            if not HAS_ONNX:
                logger.warning("ONNX Runtime not available. Skipping graph optimizations.")
                return model
            
            try:
                # Get target device
                target = params.get("target", "cpu")
                
                # This would involve exporting to ONNX and running optimization passes
                # For this simplified implementation, we just demonstrate the concept
                if HAS_OPTIMUM:
                    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
                    from optimum.onnxruntime.configuration import OptimizationConfig
                    
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Try to convert the model to ONNX format
                        try:
                            # This is a simplified example - real usage would need model-specific handling
                            logger.warning("ONNX graph optimization requires specialized setup and is not fully implemented here.")
                            
                            # The proper implementation would:
                            # 1. Export the model to ONNX format with appropriate dynamic axes
                            # 2. Use ORTOptimizer to apply optimization passes
                            # 3. Load the optimized model
                            
                            return model
                        except Exception as e:
                            logger.error(f"ONNX export failed: {e}")
                            return model
                else:
                    logger.warning("ONNX graph optimization requires the Optimum library. Skipping optimization.")
                    return model
            except Exception as e:
                logger.error(f"ONNX graph optimization failed: {e}")
                return model
        
        else:
            logger.warning(f"Unknown graph optimization method: {method}")
            return model
    
    def _get_model_size(self, model):
        """Get model size in bytes"""
        if model is None:
            return None
        
        try:
            # For PyTorch models
            if HAS_TORCH and hasattr(model, "parameters"):
                return sum(p.numel() * p.element_size() for p in model.parameters())
            
            # For ONNX models
            if HAS_ONNX and hasattr(model, "get_model_size"):
                return model.get_model_size()
            
            # Fall back to object size (less accurate)
            import sys
            return sys.getsizeof(model)
        except Exception as e:
            logger.warning(f"Error calculating model size: {e}")
            return None
    
    def validate_compression(self, model=None, reference_model=None, 
                            validation_data=None, metrics=None) -> Dict[str, Any]:
        """
        Validate compression by comparing performance of compressed and reference models.
        
        Args:
            model: Compressed model to validate (uses self.compressed_model if None)
            reference_model: Reference model for comparison (uses self.original_model if None)
            validation_data: Data for validation
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with validation results
        """
        # Use defaults if not provided
        model = model or self.compressed_model
        reference_model = reference_model or self.original_model
        metrics = metrics or ["latency", "memory", "accuracy"]
        
        if model is None or reference_model is None:
            logger.warning("Both compressed and reference models are required for validation")
            return {}
        
        validation_results = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": {}
        }
        
        # Measure latency
        if "latency" in metrics:
            validation_results["metrics"]["latency"] = self._measure_latency(model, reference_model)
        
        # Measure memory usage
        if "memory" in metrics:
            validation_results["metrics"]["memory"] = self._measure_memory(model, reference_model)
        
        # Measure accuracy (if validation data provided)
        if "accuracy" in metrics and validation_data is not None:
            validation_results["metrics"]["accuracy"] = self._measure_accuracy(model, reference_model, validation_data)
        
        # Store validation results
        self.validation_results = validation_results
        return validation_results
    
    def _measure_latency(self, model, reference_model):
        """Measure inference latency of both models"""
        if not HAS_TORCH:
            logger.warning("PyTorch required for latency measurement")
            return {}
        
        # Create dummy inputs based on model type
        try:
            dummy_inputs = self._create_dummy_inputs(model)
            
            # Measure reference model latency
            ref_latency = self._measure_model_latency(reference_model, dummy_inputs)
            
            # Measure compressed model latency
            compressed_latency = self._measure_model_latency(model, dummy_inputs)
            
            # Calculate speedup
            speedup = ref_latency / compressed_latency if compressed_latency > 0 else float('inf')
            
            return {
                "reference_latency": ref_latency,
                "compressed_latency": compressed_latency,
                "speedup": speedup
            }
        except Exception as e:
            logger.error(f"Error measuring latency: {e}")
            return {}
    
    def _measure_model_latency(self, model, inputs, num_runs=10):
        """Measure average inference latency of a model"""
        if not HAS_TORCH:
            return 0
        
        # Move model to CPU for fair comparison
        if hasattr(model, "to"):
            model = model.to("cpu")
        
        # Move inputs to CPU
        if isinstance(inputs, dict):
            inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to("cpu")
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                if isinstance(inputs, dict):
                    _ = model(**inputs)
                else:
                    _ = model(inputs)
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                
                if isinstance(inputs, dict):
                    _ = model(**inputs)
                else:
                    _ = model(inputs)
                
                latencies.append(time.time() - start_time)
        
        # Return average latency
        return sum(latencies) / len(latencies)
    
    def _measure_memory(self, model, reference_model):
        """Measure memory usage of both models"""
        # Get model sizes
        ref_size = self._get_model_size(reference_model)
        compressed_size = self._get_model_size(model)
        
        # Calculate compression ratio
        compression_ratio = ref_size / compressed_size if compressed_size > 0 else float('inf')
        
        # Measure peak memory usage during inference
        peak_memory = {}
        
        try:
            if HAS_TORCH and torch.cuda.is_available():
                # Create dummy inputs
                dummy_inputs = self._create_dummy_inputs(model)
                
                # Measure reference model peak memory
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                if hasattr(reference_model, "to"):
                    reference_model = reference_model.to("cuda")
                
                with torch.no_grad():
                    if isinstance(dummy_inputs, dict):
                        dummy_inputs_cuda = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in dummy_inputs.items()}
                        _ = reference_model(**dummy_inputs_cuda)
                    else:
                        _ = reference_model(dummy_inputs.to("cuda"))
                
                ref_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                
                # Move back to CPU to free memory
                if hasattr(reference_model, "to"):
                    reference_model = reference_model.to("cpu")
                
                # Measure compressed model peak memory
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                if hasattr(model, "to"):
                    model = model.to("cuda")
                
                with torch.no_grad():
                    if isinstance(dummy_inputs, dict):
                        dummy_inputs_cuda = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in dummy_inputs.items()}
                        _ = model(**dummy_inputs_cuda)
                    else:
                        _ = model(dummy_inputs.to("cuda"))
                
                compressed_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                
                # Move back to CPU
                if hasattr(model, "to"):
                    model = model.to("cpu")
                
                peak_memory = {
                    "reference_peak_mb": ref_peak,
                    "compressed_peak_mb": compressed_peak,
                    "peak_reduction_ratio": ref_peak / compressed_peak if compressed_peak > 0 else float('inf')
                }
        except Exception as e:
            logger.error(f"Error measuring peak memory: {e}")
        
        return {
            "reference_size_bytes": ref_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": compression_ratio,
            "reference_size_mb": ref_size / (1024 * 1024) if ref_size else None,
            "compressed_size_mb": compressed_size / (1024 * 1024) if compressed_size else None,
            "peak_memory": peak_memory
        }
    
    def _measure_accuracy(self, model, reference_model, validation_data):
        """Measure accuracy of both models on validation data"""
        # This would be a more complex implementation depending on the model type
        # For simplicity, we just log the process
        logger.warning("Accuracy measurement requires model-specific implementation and validation data")
        return {}
    
    def _create_dummy_inputs(self, model):
        """Create dummy inputs for the model based on its type"""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for creating dummy inputs")
        
        # For Hugging Face Transformers models
        if HAS_TRANSFORMERS and hasattr(model, "config"):
            try:
                # Get appropriate feature name based on model type
                if self.model_type == "embedding" or self.model_type == "text_generation":
                    # Text input
                    batch_size = 1
                    sequence_length = 32
                    
                    # Get vocab size from config
                    vocab_size = 30522  # Default for BERT
                    if hasattr(model.config, "vocab_size"):
                        vocab_size = model.config.vocab_size
                    
                    # Create dummy input IDs
                    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
                    attention_mask = torch.ones_like(input_ids)
                    
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }
                
                elif self.model_type == "vision":
                    # Image input
                    batch_size = 1
                    channels = 3
                    height = 224
                    width = 224
                    
                    # Create dummy image tensor
                    pixel_values = torch.rand(batch_size, channels, height, width)
                    
                    return {
                        "pixel_values": pixel_values
                    }
                
                elif self.model_type == "audio":
                    # Audio input
                    batch_size = 1
                    sequence_length = 16000  # 1 second at 16kHz
                    
                    # Create dummy audio tensor
                    input_values = torch.rand(batch_size, sequence_length)
                    
                    return {
                        "input_values": input_values
                    }
                
                elif self.model_type == "multimodal":
                    # Multimodal input (e.g., CLIP)
                    batch_size = 1
                    
                    # Text input
                    vocab_size = 49408  # Default for CLIP
                    text_length = 77  # Default for CLIP
                    input_ids = torch.randint(0, vocab_size, (batch_size, text_length))
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Image input
                    channels = 3
                    height = 224
                    width = 224
                    pixel_values = torch.rand(batch_size, channels, height, width)
                    
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values
                    }
                
                else:
                    # Generic input
                    logger.warning(f"No specific dummy input generator for model type {self.model_type}. Using generic input.")
                    batch_size = 1
                    feature_size = 768  # Common feature size
                    
                    return torch.rand(batch_size, feature_size)
            except Exception as e:
                logger.error(f"Error creating dummy inputs for transformer model: {e}")
                # Fall back to generic input
                return torch.rand(1, 768)
        
        # Generic fallback
        logger.warning("Using generic dummy input")
        return torch.rand(1, 768)
    
    def save_compressed_model(self, output_path: Optional[str] = None, format: str = "pytorch"):
        """
        Save the compressed model to disk.
        
        Args:
            output_path: Path to save the model (auto-generated if None)
            format: Format to save the model (pytorch, onnx, etc.)
            
        Returns:
            Path where the model was saved
        """
        if self.compressed_model is None:
            logger.warning("No compressed model available to save")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            # Create directory for model
            timestamp = int(time.time())
            model_dir = os.path.join(self.output_dir, f"{self.model_name.replace('/', '_')}_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Set output path based on format
            if format.lower() == "pytorch":
                output_path = os.path.join(model_dir, "pytorch_model")
            elif format.lower() == "onnx":
                output_path = os.path.join(model_dir, "model.onnx")
            else:
                output_path = os.path.join(model_dir, f"model.{format}")
        
        # Save model in the specified format
        try:
            if format.lower() == "pytorch":
                # Save PyTorch model
                if hasattr(self.compressed_model, "save_pretrained"):
                    # Hugging Face transformers model
                    self.compressed_model.save_pretrained(output_path)
                elif HAS_TORCH and isinstance(self.compressed_model, torch.nn.Module):
                    # Standard PyTorch model
                    torch.save(self.compressed_model.state_dict(), output_path + ".pt")
                else:
                    logger.warning(f"Don't know how to save model of type {type(self.compressed_model)}")
                    return None
            
            elif format.lower() == "onnx":
                # Save as ONNX model
                if not HAS_ONNX:
                    logger.warning("ONNX not available. Cannot save in ONNX format.")
                    return None
                
                # Get dummy inputs for ONNX export
                dummy_inputs = self._create_dummy_inputs(self.compressed_model)
                
                # Export to ONNX
                with torch.no_grad():
                    if isinstance(dummy_inputs, dict):
                        # For transformers models, use specialized export
                        if HAS_OPTIMUM:
                            from optimum.exporters import OnnxConfig
                            
                            # This is a simplified example - real export would need model-specific configuration
                            logger.warning("ONNX export requires specialized setup for transformers models")
                            logger.warning("Falling back to PyTorch format")
                            
                            # In practice, you would:
                            # 1. Create an appropriate OnnxConfig
                            # 2. Use the configuration for proper export with dynamic axes
                            return self.save_compressed_model(output_path, "pytorch")
                        else:
                            logger.warning("Optimum not available for proper ONNX export of transformers models")
                            logger.warning("Falling back to PyTorch format")
                            return self.save_compressed_model(output_path, "pytorch")
                    else:
                        # For standard PyTorch models
                        torch.onnx.export(
                            self.compressed_model,
                            dummy_inputs,
                            output_path,
                            export_params=True,
                            opset_version=12,
                            do_constant_folding=True
                        )
            
            else:
                logger.warning(f"Unsupported format: {format}")
                return None
            
            logger.info(f"Compressed model saved to {output_path}")
            
            # Also save compression stats
            stats_path = os.path.join(os.path.dirname(output_path), "compression_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(self.compression_stats, f, indent=2)
            
            # Save validation results if available
            if self.validation_results:
                validation_path = os.path.join(os.path.dirname(output_path), "validation_results.json")
                with open(validation_path, 'w') as f:
                    json.dump(self.validation_results, f, indent=2)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error saving compressed model: {e}")
            return None
    
    def generate_compression_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report about the compression process.
        
        Args:
            output_path: Path to save the report (auto-generated if None)
            
        Returns:
            Path to the generated report
        """
        if not self.compression_stats:
            logger.warning("No compression stats available for report generation")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            report_dir = os.path.join(self.output_dir, f"{self.model_name.replace('/', '_')}_{timestamp}_report")
            os.makedirs(report_dir, exist_ok=True)
            output_path = os.path.join(report_dir, "compression_report.md")
        
        try:
            # Generate report content
            report_lines = []
            
            # Add header
            report_lines.append(f"# Model Compression Report\n")
            report_lines.append(f"- **Model:** {self.model_name}")
            report_lines.append(f"- **Type:** {self.model_type}")
            report_lines.append(f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Add compression summary
            report_lines.append("## Compression Summary\n")
            
            if "overall_compression_ratio" in self.compression_stats:
                report_lines.append(f"- **Overall Compression Ratio:** {self.compression_stats['overall_compression_ratio']:.2f}x")
            if "original_size" in self.compression_stats and "compressed_size" in self.compression_stats:
                original_mb = self.compression_stats["original_size"] / (1024 * 1024)
                compressed_mb = self.compression_stats["compressed_size"] / (1024 * 1024)
                report_lines.append(f"- **Original Size:** {original_mb:.2f} MB")
                report_lines.append(f"- **Compressed Size:** {compressed_mb:.2f} MB")
                report_lines.append(f"- **Size Reduction:** {(1 - compressed_mb/original_mb) * 100:.1f}%")
            
            # Add applied methods
            report_lines.append(f"- **Methods Applied:** {', '.join(self.compression_stats.get('methods_applied', []))}")
            if self.compression_stats.get('methods_failed', []):
                report_lines.append(f"- **Methods Failed:** {', '.join(self.compression_stats.get('methods_failed', []))}")
            
            report_lines.append("\n## Compression Steps\n")
            
            # Add detailed steps
            for i, step in enumerate(self.compression_stats.get("compression_steps", [])):
                report_lines.append(f"### Step {i+1}: {step['method']}\n")
                
                # Add parameters
                report_lines.append("#### Parameters:\n")
                for param_name, param_value in step.get("parameters", {}).items():
                    report_lines.append(f"- **{param_name}:** {param_value}")
                
                # Add results
                report_lines.append("\n#### Results:\n")
                report_lines.append(f"- **Success:** {'Yes' if step.get('success', False) else 'No'}")
                
                if step.get("success", False):
                    if "compression_ratio" in step:
                        report_lines.append(f"- **Compression Ratio:** {step['compression_ratio']:.2f}x")
                    if "original_size" in step and "compressed_size" in step:
                        original_mb = step["original_size"] / (1024 * 1024)
                        compressed_mb = step["compressed_size"] / (1024 * 1024)
                        report_lines.append(f"- **Size Before:** {original_mb:.2f} MB")
                        report_lines.append(f"- **Size After:** {compressed_mb:.2f} MB")
                else:
                    if "error" in step:
                        report_lines.append(f"- **Error:** {step['error']}")
                
                report_lines.append("")
            
            # Add validation results if available
            if self.validation_results:
                report_lines.append("## Validation Results\n")
                
                metrics = self.validation_results.get("metrics", {})
                
                # Add latency
                if "latency" in metrics:
                    latency = metrics["latency"]
                    report_lines.append("### Latency\n")
                    report_lines.append(f"- **Reference Model:** {latency.get('reference_latency', 0):.4f} seconds")
                    report_lines.append(f"- **Compressed Model:** {latency.get('compressed_latency', 0):.4f} seconds")
                    report_lines.append(f"- **Speedup:** {latency.get('speedup', 0):.2f}x\n")
                
                # Add memory usage
                if "memory" in metrics:
                    memory = metrics["memory"]
                    report_lines.append("### Memory Usage\n")
                    report_lines.append(f"- **Reference Model Size:** {memory.get('reference_size_mb', 0):.2f} MB")
                    report_lines.append(f"- **Compressed Model Size:** {memory.get('compressed_size_mb', 0):.2f} MB")
                    report_lines.append(f"- **Compression Ratio:** {memory.get('compression_ratio', 0):.2f}x\n")
                    
                    # Add peak memory if available
                    peak_memory = memory.get("peak_memory", {})
                    if peak_memory:
                        report_lines.append("#### Peak Memory Usage (CUDA)\n")
                        report_lines.append(f"- **Reference Model:** {peak_memory.get('reference_peak_mb', 0):.2f} MB")
                        report_lines.append(f"- **Compressed Model:** {peak_memory.get('compressed_peak_mb', 0):.2f} MB")
                        report_lines.append(f"- **Reduction Ratio:** {peak_memory.get('peak_reduction_ratio', 0):.2f}x\n")
                
                # Add accuracy if available
                if "accuracy" in metrics and metrics["accuracy"]:
                    accuracy = metrics["accuracy"]
                    report_lines.append("### Accuracy\n")
                    for metric, value in accuracy.items():
                        report_lines.append(f"- **{metric}:** {value}\n")
            
            # Add hardware information
            if self.hardware_info:
                report_lines.append("## Hardware Information\n")
                
                # Add CPU info
                report_lines.append("### CPU\n")
                if "cpu" in self.hardware_info and isinstance(self.hardware_info["cpu"], dict):
                    cpu_info = self.hardware_info["cpu"]
                    for key, value in cpu_info.items():
                        report_lines.append(f"- **{key}:** {value}")
                else:
                    report_lines.append("- CPU information not available in detail\n")
                
                # Add GPU info if available
                if "cuda" in self.hardware_info and self.hardware_info["cuda"]:
                    report_lines.append("\n### GPU\n")
                    if "cuda_devices" in self.hardware_info:
                        for i, device in enumerate(self.hardware_info["cuda_devices"]):
                            report_lines.append(f"- **Device {i}:** {device['name']}")
                            report_lines.append(f"  - **Memory:** {device['total_memory']} GB")
                            report_lines.append(f"  - **Compute Capability:** {device['compute_capability']}")
                    else:
                        report_lines.append("- CUDA available but detailed information not available\n")
            
            # Add recommendations
            report_lines.append("## Recommendations\n")
            report_lines.append("Based on the compression results, we recommend:\n")
            
            # Generate recommendations based on results
            if self.compression_stats.get("overall_compression_ratio", 0) > 3:
                report_lines.append("- ✅ The model has been significantly compressed. Consider deploying this compressed version.")
            else:
                report_lines.append("- ⚠️ The compression ratio is modest. Consider trying additional compression methods.")
            
            if self.validation_results and "latency" in self.validation_results.get("metrics", {}):
                speedup = self.validation_results["metrics"]["latency"].get("speedup", 0)
                if speedup > 1.5:
                    report_lines.append("- ✅ The compressed model shows significant speedup. Good for latency-critical applications.")
                else:
                    report_lines.append("- ⚠️ Limited speedup observed. Consider exploring different compression methods.")
            
            # Save report
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Compression report saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating compression report: {e}")
            return None

def get_available_compression_methods():
    """Get information about available compression methods"""
    return COMPRESSION_METHODS

def main():
    """Main function for model compression from command line"""
    parser = argparse.ArgumentParser(description="Advanced model compression and optimization toolkit")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./compressed_models", help="Output directory")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for model downloads")
    parser.add_argument("--model-type", type=str, help="Model type (embedding, text_generation, vision, audio, multimodal)")
    parser.add_argument("--methods", type=str, nargs="+", help="Compression methods to apply (e.g., quantization:dynamic pruning:magnitude)")
    parser.add_argument("--params", type=str, help="JSON string with parameters for compression methods")
    parser.add_argument("--target-hardware", type=str, default="cpu", help="Target hardware (cpu, cuda, mps, openvino)")
    parser.add_argument("--list-methods", action="store_true", help="List available compression methods")
    parser.add_argument("--validate", action="store_true", help="Validate compression with basic metrics")
    parser.add_argument("--format", type=str, default="pytorch", help="Output format (pytorch, onnx)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--recommend", action="store_true", help="Get compression recommendations without applying them")
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # List available methods if requested
    if args.list_methods:
        methods = get_available_compression_methods()
        print("\nAvailable Compression Methods:\n")
        
        for category, category_methods in methods.items():
            print(f"## {category.title()}:\n")
            for method, description in category_methods.items():
                print(f"  - {category}:{method}: {description}")
            print()
        return
    
    # Create compressor
    compressor = ModelCompressor(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    if args.recommend:
        # Load model for classification
        model = compressor.load_model(args.model, args.model_type)
        
        # Get recommendations
        recommendations = compressor.get_recommended_compression(args.model_type, args.target_hardware)
        
        # Print recommendations
        print("\nCompression Recommendations:\n")
        print(f"Model: {args.model}")
        print(f"Type: {compressor.model_type}")
        print(f"Target Hardware: {args.target_hardware}\n")
        
        print("Recommended Methods:")
        for method in recommendations["recommended_methods"]:
            print(f"  - {method}")
        
        print("\nRecommended Parameters:")
        for method, params in recommendations["parameters"].items():
            print(f"  - {method}: {params}")
        
        # Generate command
        methods_str = " ".join(recommendations["recommended_methods"])
        params_str = json.dumps(recommendations["parameters"])
        
        print(f"\nTo apply these recommendations, run:")
        print(f"python model_compression.py --model {args.model} --methods {methods_str} --params '{params_str}' --target-hardware {args.target_hardware}")
        
        return
    
    # Load model
    model = compressor.load_model(args.model, args.model_type)
    logger.info(f"Model loaded: {args.model}")
    
    # Parse parameters if provided
    parameters = {}
    if args.params:
        try:
            parameters = json.loads(args.params)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing parameters JSON: {e}")
            return
    
    # Get methods to apply
    methods = args.methods
    if not methods:
        # Get recommendations if no methods specified
        recommendations = compressor.get_recommended_compression(compressor.model_type, args.target_hardware)
        methods = recommendations["recommended_methods"]
        
        if not parameters:
            parameters = recommendations["parameters"]
        
        logger.info(f"Using recommended methods: {methods}")
    
    # Apply compression
    compressed_model = compressor.apply_compression(methods, parameters)
    
    # Validate if requested
    if args.validate:
        validation_results = compressor.validate_compression()
        logger.info(f"Validation results: {validation_results}")
    
    # Save compressed model
    output_path = compressor.save_compressed_model(format=args.format)
    
    # Generate report
    report_path = compressor.generate_compression_report()
    
    if output_path:
        logger.info(f"Compressed model saved to {output_path}")
    if report_path:
        logger.info(f"Compression report saved to {report_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Unified Web Framework for ML Acceleration (August 2025)

This module provides a unified framework for integrating all web platform components,
creating a cohesive system for deploying ML models to web browsers with optimal performance.

Key features:
- Unified API for all web platform components
- Automatic feature detection and adaptation
- Standardized interfaces for model deployment
- Cross-component integration and optimization
- Progressive enhancement with fallback mechanisms
- Comprehensive configuration system
- Support for all major browsers and platforms

Usage:
    from fixed_web_platform.unified_web_framework import (
        WebPlatformAccelerator,
        create_web_endpoint,
        get_optimal_config
    )
    
    # Create web accelerator with automatic detection
    accelerator = WebPlatformAccelerator(
        model_path="models/bert-base",
        model_type="text",
        auto_detect=True  # Automatically detect and use optimal features
    )
    
    # Create inference endpoint
    endpoint = accelerator.create_endpoint()
    
    # Run inference
    result = endpoint({"text": "Example input text"})
    
    # Get detailed performance metrics
    metrics = accelerator.get_performance_metrics()
"""

import os
import sys
import json
import time
import logging
import platform
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Import web platform components
from fixed_web_platform.browser_capability_detector import BrowserCapabilityDetector
from fixed_web_platform.progressive_model_loader import ProgressiveModelLoader
from fixed_web_platform.webgpu_quantization import setup_4bit_inference
from fixed_web_platform.webgpu_ultra_low_precision import setup_ultra_low_precision
from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference
from fixed_web_platform.webgpu_wasm_fallback import setup_wasm_fallback
from fixed_web_platform.webgpu_shader_registry import WebGPUShaderRegistry
from fixed_web_platform.safari_webgpu_handler import SafariWebGPUHandler

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebPlatformAccelerator:
    """
    Unified framework for accelerating ML models on web platforms.
    
    This class provides a cohesive interface for all web platform components,
    integrating features like WebGPU acceleration, quantization, progressive loading,
    and WebAssembly fallback into a single comprehensive system.
    """
    
    def __init__(self, 
                 model_path: str, 
                 model_type: str,
                 config: Dict[str, Any] = None,
                 auto_detect: bool = True):
        """
        Initialize the web platform accelerator.
        
        Args:
            model_path: Path to the model
            model_type: Type of model (text, vision, audio, multimodal)
            config: Configuration dictionary (if None, uses auto-detection)
            auto_detect: Whether to automatically detect optimal features
        """
        self.model_path = model_path
        self.model_type = model_type
        self.config = config or {}
        
        # Initialize metrics tracking
        self._perf_metrics = {
            "initialization_time_ms": 0,
            "first_inference_time_ms": 0,
            "average_inference_time_ms": 0,
            "memory_usage_mb": 0,
            "feature_usage": {}
        }
        
        self._initialization_start = time.time()
        
        # Auto-detect capabilities if requested
        if auto_detect:
            self._detect_capabilities()
        
        # Initialize components based on configuration
        self._initialize_components()
        
        # Track initialization time
        self._perf_metrics["initialization_time_ms"] = (time.time() - self._initialization_start) * 1000
        logger.info(f"WebPlatformAccelerator initialized in {self._perf_metrics['initialization_time_ms']:.2f}ms")
    
    def _detect_capabilities(self):
        """
        Detect browser capabilities and set optimal configuration.
        """
        logger.info("Detecting browser capabilities...")
        
        # Create detector
        detector = BrowserCapabilityDetector()
        capabilities = detector.get_capabilities()
        
        # Get optimization profile
        profile = detector.get_optimization_profile()
        
        # Update configuration with detected capabilities
        self.config.update({
            "browser_capabilities": capabilities,
            "optimization_profile": profile,
            
            # Core acceleration features
            "use_webgpu": capabilities["webgpu"]["available"],
            "use_webnn": capabilities["webnn"]["available"],
            "use_wasm": True,  # Always have WASM as fallback
            
            # WebGPU features
            "compute_shaders": capabilities["webgpu"]["compute_shaders"],
            "shader_precompilation": capabilities["webgpu"]["shader_precompilation"],
            
            # Precision settings
            "quantization": profile["precision"]["default"],
            "ultra_low_precision": profile["precision"]["ultra_low_precision_enabled"],
            "adaptive_precision": True if profile["precision"]["ultra_low_precision_enabled"] else False,
            
            # Loading features
            "progressive_loading": profile["loading"]["progressive_loading"],
            "parallel_loading": profile["loading"]["parallel_loading"],
            
            # Browser-specific settings
            "browser": capabilities["browser_info"]["name"],
            "browser_version": capabilities["browser_info"]["version"],
            
            # Other features
            "streaming_inference": profile["loading"]["progressive_loading"] and self.model_type == "text",
            "kv_cache_optimization": profile["memory"]["kv_cache_optimization"] and self.model_type == "text"
        })
        
        # Set workgroup size based on browser and hardware
        self.config["workgroup_size"] = profile["compute"]["workgroup_size"]
        
        # Set streaming parameters for text generation models
        if self.model_type == "text":
            self.config["latency_optimized"] = True
            self.config["adaptive_batch_size"] = True
        
        # Set model-specific optimizations
        self._set_model_specific_config()
        
        logger.info(f"Using {self.config['browser']} {self.config['browser_version']} with "
                   f"WebGPU: {self.config['use_webgpu']}, WebNN: {self.config['use_webnn']}")
    
    def _set_model_specific_config(self):
        """
        Set model-specific configuration options.
        """
        if self.model_type == "text":
            # Text models (BERT, T5, etc.)
            if "bert" in self.model_path.lower():
                self.config.setdefault("quantization", 4)  # BERT works well with 4-bit
                self.config.setdefault("shader_precompilation", True)  # BERT benefits from shader precompilation
            elif "t5" in self.model_path.lower():
                self.config.setdefault("quantization", 4)  # T5 works well with 4-bit
                self.config.setdefault("shader_precompilation", True)
            elif "llama" in self.model_path.lower() or "gpt" in self.model_path.lower():
                self.config.setdefault("quantization", 4)  # Use 4-bit for LLMs
                self.config.setdefault("kv_cache_optimization", True)
                self.config.setdefault("streaming_inference", True)
        
        elif self.model_type == "vision":
            # Vision models (ViT, ResNet, etc.)
            self.config.setdefault("shader_precompilation", True)  # Vision models benefit from shader precompilation
            if "vit" in self.model_path.lower():
                self.config.setdefault("quantization", 4)  # ViT works well with 4-bit
            elif "resnet" in self.model_path.lower():
                self.config.setdefault("quantization", 4)  # ResNet works well with 4-bit
        
        elif self.model_type == "audio":
            # Audio models (Whisper, Wav2Vec2, etc.)
            self.config.setdefault("compute_shaders", True)  # Audio models benefit from compute shaders
            if "whisper" in self.model_path.lower():
                self.config.setdefault("quantization", 8)  # Whisper needs higher precision
            elif "wav2vec" in self.model_path.lower():
                self.config.setdefault("quantization", 8)  # wav2vec2 needs higher precision
        
        elif self.model_type == "multimodal":
            # Multimodal models (CLIP, LLaVA, etc.)
            self.config.setdefault("parallel_loading", True)  # Multimodal models benefit from parallel loading
            self.config.setdefault("progressive_loading", True)  # Multimodal models benefit from progressive loading
            if "clip" in self.model_path.lower():
                self.config.setdefault("quantization", 4)  # CLIP works well with 4-bit
            elif "llava" in self.model_path.lower():
                self.config.setdefault("quantization", 4)  # LLaVA works with 4-bit
    
    def _initialize_components(self):
        """
        Initialize all components based on configuration.
        """
        # Track initialization of each component
        self._components = {}
        self._feature_usage = {}
        
        # Initialize shader registry if using WebGPU
        if self.config.get("use_webgpu", False):
            shader_registry = WebGPUShaderRegistry(
                model_type=self.model_type,
                precompile=self.config.get("shader_precompilation", False),
                use_compute_shaders=self.config.get("compute_shaders", False),
                workgroup_size=self.config.get("workgroup_size", (128, 1, 1))
            )
            self._components["shader_registry"] = shader_registry
            self._feature_usage["shader_precompilation"] = self.config.get("shader_precompilation", False)
            self._feature_usage["compute_shaders"] = self.config.get("compute_shaders", False)
        
        # Set up progressive loading if enabled
        if self.config.get("progressive_loading", False):
            loader = ProgressiveModelLoader(
                model_path=self.model_path,
                model_type=self.model_type,
                parallel_loading=self.config.get("parallel_loading", False),
                memory_optimized=True
            )
            self._components["loader"] = loader
            self._feature_usage["progressive_loading"] = True
            self._feature_usage["parallel_loading"] = self.config.get("parallel_loading", False)
        
        # Set up quantization based on configuration
        if self.config.get("ultra_low_precision", False):
            # Use ultra-low precision (2-bit or 3-bit)
            bits = 2 if self.config.get("quantization", 4) <= 2 else 3
            quantizer = setup_ultra_low_precision(
                model=self.model_path,
                bits=bits,
                adaptive=self.config.get("adaptive_precision", True)
            )
            self._components["quantizer"] = quantizer
            self._feature_usage["ultra_low_precision"] = True
            self._feature_usage["quantization_bits"] = bits
        elif self.config.get("quantization", 16) <= 4:
            # Use 4-bit quantization
            quantizer = setup_4bit_inference(
                model_path=self.model_path,
                model_type=self.model_type,
                config={
                    "bits": 4,
                    "group_size": 128,
                    "scheme": "symmetric",
                    "mixed_precision": True,
                    "use_specialized_kernels": True,
                    "optimize_attention": self.config.get("kv_cache_optimization", False)
                }
            )
            self._components["quantizer"] = quantizer
            self._feature_usage["4bit_quantization"] = True
        
        # Set up WebGPU based on browser type
        if self.config.get("use_webgpu", False) and self.config.get("browser") == "safari":
            # Special handling for Safari
            safari_handler = SafariWebGPUHandler(
                model_path=self.model_path,
                config={
                    "safari_version": self.config.get("browser_version", 0),
                    "model_type": self.model_type,
                    "quantization": self.config.get("quantization", 8),
                    "shader_registry": self._components.get("shader_registry")
                }
            )
            self._components["webgpu_handler"] = safari_handler
            self._feature_usage["safari_metal_integration"] = True
        
        # Set up WebAssembly fallback if needed
        wasm_fallback = setup_wasm_fallback(
            model_path=self.model_path,
            model_type=self.model_type,
            use_simd=self.config.get("browser_capabilities", {}).get("webassembly", {}).get("simd", False)
        )
        self._components["wasm_fallback"] = wasm_fallback
        self._feature_usage["wasm_fallback"] = True
        self._feature_usage["wasm_simd"] = self.config.get("browser_capabilities", {}).get("webassembly", {}).get("simd", False)
        
        # Set up streaming inference for text models if enabled
        if self.model_type == "text" and self.config.get("streaming_inference", False):
            streaming_handler = WebGPUStreamingInference(
                model_path=self.model_path,
                config={
                    "quantization": f"int{self.config.get('quantization', 4)}",
                    "optimize_kv_cache": self.config.get("kv_cache_optimization", False),
                    "latency_optimized": self.config.get("latency_optimized", True),
                    "adaptive_batch_size": self.config.get("adaptive_batch_size", True)
                }
            )
            self._components["streaming"] = streaming_handler
            self._feature_usage["streaming_inference"] = True
            self._feature_usage["kv_cache_optimization"] = self.config.get("kv_cache_optimization", False)
        
        # Store feature usage in performance metrics
        self._perf_metrics["feature_usage"] = self._feature_usage
    
    def create_endpoint(self) -> Callable:
        """
        Create a unified inference endpoint function.
        
        Returns:
            Callable function for model inference
        """
        # Check if streaming inference is appropriate
        if self.model_type == "text" and self._components.get("streaming") is not None:
            endpoint = lambda input_text, **kwargs: self._handle_streaming_inference(input_text, **kwargs)
        else:
            endpoint = lambda input_data, **kwargs: self._handle_inference(input_data, **kwargs)
        
        return endpoint
    
    def _handle_streaming_inference(self, input_text, **kwargs):
        """
        Handle streaming inference for text models.
        
        Args:
            input_text: Input text or dictionary with "text" key
            kwargs: Additional parameters for inference
            
        Returns:
            Generated text or streaming iterator
        """
        # Extract prompt from input
        prompt = input_text["text"] if isinstance(input_text, dict) else input_text
        
        # Get streaming handler
        streaming = self._components["streaming"]
        
        # Check for callback
        callback = kwargs.get("callback")
        if callback:
            # Use synchronous generation with callback
            return streaming.generate(
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                callback=callback
            )
        elif kwargs.get("stream", False):
            # Return async generator for streaming
            async def stream_generator():
                result = await streaming.generate_async(
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", 100),
                    temperature=kwargs.get("temperature", 0.7)
                )
                return result
            return stream_generator
        else:
            # Use synchronous generation without callback
            return streaming.generate(
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7)
            )
    
    def _handle_inference(self, input_data, **kwargs):
        """
        Handle standard inference.
        
        Args:
            input_data: Input data (text, image, audio, etc.)
            kwargs: Additional parameters for inference
            
        Returns:
            Inference result
        """
        # Prepare input based on model type
        processed_input = self._prepare_input(input_data)
        
        # Measure first inference time
        is_first_inference = not hasattr(self, "_first_inference_done")
        if is_first_inference:
            first_inference_start = time.time()
        
        # Run inference through appropriate component
        inference_start = time.time()
        
        # Try WebGPU/WebNN first (primary method)
        if self._components.get("webgpu_handler") is not None:
            try:
                result = self._components["webgpu_handler"](processed_input, **kwargs)
            except Exception as e:
                logger.warning(f"WebGPU inference failed: {e}, falling back to WebAssembly")
                result = self._components["wasm_fallback"](processed_input, **kwargs)
        # Fall back to WebAssembly
        else:
            result = self._components["wasm_fallback"](processed_input, **kwargs)
        
        # Update inference timing metrics
        inference_time_ms = (time.time() - inference_start) * 1000
        if is_first_inference:
            self._first_inference_done = True
            self._perf_metrics["first_inference_time_ms"] = (time.time() - first_inference_start) * 1000
        
        # Update average inference time
        if not hasattr(self, "_inference_count"):
            self._inference_count = 0
            self._total_inference_time = 0
        
        self._inference_count += 1
        self._total_inference_time += inference_time_ms
        self._perf_metrics["average_inference_time_ms"] = self._total_inference_time / self._inference_count
        
        # Return processed result
        return result
    
    def _prepare_input(self, input_data):
        """
        Prepare input data based on model type.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Processed input data
        """
        # Handle different input types based on model type
        if self.model_type == "text":
            # Text input
            if isinstance(input_data, dict) and "text" in input_data:
                return input_data["text"]
            return input_data
        elif self.model_type == "vision":
            # Vision input (image data)
            if isinstance(input_data, dict) and "image" in input_data:
                return input_data["image"]
            return input_data
        elif self.model_type == "audio":
            # Audio input
            if isinstance(input_data, dict) and "audio" in input_data:
                return input_data["audio"]
            return input_data
        elif self.model_type == "multimodal":
            # Multimodal input (combination of modalities)
            return input_data
        else:
            # Default case - return as is
            return input_data
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Update memory usage if available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self._perf_metrics["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
        except (ImportError, Exception):
            pass
        
        # Return all metrics
        return self._perf_metrics
    
    def get_feature_usage(self) -> Dict[str, bool]:
        """
        Get information about which features are being used.
        
        Returns:
            Dictionary mapping feature names to usage status
        """
        return self._feature_usage
    
    def get_components(self) -> Dict[str, Any]:
        """
        Get initialized components.
        
        Returns:
            Dictionary of components
        """
        return self._components
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def get_browser_compatibility_matrix(self) -> Dict[str, Dict[str, bool]]:
        """
        Get feature compatibility matrix for current browser.
        
        Returns:
            Dictionary with feature compatibility for current browser
        """
        from fixed_web_platform.browser_capability_detector import get_browser_feature_matrix
        return get_browser_feature_matrix()


def create_web_endpoint(model_path: str, model_type: str, config: Dict[str, Any] = None) -> Callable:
    """
    Create a web-accelerated model endpoint with a single function call.
    
    Args:
        model_path: Path to the model
        model_type: Type of model (text, vision, audio, multimodal)
        config: Optional configuration dictionary
        
    Returns:
        Callable function for model inference
    """
    # Create accelerator
    accelerator = WebPlatformAccelerator(
        model_path=model_path,
        model_type=model_type,
        config=config,
        auto_detect=True
    )
    
    # Create and return endpoint
    return accelerator.create_endpoint()


def get_optimal_config(model_path: str, model_type: str) -> Dict[str, Any]:
    """
    Get optimal configuration for a specific model.
    
    Args:
        model_path: Path to the model
        model_type: Type of model
        
    Returns:
        Dictionary with optimal configuration
    """
    # Detect capabilities
    detector = BrowserCapabilityDetector()
    capabilities = detector.get_capabilities()
    profile = detector.get_optimization_profile()
    
    # Create base config
    config = {
        "browser_capabilities": capabilities,
        "optimization_profile": profile,
        "use_webgpu": capabilities["webgpu"]["available"],
        "use_webnn": capabilities["webnn"]["available"],
        "compute_shaders": capabilities["webgpu"]["compute_shaders"],
        "shader_precompilation": capabilities["webgpu"]["shader_precompilation"],
        "browser": capabilities["browser_info"]["name"],
        "browser_version": capabilities["browser_info"]["version"]
    }
    
    # Add model-specific optimizations
    if model_type == "text":
        if "bert" in model_path.lower() or "roberta" in model_path.lower():
            config.update({
                "quantization": 4,
                "shader_precompilation": True,
                "ultra_low_precision": False
            })
        elif "t5" in model_path.lower():
            config.update({
                "quantization": 4,
                "shader_precompilation": True,
                "ultra_low_precision": False
            })
        elif "llama" in model_path.lower() or "gpt" in model_path.lower():
            config.update({
                "quantization": 4,
                "kv_cache_optimization": True,
                "streaming_inference": True,
                "ultra_low_precision": profile["precision"]["ultra_low_precision_enabled"]
            })
    elif model_type == "vision":
        config.update({
            "quantization": 4,
            "shader_precompilation": True,
            "ultra_low_precision": False
        })
    elif model_type == "audio":
        config.update({
            "quantization": 8,
            "compute_shaders": True,
            "ultra_low_precision": False
        })
    elif model_type == "multimodal":
        config.update({
            "quantization": 4,
            "parallel_loading": True,
            "progressive_loading": True,
            "ultra_low_precision": False
        })
    
    return config


def get_browser_capabilities() -> Dict[str, Any]:
    """
    Get current browser capabilities.
    
    Returns:
        Dictionary with browser capabilities
    """
    detector = BrowserCapabilityDetector()
    return detector.get_capabilities()


if __name__ == "__main__":
    print("Unified Web Framework")
    
    # Get browser capabilities 
    detector = BrowserCapabilityDetector()
    capabilities = detector.get_capabilities()
    
    print(f"Browser: {capabilities['browser_info']['name']} {capabilities['browser_info']['version']}")
    print(f"WebGPU available: {capabilities['webgpu']['available']}")
    print(f"WebNN available: {capabilities['webnn']['available']}")
    print(f"WebAssembly SIMD: {capabilities['webassembly']['simd']}")
    
    # Example usage
    model_path = "models/bert-base-uncased"
    model_type = "text"
    
    # Create accelerator with auto-detection
    accelerator = WebPlatformAccelerator(
        model_path=model_path,
        model_type=model_type,
        auto_detect=True
    )
    
    # Print configuration
    config = accelerator.get_config()
    print("\nConfiguration:")
    print(f"  Quantization: {config.get('quantization')}-bit")
    print(f"  Ultra-low precision: {config.get('ultra_low_precision', False)}")
    print(f"  Shader precompilation: {config.get('shader_precompilation', False)}")
    print(f"  Compute shaders: {config.get('compute_shaders', False)}")
    print(f"  Progressive loading: {config.get('progressive_loading', False)}")
    print(f"  Parallel loading: {config.get('parallel_loading', False)}")
    
    # Print feature usage
    feature_usage = accelerator.get_feature_usage()
    print("\nFeature Usage:")
    for feature, used in feature_usage.items():
        print(f"  {feature}: {'✅' if used else '❌'}")
    
    # Get performance metrics
    metrics = accelerator.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"  Initialization time: {metrics['initialization_time_ms']:.2f}ms")
    
    # Create endpoint
    endpoint = accelerator.create_endpoint()
    
    # Example inference
    result = endpoint("Example text for inference")
    
    # Get updated metrics
    metrics = accelerator.get_performance_metrics()
    print(f"  First inference time: {metrics['first_inference_time_ms']:.2f}ms")
    print(f"  Average inference time: {metrics['average_inference_time_ms']:.2f}ms")
    print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
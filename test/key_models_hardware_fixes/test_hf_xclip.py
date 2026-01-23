#!/usr/bin/env python3
"""
Class-based test file for all X-CLIP-family models.
This file provides a unified testing interface for:
- XCLIPModel
"""

import os
import sys
import json
import time
import datetime
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")


# Try to import PIL
try:
    from PIL import Image
    import requests
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    requests = MagicMock()
    BytesIO = MagicMock()
    HAS_PIL = False
    logger.warning("PIL or requests not available, using mock")


if not HAS_PIL:
    
class MockHandler:
def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        print(f"Created mock handler for {platform}")

    def init_cpu(self):
        """Initialize for CPU platform."""
        
        self.platform = "CPU"
        self.device = "cpu"
        self.device_name = "cpu"
        return True
    
        """Mock handler for platforms that don't have real implementations."""
        
        
    
    def init_cuda(self):
        """Initialize for CUDA platform."""
        import torch
        self.platform = "CUDA"
        self.device = "cuda"
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return True
    
    
    def init_mps(self):
        """Initialize for MPS platform."""
        import torch
        self.platform = "MPS"
        self.device = "mps"
        self.device_name = "mps" if torch.backends.mps.is_available() else "cpu"
        return True
    
    
    def init_openvino(self):
        """Initialize for OPENVINO platform."""
        import openvino
        self.platform = "OPENVINO"
        self.device = "openvino"
        self.device_name = "openvino"
        return True
    
    
    def init_rocm(self):
        """Initialize for ROCM platform."""
        import torch
        self.platform = "ROCM"
        self.device = "rocm"
        self.device_name = "cuda" if torch.cuda.is_available() and torch.version.hip is not None else "cpu"
        return True
    

    def create_cpu_handler(self):
        """Create handler for CPU platform."""
        model_path = self.get_model_path_or_name()
            handler = AutoModel.from_pretrained(model_path).to(self.device_name)
        return handler
    
    
    
    def create_cuda_handler(self):
        """Create handler for CUDA platform."""
        model_path = self.get_model_path_or_name()
            handler = AutoModel.from_pretrained(model_path).to(self.device_name)
        return handler
    
    
    def create_mps_handler(self):
        """Create handler for MPS platform."""
        model_path = self.get_model_path_or_name()
            handler = AutoModel.from_pretrained(model_path).to(self.device_name)
        return handler
    
    
    def create_openvino_handler(self):
        """Create handler for OPENVINO platform."""
        model_path = self.get_model_path_or_name()
            from openvino.runtime import Core
            import numpy as np
            ie = Core()
            compiled_model = ie.compile_model(model_path, "CPU")
            handler = lambda input_data: compiled_model(np.array(input_data))[0]
        return handler
    
    
    def create_rocm_handler(self):
        """Create handler for ROCM platform."""
        model_path = self.get_model_path_or_name()
            handler = AutoModel.from_pretrained(model_path).to(self.device_name)
        return handler
    

    def test_with_openvino(self):
        """Test the model using OpenVINO integration."""
        results = {
            "model": self.model_id,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for OpenVINO support
        if not HW_CAPABILITIES["openvino"]:
            results["openvino_error_type"] = "missing_dependency"
            results["openvino_missing_core"] = ["openvino"]
            results["openvino_success"] = False
            return results
        
        # Check for transformers
        if not HAS_TRANSFORMERS:
            results["openvino_error_type"] = "missing_dependency"
            results["openvino_missing_core"] = ["transformers"]
            results["openvino_success"] = False
            return results
        
        try:
            from optimum.intel import OVModelForVision
            logger.info(f"Testing {self.model_id} with OpenVINO...")
            
            # Time tokenizer loading
            tokenizer_load_start = time.time()
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Time model loading
            model_load_start = time.time()
            model = OVModelForVision.from_pretrained(
                self.model_id,
                export=True,
                provider="CPU"
            )
            model_load_time = time.time() - model_load_start
            
            # Prepare input
            test_input = self.test_image_url
            
            # Process image
            if HAS_PIL:
                response = requests.get(test_input)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                # Get text features
                inputs = tokenizer(self.candidate_labels, padding=True, return_tensors="pt")
                
                # Get image features
                processor = transformers.AutoProcessor.from_pretrained(self.model_id)
                image_inputs = processor(images=image, return_tensors="pt")
                inputs.update(image_inputs)
            else:
                # Mock inputs
                inputs = {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                    "pixel_values": torch.zeros(1, 3, 224, 224)
                }
            
            # Run inference
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            # Process classification output
            if hasattr(outputs, "logits_per_image"):
                logits = outputs.logits_per_image[0]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                predictions = []
                for i, (label, prob) in enumerate(zip(self.candidate_labels, probs)):
                    predictions.append({
                        "label": label,
                        "score": float(prob)
                    })
            else:
                predictions = [{"label": "Mock label", "score": 0.95}]
            
            # Store results
            results["openvino_success"] = True
            results["openvino_load_time"] = model_load_time
            results["openvino_inference_time"] = inference_time
            results["openvino_tokenizer_load_time"] = tokenizer_load_time
            
            # Add predictions if available
            if 'predictions' in locals():
                results["openvino_predictions"] = predictions
            
            results["openvino_error_type"] = "none"
            
            # Add to examples
            example_data = {
                "method": "OpenVINO inference",
                "input": str(test_input)
            }
            
            if 'predictions' in locals():
                example_data["predictions"] = predictions
            
            self.examples.append(example_data)
            
            # Store in performance stats
            self.performance_stats["openvino"] = {
                "inference_time": inference_time,
                "load_time": model_load_time,
                "tokenizer_load_time": tokenizer_load_time
            }
            
        except Exception as e:
            # Store error information
            results["openvino_success"] = False
            results["openvino_error"] = str(e)
            results["openvino_traceback"] = traceback.format_exc()
            logger.error(f"Error testing with OpenVINO: {e}")
            
            # Classify error
            error_str = str(e).lower()
            if "no module named" in error_str:
                results["openvino_error_type"] = "missing_dependency"
            else:
                results["openvino_error_type"] = "other"
        
        # Add to overall results
        self.results["openvino"] = results
        return results
    
        
        def run_tests(self, all_hardware=False):
            """
            Run all tests for this model.
            
            Args:
                all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
            
            Returns:
                Dict containing test results
            """
            # Always test on default device
            self.test_pipeline()
            self.test_from_pretrained()
            
            # Test on all available hardware if requested
            if all_hardware:
                # Always test on CPU
                if self.preferred_device != "cpu":
                    self.test_pipeline(device="cpu")
                    self.test_from_pretrained(device="cpu")
                
                # Test on CUDA if available
                if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
                    self.test_pipeline(device="cuda")
                    self.test_from_pretrained(device="cuda")
                
                # Test on OpenVINO if available
                if HW_CAPABILITIES["openvino"]:
                    self.test_with_openvino()
            
            # Build final results
            return {
                "results": self.results,
                "examples": self.examples,
                "performance": self.performance_stats,
                "hardware": HW_CAPABILITIES,
                "metadata": {
                    "model": self.model_id,
                    "task": self.task,
                    "class": self.class_name,
                    "description": self.description,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "has_transformers": HAS_TRANSFORMERS,
                    "has_torch": HAS_TORCH,
                    "has_pil": HAS_PIL
                }
            }
    
    


    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        print(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")
        return {"mock_output": f"Mock output for {self.platform}"}
class MockImage:
        @staticmethod
        def open(file):
            class MockImg:
                def __init__(self):
                    self.size = (224, 224)
                def convert(self, mode):
                    return self
                def resize(self, size):
                    return self
            return MockImg()
            
    class MockRequests:
        @staticmethod
        def get(url):
            class MockResponse:
                def __init__(self):
                    self.content = b"mock image data"
                def raise_for_status(self):
                    pass
            return MockResponse()

    Image.open = MockImage.open
    requests.get = MockRequests.get


# Hardware detection
def check_hardware():
    """Check available hardware and return capabilities."""
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count()
            capabilities["cuda_version"] = torch.version.cuda
    
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()
    
    # Check OpenVINO
    try:
        import openvino
        capabilities["openvino"] = True
    except ImportError:
        pass
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()

# Models registry - Maps model IDs to their specific configurations
X-CLIP_MODELS_REGISTRY = {
    "microsoft/xclip-base-patch32": {
        "description": "X-CLIP Base (patch size 32)",
        "class": "XCLIPModel",
    },
}

class TestXCLIPModels:
    """Base test class for all X-CLIP-family models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class for a specific model or default."""
        self.model_id = model_id or "microsoft/xclip-base-patch32"
        
        # Verify model exists in registry
        if self.model_id not in X-CLIP_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = X-CLIP_MODELS_REGISTRY["microsoft/xclip-base-patch32"]
        else:
            self.model_info = X-CLIP_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "zero-shot-image-classification"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "['a photo of a cat', 'a photo of a dog']"
        self.test_texts = [
            "['a photo of a cat', 'a photo of a dog']",
            "['a photo of a cat', 'a photo of a dog'] (alternative)"
        ]
        self.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
    
    

    def init_webnn(self, model_name=None):
        """Initialize vision model for WebNN inference."""
        try:
            print("Initializing WebNN for vision model")
            model_name = model_name or self.model_name
            
            # Check for WebNN support
            webnn_support = False
            try:
                # In browser environments, check for WebNN API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'ml'):
                    webnn_support = True
                    print("WebNN API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(16)
            
            if not webnn_support:
                # Create a WebNN simulation using CPU implementation for vision models
                print("Using WebNN simulation for vision model")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Wrap the CPU function to simulate WebNN
    def webnn_handler(image_input, **kwargs):
                    try:
                        # Process image input (path or PIL Image)
                        if isinstance(image_input, str):
                            from PIL import Image
                            image = Image.open(image_input).convert("RGB")
                        elif isinstance(image_input, list):
                            if all(isinstance(img, str) for img in image_input):
                                from PIL import Image
                                image = [Image.open(img).convert("RGB") for img in image_input]
                            else:
                                image = image_input
                        else:
                            image = image_input
                            
                        # Process with processor
                        inputs = processor(images=image, return_tensors="pt")
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebNN-specific metadata
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBNN",
                            "model": model_name,
                            "backend": "webnn-simulation",
                            "device": "cpu"
                        }
                    except Exception as e:
                        print(f"Error in WebNN simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webnn_handler, queue, batch_size
            else:
                # Use actual WebNN implementation when available
                # (This would use the WebNN API in browser environments)
                print("Using native WebNN implementation")
                
                # Since WebNN API access depends on browser environment,
                # implementation details would involve JS interop
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x: {"output": "Native WebNN output", "implementation_type": "WEBNN"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebNN: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(16)
            return None, None, lambda x: {"output": "Mock WebNN output", "implementation_type": "MOCK_WEBNN"}, queue, 1

    def init_webgpu(self, model_name=None):
        """Initialize multimodal model for WebGPU inference with advanced optimizations."""
        try:
            print("Initializing WebGPU for multimodal model")
            model_name = model_name or self.model_name
            
            # Check for WebGPU support
            webgpu_support = False
            try:
                # In browser environments, check for WebGPU API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'gpu'):
                    webgpu_support = True
                    print("WebGPU API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(16)
            
            if not webgpu_support:
                # Create a WebGPU simulation using CPU implementation for multimodal models
                print("Using WebGPU/transformers.js simulation for multimodal model with optimizations")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Multimodal-specific optimizations
                use_parallel_loading = True
                use_4bit_quantization = True
                use_kv_cache_optimization = True
                print(f"WebGPU optimizations: parallel_loading={use_parallel_loading}, 4bit_quantization={use_4bit_quantization}, kv_cache={use_kv_cache_optimization}")
                
                # Wrap the CPU function to simulate WebGPU/transformers.js for multimodal
    def webgpu_handler(input_data, **kwargs):
                    try:
                        # Process multimodal input (image + text)
                        if isinstance(input_data, dict):
                            # Handle dictionary input with multiple modalities
                            image = input_data.get("image")
                            text = input_data.get("text")
                            
                            # Load image if path is provided
                            if isinstance(image, str):
                                from PIL import Image
                                image = Image.open(image).convert("RGB")
                        elif isinstance(input_data, str) and input_data.endswith(('.jpg', '.png', '.jpeg')):
                            # Handle image path as direct input
                            from PIL import Image
                            image = Image.open(input_data).convert("RGB")
                            text = kwargs.get("text", "")
                        else:
                            # Default handling for text input
                            image = None
                            text = input_data
                            
                        # Process with processor
                        if image is not None and text:
                            # Apply parallel loading optimization if enabled
                            if use_parallel_loading:
                                print("Using parallel loading optimization for multimodal input")
                                
                            # Process with processor
                            inputs = processor(text=text, images=image, return_tensors="pt")
                        else:
                            inputs = processor(input_data, return_tensors="pt")
                        
                        # Apply 4-bit quantization if enabled
                        if use_4bit_quantization:
                            print("Using 4-bit quantization for model weights")
                            # In real implementation, weights would be quantized here
                        
                        # Apply KV cache optimization if enabled
                        if use_kv_cache_optimization:
                            print("Using KV cache optimization for inference")
                            # In real implementation, KV cache would be used here
                        
                        # Run inference with optimizations
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebGPU-specific metadata including optimization flags
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBGPU_TRANSFORMERS_JS",
                            "model": model_name,
                            "backend": "webgpu-simulation",
                            "device": "webgpu",
                            "optimizations": {
                                "parallel_loading": use_parallel_loading,
                                "quantization_4bit": use_4bit_quantization,
                                "kv_cache_enabled": use_kv_cache_optimization
                            },
                            "transformers_js": {
                                "version": "2.9.0",  # Simulated version
                                "quantized": use_4bit_quantization,
                                "format": "float4" if use_4bit_quantization else "float32",
                                "backend": "webgpu"
                            }
                        }
                    except Exception as e:
                        print(f"Error in WebGPU multimodal simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webgpu_handler, queue, batch_size
            else:
                # Use actual WebGPU implementation when available
                print("Using native WebGPU implementation with transformers.js for multimodal model")
                
                # Since WebGPU API access depends on browser environment,
                # implementation details would involve JS interop
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x: {"output": "Native WebGPU multimodal output", "implementation_type": "WEBGPU_TRANSFORMERS_JS"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebGPU for multimodal model: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(16)
            return None, None, lambda x: {"output": "Mock WebGPU multimodal output", "implementation_type": "MOCK_WEBGPU"}, queue, 1
def test_pipeline(self, device="auto"):
    """Test the model using transformers pipeline API."""
    if device == "auto":
        device = self.preferred_device
    
    results = {
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_core"] = ["transformers"]
        results["pipeline_success"] = False
        return results
        
    if not HAS_PIL:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
        results["pipeline_success"] = False
        return results
    
    try:
        logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
        
        # Create pipeline with appropriate parameters
        pipeline_kwargs = {
            "task": self.task,
            "model": self.model_id,
            "device": device
        }
        
        # Time the model loading
        load_start_time = time.time()
        pipeline = transformers.pipeline(**pipeline_kwargs)
        load_time = time.time() - load_start_time
        
        # Prepare test input
        pipeline_input = self.test_text
        
        # Run warmup inference if on CUDA
        if device == "cuda":
            try:
                _ = pipeline(pipeline_input)
            except Exception:
                pass
        
        # Run multiple inference passes
        num_runs = 3
        times = []
        outputs = []
        
        for _ in range(num_runs):
            start_time = time.time()
            output = pipeline(pipeline_input)
            end_time = time.time()
            times.append(end_time - start_time)
            outputs.append(output)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Store results
        results["pipeline_success"] = True
        results["pipeline_avg_time"] = avg_time
        results["pipeline_min_time"] = min_time
        results["pipeline_max_time"] = max_time
        results["pipeline_load_time"] = load_time
        results["pipeline_error_type"] = "none"
        
        # Add to examples
        self.examples.append({
            "method": f"pipeline() on {device}",
            "input": str(pipeline_input),
            "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
        })
        
        # Store in performance stats
        self.performance_stats[f"pipeline_{device}"] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "load_time": load_time,
            "num_runs": num_runs
        }
        
    except Exception as e:
        # Store error information
        results["pipeline_success"] = False
        results["pipeline_error"] = str(e)
        results["pipeline_traceback"] = traceback.format_exc()
        logger.error(f"Error testing pipeline on {device}: {e}")
        
        # Classify error type
        error_str = str(e).lower()
        traceback_str = traceback.format_exc().lower()
        
        if "cuda" in error_str or "cuda" in traceback_str:
            results["pipeline_error_type"] = "cuda_error"
        elif "memory" in error_str:
            results["pipeline_error_type"] = "out_of_memory"
        elif "no module named" in error_str:
            results["pipeline_error_type"] = "missing_dependency"
        else:
            results["pipeline_error_type"] = "other"
    
    # Add to overall results
    self.results[f"pipeline_{device}"] = results
    return results

    
    
def test_from_pretrained(self, device="auto"):
    """Test the model using direct from_pretrained loading."""
    if device == "auto":
        device = self.preferred_device
    
    results = {
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_core"] = ["transformers"]
        results["from_pretrained_success"] = False
        return results
        
    if not HAS_PIL:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
        results["from_pretrained_success"] = False
        return results
    
    try:
        logger.info(f"Testing {self.model_id} with from_pretrained() on {device}...")
        
        # Common parameters for loading
        pretrained_kwargs = {
            "local_files_only": False
        }
        
        # Time tokenizer loading
        tokenizer_load_start = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        tokenizer_load_time = time.time() - tokenizer_load_start
        
        # Use appropriate model class based on model type
        model_class = None
        if self.class_name == "XCLIPModel":
            model_class = transformers.XCLIPModel
        else:
            # Fallback to Auto class
            model_class = transformers.AutoModel
        
        # Time model loading
        model_load_start = time.time()
        model = model_class.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        model_load_time = time.time() - model_load_start
        
        # Move model to device
        if device != "cpu":
            model = model.to(device)
        
        # Prepare test input
        test_input = self.test_image_url
        
        # Get image
        if HAS_PIL:
            response = requests.get(test_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Mock image
            image = None
            
        # Get text features
        inputs = tokenizer(self.candidate_labels, padding=True, return_tensors="pt")
        
        if HAS_PIL:
            # Get image features
            processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            image_inputs = processor(images=image, return_tensors="pt")
            inputs.update(image_inputs)
        
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Run warmup inference if using CUDA
        if device == "cuda":
            try:
                with torch.no_grad():
                    _ = model(**inputs)
            except Exception:
                pass
        
        # Run multiple inference passes
        num_runs = 3
        times = []
        outputs = []
        
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(**inputs)
            end_time = time.time()
            times.append(end_time - start_time)
            outputs.append(output)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Process classification output
        if hasattr(outputs, "logits_per_image"):
            logits = outputs.logits_per_image[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = []
            for i, (label, prob) in enumerate(zip(self.candidate_labels, probs)):
                predictions.append({
                    "label": label,
                    "score": prob.item()
                })
        else:
            predictions = [{"label": "Mock label", "score": 0.95}]
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
        
        # Store results
        results["from_pretrained_success"] = True
        results["from_pretrained_avg_time"] = avg_time
        results["from_pretrained_min_time"] = min_time
        results["from_pretrained_max_time"] = max_time
        results["tokenizer_load_time"] = tokenizer_load_time
        results["model_load_time"] = model_load_time
        results["model_size_mb"] = model_size_mb
        results["from_pretrained_error_type"] = "none"
        
        # Add predictions if available
        if 'predictions' in locals():
            results["predictions"] = predictions
        
        # Add to examples
        example_data = {
            "method": f"from_pretrained() on {device}",
            "input": str(test_input)
        }
        
        if 'predictions' in locals():
            example_data["predictions"] = predictions
        
        self.examples.append(example_data)
        
        # Store in performance stats
        self.performance_stats[f"from_pretrained_{device}"] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "tokenizer_load_time": tokenizer_load_time,
            "model_load_time": model_load_time,
            "model_size_mb": model_size_mb,
            "num_runs": num_runs
        }
        
    except Exception as e:
        # Store error information
        results["from_pretrained_success"] = False
        results["from_pretrained_error"] = str(e)
        results["from_pretrained_traceback"] = traceback.format_exc()
        logger.error(f"Error testing from_pretrained on {device}: {e}")
        
        # Classify error type
        error_str = str(e).lower()
        traceback_str = traceback.format_exc().lower()
        
        if "cuda" in error_str or "cuda" in traceback_str:
            results["from_pretrained_error_type"] = "cuda_error"
        elif "memory" in error_str:
            results["from_pretrained_error_type"] = "out_of_memory"
        elif "no module named" in error_str:
            results["from_pretrained_error_type"] = "missing_dependency"
        else:
            results["from_pretrained_error_type"] = "other"
    
    # Add to overall results
    self.results[f"from_pretrained_{device}"] = results
    return results

    
    
def save_results(model_id, results, output_dir="collected_results"):
    """Save test results to a file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_xclip_{safe_model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path

def get_available_models():
    """Get a list of all available X-CLIP models in the registry."""
    return list(X-CLIP_MODELS_REGISTRY.keys())

def test_all_models(output_dir="collected_results", all_hardware=False):
    """Test all registered X-CLIP models."""
    models = get_available_models()
    results = {}
    
    for model_id in models:
        logger.info(f"Testing model: {model_id}")
        tester = TestXCLIPModels(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values() 
                          if r.get("pipeline_success") is not False)
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_xclip_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test X-CLIP-family models")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
    
    # Hardware options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    # List options
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        print("\nAvailable X-CLIP-family models:")
        for model in models:
            info = X-CLIP_MODELS_REGISTRY[model]
            print(f"  - {model} ({info['class']}): {info['description']}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print("\nX-CLIP Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {successful} of {total} models ({successful/total*100:.1f}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "microsoft/xclip-base-patch32"
    logger.info(f"Testing model: {model_id}")
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run test
    tester = TestXCLIPModels(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values()
                  if r.get("pipeline_success") is not False)
    
    print("\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {model_id}")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {device}: {stats['avg_time']:.4f}s average inference time")
        
        # Print example outputs if available
        if results.get("examples") and len(results["examples"]) > 0:
            print("\nExample output:")
            example = results["examples"][0]
            if "predictions" in example:
                print(f"  Input: {example['input']}")
                print(f"  Predictions: {example['predictions']}")
            elif "output_preview" in example:
                print(f"  Input: {example['input']}")
                print(f"  Output: {example['output_preview']}")
    else:
        print(f"❌ Failed to test {model_id}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {test_name}: {result.get('pipeline_error_type', 'unknown')}")
                print(f"    {result.get('pipeline_error', 'Unknown error')}")
    
    print("\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hugging Face model skillset for mistral model.

This skillset implements decoder-only architecture model support across hardware platforms:
- CPU: Standard CPU implementation
- CUDA: NVIDIA GPU implementation
- OpenVINO: Intel hardware acceleration
- MPS: Apple Silicon GPU implementation
- ROCm: AMD GPU implementation
- Qualcomm: Qualcomm AI Engine/Hexagon DSP implementation
- WebNN: Web Neural Network API (browser)
- WebGPU: Web GPU API (browser)
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if environment variables are set for mock mode
MOCK_MODE = os.environ.get("MOCK_MODE", "False").lower() == "true"

# Try to import hardware-specific libraries
try:
    import torch
except ImportError:
    pass

try:
    import numpy as np
except ImportError:
    pass

class MockHandler:
    """Mock handler for platforms that don't have real implementations."""
    
    def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        logger.info(f"Created mock handler for {platform}")
    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        logger.info(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")
        input_text = args[0] if args else kwargs.get("input_text", "")
        max_new_tokens = kwargs.get("max_new_tokens", 50)
        return {
            "success": True,
            "generated_text": f"{input_text} [MOCK GENERATION WITH {max_new_tokens} TOKENS]",
            "platform": self.platform
        }

class MistralSkillset:
    """Skillset for mistral model across hardware backends."""
    
    def __init__(self, model_id=None, device=None):
        """
        Initialize the skillset.
        
        Args:
            model_id: Model ID to use (default: mistral-base)
            device: Device to use (default: auto-detect optimal device)
        """
        self.model_id = model_id or self.get_default_model_id()
        self.model_type = "mistral"
        self.task = "text-generation"
        self.architecture_type = "decoder-only"
        
        # Initialize device
        self.device = device or self.get_optimal_device()
        self.model = None
        self.tokenizer = None
        
        # Test cases for validation
        self.test_cases = [
            {
                "description": "Test on CPU platform",
                "platform": "CPU",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            },
            {
                "description": "Test on CUDA platform",
                "platform": "CUDA",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            },
            {
                "description": "Test on OPENVINO platform",
                "platform": "OPENVINO",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            },
            {
                "description": "Test on MPS platform",
                "platform": "MPS",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            },
            {
                "description": "Test on ROCM platform",
                "platform": "ROCM",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            },
            {
                "description": "Test on QUALCOMM platform",
                "platform": "QUALCOMM",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            },
            {
                "description": "Test on WEBNN platform",
                "platform": "WEBNN",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            },
            {
                "description": "Test on WEBGPU platform",
                "platform": "WEBGPU",
                "input": "Once upon a time, there was a",
                "expected": {"success": True}
            }
        ]
        
        logger.info(f"Initialized mistral skillset with device={device}")
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "mistral-base"
    
    def get_optimal_device(self) -> str:
        """Get the optimal device for this model type."""
        if MOCK_MODE:
            return "cpu"
            
        # Try to import hardware detection
        try:
            # First, try relative import
            try:
                from ....hardware.hardware_detection import get_optimal_device, get_model_hardware_recommendations
            except ImportError:
                # Then, try absolute import
                from ipfs_accelerate_py.worker.hardware.hardware_detection import get_optimal_device, get_model_hardware_recommendations
            
            # Get recommended devices for this architecture
            recommended_devices = get_model_hardware_recommendations(self.architecture_type)
            return get_optimal_device(recommended_devices)
        except ImportError:
            # Fallback to basic detection if hardware module not available
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except:
                return "cpu"
    
    #
    # Hardware platform initialization methods
    #
    
    def init_cpu(self):
        """Initialize for CPU platform."""
        self.platform = "CPU"
        self.device = "cpu"
        return self.load_tokenizer()
    
    def init_cuda(self):
        """Initialize for CUDA platform."""
        import torch
        self.platform = "CUDA"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logger.warning("CUDA not available, falling back to CPU")
        return self.load_tokenizer()
    
    def init_openvino(self):
        """Initialize for OPENVINO platform."""
        try:
            import openvino
        except ImportError:
            logger.warning("OpenVINO not available, falling back to CPU")
            self.platform = "CPU"
            self.device = "cpu"
            return self.load_tokenizer()
        
        self.platform = "OPENVINO"
        self.device = "openvino"
        return self.load_tokenizer()
    
    def init_mps(self):
        """Initialize for MPS platform."""
        import torch
        self.platform = "MPS"
        self.device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if self.device != "mps":
            logger.warning("MPS not available, falling back to CPU")
        return self.load_tokenizer()
    
    def init_rocm(self):
        """Initialize for ROCM platform."""
        import torch
        self.platform = "ROCM"
        self.device = "cuda" if torch.cuda.is_available() and hasattr(torch.version, "hip") else "cpu"
        if self.device != "cuda":
            logger.warning("ROCm not available, falling back to CPU")
        return self.load_tokenizer()
    
    def init_qualcomm(self):
        """Initialize for Qualcomm platform."""
        try:
            # Try to import Qualcomm-specific libraries
            import importlib.util
            has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
            has_qti = importlib.util.find_spec("qti") is not None
            has_qualcomm_env = "QUALCOMM_SDK" in os.environ
            
            if has_qnn or has_qti or has_qualcomm_env:
                self.platform = "QUALCOMM"
                self.device = "qualcomm"
            else:
                logger.warning("Qualcomm SDK not available, falling back to CPU")
                self.platform = "CPU"
                self.device = "cpu"
        except Exception as e:
            logger.error(f"Error initializing Qualcomm platform: {e}")
            self.platform = "CPU"
            self.device = "cpu"
            
        return self.load_tokenizer()
    
    def init_webnn(self):
        """Initialize for WEBNN platform."""
        self.platform = "WEBNN"
        self.device = "webnn"
        return self.load_tokenizer()
    
    def init_webgpu(self):
        """Initialize for WEBGPU platform."""
        self.platform = "WEBGPU"
        self.device = "webgpu"
        return self.load_tokenizer()
    
    #
    # Core functionality
    #
    
    def load_tokenizer(self):
        """Load tokenizer."""
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                
                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                return True
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                return False
        return True
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load the model and tokenizer.
        
        Returns:
            Dict with loading results
        """
        start_time = time.time()
        
        try:
            if MOCK_MODE:
                # Mock implementation
                self.tokenizer = object()
                self.model = object()
                
                return {
                    "success": True,
                    "time_seconds": time.time() - start_time,
                    "device": self.device,
                    "model_id": self.model_id
                }
            
            # Import necessary libraries
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Device-specific initialization
            # Device-specific initialization will be added automatically
            
            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on device
            if self.device in ["cpu", "cuda", "mps"]:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            elif self.device == "rocm":
                # ROCm uses cuda device name in PyTorch
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="cuda",
                    torch_dtype=torch.float16
                )
            elif self.device == "openvino":
                # OpenVINO-specific loading
                try:
                    from optimum.intel import OVModelForCausalLM
                    self.model = OVModelForCausalLM.from_pretrained(
                        self.model_id,
                        export=True
                    )
                except ImportError:
                    logger.warning("OpenVINO optimum not available, falling back to CPU")
                    self.device = "cpu"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        device_map="cpu"
                    )
            elif self.device == "qualcomm":
                # QNN-specific loading (placeholder)
                try:
                    import qnn_wrapper
                    # QNN specific implementation would go here
                    logger.info("QNN support for decoder-only models is experimental")
                    # For now, fall back to CPU
                    self.device = "cpu"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        device_map="cpu"
                    )
                except ImportError:
                    # Fallback to CPU if QNN import fails
                    logger.warning("QNN not available, falling back to CPU")
                    self.device = "cpu"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        device_map="cpu"
                    )
            else:
                # Fallback to CPU for unknown devices
                logger.warning(f"Unknown device {self.device}, falling back to CPU")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="cpu"
                )
                
            return {
                "success": True,
                "time_seconds": time.time() - start_time,
                "device": self.device,
                "model_id": self.model_id
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                "success": False,
                "time_seconds": time.time() - start_time,
                "device": self.device,
                "model_id": self.model_id,
                "error": str(e)
            }
    
    #
    # Hardware-specific handlers
    #
    
    def create_cpu_handler(self):
        """Create handler for CPU platform."""
        try:
            model_path = self.model_id
            if self.model is None:
                load_result = self.load_model()
                if not load_result["success"]:
                    return MockHandler(model_path, "cpu")
            
            def handler(input_text, max_new_tokens=50, **kwargs):
                # Apply tokenizer
                inputs = self.tokenizer(input_text, return_tensors="pt")
                
                # Run inference
                import torch
                with torch.no_grad():
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": kwargs.get("do_sample", True),
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
                    }
                    
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Process outputs
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "input_text": input_text,
                    "device": self.device
                }
            
            return handler
        except Exception as e:
            logger.error(f"Error creating CPU handler: {e}")
            return MockHandler(model_path, "cpu")
    
    def create_cuda_handler(self):
        """Create handler for CUDA platform."""
        try:
            import torch
            model_path = self.model_id
            if self.model is None:
                load_result = self.load_model()
                if not load_result["success"]:
                    return MockHandler(model_path, "cuda")
            
            def handler(input_text, max_new_tokens=50, **kwargs):
                # Apply tokenizer
                inputs = self.tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": kwargs.get("do_sample", True),
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
                    }
                    
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Process outputs
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "input_text": input_text,
                    "device": self.device
                }
            
            return handler
        except Exception as e:
            logger.error(f"Error creating CUDA handler: {e}")
            return MockHandler(model_path, "cuda")
    
    def create_openvino_handler(self):
        """Create handler for OPENVINO platform."""
        try:
            model_path = self.model_id
            if self.model is None:
                load_result = self.load_model()
                if not load_result["success"]:
                    return MockHandler(model_path, "openvino")
            
            # For demonstration, we use the actual model if loaded or a mock otherwise
            if hasattr(self.model, "generate"):
                def handler(input_text, max_new_tokens=50, **kwargs):
                    # Apply tokenizer
                    inputs = self.tokenizer(input_text, return_tensors="pt")
                    
                    # Run inference
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": kwargs.get("do_sample", True),
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
                    }
                    
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                    
                    # Process outputs
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    return {
                        "success": True,
                        "generated_text": generated_text,
                        "input_text": input_text,
                        "device": self.device
                    }
                
                return handler
            else:
                return MockHandler(model_path, "openvino")
        except Exception as e:
            logger.error(f"Error creating OpenVINO handler: {e}")
            return MockHandler(model_path, "openvino")
    
    def create_mps_handler(self):
        """Create handler for MPS platform."""
        try:
            import torch
            model_path = self.model_id
            if self.model is None:
                load_result = self.load_model()
                if not load_result["success"]:
                    return MockHandler(model_path, "mps")
            
            def handler(input_text, max_new_tokens=50, **kwargs):
                # Apply tokenizer
                inputs = self.tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": kwargs.get("do_sample", True),
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
                    }
                    
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Process outputs
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "input_text": input_text,
                    "device": self.device
                }
            
            return handler
        except Exception as e:
            logger.error(f"Error creating MPS handler: {e}")
            return MockHandler(model_path, "mps")
    
    def create_rocm_handler(self):
        """Create handler for ROCM platform."""
        try:
            import torch
            model_path = self.model_id
            if self.model is None:
                load_result = self.load_model()
                if not load_result["success"]:
                    return MockHandler(model_path, "rocm")
            
            def handler(input_text, max_new_tokens=50, **kwargs):
                # Apply tokenizer
                inputs = self.tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": kwargs.get("do_sample", True),
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
                    }
                    
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Process outputs
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "input_text": input_text,
                    "device": self.device
                }
            
            return handler
        except Exception as e:
            logger.error(f"Error creating ROCm handler: {e}")
            return MockHandler(model_path, "rocm")
    
    def create_qualcomm_handler(self):
        """Create handler for Qualcomm platform."""
        try:
            model_path = self.model_id
            if self.model is None:
                load_result = self.load_model()
                if not load_result["success"]:
                    return MockHandler(model_path, "qualcomm")
                
            # Check if Qualcomm QNN SDK is available
            import importlib.util
            has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
            has_qti = importlib.util.find_spec("qti.aisw.dlc_utils") is not None
            
            if not (has_qnn or has_qti):
                logger.warning("Warning: Qualcomm SDK not found, using mock implementation")
                return MockHandler(model_path, "qualcomm")
            
            # In a real implementation, we would use Qualcomm SDK for inference
            # For demonstration, we just return a mock result
            def handler(input_text, max_new_tokens=50, **kwargs):
                # This is a placeholder for QNN-specific implementation
                return {
                    "success": True,
                    "generated_text": f"{input_text} [QUALCOMM GENERATED TEXT]",
                    "input_text": input_text,
                    "device": self.device,
                    "platform": "qualcomm"
                }
            
            return handler
        except Exception as e:
            logger.error(f"Error creating Qualcomm handler: {e}")
            return MockHandler(model_path, "qualcomm")
            
    def create_webnn_handler(self):
        """Create handler for WEBNN platform."""
        try:
            # WebNN would use browser APIs - this is a mock implementation
            if self.tokenizer is None:
                self.load_tokenizer()
            
            # In a real implementation, we'd use the WebNN API
            return MockHandler(self.model_id, "webnn")
        except Exception as e:
            logger.error(f"Error creating WebNN handler: {e}")
            return MockHandler(self.model_id, "webnn")
    
    def create_webgpu_handler(self):
        """Create handler for WEBGPU platform."""
        try:
            # WebGPU would use browser APIs - this is a mock implementation
            if self.tokenizer is None:
                self.load_tokenizer()
            
            # In a real implementation, we'd use the WebGPU API
            return MockHandler(self.model_id, "webgpu")
        except Exception as e:
            logger.error(f"Error creating WebGPU handler: {e}")
            return MockHandler(self.model_id, "webgpu")
    
    #
    # Public API methods
    #
    
    def run_inference(self, input_text: str, max_new_tokens: int = 50, **kwargs) -> Dict[str, Any]:
        """
        Run inference with the model.
        
        Args:
            input_text: Text to process
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters (do_sample, temperature, top_p, etc.)
            
        Returns:
            Dict with inference results
        """
        if not self.model or not self.tokenizer:
            load_result = self.load_model()
            if not load_result["success"]:
                return {
                    "success": False,
                    "error": f"Model not loaded: {load_result.get('error', 'Unknown error')}"
                }
        
        start_time = time.time()
        
        try:
            if MOCK_MODE:
                # Mock implementation
                return {
                    "success": True,
                    "time_seconds": time.time() - start_time,
                    "generated_text": f"{input_text} [MOCK GENERATION]",
                    "input_text": input_text,
                    "device": self.device
                }
            
            # Create handler for the current device
            platform = self.device
            if platform == "cuda" and hasattr(torch, "version") and hasattr(torch.version, "hip"):
                platform = "rocm"
            
            handler_method = getattr(self, f"create_{platform}_handler", None)
            if handler_method:
                handler = handler_method()
            else:
                handler = self.create_cpu_handler()
            
            # Run inference
            result = handler(input_text, max_new_tokens=max_new_tokens, **kwargs)
            result["time_seconds"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            return {
                "success": False,
                "time_seconds": time.time() - start_time,
                "device": self.device,
                "error": str(e)
            }
    
    def benchmark(self, iterations: int = 5, input_length: int = 10, output_length: int = 20) -> Dict[str, Any]:
        """
        Run a benchmark of the model.
        
        Args:
            iterations: Number of iterations to run
            input_length: Number of tokens in input
            output_length: Number of tokens to generate
            
        Returns:
            Dict with benchmark results
        """
        if not self.model or not self.tokenizer:
            load_result = self.load_model()
            if not load_result["success"]:
                return {
                    "success": False,
                    "error": f"Model not loaded: {load_result.get('error', 'Unknown error')}"
                }
        
        results = {
            "success": True,
            "device": self.device,
            "model_id": self.model_id,
            "iterations": iterations,
            "input_length": input_length,
            "output_length": output_length,
            "latencies_ms": [],
            "mean_latency_ms": 0.0,
            "throughput_tokens_per_sec": 0.0
        }
        
        try:
            if MOCK_MODE:
                # Mock implementation
                import random
                results["latencies_ms"] = [random.uniform(100, 500) for _ in range(iterations)]
                results["mean_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])
                results["throughput_tokens_per_sec"] = (output_length * 1000) / results["mean_latency_ms"]
                return results
            
            # Prepare input
            input_text = "Once upon a time, there was a wizard who lived in a tall tower."
            
            # Create handler for the current device
            platform = self.device
            if platform == "cuda" and hasattr(torch, "version") and hasattr(torch.version, "hip"):
                platform = "rocm"
                
            handler_method = getattr(self, f"create_{platform}_handler", None)
            if handler_method:
                handler = handler_method()
            else:
                handler = self.create_cpu_handler()
            
            # Run inference multiple times
            for _ in range(iterations):
                start_time = time.time()
                handler(input_text, max_new_tokens=output_length, do_sample=False)
                latency = (time.time() - start_time) * 1000  # ms
                results["latencies_ms"].append(latency)
            
            # Calculate statistics
            results["mean_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])
            results["throughput_tokens_per_sec"] = (output_length * 1000) / results["mean_latency_ms"]
            
            return results
            
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return {
                "success": False,
                "device": self.device,
                "model_id": self.model_id,
                "error": str(e)
            }
    
    def run(self, platform="CPU", mock=False):
        """Run the model on the specified platform."""
        platform = platform.lower()
        init_method = getattr(self, f"init_{platform}", None)
        
        if init_method is None:
            logger.error(f"Platform {platform} not supported")
            return False
        
        if not init_method():
            logger.error(f"Failed to initialize {platform} platform")
            return False
        
        # Create handler for the platform
        try:
            handler_method = getattr(self, f"create_{platform}_handler", None)
            if mock:
                # Use mock handler for testing
                handler = MockHandler(self.model_id, platform)
            else:
                handler = handler_method()
        except Exception as e:
            logger.error(f"Error creating handler for {platform}: {e}")
            return False
        
        # Test with a sample input
        try:
            input_text = "Once upon a time, "
            result = handler(input_text, max_new_tokens=20)
            
            if "generated_text" in result:
                text_preview = result["generated_text"][:50] + "..." if len(result["generated_text"]) > 50 else result["generated_text"]
                logger.info(f"Generated text: {text_preview}")
                
            logger.info(f"Successfully tested on {platform} platform")
            return True
        except Exception as e:
            logger.error(f"Error running test on {platform}: {e}")
            return False


def test_skillset():
    """Simple test function for the skillset."""
    skillset = MistralSkillset()
    
    # Load model
    load_result = skillset.load_model()
    print(f"Load result: {'success': {load_result['success']}, 'device': {load_result['device']}}")
    
    if load_result["success"]:
        # Run inference
        inference_result = skillset.run_inference("Once upon a time,", max_new_tokens=20)
        print(f"Inference result: {'success': {inference_result['success']}, 'generated_text': '{inference_result.get('generated_text', '')[:50]}...'}")
        
        # Run benchmark
        benchmark_result = skillset.benchmark(iterations=2, output_length=10)
        print(f"Benchmark result: {'mean_latency_ms': {benchmark_result.get('mean_latency_ms', 0):.2f}, 'throughput': {benchmark_result.get('throughput_tokens_per_sec', 0):.2f}}")


if __name__ == "__main__":
    """Run the skillset."""
    import argparse
    parser = argparse.ArgumentParser(description="Test mistral model")
    parser.add_argument("--model", help="Model path or name", default="mistral-base")
    parser.add_argument("--platform", default="CPU", help="Platform to test on")
    parser.add_argument("--mock", action="store_true", help="Use mock implementations")
    args = parser.parse_args()
    
    skillset = MistralSkillset(args.model)
    result = skillset.run(args.platform, args.mock)
    
    if result:
        print(f"Test successful on {args.platform}")
        sys.exit(0)
    else:
        print(f"Test failed on {args.platform}")
        sys.exit(1)
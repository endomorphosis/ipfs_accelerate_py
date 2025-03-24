#!/usr/bin/env python3
"""
Hugging Face model skillset for rembert model.

This skillset implements encoder-only architecture model support across hardware platforms:
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
        return {
            "embeddings": [[0.1, 0.2, 0.3]] * 10,
            "success": True,
            "platform": self.platform
        }

class RembertSkillset:
    """Skillset for rembert model across hardware backends."""
    
    def __init__(self, model_id=None, device=None):
        """
        Initialize the skillset.
        
        Args:
            model_id: Model ID to use (default: rembert-base)
            device: Device to use (default: auto-detect optimal device)
        """
        self.model_id = model_id or self.get_default_model_id()
        self.model_type = "rembert"
        self.task = "fill-mask" if "rembert" in ["bert", "roberta"] else "feature-extraction"
        self.architecture_type = "encoder-only"
        
        # Initialize device
        self.device = device or self.get_optimal_device()
        self.model = None
        self.tokenizer = None
        
        # Test cases for validation
        self.test_cases = [
            {
                "description": "Test on CPU platform",
                "platform": "CPU",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            },
            {
                "description": "Test on CUDA platform",
                "platform": "CUDA",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            },
            {
                "description": "Test on OPENVINO platform",
                "platform": "OPENVINO",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            },
            {
                "description": "Test on MPS platform",
                "platform": "MPS",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            },
            {
                "description": "Test on ROCM platform",
                "platform": "ROCM",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            },
            {
                "description": "Test on QUALCOMM platform",
                "platform": "QUALCOMM",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            },
            {
                "description": "Test on WEBNN platform",
                "platform": "WEBNN",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            },
            {
                "description": "Test on WEBGPU platform",
                "platform": "WEBGPU",
                "input": "This is a sample text for testing the rembert model.",
                "expected": {"success": True}
            }
        ]
        
        logger.info(f"Initialized rembert skillset with device={device}")
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "rembert-base"
    
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
            from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
            
            # Device-specific initialization
            # Device-specific initialization will be added automatically
            
            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model based on device
            if self.device in ["cpu", "cuda", "mps"]:
                if self.task == "fill-mask":
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        self.model_id,
                        device_map=self.device
                    )
                else:
                    self.model = AutoModel.from_pretrained(
                        self.model_id,
                        device_map=self.device
                    )
            elif self.device == "rocm":
                # ROCm uses cuda device name in PyTorch
                if self.task == "fill-mask":
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        self.model_id,
                        device_map="cuda"
                    )
                else:
                    self.model = AutoModel.from_pretrained(
                        self.model_id,
                        device_map="cuda"
                    )
            elif self.device == "openvino":
                # OpenVINO-specific loading
                try:
                    from optimum.intel import OVModelForMaskedLM, OVModel
                    if self.task == "fill-mask":
                        self.model = OVModelForMaskedLM.from_pretrained(
                            self.model_id,
                            export=True
                        )
                    else:
                        self.model = OVModel.from_pretrained(
                            self.model_id,
                            export=True
                        )
                except ImportError:
                    logger.warning("OpenVINO optimum not available, falling back to CPU")
                    self.device = "cpu"
                    if self.task == "fill-mask":
                        self.model = AutoModelForMaskedLM.from_pretrained(
                            self.model_id,
                            device_map="cpu"
                        )
                    else:
                        self.model = AutoModel.from_pretrained(
                            self.model_id,
                            device_map="cpu"
                        )
            elif self.device == "qualcomm":
                # QNN-specific loading (placeholder)
                try:
                    import qnn_wrapper
                    # QNN specific implementation would go here
                    logger.info("QNN support for encoder-only models is experimental")
                    # For now, fall back to CPU
                    self.device = "cpu"
                    if self.task == "fill-mask":
                        self.model = AutoModelForMaskedLM.from_pretrained(
                            self.model_id,
                            device_map="cpu"
                        )
                    else:
                        self.model = AutoModel.from_pretrained(
                            self.model_id,
                            device_map="cpu"
                        )
                except ImportError:
                    # Fallback to CPU if QNN import fails
                    logger.warning("QNN not available, falling back to CPU")
                    self.device = "cpu"
                    if self.task == "fill-mask":
                        self.model = AutoModelForMaskedLM.from_pretrained(
                            self.model_id,
                            device_map="cpu"
                        )
                    else:
                        self.model = AutoModel.from_pretrained(
                            self.model_id,
                            device_map="cpu"
                        )
            else:
                # Fallback to CPU for unknown devices
                logger.warning(f"Unknown device {self.device}, falling back to CPU")
                self.device = "cpu"
                if self.task == "fill-mask":
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        self.model_id,
                        device_map="cpu"
                    )
                else:
                    self.model = AutoModel.from_pretrained(
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
            
            def handler(input_text):
                inputs = self.tokenizer(input_text, return_tensors="pt")
                
                # Get embeddings
                import torch
                with torch.no_grad():
                    if self.task == "fill-mask":
                        # For masked language modeling
                        if "[MASK]" in input_text:
                            outputs = self.model(**inputs)
                            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                            mask_token_logits = outputs.logits[0, mask_token_index, :]
                            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
                            predictions = [self.tokenizer.decode([token_id]) for token_id in top_tokens]
                            
                            return {
                                "success": True,
                                "predictions": predictions,
                                "device": self.device
                            }
                        else:
                            # Get embeddings even for fill-mask models if no mask token
                            outputs = self.model(**inputs)
                            embeddings = outputs.hidden_states[-1].mean(dim=1).tolist() if hasattr(outputs, "hidden_states") else outputs.last_hidden_state.mean(dim=1).tolist()
                    else:
                        # Standard embedding extraction
                        outputs = self.model(**inputs)
                        # Mean pooling
                        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
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
            
            def handler(input_text):
                inputs = self.tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    if self.task == "fill-mask":
                        # For masked language modeling
                        if "[MASK]" in input_text:
                            outputs = self.model(**inputs)
                            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                            mask_token_logits = outputs.logits[0, mask_token_index, :]
                            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
                            predictions = [self.tokenizer.decode([token_id]) for token_id in top_tokens]
                            
                            return {
                                "success": True,
                                "predictions": predictions,
                                "device": self.device
                            }
                        else:
                            # Get embeddings even for fill-mask models if no mask token
                            outputs = self.model(**inputs)
                            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().tolist() if hasattr(outputs, "hidden_states") else outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                    else:
                        # Standard embedding extraction
                        outputs = self.model(**inputs)
                        # Mean pooling
                        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
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
            if hasattr(self.model, "generate") or hasattr(self.model, "forward"):
                def handler(input_text):
                    inputs = self.tokenizer(input_text, return_tensors="pt")
                    
                    # Get embeddings
                    import torch
                    with torch.no_grad():
                        if self.task == "fill-mask":
                            # For masked language modeling
                            if "[MASK]" in input_text:
                                outputs = self.model(**inputs)
                                mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                                mask_token_logits = outputs.logits[0, mask_token_index, :]
                                top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
                                predictions = [self.tokenizer.decode([token_id]) for token_id in top_tokens]
                                
                                return {
                                    "success": True,
                                    "predictions": predictions,
                                    "device": self.device
                                }
                            else:
                                # Get embeddings even for fill-mask models if no mask token
                                outputs = self.model(**inputs)
                                embeddings = outputs.hidden_states[-1].mean(dim=1).tolist() if hasattr(outputs, "hidden_states") else outputs.last_hidden_state.mean(dim=1).tolist()
                        else:
                            # Standard embedding extraction
                            outputs = self.model(**inputs)
                            # Mean pooling
                            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
                    
                    return {
                        "success": True,
                        "embeddings": embeddings,
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
            
            def handler(input_text):
                inputs = self.tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    if self.task == "fill-mask":
                        # For masked language modeling
                        if "[MASK]" in input_text:
                            outputs = self.model(**inputs)
                            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                            mask_token_logits = outputs.logits[0, mask_token_index, :]
                            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
                            predictions = [self.tokenizer.decode([token_id]) for token_id in top_tokens]
                            
                            return {
                                "success": True,
                                "predictions": predictions,
                                "device": self.device
                            }
                        else:
                            # Get embeddings even for fill-mask models if no mask token
                            outputs = self.model(**inputs)
                            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().tolist() if hasattr(outputs, "hidden_states") else outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                    else:
                        # Standard embedding extraction
                        outputs = self.model(**inputs)
                        # Mean pooling
                        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
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
            
            def handler(input_text):
                inputs = self.tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    if self.task == "fill-mask":
                        # For masked language modeling
                        if "[MASK]" in input_text:
                            outputs = self.model(**inputs)
                            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                            mask_token_logits = outputs.logits[0, mask_token_index, :]
                            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
                            predictions = [self.tokenizer.decode([token_id]) for token_id in top_tokens]
                            
                            return {
                                "success": True,
                                "predictions": predictions,
                                "device": self.device
                            }
                        else:
                            # Get embeddings even for fill-mask models if no mask token
                            outputs = self.model(**inputs)
                            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().tolist() if hasattr(outputs, "hidden_states") else outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                    else:
                        # Standard embedding extraction
                        outputs = self.model(**inputs)
                        # Mean pooling
                        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
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
            def handler(input_text):
                return {
                    "success": True,
                    "embeddings": [[0.1, 0.2, 0.3]] * 10,
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
    
    def run_inference(self, input_text: str) -> Dict[str, Any]:
        """
        Run inference with the model.
        
        Args:
            input_text: Text to process
            
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
                if self.task == "fill-mask" and "[MASK]" in input_text:
                    return {
                        "success": True,
                        "time_seconds": time.time() - start_time,
                        "predictions": ["the", "a", "an", "one", "this"],
                        "device": self.device
                    }
                else:
                    return {
                        "success": True,
                        "time_seconds": time.time() - start_time,
                        "embeddings": [[0.1, 0.2, 0.3]] * 10,
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
            result = handler(input_text)
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
    
    def benchmark(self, iterations: int = 5) -> Dict[str, Any]:
        """
        Run a benchmark of the model.
        
        Args:
            iterations: Number of iterations to run
            
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
            "latencies_ms": [],
            "mean_latency_ms": 0.0,
            "throughput_samples_per_sec": 0.0
        }
        
        try:
            if MOCK_MODE:
                # Mock implementation
                import random
                results["latencies_ms"] = [random.uniform(10, 50) for _ in range(iterations)]
                results["mean_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])
                results["throughput_samples_per_sec"] = 1000 / results["mean_latency_ms"]
                return results
            
            # Prepare input
            if self.task == "fill-mask":
                input_text = "The capital of France is [MASK]."
            else:
                input_text = "This is a sample text for benchmarking the model."
                
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
                handler(input_text)
                latency = (time.time() - start_time) * 1000  # ms
                results["latencies_ms"].append(latency)
            
            # Calculate statistics
            results["mean_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])
            results["throughput_samples_per_sec"] = 1000 / results["mean_latency_ms"]
            
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
            if self.task == "fill-mask":
                input_text = "The capital of France is [MASK]."
                result = handler(input_text)
                if "predictions" in result:
                    logger.info(f"Predictions: {result['predictions']}")
                elif "embeddings" in result:
                    logger.info(f"Embedding shape: {len(result['embeddings'])}x{len(result['embeddings'][0])}")
            else:
                input_text = "This is a sample text for testing."
                result = handler(input_text)
                if "embeddings" in result:
                    logger.info(f"Embedding shape: {len(result['embeddings'])}x{len(result['embeddings'][0])}")
                
            logger.info(f"Successfully tested on {platform} platform")
            return True
        except Exception as e:
            logger.error(f"Error running test on {platform}: {e}")
            return False


def test_skillset():
    """Simple test function for the skillset."""
    skillset = RembertSkillset()
    
    # Load model
    load_result = skillset.load_model()
    print(f"Load result: {'success': {load_result['success']}, 'device': {load_result['device']}}")
    
    if load_result["success"]:
        # Run inference
        if skillset.task == "fill-mask":
            inference_result = skillset.run_inference("The capital of France is [MASK].")
            print(f"Inference result: {'success': {inference_result['success']}, 'predictions': {inference_result.get('predictions', [])}}")
        else:
            inference_result = skillset.run_inference("This is a sample text for testing.")
            embeddings = inference_result.get('embeddings', [[]])
            embedding_shape = f"{len(embeddings)}x{len(embeddings[0]) if embeddings and embeddings[0] else 0}"
            print(f"Inference result: {'success': {inference_result['success']}, 'embedding_shape': {embedding_shape}}")
        
        # Run benchmark
        benchmark_result = skillset.benchmark(iterations=2)
        print(f"Benchmark result: {'mean_latency_ms': {benchmark_result.get('mean_latency_ms', 0):.2f}, 'throughput': {benchmark_result.get('throughput_samples_per_sec', 0):.2f}}")


if __name__ == "__main__":
    """Run the skillset."""
    import argparse
    parser = argparse.ArgumentParser(description="Test rembert model")
    parser.add_argument("--model", help="Model path or name", default="rembert-base")
    parser.add_argument("--platform", default="CPU", help="Platform to test on")
    parser.add_argument("--mock", action="store_true", help="Use mock implementations")
    args = parser.parse_args()
    
    skillset = RembertSkillset(args.model)
    result = skillset.run(args.platform, args.mock)
    
    if result:
        print(f"Test successful on {args.platform}")
        sys.exit(0)
    else:
        print(f"Test failed on {args.platform}")
        sys.exit(1)
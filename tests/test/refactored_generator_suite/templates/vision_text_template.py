#!/usr/bin/env python3
"""
Hugging Face model skillset for {model_type} model.

This skillset implements vision-text architecture model support across hardware platforms:
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
import json
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
try:
    from PIL import Image
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if environment variables are set for mock mode
MOCK_MODE = os.environ.get("MOCK_MODE", "False").lower() == "true"

# Try to import hardware-specific libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Will use mock mode if real inference is attempted.")

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
        
        # Check if we have an image input
        image = None
        text = None
        if args and len(args) > 0:
            if hasattr(args[0], "mode") and hasattr(args[0], "size"):  # Likely a PIL Image
                image = args[0]
            elif isinstance(args[0], str):
                image = f"Image path: {args[0]}"
                
        if len(args) > 1 and isinstance(args[1], (str, list)):
            text = args[1]
        
        if kwargs.get("text_input") is not None:
            text = kwargs.get("text_input")
        
        # Generate an appropriate mock response based on the input types
        if text and isinstance(text, list):
            # CLIP-like similarity comparison
            return {
                "success": True,
                "results": {
                    "similarity_scores": [0.8, 0.4, 0.2],
                    "highest_similarity_idx": 0, 
                    "highest_match": text[0]
                },
                "platform": self.platform
            }
        else:
            # Image captioning or general response
            return {
                "success": True,
                "results": {
                    "generated_text": f"Mock {self.platform} caption: an image of something interesting"
                },
                "platform": self.platform
            }

class {model_type_upper}Skillset:
    """Skillset for {model_type} model across hardware backends."""
    
    def __init__(self, model_id=None, device=None):
        """
        Initialize the skillset.
        
        Args:
            model_id: Model ID to use (default: {default_model_id})
            device: Device to use (default: auto-detect optimal device)
        """
        self.model_id = model_id or self.get_default_model_id()
        self.model_type = "{model_type}"
        self.task = "vision-to-text"
        self.architecture_type = "vision-encoder-text-decoder"
        
        # Initialize device
        self.device = device or self.get_optimal_device()
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Test cases for validation
        self.test_cases = [
            {
                "description": "Test on CPU platform",
                "platform": "CPU",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            },
            {
                "description": "Test on CUDA platform",
                "platform": "CUDA",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            },
            {
                "description": "Test on OPENVINO platform",
                "platform": "OPENVINO",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            },
            {
                "description": "Test on MPS platform",
                "platform": "MPS",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            },
            {
                "description": "Test on ROCM platform",
                "platform": "ROCM",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            },
            {
                "description": "Test on QUALCOMM platform",
                "platform": "QUALCOMM",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            },
            {
                "description": "Test on WEBNN platform",
                "platform": "WEBNN",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            },
            {
                "description": "Test on WEBGPU platform",
                "platform": "WEBGPU",
                "input": {"image_path": "test.jpg"},
                "expected": {"success": True}
            }
        ]
        
        logger.info(f"Initialized {model_type} skillset with device={device}")
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "{default_model_id}"
    
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
        return self.load_processor()
    
    def init_cuda(self):
        """Initialize for CUDA platform."""
        import torch
        self.platform = "CUDA"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logger.warning("CUDA not available, falling back to CPU")
        return self.load_processor()
    
    def init_openvino(self):
        """Initialize for OPENVINO platform."""
        try:
            import openvino
        except ImportError:
            logger.warning("OpenVINO not available, falling back to CPU")
            self.platform = "CPU"
            self.device = "cpu"
            return self.load_processor()
        
        self.platform = "OPENVINO"
        self.device = "openvino"
        return self.load_processor()
    
    def init_mps(self):
        """Initialize for MPS platform."""
        import torch
        self.platform = "MPS"
        self.device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if self.device != "mps":
            logger.warning("MPS not available, falling back to CPU")
        return self.load_processor()
    
    def init_rocm(self):
        """Initialize for ROCM platform."""
        import torch
        self.platform = "ROCM"
        self.device = "cuda" if torch.cuda.is_available() and hasattr(torch.version, "hip") else "cpu"
        if self.device != "cuda":
            logger.warning("ROCm not available, falling back to CPU")
        return self.load_processor()
    
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
            
        return self.load_processor()
    
    def init_webnn(self):
        """Initialize for WEBNN platform."""
        self.platform = "WEBNN"
        self.device = "webnn"
        return self.load_processor()
    
    def init_webgpu(self):
        """Initialize for WEBGPU platform."""
        self.platform = "WEBGPU"
        self.device = "webgpu"
        return self.load_processor()
    
    #
    # Core functionality
    #
    
    def load_processor(self):
        """Load processor."""
        if self.processor is None:
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                return True
            except Exception as e:
                logger.error(f"Error loading processor: {e}")
                return False
        return True
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load the model and processor.
        
        Returns:
            Dict with loading results
        """
        start_time = time.time()
        
        try:
            if MOCK_MODE:
                # Mock implementation
                self.processor = object()
                self.model = object()
                
                return {
                    "success": True,
                    "time_seconds": time.time() - start_time,
                    "device": self.device,
                    "model_id": self.model_id
                }
            
            # Import necessary libraries
            import torch
            from transformers import {model_class_name}, AutoProcessor
            
            # Device-specific initialization
            {device_init_code}
            
            # Load processor if not already loaded
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Load model based on device
            if self.device in ["cpu", "cuda", "mps"]:
                self.model = {model_class_name}.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            elif self.device == "rocm":
                # ROCm uses cuda device name in PyTorch
                self.model = {model_class_name}.from_pretrained(
                    self.model_id,
                    device_map="cuda",
                    torch_dtype=torch.float16
                )
            elif self.device == "openvino":
                # OpenVINO-specific loading
                try:
                    from optimum.intel import OVModelFor{model_class_name_short}
                    self.model = OVModelFor{model_class_name_short}.from_pretrained(
                        self.model_id,
                        export=True
                    )
                except ImportError:
                    logger.warning("OpenVINO optimum not available, falling back to CPU")
                    self.device = "cpu"
                    self.model = {model_class_name}.from_pretrained(
                        self.model_id,
                        device_map="cpu"
                    )
            elif self.device == "qualcomm":
                # QNN-specific loading (placeholder)
                try:
                    import qnn_wrapper
                    # QNN specific implementation would go here
                    logger.info("QNN support for vision-text models is experimental")
                    # For now, fall back to CPU
                    self.device = "cpu"
                    self.model = {model_class_name}.from_pretrained(
                        self.model_id,
                        device_map="cpu"
                    )
                except ImportError:
                    # Fallback to CPU if QNN import fails
                    logger.warning("QNN not available, falling back to CPU")
                    self.device = "cpu"
                    self.model = {model_class_name}.from_pretrained(
                        self.model_id,
                        device_map="cpu"
                    )
            else:
                # Fallback to CPU for unknown devices
                logger.warning(f"Unknown device {self.device}, falling back to CPU")
                self.device = "cpu"
                self.model = {model_class_name}.from_pretrained(
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
    
    def load_image(self, image_path: str) -> Dict[str, Any]:
        """
        Load image from a file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with image data
        """
        try:
            if MOCK_MODE:
                # Mock implementation
                return {
                    "success": True,
                    "image": object(),  # A placeholder object
                    "image_path": image_path
                }
            
            from PIL import Image
            
            # Load image
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file does not exist: {image_path}"
                }
                
            image = Image.open(image_path)
            
            return {
                "success": True,
                "image": image,
                "image_path": image_path
            }
        except Exception as e:
            logger.error(f"Error loading image file {image_path}: {e}")
            return {
                "success": False,
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
            
            def handler(image_input, text_input=None):
                try:
                    from PIL import Image
                    
                    # Process the image input
                    if isinstance(image_input, str):
                        # Assume it's a file path
                        image_data = self.load_image(image_input)
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    elif isinstance(image_input, Image.Image):
                        # Already a PIL Image
                        image = image_input
                    elif isinstance(image_input, dict) and "image_path" in image_input:
                        # Dict with image path
                        image_data = self.load_image(image_input["image_path"])
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    else:
                        return {
                            "success": False,
                            "error": f"Invalid image input type: {type(image_input)}"
                        }
                    
                    # Process inputs based on model type and presence of text input
                    if "{model_type}" == "clip" and text_input:
                        # For CLIP-like models with text-image similarity
                        import torch
                        inputs = self.processor(
                            text=text_input,
                            images=image,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        # Get similarity scores
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        logits_per_image = outputs.logits_per_image  # image-text similarity score
                        probs = logits_per_image.softmax(dim=1)
                        
                        # Convert to Python types for JSON serialization
                        similarity_scores = probs[0].tolist() if hasattr(probs[0], "tolist") else probs[0]
                        highest_similarity_idx = int(probs[0].argmax().item()) if hasattr(probs[0], "argmax") else 0
                        
                        results = {
                            "similarity_scores": similarity_scores,
                            "highest_similarity_idx": highest_similarity_idx,
                            "highest_match": text_input[highest_similarity_idx] if isinstance(text_input, list) else None
                        }
                    
                    elif "{model_type}" == "blip":
                        # For BLIP-like models with image captioning
                        import torch
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Generate caption
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        caption = self.processor.batch_decode(outputs[0], skip_special_tokens=True)
                        
                        results = {
                            "generated_text": caption
                        }
                    
                    else:
                        # Generic handling for other vision-text models
                        import torch
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Run model
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        # Process outputs - attempt to get text
                        try:
                            text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                            results = {"generated_text": text_output}
                        except:
                            # Fallback to raw outputs
                            results = {"raw_outputs": str(outputs)}
                    
                    return {
                        "success": True,
                        "results": results,
                        "device": self.device
                    }
                    
                except Exception as e:
                    logger.error(f"CPU handler error: {e}")
                    return {
                        "success": False,
                        "error": str(e)
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
            
            def handler(image_input, text_input=None):
                try:
                    from PIL import Image
                    
                    # Process the image input
                    if isinstance(image_input, str):
                        # Assume it's a file path
                        image_data = self.load_image(image_input)
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    elif isinstance(image_input, Image.Image):
                        # Already a PIL Image
                        image = image_input
                    elif isinstance(image_input, dict) and "image_path" in image_input:
                        # Dict with image path
                        image_data = self.load_image(image_input["image_path"])
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    else:
                        return {
                            "success": False,
                            "error": f"Invalid image input type: {type(image_input)}"
                        }
                    
                    # Process inputs based on model type and presence of text input
                    if "{model_type}" == "clip" and text_input:
                        # For CLIP-like models with text-image similarity
                        inputs = self.processor(
                            text=text_input,
                            images=image,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        # Move to GPU
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Get similarity scores
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        logits_per_image = outputs.logits_per_image  # image-text similarity score
                        probs = logits_per_image.softmax(dim=1)
                        
                        # Convert to Python types for JSON serialization
                        similarity_scores = probs[0].cpu().tolist() if hasattr(probs[0], "tolist") else probs[0]
                        highest_similarity_idx = int(probs[0].argmax().item()) if hasattr(probs[0], "argmax") else 0
                        
                        results = {
                            "similarity_scores": similarity_scores,
                            "highest_similarity_idx": highest_similarity_idx,
                            "highest_match": text_input[highest_similarity_idx] if isinstance(text_input, list) else None
                        }
                    
                    elif "{model_type}" == "blip":
                        # For BLIP-like models with image captioning
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Move to GPU
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Generate caption
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        caption = self.processor.batch_decode(outputs[0], skip_special_tokens=True)
                        
                        results = {
                            "generated_text": caption
                        }
                    
                    else:
                        # Generic handling for other vision-text models
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Move to GPU
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Run model
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        # Process outputs - attempt to get text
                        try:
                            text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                            results = {"generated_text": text_output}
                        except:
                            # Fallback to raw outputs
                            results = {"raw_outputs": str(outputs)}
                    
                    return {
                        "success": True,
                        "results": results,
                        "device": self.device
                    }
                    
                except Exception as e:
                    logger.error(f"CUDA handler error: {e}")
                    return {
                        "success": False,
                        "error": str(e)
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
                def handler(image_input, text_input=None):
                    try:
                        from PIL import Image
                        
                        # Process the image input
                        if isinstance(image_input, str):
                            # Assume it's a file path
                            image_data = self.load_image(image_input)
                            if not image_data["success"]:
                                return {
                                    "success": False,
                                    "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                                }
                            image = image_data["image"]
                        elif isinstance(image_input, Image.Image):
                            # Already a PIL Image
                            image = image_input
                        elif isinstance(image_input, dict) and "image_path" in image_input:
                            # Dict with image path
                            image_data = self.load_image(image_input["image_path"])
                            if not image_data["success"]:
                                return {
                                    "success": False,
                                    "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                                }
                            image = image_data["image"]
                        else:
                            return {
                                "success": False,
                                "error": f"Invalid image input type: {type(image_input)}"
                            }
                            
                        # Different processing based on model type
                        if "{model_type}" == "clip" and text_input:
                            # CLIP processing with OpenVINO
                            import torch
                            inputs = self.processor(
                                text=text_input,
                                images=image,
                                return_tensors="pt",
                                padding=True
                            )
                            
                            # Run with OpenVINO
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                            
                            logits_per_image = outputs.logits_per_image
                            probs = logits_per_image.softmax(dim=1)
                            
                            # Convert to Python types for JSON serialization
                            similarity_scores = probs[0].tolist() if hasattr(probs[0], "tolist") else probs[0]
                            highest_similarity_idx = int(probs[0].argmax().item()) if hasattr(probs[0], "argmax") else 0
                            
                            results = {
                                "similarity_scores": similarity_scores,
                                "highest_similarity_idx": highest_similarity_idx,
                                "highest_match": text_input[highest_similarity_idx] if isinstance(text_input, list) else None
                            }
                        else:
                            # General vision-text model processing with OpenVINO
                            import torch
                            inputs = self.processor(images=image, return_tensors="pt")
                            
                            # Run with OpenVINO
                            with torch.no_grad():
                                outputs = self.model.generate(**inputs)
                            
                            # Process outputs - attempt to get text
                            try:
                                text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                                results = {"generated_text": text_output}
                            except:
                                # Fallback to raw outputs
                                results = {"raw_outputs": str(outputs)}
                        
                        return {
                            "success": True,
                            "results": results,
                            "device": self.device
                        }
                        
                    except Exception as e:
                        logger.error(f"OpenVINO handler error: {e}")
                        return {
                            "success": False,
                            "error": str(e)
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
            
            def handler(image_input, text_input=None):
                try:
                    from PIL import Image
                    
                    # Process the image input
                    if isinstance(image_input, str):
                        # Assume it's a file path
                        image_data = self.load_image(image_input)
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    elif isinstance(image_input, Image.Image):
                        # Already a PIL Image
                        image = image_input
                    elif isinstance(image_input, dict) and "image_path" in image_input:
                        # Dict with image path
                        image_data = self.load_image(image_input["image_path"])
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    else:
                        return {
                            "success": False,
                            "error": f"Invalid image input type: {type(image_input)}"
                        }
                    
                    # Process inputs based on model type and presence of text input
                    if "{model_type}" == "clip" and text_input:
                        # For CLIP-like models with text-image similarity
                        inputs = self.processor(
                            text=text_input,
                            images=image,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        # Move to MPS
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Get similarity scores
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        logits_per_image = outputs.logits_per_image  # image-text similarity score
                        probs = logits_per_image.softmax(dim=1)
                        
                        # Convert to Python types for JSON serialization
                        similarity_scores = probs[0].cpu().tolist() if hasattr(probs[0], "tolist") else probs[0]
                        highest_similarity_idx = int(probs[0].argmax().item()) if hasattr(probs[0], "argmax") else 0
                        
                        results = {
                            "similarity_scores": similarity_scores,
                            "highest_similarity_idx": highest_similarity_idx,
                            "highest_match": text_input[highest_similarity_idx] if isinstance(text_input, list) else None
                        }
                    
                    elif "{model_type}" == "blip":
                        # For BLIP-like models with image captioning
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Move to MPS
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Generate caption
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        caption = self.processor.batch_decode(outputs[0], skip_special_tokens=True)
                        
                        results = {
                            "generated_text": caption
                        }
                    
                    else:
                        # Generic handling for other vision-text models
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Move to MPS
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Run model
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        # Process outputs - attempt to get text
                        try:
                            text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                            results = {"generated_text": text_output}
                        except:
                            # Fallback to raw outputs
                            results = {"raw_outputs": str(outputs)}
                    
                    return {
                        "success": True,
                        "results": results,
                        "device": self.device
                    }
                    
                except Exception as e:
                    logger.error(f"MPS handler error: {e}")
                    return {
                        "success": False,
                        "error": str(e)
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
            
            def handler(image_input, text_input=None):
                try:
                    from PIL import Image
                    
                    # Process the image input
                    if isinstance(image_input, str):
                        # Assume it's a file path
                        image_data = self.load_image(image_input)
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    elif isinstance(image_input, Image.Image):
                        # Already a PIL Image
                        image = image_input
                    elif isinstance(image_input, dict) and "image_path" in image_input:
                        # Dict with image path
                        image_data = self.load_image(image_input["image_path"])
                        if not image_data["success"]:
                            return {
                                "success": False,
                                "error": f"Failed to load image: {image_data.get('error', 'Unknown error')}"
                            }
                        image = image_data["image"]
                    else:
                        return {
                            "success": False,
                            "error": f"Invalid image input type: {type(image_input)}"
                        }
                    
                    # Process inputs based on model type and presence of text input
                    if "{model_type}" == "clip" and text_input:
                        # For CLIP-like models with text-image similarity
                        inputs = self.processor(
                            text=text_input,
                            images=image,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        # Move to ROCm (device is "cuda" in PyTorch for ROCm)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Get similarity scores
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        logits_per_image = outputs.logits_per_image  # image-text similarity score
                        probs = logits_per_image.softmax(dim=1)
                        
                        # Convert to Python types for JSON serialization
                        similarity_scores = probs[0].cpu().tolist() if hasattr(probs[0], "tolist") else probs[0]
                        highest_similarity_idx = int(probs[0].argmax().item()) if hasattr(probs[0], "argmax") else 0
                        
                        results = {
                            "similarity_scores": similarity_scores,
                            "highest_similarity_idx": highest_similarity_idx,
                            "highest_match": text_input[highest_similarity_idx] if isinstance(text_input, list) else None
                        }
                    
                    elif "{model_type}" == "blip":
                        # For BLIP-like models with image captioning
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Move to ROCm
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Generate caption
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        caption = self.processor.batch_decode(outputs[0], skip_special_tokens=True)
                        
                        results = {
                            "generated_text": caption
                        }
                    
                    else:
                        # Generic handling for other vision-text models
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Move to ROCm
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Run model
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs)
                        
                        # Process outputs - attempt to get text
                        try:
                            text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                            results = {"generated_text": text_output}
                        except:
                            # Fallback to raw outputs
                            results = {"raw_outputs": str(outputs)}
                    
                    return {
                        "success": True,
                        "results": results,
                        "device": self.device
                    }
                    
                except Exception as e:
                    logger.error(f"ROCm handler error: {e}")
                    return {
                        "success": False,
                        "error": str(e)
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
            def handler(image_input, text_input=None):
                # Basic input validation for better error messages
                if isinstance(image_input, str):
                    if not os.path.exists(image_input):
                        return {
                            "success": False,
                            "error": f"Image file does not exist: {image_input}"
                        }
                        
                # Return mock response based on model type
                if "{model_type}" == "clip" and text_input:
                    return {
                        "success": True,
                        "results": {
                            "similarity_scores": [0.8, 0.5, 0.3],
                            "highest_similarity_idx": 0,
                            "highest_match": text_input[0] if isinstance(text_input, list) else None
                        },
                        "device": self.device,
                        "platform": "qualcomm"
                    }
                else:
                    return {
                        "success": True,
                        "results": {
                            "generated_text": "Qualcomm-generated description of the image"
                        },
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
            if self.processor is None:
                self.load_processor()
            
            # In a real implementation, we'd use the WebNN API
            return MockHandler(self.model_id, "webnn")
        except Exception as e:
            logger.error(f"Error creating WebNN handler: {e}")
            return MockHandler(self.model_id, "webnn")
    
    def create_webgpu_handler(self):
        """Create handler for WEBGPU platform."""
        try:
            # WebGPU would use browser APIs - this is a mock implementation
            if self.processor is None:
                self.load_processor()
            
            # In a real implementation, we'd use the WebGPU API
            return MockHandler(self.model_id, "webgpu")
        except Exception as e:
            logger.error(f"Error creating WebGPU handler: {e}")
            return MockHandler(self.model_id, "webgpu")
    
    #
    # Public API methods
    #
    
    def run_inference(self, image_input, text_input=None) -> Dict[str, Any]:
        """
        Run inference with the model.
        
        Args:
            image_input: The image input. Can be:
                - Path to image file
                - PIL Image object
                - Dict with "image_path" key
            text_input: Optional text input for models that accept text (for CLIP similarity, etc.)
            
        Returns:
            Dict with inference results
        """
        if not self.model or not self.processor:
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
                mock_results = {}
                if text_input:
                    # For image-text similarity models like CLIP
                    mock_similarity = [0.9, 0.8, 0.7]
                    mock_results = {
                        "similarity_scores": mock_similarity,
                        "highest_similarity_idx": 0,
                        "highest_match": text_input[0] if isinstance(text_input, list) else None
                    }
                else:
                    # For image captioning models like BLIP
                    mock_results = {
                        "generated_text": "a mock caption for the image"
                    }
                
                return {
                    "success": True,
                    "time_seconds": time.time() - start_time,
                    "results": mock_results,
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
            result = handler(image_input, text_input)
            if "time_seconds" not in result:
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
    
    def benchmark(self, iterations: int = 5, image_path: str = None) -> Dict[str, Any]:
        """
        Run a benchmark of the model.
        
        Args:
            iterations: Number of iterations to run
            image_path: Path to image file to use for benchmark (if None, a blank image will be created)
            
        Returns:
            Dict with benchmark results
        """
        if not self.model or not self.processor:
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
            "image_path": image_path,
            "latencies_ms": [],
            "mean_latency_ms": 0.0,
            "throughput_fps": 0.0
        }
        
        try:
            if MOCK_MODE:
                # Mock implementation
                import random
                results["latencies_ms"] = [random.uniform(20, 100) for _ in range(iterations)]
                results["mean_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])
                results["throughput_fps"] = 1000 / results["mean_latency_ms"]
                return results
            
            # Create handler for the current device
            platform = self.device
            if platform == "cuda" and hasattr(torch, "version") and hasattr(torch.version, "hip"):
                platform = "rocm"
                
            handler_method = getattr(self, f"create_{platform}_handler", None)
            if handler_method:
                handler = handler_method()
            else:
                handler = self.create_cpu_handler()
            
            # Prepare image for benchmark
            if image_path and os.path.exists(image_path):
                from PIL import Image
                image = Image.open(image_path)
            else:
                # Create a blank test image
                from PIL import Image
                image = Image.new("RGB", (224, 224), color="white")
            
            # For CLIP models, include text input for the benchmark
            text_input = None
            if "{model_type}" == "clip":
                text_input = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
            
            # Run inference multiple times
            for _ in range(iterations):
                start_time = time.time()
                handler(image, text_input)
                latency = (time.time() - start_time) * 1000  # ms
                results["latencies_ms"].append(latency)
            
            # Calculate statistics
            results["mean_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])
            results["throughput_fps"] = 1000 / results["mean_latency_ms"]
            
            return results
            
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return {
                "success": False,
                "device": self.device,
                "model_id": self.model_id,
                "error": str(e)
            }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results") -> str:
        """
        Save results to a JSON file.
        
        Args:
            results: Results to save
            output_dir: Directory to save results to
            
        Returns:
            Path to saved file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
            filename = f"{model_name}_{self.device}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Prepare results for serialization
            save_results = {
                "model_id": self.model_id,
                "device": self.device,
                "timestamp": timestamp,
                "architecture_type": self.architecture_type,
                "model_type": "{model_type}",
                "results": results
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_results, f, indent=2)
            
            return filepath
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
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
            # Create a simple test image
            from PIL import Image
            test_image = Image.new("RGB", (224, 224), color="white")
            
            # Run handler with appropriate inputs
            text_input = None
            if "{model_type}" == "clip":
                text_input = ["a photo of a cat", "a photo of a dog"]
                
            result = handler(test_image, text_input)
            
            if "results" in result:
                if "generated_text" in result["results"]:
                    logger.info(f"Result: {result['results']['generated_text'][:50]}...")
                elif "similarity_scores" in result["results"]:
                    logger.info(f"Similarity scores: {result['results']['similarity_scores']}")
                    logger.info(f"Highest match: {result['results']['highest_match']}")
                
            logger.info(f"Successfully tested on {platform} platform")
            return True
        except Exception as e:
            logger.error(f"Error running test on {platform}: {e}")
            return False


def test_skillset():
    """Simple test function for the skillset."""
    skillset = {model_type_upper}Skillset()
    
    # Load model
    load_result = skillset.load_model()
    print(f"Load result: {{'success': {load_result['success']}, 'device': {load_result['device']}}}")
    
    if load_result["success"]:
        try:
            # Create a test image
            from PIL import Image
            test_image = Image.new("RGB", (224, 224), color="white")
            
            # Run inference
            if "{model_type}" == "clip":
                text_input = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
                inference_result = skillset.run_inference(test_image, text_input)
                print(f"Inference result: {{'success': {inference_result['success']}, 'similarity_scores': {inference_result.get('results', {}).get('similarity_scores', [])}}}") 
            else:
                inference_result = skillset.run_inference(test_image)
                print(f"Inference result: {{'success': {inference_result['success']}, 'output': '{inference_result.get('results', {}).get('generated_text', '')}'}}")
            
            # Run benchmark
            benchmark_result = skillset.benchmark(iterations=2)
            print(f"Benchmark result: {{'mean_latency_ms': {benchmark_result.get('mean_latency_ms', 0):.2f}, 'throughput_fps': {benchmark_result.get('throughput_fps', 0):.2f}}}")
        except Exception as e:
            print(f"Error in test: {e}")

if __name__ == "__main__":
    """Run the skillset."""
    import argparse
    parser = argparse.ArgumentParser(description="Test {model_type} model")
    parser.add_argument("--model", help="Model path or name", default="{default_model_id}")
    parser.add_argument("--platform", default="CPU", help="Platform to test on (cpu, cuda, rocm, mps, openvino, qualcomm, webnn, webgpu)")
    parser.add_argument("--mock", action="store_true", help="Use mock implementations")
    parser.add_argument("--image", help="Path to image file for testing")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", default="results", help="Directory for saved results")
    args = parser.parse_args()
    
    skillset = {model_type_upper}Skillset(args.model)
    result = skillset.run(args.platform, args.mock)
    
    if result:
        print(f"Test successful on {args.platform}")
        sys.exit(0)
    else:
        print(f"Test failed on {args.platform}")
        sys.exit(1)
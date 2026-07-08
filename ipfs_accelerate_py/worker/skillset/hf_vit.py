"""
Vision Transformer (ViT) implementation for image classification using Hugging Face transformers.
Supports CPU, CUDA, and OpenVINO backends.
"""

import os
import time
import traceback
from typing import Tuple, Callable, Dict, Any, Optional, Union, List
import logging

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

# Set up logging
logger = logging.getLogger(__name__)

class hf_vit:
    """
    Hugging Face Vision Transformer (ViT) implementation for image classification.
    Supports image classification with different ViT architectures.
    """
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the Vision Transformer module.
        
        Args:
            resources (dict, optional): Resource dictionary containing dependencies
            metadata (dict, optional): Metadata dictionary
        """
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper(auto_detect_ci=True)
            except Exception:
                self._storage = None
        else:
            self._storage = None
        
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Flag for tracking if we're using mocked dependencies
        self.using_mocks = False
        
        # Check for required packages
        for pkg_name in ["torch", "transformers", "numpy"]:
            if pkg_name not in self.resources:
                try:
                    if pkg_name == "torch":
                        import torch
                        self.resources["torch"] = torch
                    elif pkg_name == "transformers":
                        import transformers
                        self.resources["transformers"] = transformers
                    elif pkg_name == "numpy":
                        import numpy
                        self.resources["numpy"] = numpy
                except ImportError:
                    from unittest.mock import MagicMock
                    self.resources[pkg_name] = MagicMock()
                    self.using_mocks = True
                    logger.warning(f"{pkg_name} not available, using mock implementation")
        
        # Try to import PIL, which is needed for image processing
        if "PIL" not in self.resources:
            try:
                from PIL import Image
                self.resources["PIL"] = Image
            except ImportError:
                from unittest.mock import MagicMock
                self.resources["PIL"] = MagicMock()
                self.using_mocks = True
                logger.warning("PIL not available, using mock implementation")
        
        return None

    def init_cpu(self, model_name: str, model_type: str, device_label: str = "cpu", **kwargs) -> Tuple:
        """
        Initialize Vision Transformer model for CPU inference.
        
        Args:
            model_name (str): Name or path of the model (e.g., "google/vit-base-patch16-224")
            model_type (str): Type of model (typically "image-classification")
            device_label (str): CPU device label (typically "cpu")
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (endpoint, processor, handler, queue, batch_size)
        """
        logger.info(f"Loading {model_name} for CPU inference...")
        
        torch = self.resources.get("torch")
        transformers = self.resources.get("transformers")
        Image = self.resources.get("PIL")
        
        # Check if we're using mocks
        if self.using_mocks or isinstance(transformers, type(type)) or isinstance(torch, type(type)):
            logger.warning("Using mock implementation for CPU")
            return self._create_mock_implementation(model_name, device_label)
        
        try:
            # Import the necessary components for ViT
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Load the processor and model
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            
            logger.info(f"Successfully loaded ViT model and processor for {model_name}")
            
            # Create handler function
            def handler(image_input):
                """
                Handle image classification requests.
                
                Args:
                    image_input (str or PIL.Image): Path to image or PIL Image object
                    
                Returns:
                    dict: Classification results including class label, confidence, and top predictions
                """
                try:
                    start_time = time.time()
                    
                    # Load image
                    if isinstance(image_input, str):
                        if os.path.exists(image_input):
                            image = Image.open(image_input).convert("RGB")
                        else:
                            raise ValueError(f"Image path {image_input} does not exist")
                    elif isinstance(image_input, Image.Image):
                        image = image_input
                    else:
                        raise ValueError(f"Unsupported image input type: {type(image_input)}")
                    
                    # Process image
                    inputs = processor(images=image, return_tensors="pt")
                    
                    # Run model inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Get classification results
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    
                    # Get class label if available, otherwise return index
                    if hasattr(model.config, "id2label") and predicted_class_idx in model.config.id2label:
                        class_label = model.config.id2label[predicted_class_idx]
                    else:
                        class_label = f"Class {predicted_class_idx}"
                    
                    # Calculate confidence scores
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs[0, predicted_class_idx].item()
                    
                    # Get top 5 predictions if available
                    top_5_indices = probs[0].topk(min(5, probs.shape[1])).indices.tolist()
                    top_5_probs = probs[0].topk(min(5, probs.shape[1])).values.tolist()
                    
                    top_5_predictions = []
                    for idx, prob in zip(top_5_indices, top_5_probs):
                        if hasattr(model.config, "id2label") and idx in model.config.id2label:
                            label = model.config.id2label[idx]
                        else:
                            label = f"Class {idx}"
                        top_5_predictions.append({"label": label, "confidence": prob})
                    
                    return {
                        "class": class_label,
                        "confidence": confidence,
                        "top_predictions": top_5_predictions,
                        "processing_time": time.time() - start_time,
                        "implementation_type": "REAL"
                    }
                    
                except Exception as e:
                    logger.error(f"Error in CPU handler: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return {
                        "error": str(e),
                        "implementation_type": "REAL",
                        "is_error": True
                    }
            
            return model, processor, handler, None, 1  # batch size 1
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Falling back to mock implementation")
            
            return self._create_mock_implementation(model_name, device_label)

    def init_cuda(self, model_name: str, model_type: str, device_label: str = "cuda:0", **kwargs) -> Tuple:
        """
        Initialize Vision Transformer model with CUDA support.
        
        Args:
            model_name (str): Name or path of the model (e.g., "google/vit-base-patch16-224")
            model_type (str): Type of model (typically "image-classification")
            device_label (str): CUDA device label (e.g., "cuda:0")
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (endpoint, processor, handler, queue, batch_size)
        """
        logger.info(f"Loading {model_name} for CUDA inference...")
        
        torch = self.resources.get("torch")
        transformers = self.resources.get("transformers")
        Image = self.resources.get("PIL")
        
        # Check if we're using mocks or if CUDA is not available
        if self.using_mocks or isinstance(transformers, type(type)) or isinstance(torch, type(type)):
            logger.warning("Using mock implementation for CUDA due to missing dependencies")
            return self._create_mock_implementation(model_name, device_label)
        
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU implementation")
                return self.init_cpu(model_name, model_type, device_label="cpu")
            
            # Import the necessary components for ViT
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Initialize CUDA device
            torch_device = torch.device(device_label)
            logger.info(f"Using CUDA device: {torch_device}")
            
            # Load the processor and model
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            
            # Move model to CUDA and optimize
            model = model.to(torch_device)
            model = model.eval()
            
            # Try to use half-precision for better CUDA performance
            try:
                model = model.half()  # Convert to FP16
                logger.info("Using FP16 precision for faster inference")
            except Exception as half_err:
                logger.warning(f"Unable to use half precision: {half_err}")
            
            logger.info(f"Successfully loaded ViT model to {torch_device}")
            
            # Create handler function
            def handler(image_input):
                """
                Handle image classification requests with CUDA acceleration.
                
                Args:
                    image_input (str or PIL.Image): Path to image or PIL Image object
                    
                Returns:
                    dict: Classification results including class label, confidence, and top predictions
                """
                try:
                    start_time = time.time()
                    
                    # Track GPU memory
                    gpu_mem_before = torch.cuda.memory_allocated(torch_device) / (1024 * 1024)
                    
                    # Load image
                    if isinstance(image_input, str):
                        if os.path.exists(image_input):
                            image = Image.open(image_input).convert("RGB")
                        else:
                            raise ValueError(f"Image path {image_input} does not exist")
                    elif isinstance(image_input, Image.Image):
                        image = image_input
                    else:
                        raise ValueError(f"Unsupported image input type: {type(image_input)}")
                    
                    # Process image
                    inputs = processor(images=image, return_tensors="pt")
                    
                    # Move inputs to CUDA
                    inputs = {key: val.to(torch_device) for key, val in inputs.items()}
                    
                    # Run model inference with CUDA synchronization
                    torch.cuda.synchronize()
                    inference_start = time.time()
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    torch.cuda.synchronize()
                    inference_time = time.time() - inference_start
                    
                    # Measure GPU memory
                    gpu_mem_after = torch.cuda.memory_allocated(torch_device) / (1024 * 1024)
                    gpu_mem_used = gpu_mem_after - gpu_mem_before
                    
                    # Get classification results
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    
                    # Get class label if available
                    if hasattr(model.config, "id2label") and predicted_class_idx in model.config.id2label:
                        class_label = model.config.id2label[predicted_class_idx]
                    else:
                        class_label = f"Class {predicted_class_idx}"
                    
                    # Calculate confidence scores
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs[0, predicted_class_idx].item()
                    
                    # Get top 5 predictions if available
                    top_5_indices = probs[0].topk(min(5, probs.shape[1])).indices.cpu().tolist()
                    top_5_probs = probs[0].topk(min(5, probs.shape[1])).values.cpu().tolist()
                    
                    top_5_predictions = []
                    for idx, prob in zip(top_5_indices, top_5_probs):
                        if hasattr(model.config, "id2label") and idx in model.config.id2label:
                            label = model.config.id2label[idx]
                        else:
                            label = f"Class {idx}"
                        top_5_predictions.append({"label": label, "confidence": prob})
                    
                    return {
                        "class": class_label,
                        "confidence": confidence,
                        "top_predictions": top_5_predictions,
                        "processing_time": time.time() - start_time,
                        "inference_time": inference_time,
                        "gpu_memory_mb": gpu_mem_used,
                        "device": str(torch_device),
                        "implementation_type": "REAL"
                    }
                    
                except Exception as e:
                    logger.error(f"Error in CUDA handler: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return {
                        "error": str(e),
                        "implementation_type": "REAL",
                        "is_error": True
                    }
            
            return model, processor, handler, None, 4  # batch size 4
            
        except Exception as e:
            logger.error(f"Error loading model with CUDA: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Falling back to mock implementation")
            
            return self._create_mock_implementation(model_name, device_label)

    def init_openvino(self, model_name: str, model_type: str, device: str = "CPU", 
                     openvino_label: str = "openvino:0", **kwargs) -> Tuple:
        """
        Initialize Vision Transformer model with OpenVINO support.
        
        Args:
            model_name (str): Name or path of the model (e.g., "google/vit-base-patch16-224")
            model_type (str): Type of model (typically "image-classification")
            device (str): OpenVINO device (e.g., "CPU", "GPU")
            openvino_label (str): OpenVINO device label
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (endpoint, processor, handler, queue, batch_size)
        """
        logger.info(f"Loading {model_name} for OpenVINO inference...")
        
        torch = self.resources.get("torch")
        transformers = self.resources.get("transformers")
        Image = self.resources.get("PIL")
        
        # Check if we're using mocks
        if self.using_mocks or isinstance(transformers, type(type)) or isinstance(torch, type(type)):
            logger.warning("Using mock implementation for OpenVINO due to missing dependencies")
            return self._create_mock_implementation(model_name, device)
        
        try:
            # Check if OpenVINO is available
            try:
                import openvino
                from openvino.runtime import Core
                has_openvino = True
                logger.info("OpenVINO is installed")
            except ImportError:
                has_openvino = False
                logger.warning("OpenVINO not installed, falling back to mock implementation")
                return self._create_mock_implementation(model_name, device)
            
            # Try to use optimum-intel for OpenVINO
            try:
                from optimum.intel.openvino import OVModelForImageClassification
                from transformers import AutoImageProcessor
                
                # Load the processor and model
                processor = AutoImageProcessor.from_pretrained(model_name)
                ov_model = OVModelForImageClassification.from_pretrained(
                    model_name, 
                    export=True,
                    device=device
                )
                
                logger.info(f"Successfully loaded ViT model with OpenVINO")
                
                # Create handler function
                def handler(image_input):
                    """
                    Handle image classification requests with OpenVINO acceleration.
                    
                    Args:
                        image_input (str or PIL.Image): Path to image or PIL Image object
                        
                    Returns:
                        dict: Classification results including class label, confidence, and top predictions
                    """
                    try:
                        start_time = time.time()
                        
                        # Load image
                        if isinstance(image_input, str):
                            if os.path.exists(image_input):
                                image = Image.open(image_input).convert("RGB")
                            else:
                                raise ValueError(f"Image path {image_input} does not exist")
                        elif isinstance(image_input, Image.Image):
                            image = image_input
                        else:
                            raise ValueError(f"Unsupported image input type: {type(image_input)}")
                        
                        # Process image
                        inputs = processor(images=image, return_tensors="pt")
                        
                        # Run model inference
                        outputs = ov_model(**inputs)
                        
                        # Get classification results
                        logits = outputs.logits
                        predicted_class_idx = logits.argmax(-1).item()
                        
                        # Get class label if available
                        if hasattr(ov_model.config, "id2label") and predicted_class_idx in ov_model.config.id2label:
                            class_label = ov_model.config.id2label[predicted_class_idx]
                        else:
                            class_label = f"Class {predicted_class_idx}"
                        
                        # Calculate confidence scores
                        probs = torch.softmax(logits, dim=-1)
                        confidence = probs[0, predicted_class_idx].item()
                        
                        # Get top 5 predictions if available
                        top_5_indices = probs[0].topk(min(5, probs.shape[1])).indices.tolist()
                        top_5_probs = probs[0].topk(min(5, probs.shape[1])).values.tolist()
                        
                        top_5_predictions = []
                        for idx, prob in zip(top_5_indices, top_5_probs):
                            if hasattr(ov_model.config, "id2label") and idx in ov_model.config.id2label:
                                label = ov_model.config.id2label[idx]
                            else:
                                label = f"Class {idx}"
                            top_5_predictions.append({"label": label, "confidence": prob})
                        
                        return {
                            "class": class_label,
                            "confidence": confidence,
                            "top_predictions": top_5_predictions,
                            "processing_time": time.time() - start_time,
                            "device": device,
                            "implementation_type": "REAL"
                        }
                        
                    except Exception as e:
                        logger.error(f"Error in OpenVINO handler: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        return {
                            "error": str(e),
                            "implementation_type": "REAL",
                            "is_error": True
                        }
                
                return ov_model, processor, handler, None, 1
                
            except Exception as optimum_err:
                logger.warning(f"Error using optimum-intel: {optimum_err}")
                logger.info("Falling back to direct OpenVINO implementation")
                
                # Manual conversion to OpenVINO IR
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                import numpy as np
                
                # Load the processor and model
                processor = AutoImageProcessor.from_pretrained(model_name)
                original_model = AutoModelForImageClassification.from_pretrained(model_name)
                
                # Cache directory for converted models
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "openvino_models")
                os.makedirs(cache_dir, exist_ok=True)
                
                model_hash = hash(model_name) % 10000
                ov_model_path = os.path.join(cache_dir, f"vit_{model_hash}.xml")
                
                # Convert to ONNX and then to OpenVINO IR if not already converted
                if not os.path.exists(ov_model_path):
                    logger.info(f"Converting {model_name} to OpenVINO IR format...")
                    
                    # Create a temp directory for ONNX
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        onnx_path = os.path.join(tmpdirname, "model.onnx")
                        
                        # Create dummy input for tracing
                        dummy_input = {
                            "pixel_values": torch.randn(1, 3, 224, 224)
                        }
                        
                        # Export to ONNX
                        torch.onnx.export(
                            original_model,
                            (dummy_input,),
                            onnx_path,
                            opset_version=12,
                            input_names=["pixel_values"],
                            output_names=["logits"],
                            dynamic_axes={
                                "pixel_values": {0: "batch_size"},
                                "logits": {0: "batch_size"}
                            }
                        )
                        
                        # Convert ONNX to OpenVINO IR
                        core = Core()
                        ov_model = core.read_model(onnx_path)
                        compiled_model = core.compile_model(ov_model, device)
                        
                        # Save the model
                        from openvino.runtime import serialize
                        serialize(ov_model, ov_model_path)
                        
                    logger.info(f"Model converted and saved to {ov_model_path}")
                    
                # Load OpenVINO model
                core = Core()
                ov_model = core.read_model(ov_model_path)
                compiled_model = core.compile_model(ov_model, device)
                
                output_layer = compiled_model.output(0)
                
                # Create handler function
                def handler(image_input):
                    """
                    Handle image classification requests with direct OpenVINO implementation.
                    
                    Args:
                        image_input (str or PIL.Image): Path to image or PIL Image object
                        
                    Returns:
                        dict: Classification results including class label, confidence, and top predictions
                    """
                    try:
                        start_time = time.time()
                        
                        # Load image
                        if isinstance(image_input, str):
                            if os.path.exists(image_input):
                                image = Image.open(image_input).convert("RGB")
                            else:
                                raise ValueError(f"Image path {image_input} does not exist")
                        elif isinstance(image_input, Image.Image):
                            image = image_input
                        else:
                            raise ValueError(f"Unsupported image input type: {type(image_input)}")
                        
                        # Process image with processor
                        inputs = processor(images=image, return_tensors="np")
                        
                        # Run inference
                        results = compiled_model(inputs["pixel_values"])[output_layer]
                        
                        # Create torch tensors for processing
                        logits = torch.from_numpy(results)
                        
                        # Get classification results
                        predicted_class_idx = logits.argmax(-1).item()
                        
                        # Get class label if available
                        if hasattr(original_model.config, "id2label") and predicted_class_idx in original_model.config.id2label:
                            class_label = original_model.config.id2label[predicted_class_idx]
                        else:
                            class_label = f"Class {predicted_class_idx}"
                        
                        # Calculate confidence scores
                        probs = torch.softmax(logits, dim=-1)
                        confidence = probs[0, predicted_class_idx].item()
                        
                        # Get top 5 predictions if available
                        top_5_indices = probs[0].topk(min(5, probs.shape[1])).indices.tolist()
                        top_5_probs = probs[0].topk(min(5, probs.shape[1])).values.tolist()
                        
                        top_5_predictions = []
                        for idx, prob in zip(top_5_indices, top_5_probs):
                            if hasattr(original_model.config, "id2label") and idx in original_model.config.id2label:
                                label = original_model.config.id2label[idx]
                            else:
                                label = f"Class {idx}"
                            top_5_predictions.append({"label": label, "confidence": prob})
                        
                        return {
                            "class": class_label,
                            "confidence": confidence,
                            "top_predictions": top_5_predictions,
                            "processing_time": time.time() - start_time,
                            "device": device,
                            "implementation_type": "REAL"
                        }
                        
                    except Exception as e:
                        logger.error(f"Error in OpenVINO handler: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        return {
                            "error": str(e),
                            "implementation_type": "REAL",
                            "is_error": True
                        }
                
                return compiled_model, processor, handler, None, 1
                
        except Exception as e:
            logger.error(f"Error setting up OpenVINO: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Falling back to mock implementation")
            
            return self._create_mock_implementation(model_name, device)

    def _create_mock_implementation(self, model_name: str, device: str) -> Tuple:
        """
        Create a mock implementation for the ViT model.
        
        Args:
            model_name (str): Name or path of the model
            device (str): Device label
            
        Returns:
            tuple: (endpoint, processor, handler, queue, batch_size)
        """
        from unittest.mock import MagicMock
        
        # Create mock implementations
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Set some attributes to make it look more realistic
        config = MagicMock()
        config.id2label = {
            0: "cat",
            1: "dog",
            2: "bird",
            3: "fish",
            4: "person"
        }
        mock_model.config = config
        
        # Create a mock handler with realistic outputs
        def mock_handler(image_input):
            """
            Mock handler for image classification.
            
            Args:
                image_input (str or PIL.Image): Path to image or PIL Image object
                
            Returns:
                dict: Mock classification results
            """
            # Generate deterministic class based on input to make it more realistic
            if isinstance(image_input, str):
                # Use the filename to determine the mock class
                if "cat" in image_input.lower():
                    class_idx = 0
                elif "dog" in image_input.lower():
                    class_idx = 1
                elif "bird" in image_input.lower():
                    class_idx = 2
                elif "fish" in image_input.lower():
                    class_idx = 3
                else:
                    class_idx = 4
            else:
                # Default to person
                class_idx = 4
                
            class_label = mock_model.config.id2label[class_idx]
            
            # Create realistic top predictions
            top_predictions = []
            confidences = [0.8, 0.1, 0.05, 0.03, 0.02]
            
            for i, conf in enumerate(confidences):
                idx = (class_idx + i) % 5
                top_predictions.append({
                    "label": mock_model.config.id2label[idx],
                    "confidence": conf
                })
            
            # Add device-specific information
            if "cuda" in str(device).lower():
                return {
                    "class": class_label,
                    "confidence": 0.8,
                    "top_predictions": top_predictions,
                    "processing_time": 0.05,
                    "inference_time": 0.03,
                    "gpu_memory_mb": 120,
                    "device": device,
                    "implementation_type": "MOCK"
                }
            elif "openvino" in str(device).lower() or device == "CPU":
                return {
                    "class": class_label,
                    "confidence": 0.8,
                    "top_predictions": top_predictions,
                    "processing_time": 0.15,
                    "device": device,
                    "implementation_type": "MOCK"
                }
            else:
                return {
                    "class": class_label,
                    "confidence": 0.8,
                    "top_predictions": top_predictions,
                    "processing_time": 0.1,
                    "implementation_type": "MOCK"
                }
        
        logger.info(f"Created mock implementation for {model_name} on {device}")
        return mock_model, mock_processor, mock_handler, None, 1
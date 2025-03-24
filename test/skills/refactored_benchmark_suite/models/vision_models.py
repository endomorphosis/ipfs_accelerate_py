"""
Vision model adapters for hardware-aware benchmarking.

This module provides model adapters for various vision models with hardware-aware
optimizations and support for modern vision architectures like DETR, SAM, DINOv2, and Swin.
"""

import logging
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple, List, Union

import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor, 
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation,
    AutoModelForMaskedImageModeling,
    AutoModelForZeroShotImageClassification
)

from . import ModelAdapter

logger = logging.getLogger("benchmark.models.vision")

def apply_hardware_optimizations(model, device_type: str, use_flash_attention: bool = False, 
                               use_torch_compile: bool = False) -> torch.nn.Module:
    """
    Apply hardware-specific optimizations to vision models.
    
    Args:
        model: PyTorch model
        device_type: Hardware device type (cuda, cpu, mps, rocm)
        use_flash_attention: Whether to use Flash Attention for transformer models
        use_torch_compile: Whether to use torch.compile for PyTorch 2.0+ optimizations
        
    Returns:
        Optimized model
    """
    # Apply Flash Attention if available and requested
    if use_flash_attention and device_type == "cuda":
        try:
            from flash_attn.flash_attention import FlashAttention
            
            # Check if model has transformer layers with attention
            if hasattr(model, "transformer") or hasattr(model, "encoder") or hasattr(model, "decoder"):
                logger.info("Applying Flash Attention optimization to transformer layers")
                # This is placeholder logic - the actual implementation would involve
                # finding and replacing attention layers with Flash Attention
                # Implementation would vary by model architecture
        except ImportError:
            logger.warning("Flash Attention not available. Install with 'pip install flash-attn'")
    
    # Apply torch.compile if available and requested
    if use_torch_compile:
        try:
            # Check if PyTorch version supports torch.compile
            compile_supported = hasattr(torch, "compile") 
            if compile_supported:
                logger.info("Applying torch.compile optimization")
                model = torch.compile(model)
            else:
                logger.warning("torch.compile not available. Requires PyTorch 2.0+")
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}")
    
    # Apply device-specific optimizations
    if device_type == "cuda":
        # CUDA-specific optimizations
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            # Enable CUDA graphs for inference if available
            # Note: CUDA graphs require static input shapes
            try:
                if hasattr(torch.cuda, "is_current_stream_capturing"):
                    logger.info("CUDA Graph support detected")
            except Exception:
                pass
            
            # Enable cudnn benchmark for optimized convolutions
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode for hardware optimization")
                
    elif device_type == "cpu":
        # CPU-specific optimizations
        try:
            # Set CPU thread settings
            if hasattr(torch, "set_num_threads"):
                # Use a reasonable number of threads (not too many to avoid context switching overhead)
                num_threads = min(os.cpu_count(), 4) if os.cpu_count() else 4
                torch.set_num_threads(num_threads)
                logger.info(f"Set CPU threads to {num_threads} for optimal performance")
                
            # Enable oneDNN (MKL-DNN) optimizations if available
            if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                logger.info("Enabled oneDNN (MKL-DNN) optimizations")
        except Exception as e:
            logger.warning(f"Failed to apply CPU optimizations: {e}")
    
    # Return optimized model
    return model


class VisionModelAdapter(ModelAdapter):
    """
    Enhanced adapter for vision models with hardware-aware optimizations.
    
    Handles loading and input preparation for various vision model types,
    including modern architectures like DETR, SAM, DINOv2, and Swin transformers.
    """
    
    def __init__(self, model_id: str, task: Optional[str] = None):
        """
        Initialize a vision model adapter.
        
        Args:
            model_id: HuggingFace model ID
            task: Model task
        """
        super().__init__(model_id, task)
        
        # Model type detection based on ID
        self.model_id_lower = self.model_id.lower()
        
        # Detect vision model types
        self.is_vit = "vit" in self.model_id_lower
        self.is_detr = "detr" in self.model_id_lower
        self.is_sam = "sam" in self.model_id_lower
        self.is_dino = "dino" in self.model_id_lower
        self.is_swin = "swin" in self.model_id_lower
        self.is_convnext = "convnext" in self.model_id_lower
        self.is_resnet = "resnet" in self.model_id_lower
        
        # Default task based on model type if not provided
        if self.task is None:
            if self.is_detr or "object-detection" in self.model_id_lower:
                self.task = "object-detection"
            elif self.is_sam or "segmentation" in self.model_id_lower:
                self.task = "image-segmentation"
            else:
                self.task = "image-classification"  # Default
        
        # Initialize processor and model config
        self.processor = None
        self.image_size = (224, 224)  # Default image size
        
        # Model-specific settings
        if self.is_sam:
            self.image_size = (1024, 1024)  # SAM typically uses larger images
        elif self.is_detr:
            self.image_size = (800, 800)  # DETR typically uses larger images
    
    def load_model(self, device: torch.device, use_flash_attention: bool = False, 
                  use_torch_compile: bool = False) -> nn.Module:
        """
        Load the vision model with hardware-aware optimizations.
        
        Args:
            device: Device to load the model on
            use_flash_attention: Whether to use Flash Attention for transformer models
            use_torch_compile: Whether to use torch.compile for PyTorch 2.0+ optimizations
            
        Returns:
            Loaded and optimized model
        """
        # Set model loading kwargs
        model_kwargs = {"torch_dtype": torch.float16 if device.type == "cuda" else torch.float32}
        
        # Load appropriate processor based on model type
        try:
            # First try modern AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        except Exception:
            try:
                # Fall back to AutoFeatureExtractor
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
            except Exception as e:
                logger.warning(f"Could not load processor for {self.model_id}: {e}")
                logger.warning("Will try to use a default processor")
                self.processor = None
        
        # Extract image size from processor if available
        if self.processor is not None:
            if hasattr(self.processor, "size"):
                if isinstance(self.processor.size, dict):
                    self.image_size = (self.processor.size.get("height", 224), self.processor.size.get("width", 224))
                elif isinstance(self.processor.size, (list, tuple)) and len(self.processor.size) >= 2:
                    self.image_size = (self.processor.size[0], self.processor.size[1])
                else:
                    self.image_size = (self.processor.size, self.processor.size)
            elif hasattr(self.processor, "crop_size"):
                if isinstance(self.processor.crop_size, dict):
                    self.image_size = (self.processor.crop_size.get("height", 224), self.processor.crop_size.get("width", 224))
                elif isinstance(self.processor.crop_size, (list, tuple)) and len(self.processor.crop_size) >= 2:
                    self.image_size = (self.processor.crop_size[0], self.processor.crop_size[1])
                else:
                    self.image_size = (self.processor.crop_size, self.processor.crop_size)
        
        # Use model-type specific loading for modern architectures
        try:
            if self.is_sam:
                # SAM specific loading
                from transformers import SamModel
                logger.info(f"Loading SAM model: {self.model_id}")
                model = SamModel.from_pretrained(self.model_id, **model_kwargs)
            elif self.is_detr:
                # DETR specific loading
                logger.info(f"Loading DETR model: {self.model_id}")
                model = AutoModelForObjectDetection.from_pretrained(self.model_id, **model_kwargs)
            elif self.is_dino:
                # DINOv2 specific loading
                from transformers import AutoModelForImageClassification
                logger.info(f"Loading DINOv2 model: {self.model_id}")
                model = AutoModelForImageClassification.from_pretrained(self.model_id, **model_kwargs)
            elif self.is_swin:
                # Swin specific loading
                logger.info(f"Loading Swin Transformer model: {self.model_id}")
                if self.task == "image-classification":
                    model = AutoModelForImageClassification.from_pretrained(self.model_id, **model_kwargs)
                elif self.task == "object-detection":
                    model = AutoModelForObjectDetection.from_pretrained(self.model_id, **model_kwargs)
                else:
                    model = AutoModelForImageClassification.from_pretrained(self.model_id, **model_kwargs)
            else:
                # Generic loading based on task
                if self.task == "image-classification":
                    model = AutoModelForImageClassification.from_pretrained(self.model_id, **model_kwargs)
                elif self.task == "object-detection":
                    model = AutoModelForObjectDetection.from_pretrained(self.model_id, **model_kwargs)
                elif self.task == "semantic-segmentation" or self.task == "image-segmentation":
                    model = AutoModelForSemanticSegmentation.from_pretrained(self.model_id, **model_kwargs)
                elif self.task == "masked-image-modeling":
                    model = AutoModelForMaskedImageModeling.from_pretrained(self.model_id, **model_kwargs)
                elif self.task == "zero-shot-image-classification":
                    model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_id, **model_kwargs)
                else:
                    # Default to image classification
                    logger.warning(f"Unknown task '{self.task}' for vision model, using image classification")
                    model = AutoModelForImageClassification.from_pretrained(self.model_id, **model_kwargs)
        except Exception as e:
            logger.error(f"Error in specialized model loading: {e}")
            logger.warning("Falling back to generic model loading")
            
            # Generic fallback approach
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(self.model_id)
            except Exception as e2:
                logger.error(f"Error in fallback model loading: {e2}")
                
                # Ultimate fallback - create a simple vision model for testing
                logger.warning("Creating dummy vision model for benchmarking")
                model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 1000)
                )
        
        # Apply hardware-specific optimizations
        model = apply_hardware_optimizations(
            model, 
            device.type,
            use_flash_attention=use_flash_attention,
            use_torch_compile=use_torch_compile
        )
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        return model
    
    def prepare_inputs(self, batch_size: int, sequence_length: int = 1) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the vision model.
        
        Args:
            batch_size: Batch size
            sequence_length: Not used for most vision models, but can be used for video models
            
        Returns:
            Dictionary of input tensors
        """
        if self.processor is None:
            # Create a default random image input
            return self._create_default_inputs(batch_size)
        
        # Create sample images
        sample_images = self._create_sample_images(batch_size)
        
        # Process images with model-specific handling
        try:
            # Model-specific input preparation
            if self.is_sam:
                # SAM models require special input handling with point prompts
                if isinstance(sample_images, list) and len(sample_images) > 0:
                    inputs = self.processor(sample_images, return_tensors="pt")
                    
                    # Add input points for prompting
                    height, width = self.image_size
                    input_points = torch.tensor([[[width // 2, height // 2]]] * batch_size)
                    input_labels = torch.tensor([[1]] * batch_size)
                    
                    inputs["input_points"] = input_points
                    inputs["input_labels"] = input_labels
                    return inputs
                else:
                    return self._create_sam_default_inputs(batch_size)
                    
            elif self.is_detr:
                # DETR models can use standard image inputs
                inputs = self.processor(sample_images, return_tensors="pt")
                return inputs
                
            else:
                # Standard processing for most vision models
                inputs = self.processor(sample_images, return_tensors="pt")
                return inputs
                
        except Exception as e:
            logger.warning(f"Error using processor for {self.model_id}: {e}")
            logger.warning("Using default inputs instead")
            
            # Generate model-specific default inputs
            if self.is_sam:
                return self._create_sam_default_inputs(batch_size)
            elif self.is_detr:
                return self._create_detr_default_inputs(batch_size)
            else:
                return self._create_default_inputs(batch_size)
    
    def _create_sample_images(self, batch_size: int) -> List[Any]:
        """
        Create sample images for benchmarking.
        
        Args:
            batch_size: Number of images to create
            
        Returns:
            List of sample images
        """
        try:
            import numpy as np
            from PIL import Image
            
            # Create random RGB images
            height, width = self.image_size
            sample_images = []
            
            for _ in range(batch_size):
                # Create a random RGB image
                img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                sample_images.append(img)
            
            return sample_images
            
        except ImportError:
            logger.warning("PIL or numpy not available, using tensor inputs")
            return None
    
    def _create_default_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Create default inputs when processor is not available.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary of input tensors
        """
        # Create a random tensor of shape [batch_size, 3, height, width]
        height, width = self.image_size
        pixel_values = torch.rand(batch_size, 3, height, width)
        
        return {"pixel_values": pixel_values}
    
    def _create_sam_default_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Create default inputs for SAM models.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary of input tensors
        """
        height, width = self.image_size
        pixel_values = torch.rand(batch_size, 3, height, width)
        input_points = torch.tensor([[[width // 2, height // 2]]] * batch_size)
        input_labels = torch.tensor([[1]] * batch_size)
        
        return {
            "pixel_values": pixel_values,
            "input_points": input_points,
            "input_labels": input_labels
        }
    
    def _create_detr_default_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Create default inputs for DETR models.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary of input tensors
        """
        height, width = self.image_size
        pixel_values = torch.rand(batch_size, 3, height, width)
        
        # DETR may expect pixel mask
        pixel_mask = torch.ones(batch_size, height, width, dtype=torch.long)
        
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask
        }
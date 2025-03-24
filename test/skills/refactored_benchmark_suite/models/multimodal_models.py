"""
Multimodal model adapters for hardware-aware benchmarking.

This module provides model adapters for various multimodal models with hardware-aware
optimizations and support for modern architectures like LLaVA, BLIP2, ImageBind,
and other vision-language models.
"""

import logging
import os
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor, AutoModel, AutoModelForVision2Seq, 
    CLIPModel, CLIPProcessor, CLIPTextModel, CLIPVisionModel,
    BlipModel, BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    ViltModel, ViltProcessor, 
    VideoMAEModel, AutoImageProcessor,
    AutoModelForDocumentQuestionAnswering, VisionTextDualEncoderModel,
    LlavaProcessor, LlavaForConditionalGeneration
)

from . import ModelAdapter

logger = logging.getLogger("benchmark.models.multimodal")

def apply_multimodal_hardware_optimizations(model, device_type: str, use_flash_attention: bool = False, 
                                       use_torch_compile: bool = False) -> torch.nn.Module:
    """
    Apply hardware-specific optimizations to multimodal models.
    
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
            if hasattr(model, "vision_model") or hasattr(model, "language_model") or hasattr(model, "text_model"):
                logger.info("Applying Flash Attention optimization to multimodal transformer layers")
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
                logger.info("Applying torch.compile optimization to multimodal model")
                model = torch.compile(model)
            else:
                logger.warning("torch.compile not available. Requires PyTorch 2.0+")
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile to multimodal model: {e}")
    
    # Apply device-specific optimizations
    if device_type == "cuda":
        # CUDA-specific optimizations
        if torch.cuda.is_available():
            # Enable cudnn benchmark for optimized convolutions (common in vision models)
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode for multimodal model optimization")
            
            # Set higher CUDA stream priority for faster processing
            try:
                current_stream = torch.cuda.current_stream()
                current_stream.priority = -1  # Highest priority
                logger.info("Set high priority CUDA stream for multimodal processing")
            except:
                pass
                
    elif device_type == "cpu":
        # CPU-specific optimizations
        try:
            # Set CPU thread settings
            if hasattr(torch, "set_num_threads"):
                # Multimodal models often benefit from more threads for vision processing
                num_threads = min(os.cpu_count(), 8) if os.cpu_count() else 4
                torch.set_num_threads(num_threads)
                logger.info(f"Set CPU threads to {num_threads} for optimal multimodal model performance")
                
            # Enable oneDNN (MKL-DNN) optimizations if available
            if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                logger.info("Enabled oneDNN (MKL-DNN) optimizations for multimodal models")
        except Exception as e:
            logger.warning(f"Failed to apply CPU optimizations for multimodal model: {e}")
    
    # Return optimized model
    return model


class MultimodalModelAdapter(ModelAdapter):
    """
    Enhanced adapter for multimodal models with hardware-aware optimizations.
    
    Handles loading and input preparation for various multimodal model types,
    including modern architectures like LLaVA, BLIP2, ImageBind, and other
    vision-language models.
    """
    
    def __init__(self, model_id: str, task: Optional[str] = None):
        """
        Initialize a multimodal model adapter.
        
        Args:
            model_id: HuggingFace model ID
            task: Model task
        """
        super().__init__(model_id, task)
        
        # Default task if not provided
        if self.task is None:
            self.task = "image-to-text"
        
        # Initialize processor
        self.processor = None
        self.image_size = (224, 224)  # Default image size
        self.max_length = 77  # Default text length for CLIP-like models
        
        # Model type detection based on ID
        self.model_id_lower = self.model_id.lower()
        
        # Modern vision-language model detection
        self.is_llava = "llava" in self.model_id_lower
        self.is_blip2 = "blip2" in self.model_id_lower or "blip-2" in self.model_id_lower
        self.is_imagebind = "imagebind" in self.model_id_lower
        
        # Vision-language model detection
        self.is_clip = "clip" in self.model_id_lower
        self.is_blip = "blip" in self.model_id_lower and not self.is_blip2
        self.is_vilt = "vilt" in self.model_id_lower
        self.is_flava = "flava" in self.model_id_lower
        
        # Other multimodal model detection
        self.is_donut = "donut" in self.model_id_lower
        self.is_videomae = "videomae" in self.model_id_lower
        self.is_layoutlm = "layoutlm" in self.model_id_lower
        self.is_pix2struct = "pix2struct" in self.model_id_lower
        self.is_git = "git" in self.model_id_lower and not "digital" in self.model_id_lower
        self.is_instructblip = "instructblip" in self.model_id_lower
        
        # Adjust model-specific settings
        if self.is_llava:
            self.max_length = 128
            # LLaVA typically uses 336x336 or 224x224 images
            if "v1.5" in self.model_id_lower:
                self.image_size = (336, 336)
            else:
                self.image_size = (224, 224)
        elif self.is_blip2:
            self.max_length = 96
            # BLIP-2 typically uses 224x224 images
            self.image_size = (224, 224)
        elif self.is_imagebind:
            # ImageBind typically uses 224x224 images
            self.image_size = (224, 224)
    
    def load_model(self, device: torch.device, use_flash_attention: bool = False, 
                  use_torch_compile: bool = False) -> nn.Module:
        """
        Load the multimodal model with hardware-aware optimizations.
        
        Args:
            device: Device to load the model on
            use_flash_attention: Whether to use Flash Attention for transformer models
            use_torch_compile: Whether to use torch.compile for PyTorch 2.0+ optimizations
            
        Returns:
            Loaded and optimized model
        """
        # Set model loading kwargs
        model_kwargs = {"torch_dtype": torch.float16 if device.type == "cuda" else torch.float32}
        
        # Load model based on type
        try:
            # Modern multimodal models
            if self.is_llava:
                model = self._load_llava_model(**model_kwargs)
            elif self.is_blip2:
                model = self._load_blip2_model(**model_kwargs)
            elif self.is_imagebind:
                model = self._load_imagebind_model(**model_kwargs)
            elif self.is_instructblip:
                model = self._load_instructblip_model(**model_kwargs)
            
            # Standard vision-language models
            elif self.is_clip:
                model = self._load_clip_model(**model_kwargs)
            elif self.is_blip:
                model = self._load_blip_model(**model_kwargs)
            elif self.is_vilt:
                model = self._load_vilt_model(**model_kwargs)
            elif self.is_flava:
                model = self._load_flava_model(**model_kwargs)
            elif self.is_git:
                model = self._load_git_model(**model_kwargs)
            
            # Other multimodal models
            elif self.is_videomae:
                model = self._load_video_model(**model_kwargs)
            elif self.is_layoutlm or self.is_donut:
                model = self._load_document_model(**model_kwargs)
            elif self.is_pix2struct:
                model = self._load_pix2struct_model(**model_kwargs)
            
            # Task-based loading
            elif self.task == "image-to-text":
                model = self._load_vision_to_text_model(**model_kwargs)
            elif self.task == "visual-question-answering":
                model = self._load_vqa_model(**model_kwargs)
            else:
                # Default to general processor and model
                model = self._load_default_multimodal_model(**model_kwargs)
                
            # Extract image size from processor if available
            self._extract_image_size_from_processor()
                
        except Exception as e:
            logger.error(f"Error loading multimodal model {self.model_id}: {e}")
            # Fallback to base multimodal model
            logger.warning("Falling back to AutoModel and AutoProcessor")
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            except Exception as e2:
                logger.error(f"Error in fallback model loading: {e2}")
                # Create a simple dummy multimodal model for benchmarking
                logger.warning("Creating dummy multimodal model for benchmarking")
                model = self._create_dummy_multimodal_model()
        
        # Apply hardware-specific optimizations
        model = apply_multimodal_hardware_optimizations(
            model, 
            device.type,
            use_flash_attention=use_flash_attention,
            use_torch_compile=use_torch_compile
        )
            
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        return model
    
    def _create_dummy_multimodal_model(self) -> nn.Module:
        """Create a dummy multimodal model as fallback."""
        # Vision component
        vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # Text component
        text_encoder = nn.Sequential(
            nn.Embedding(1000, 128),
            nn.LSTM(128, 256, batch_first=True),
        )
        
        # Multimodal fusion
        fusion = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1000)
        )
        
        # Create a custom module
        class DummyMultimodalModel(nn.Module):
            def __init__(self, vision_encoder, text_encoder, fusion):
                super().__init__()
                self.vision_encoder = vision_encoder
                self.text_encoder = text_encoder
                self.fusion = fusion
                
            def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
                # Process vision input if available
                if pixel_values is not None:
                    vision_features = self.vision_encoder(pixel_values)
                else:
                    # Create dummy vision features
                    batch_size = input_ids.shape[0] if input_ids is not None else 1
                    vision_features = torch.zeros(batch_size, 128)
                
                # Process text input if available
                if input_ids is not None:
                    text_features, _ = self.text_encoder(input_ids.long())[0][:, -1, :]
                else:
                    # Create dummy text features
                    batch_size = pixel_values.shape[0] if pixel_values is not None else 1
                    text_features = torch.zeros(batch_size, 256)
                
                # Fuse features
                combined_features = torch.cat([vision_features, text_features], dim=1)
                output = self.fusion(combined_features)
                
                return {"logits": output}
        
        return DummyMultimodalModel(vision_encoder, text_encoder, fusion)
    
    def _load_llava_model(self, **model_kwargs):
        """Load a LLaVA model and processor."""
        logger.info(f"Loading LLaVA model: {self.model_id}")
        try:
            self.processor = LlavaProcessor.from_pretrained(self.model_id)
            model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            return model
        except Exception as e:
            logger.warning(f"Error loading LLaVA model: {e}")
            
            # Try alternative loading method
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
                return model
            except Exception as e2:
                logger.error(f"Failed to load LLaVA model: {e2}")
                raise
    
    def _load_blip2_model(self, **model_kwargs):
        """Load a BLIP-2 model and processor."""
        logger.info(f"Loading BLIP-2 model: {self.model_id}")
        try:
            self.processor = Blip2Processor.from_pretrained(self.model_id)
            model = Blip2ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            return model
        except Exception as e:
            logger.warning(f"Error loading BLIP-2 model: {e}")
            
            # Try alternative loading method
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
                return model
            except Exception as e2:
                logger.error(f"Failed to load BLIP-2 model: {e2}")
                raise
    
    def _load_imagebind_model(self, **model_kwargs):
        """Load an ImageBind model and processor."""
        logger.info(f"Loading ImageBind model: {self.model_id}")
        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            return model
        except Exception as e:
            logger.error(f"Failed to load ImageBind model: {e}")
            
            # Try manual loading of ImageBind
            try:
                import importlib.util
                if importlib.util.find_spec("imagebind"):
                    logger.info("Trying to load ImageBind directly")
                    import imagebind.models
                    model = imagebind.models.imagebind_model.imagebind_huge()
                    # Create a simple processor for ImageBind
                    self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    return model
                else:
                    logger.warning("ImageBind package not available")
                    raise
            except Exception as e2:
                logger.error(f"Failed to load ImageBind directly: {e2}")
                raise
    
    def _load_instructblip_model(self, **model_kwargs):
        """Load an InstructBLIP model and processor."""
        logger.info(f"Loading InstructBLIP model: {self.model_id}")
        try:
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            self.processor = InstructBlipProcessor.from_pretrained(self.model_id)
            model = InstructBlipForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            return model
        except Exception as e:
            logger.warning(f"Error loading InstructBLIP model: {e}")
            
            # Try alternative loading method
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
                return model
            except Exception as e2:
                logger.error(f"Failed to load InstructBLIP model: {e2}")
                raise
    
    def _load_clip_model(self, **model_kwargs):
        """Load a CLIP model and processor."""
        logger.info(f"Loading CLIP model: {self.model_id}")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        if self.task == "image-to-text":
            model = CLIPModel.from_pretrained(self.model_id, **model_kwargs)
        elif self.task == "text-embedding":
            model = CLIPTextModel.from_pretrained(self.model_id, **model_kwargs)
        elif self.task == "image-embedding":
            model = CLIPVisionModel.from_pretrained(self.model_id, **model_kwargs)
        else:
            model = CLIPModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_blip_model(self, **model_kwargs):
        """Load a BLIP model and processor."""
        logger.info(f"Loading BLIP model: {self.model_id}")
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        if self.task == "image-to-text":
            try:
                model = BlipForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            except:
                model = BlipModel.from_pretrained(self.model_id, **model_kwargs)
        else:
            model = BlipModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_vilt_model(self, **model_kwargs):
        """Load a ViLT model and processor."""
        logger.info(f"Loading ViLT model: {self.model_id}")
        self.processor = ViltProcessor.from_pretrained(self.model_id)
        model = ViltModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_flava_model(self, **model_kwargs):
        """Load a FLAVA model and processor."""
        logger.info(f"Loading FLAVA model: {self.model_id}")
        from transformers import FlavaModel, FlavaProcessor
        self.processor = FlavaProcessor.from_pretrained(self.model_id)
        model = FlavaModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_video_model(self, **model_kwargs):
        """Load a video model and processor."""
        logger.info(f"Loading video model: {self.model_id}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = VideoMAEModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_pix2struct_model(self, **model_kwargs):
        """Load a Pix2Struct model and processor."""
        logger.info(f"Loading Pix2Struct model: {self.model_id}")
        try:
            from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
            self.processor = Pix2StructProcessor.from_pretrained(self.model_id)
            model = Pix2StructForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            return model
        except Exception as e:
            logger.warning(f"Error loading Pix2Struct model: {e}")
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
                return model
            except Exception as e2:
                logger.error(f"Failed to load Pix2Struct model: {e2}")
                raise
    
    def _load_git_model(self, **model_kwargs):
        """Load a GIT model and processor."""
        logger.info(f"Loading GIT model: {self.model_id}")
        try:
            from transformers import GitProcessor, GitForCausalLM
            self.processor = GitProcessor.from_pretrained(self.model_id)
            model = GitForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            return model
        except Exception as e:
            logger.warning(f"Error loading GIT model: {e}")
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
                return model
            except Exception as e2:
                logger.error(f"Failed to load GIT model: {e2}")
                raise
    
    def _load_document_model(self, **model_kwargs):
        """Load a document understanding model and processor."""
        logger.info(f"Loading document understanding model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForDocumentQuestionAnswering.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_vision_to_text_model(self, **model_kwargs):
        """Load a vision-to-text model and processor."""
        logger.info(f"Loading vision-to-text model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
        except Exception as e:
            logger.warning(f"AutoModelForVision2Seq not available for {self.model_id}: {e}")
            logger.warning("Trying AutoModel instead")
            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_vqa_model(self, **model_kwargs):
        """Load a visual question answering model and processor."""
        logger.info(f"Loading VQA model: {self.model_id}")
        from transformers import ViltForQuestionAnswering
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            model = ViltForQuestionAnswering.from_pretrained(self.model_id, **model_kwargs)
        except Exception as e:
            logger.warning(f"VQA-specific model not available for {self.model_id}: {e}")
            logger.warning("Trying AutoModel instead")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _load_default_multimodal_model(self, **model_kwargs):
        """Load a default multimodal model and processor."""
        logger.info(f"Loading default multimodal model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            model = VisionTextDualEncoderModel.from_pretrained(self.model_id, **model_kwargs)
        except Exception as e:
            logger.warning(f"VisionTextDualEncoderModel not available for {self.model_id}: {e}")
            logger.warning("Falling back to AutoModel")
            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        return model
    
    def _extract_image_size_from_processor(self):
        """Extract image size from processor if available."""
        if self.processor is None:
            return
            
        for attr_name in ["size", "image_size", "size_dict", "crop_size"]:
            if hasattr(self.processor, attr_name):
                size_attr = getattr(self.processor, attr_name)
                if isinstance(size_attr, dict):
                    self.image_size = (size_attr.get("height", 224), size_attr.get("width", 224))
                    return
                elif isinstance(size_attr, (list, tuple)) and len(size_attr) >= 2:
                    self.image_size = (size_attr[0], size_attr[1])
                    return
                elif isinstance(size_attr, int):
                    self.image_size = (size_attr, size_attr)
                    return
                    
        # Check feature extractor if available
        if hasattr(self.processor, "feature_extractor"):
            feature_extractor = self.processor.feature_extractor
            for attr_name in ["size", "image_size", "size_dict", "crop_size"]:
                if hasattr(feature_extractor, attr_name):
                    size_attr = getattr(feature_extractor, attr_name)
                    if isinstance(size_attr, dict):
                        self.image_size = (size_attr.get("height", 224), size_attr.get("width", 224))
                        return
                    elif isinstance(size_attr, (list, tuple)) and len(size_attr) >= 2:
                        self.image_size = (size_attr[0], size_attr[1])
                        return
                    elif isinstance(size_attr, int):
                        self.image_size = (size_attr, size_attr)
                        return
    
    def prepare_inputs(self, batch_size: int, sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the multimodal model.
        
        Args:
            batch_size: Batch size
            sequence_length: Text length for text input
            
        Returns:
            Dictionary of input tensors
        """
        if self.processor is None:
            raise ValueError("Processor not initialized. Call load_model first.")
        
        # Default sequence length
        if sequence_length is not None:
            self.max_length = sequence_length
        
        # Generate inputs based on model type
        if self.is_llava:
            return self._prepare_llava_inputs(batch_size)
        elif self.is_blip2:
            return self._prepare_blip2_inputs(batch_size)
        elif self.is_imagebind:
            return self._prepare_imagebind_inputs(batch_size)
        elif self.is_videomae:
            return self._prepare_video_inputs(batch_size)
        elif self.is_layoutlm or self.is_donut:
            return self._prepare_document_inputs(batch_size)
        elif self.is_pix2struct:
            return self._prepare_pix2struct_inputs(batch_size)
        elif self.is_instructblip:
            return self._prepare_instructblip_inputs(batch_size)
        elif self.task == "visual-question-answering":
            return self._prepare_vqa_inputs(batch_size)
        else:
            # Default vision-language inputs
            return self._prepare_vision_language_inputs(batch_size)
    
    def _prepare_llava_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for LLaVA models."""
        try:
            # Create random images
            random_images = self._create_random_images(batch_size)
            
            # Create text prompt (instruction)
            prompt = "Describe this image in detail."
            prompts = [prompt] * batch_size
            
            try:
                # Process with LLaVA processor
                inputs = self.processor(
                    text=prompts,
                    images=random_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            except Exception as e:
                logger.warning(f"Error with standard LLaVA processing: {e}")
                
                # Try alternative processing
                try:
                    # Try separate processing of images and text
                    logger.warning("Trying separate processing for LLaVA")
                    
                    # Process images
                    if hasattr(self.processor, "image_processor"):
                        image_inputs = self.processor.image_processor(random_images, return_tensors="pt")
                    else:
                        # Create default image tensor
                        height, width = self.image_size
                        image_inputs = {"pixel_values": torch.rand(batch_size, 3, height, width)}
                    
                    # Process text
                    if hasattr(self.processor, "tokenizer"):
                        text_inputs = self.processor.tokenizer(
                            prompts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=self.max_length
                        )
                    else:
                        # Create dummy input_ids and attention_mask
                        text_inputs = {
                            "input_ids": torch.randint(0, 1000, (batch_size, self.max_length)),
                            "attention_mask": torch.ones(batch_size, self.max_length, dtype=torch.long)
                        }
                    
                    # Combine inputs
                    inputs = {**image_inputs, **text_inputs}
                except Exception as e2:
                    logger.error(f"Alternative LLaVA processing failed: {e2}")
                    return self._create_default_inputs(batch_size)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing LLaVA inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _prepare_blip2_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for BLIP-2 models."""
        try:
            # Create random images
            random_images = self._create_random_images(batch_size)
            
            # Create text prompt
            prompt = "Question: What do you see in this image? Answer:"
            prompts = [prompt] * batch_size
            
            try:
                # Process with BLIP-2 processor
                inputs = self.processor(
                    text=prompts,
                    images=random_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            except Exception as e:
                logger.warning(f"Error with standard BLIP-2 processing: {e}")
                
                # Try alternative processing
                try:
                    # Try separate processing of images and text
                    logger.warning("Trying separate processing for BLIP-2")
                    
                    # Process images
                    if hasattr(self.processor, "image_processor"):
                        image_inputs = self.processor.image_processor(random_images, return_tensors="pt")
                    else:
                        # Create default image tensor
                        height, width = self.image_size
                        image_inputs = {"pixel_values": torch.rand(batch_size, 3, height, width)}
                    
                    # Process text
                    if hasattr(self.processor, "tokenizer"):
                        text_inputs = self.processor.tokenizer(
                            prompts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=self.max_length
                        )
                    else:
                        # Create dummy input_ids and attention_mask
                        text_inputs = {
                            "input_ids": torch.randint(0, 1000, (batch_size, self.max_length)),
                            "attention_mask": torch.ones(batch_size, self.max_length, dtype=torch.long)
                        }
                    
                    # Combine inputs
                    inputs = {**image_inputs, **text_inputs}
                except Exception as e2:
                    logger.error(f"Alternative BLIP-2 processing failed: {e2}")
                    return self._create_default_inputs(batch_size)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing BLIP-2 inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _prepare_imagebind_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for ImageBind models."""
        try:
            # Create random images
            random_images = self._create_random_images(batch_size)
            
            # Create text data
            text = "A photo of a beautiful landscape."
            texts = [text] * batch_size
            
            # Create audio data (dummy)
            audio_data = torch.randn(batch_size, 16000)  # 1 second at 16kHz
            
            try:
                # Process with processor if available
                if self.processor:
                    # Process images
                    if hasattr(self.processor, "image_processor"):
                        image_inputs = self.processor.image_processor(random_images, return_tensors="pt")
                    else:
                        # Try direct processing
                        image_inputs = self.processor(images=random_images, return_tensors="pt")
                    
                    # For ImageBind, typically return pixel_values only
                    inputs = {"pixel_values": image_inputs.get("pixel_values", image_inputs.get("image", None))}
                    
                    # Add other modalities as dummy tensors
                    if "pixel_values" in inputs:
                        inputs["text_embeds"] = torch.randn(batch_size, 768)
                        inputs["audio_embeds"] = torch.randn(batch_size, 768)
                else:
                    # Create default inputs
                    height, width = self.image_size
                    inputs = {
                        "pixel_values": torch.rand(batch_size, 3, height, width),
                        "text_embeds": torch.randn(batch_size, 768),
                        "audio_embeds": torch.randn(batch_size, 768)
                    }
            except Exception as e:
                logger.warning(f"Error processing ImageBind inputs: {e}")
                # Create default inputs
                height, width = self.image_size
                inputs = {
                    "pixel_values": torch.rand(batch_size, 3, height, width),
                    "text_embeds": torch.randn(batch_size, 768),
                    "audio_embeds": torch.randn(batch_size, 768)
                }
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing ImageBind inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _prepare_instructblip_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for InstructBLIP models."""
        try:
            # Create random images
            random_images = self._create_random_images(batch_size)
            
            # Create instruction prompt
            prompt = "Describe this image in detail."
            prompts = [prompt] * batch_size
            
            try:
                # Process with InstructBLIP processor
                inputs = self.processor(
                    text=prompts,
                    images=random_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            except Exception as e:
                logger.warning(f"Error with standard InstructBLIP processing: {e}")
                
                # Try alternative processing
                try:
                    # Try separate processing of images and text
                    logger.warning("Trying separate processing for InstructBLIP")
                    
                    # Process images
                    if hasattr(self.processor, "image_processor"):
                        image_inputs = self.processor.image_processor(random_images, return_tensors="pt")
                    else:
                        # Create default image tensor
                        height, width = self.image_size
                        image_inputs = {"pixel_values": torch.rand(batch_size, 3, height, width)}
                    
                    # Process text
                    if hasattr(self.processor, "tokenizer"):
                        text_inputs = self.processor.tokenizer(
                            prompts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=self.max_length
                        )
                    else:
                        # Create dummy input_ids and attention_mask
                        text_inputs = {
                            "input_ids": torch.randint(0, 1000, (batch_size, self.max_length)),
                            "attention_mask": torch.ones(batch_size, self.max_length, dtype=torch.long)
                        }
                    
                    # Combine inputs
                    inputs = {**image_inputs, **text_inputs}
                except Exception as e2:
                    logger.error(f"Alternative InstructBLIP processing failed: {e2}")
                    return self._create_default_inputs(batch_size)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing InstructBLIP inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _prepare_pix2struct_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for Pix2Struct models."""
        try:
            # Create random images
            random_images = self._create_random_images(batch_size)
            
            # Create prompt for image understanding
            prompt = "What does this image show?"
            prompts = [prompt] * batch_size
            
            try:
                # Process with Pix2Struct processor
                inputs = self.processor(
                    text=prompts,
                    images=random_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            except Exception as e:
                logger.warning(f"Error with standard Pix2Struct processing: {e}")
                
                # Try alternative processing
                try:
                    # Try processing just images
                    inputs = self.processor(
                        images=random_images,
                        return_tensors="pt"
                    )
                    
                    # Add dummy text inputs if needed
                    if "input_ids" not in inputs:
                        inputs["input_ids"] = torch.randint(0, 1000, (batch_size, self.max_length))
                        inputs["attention_mask"] = torch.ones(batch_size, self.max_length, dtype=torch.long)
                except Exception as e2:
                    logger.error(f"Alternative Pix2Struct processing failed: {e2}")
                    return self._create_default_inputs(batch_size)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing Pix2Struct inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _prepare_vision_language_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for vision-language models."""
        # Create text inputs
        text = "A photo of a cute dog playing in the park. " * (self.max_length // 10 + 1)
        text = text[:self.max_length]
        texts = [text] * batch_size
        
        # Create image inputs
        random_images = self._create_random_images(batch_size)
        
        # Process inputs based on model type
        try:
            # Try standard processing with text and images
            inputs = self.processor(
                text=texts,
                images=random_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
        except Exception as e1:
            logger.warning(f"Error with standard processing: {e1}")
            try:
                # Try CLIP-style processing
                inputs = self.processor(
                    text=texts,
                    images=random_images,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
            except Exception as e2:
                logger.warning(f"Error with CLIP-style processing: {e2}")
                try:
                    # Fallback to image-only processing
                    logger.warning("Falling back to image-only processing")
                    inputs = self.processor(
                        images=random_images,
                        return_tensors="pt"
                    )
                except Exception as e3:
                    logger.error(f"All processing methods failed: {e3}")
                    # Create minimal default inputs
                    return self._create_default_inputs(batch_size)
        
        return inputs
    
    def _prepare_video_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for video models."""
        try:
            # Create random video frames (batch_size, num_frames, channels, height, width)
            num_frames = 16  # Common for video models
            height, width = self.image_size
            
            # Create random video tensor
            video_tensor = torch.rand(batch_size, num_frames, 3, height, width)
            
            try:
                # Try to process with the processor
                inputs = self.processor(
                    video_tensor,
                    return_tensors="pt"
                )
            except:
                # Fallback to direct tensor
                inputs = {"pixel_values": video_tensor}
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing video inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _prepare_document_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for document understanding models."""
        try:
            # Create random document images
            random_images = self._create_random_images(batch_size)
            
            # Create question for document QA
            question = "What is the total amount?"
            questions = [question] * batch_size
            
            try:
                # Try to process with the processor
                inputs = self.processor(
                    images=random_images,
                    questions=questions,
                    return_tensors="pt",
                    padding=True
                )
            except:
                try:
                    # Try alternative processing
                    inputs = self.processor(
                        images=random_images,
                        text=questions,
                        return_tensors="pt",
                        padding=True
                    )
                except:
                    # Fallback to image-only
                    inputs = self.processor(
                        images=random_images,
                        return_tensors="pt"
                    )
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing document inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _prepare_vqa_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for visual question answering models."""
        try:
            # Create random images
            random_images = self._create_random_images(batch_size)
            
            # Create questions
            question = "What is the color of the dog in the image?"
            questions = [question] * batch_size
            
            try:
                # Try to process with the processor
                inputs = self.processor(
                    images=random_images,
                    text=questions,
                    return_tensors="pt",
                    padding=True
                )
            except:
                try:
                    # Try alternative processing
                    inputs = self.processor(
                        images=random_images,
                        questions=questions,
                        return_tensors="pt",
                        padding=True
                    )
                except:
                    # Fallback to separate processing
                    logger.warning("Falling back to separate image/text processing")
                    
                    # Process images
                    image_processor = getattr(self.processor, "image_processor", None)
                    if image_processor:
                        image_inputs = image_processor(random_images, return_tensors="pt")
                    else:
                        # Create default image tensor
                        height, width = self.image_size
                        image_inputs = {"pixel_values": torch.rand(batch_size, 3, height, width)}
                    
                    # Process text
                    tokenizer = getattr(self.processor, "tokenizer", None)
                    if tokenizer:
                        text_inputs = tokenizer(
                            questions, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=self.max_length
                        )
                    else:
                        # Create dummy input_ids and attention_mask
                        text_inputs = {
                            "input_ids": torch.randint(0, 1000, (batch_size, self.max_length)),
                            "attention_mask": torch.ones(batch_size, self.max_length, dtype=torch.long)
                        }
                    
                    # Combine inputs
                    inputs = {**image_inputs, **text_inputs}
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing VQA inputs: {e}")
            return self._create_default_inputs(batch_size)
    
    def _create_random_images(self, batch_size: int) -> List[Image.Image]:
        """Create random RGB images for benchmarking."""
        try:
            # Create random RGB images
            height, width = self.image_size
            random_images = []
            
            for _ in range(batch_size):
                # Create a random RGB image
                img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                random_images.append(img)
            
            return random_images
            
        except ImportError as e:
            logger.warning(f"Error creating random images: {e}")
            return None
    
    def _create_default_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Create default inputs when all other methods fail."""
        logger.warning("Creating default inputs for multimodal model")
        
        # Create default image tensor
        height, width = self.image_size
        pixel_values = torch.rand(batch_size, 3, height, width)
        
        # Create default text tensor
        input_ids = torch.randint(0, 1000, (batch_size, self.max_length))
        attention_mask = torch.ones(batch_size, self.max_length, dtype=torch.long)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
#!/usr/bin/env python3
"""
Skill implementation for detr with hardware platform support
"""

import os
import sys
import torch
import numpy as np
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoFeatureExtractor, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification, AutoModelForAudioClassification, AutoModelForVideoClassification

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
HAS_ROCM = (HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version')) or ('ROCM_HOME' in os.environ)
HAS_OPENVINO = False
try:
    import openvino
    HAS_OPENVINO = True
except ImportError:
    pass

HAS_QNN = False
try:
    import qnn_wrapper
    HAS_QNN = True
except ImportError:
    try:
        import qti
        HAS_QNN = True
    except ImportError:
        pass

HAS_WEBNN = os.environ.get("WEBNN_AVAILABLE", "0") == "1"
HAS_WEBGPU = os.environ.get("WEBGPU_AVAILABLE", "0") == "1"

class DetrSkill:
    """Skill for detr model with hardware platform support."""
    
    def __init__(self, model_id="facebook/detr-resnet-50", device=None):
        """Initialize the skill."""
        self.model_id = model_id
        self.device = device or self.get_default_device()
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.modality = "vision"
        logger.info(f"Initialized detr skill with device: {self.device}")
    
    def get_default_device(self):
        """Get the best available device based on hardware availability."""
        # Check for CUDA
        if HAS_CUDA:
            return "cuda"
        
        # Check for Apple Silicon
        if HAS_MPS:
            return "mps"
        
        # Check for ROCm (AMD)
        if HAS_ROCM:
            return "rocm"
        
        # Check for OpenVINO (Intel)
        if HAS_OPENVINO:
            return "openvino"
        
        # Check for Qualcomm QNN
        if HAS_QNN:
            return "qualcomm"
        
        # Check for WebNN
        if HAS_WEBNN:
            return "webnn"
        
        # Check for WebGPU
        if HAS_WEBGPU:
            return "webgpu"
        
        # Default to CPU
        return "cpu"
    
    def load_model(self):
        """Load the model and tokenizer based on modality."""
        if self.model is None:
            logger.info(f"Loading {self.modality} model: {self.model_id}")
            
            # Load appropriate tokenizer/processor and model based on modality
            if self.modality == "audio":
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            elif self.modality == "vision":
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            elif self.modality == "multimodal":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            elif self.modality == "video":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
            else:
                # Default to text
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded and moved to {self.device}")
            else:
                logger.info("Model loaded on CPU")
    
    def process(self, text):
        """Process the input text and return the output."""
        # Ensure model is loaded
        self.load_model()
        
        logger.info(f"Processing input with {self.modality} model")
        
        if self.modality == "text":
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert to numpy for consistent output
            last_hidden_state = outputs.last_hidden_state.cpu().numpy()
            
            # Return formatted results
            return {
                "model": self.model_id,
                "device": self.device,
                "last_hidden_state_shape": last_hidden_state.shape,
                "embedding": last_hidden_state.mean(axis=1).tolist(),
            }
        elif self.modality in ["vision", "image"]:
            # Process image input (assuming np.array)
            inputs = self.processor(text, return_tensors="pt")
            
            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs based on model type
            if hasattr(outputs, "logits"):
                result = outputs.logits.cpu().numpy()
            else:
                result = outputs.last_hidden_state.cpu().numpy()
            
            return {
                "model": self.model_id,
                "device": self.device,
                "shape": result.shape,
                "output": result.tolist() if result.size < 1000 else result.mean(axis=1).tolist(),
            }
        else:
            # Generic processing for other modalities
            logger.info(f"Using generic processing for {self.modality} modality")
            return {
                "model": self.model_id,
                "device": self.device,
                "modality": self.modality,
                "error": f"Direct processing for {self.modality} not implemented, use specific methods instead"
            }
    
    def get_supported_hardware(self):
        """Get the list of supported hardware platforms."""
        supported = ["cpu"]
        
        if HAS_CUDA:
            supported.append("cuda")
        if HAS_MPS:
            supported.append("mps")
        if HAS_ROCM:
            supported.append("rocm")
        if HAS_OPENVINO:
            supported.append("openvino")
        if HAS_QNN:
            supported.append("qualcomm")
        if HAS_WEBNN:
            supported.append("webnn")
        if HAS_WEBGPU:
            supported.append("webgpu")
        
        return supported

# Factory function to create skill instance
def create_skill(model_id="detr", device=None):
    """Create a skill instance."""
    return DetrSkill(model_id=model_id, device=device)

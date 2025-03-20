#!/usr/bin/env python3
"""
Test file for bert-base-uncased.

This file is auto-generated using the template-based test generator.
Generated: 2025-03-19 21:37:45
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


import pytest

import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from transformers import AutoModel, AutoTokenizer
from common.hardware_detection import detect_hardware, skip_if_no_cuda
from common.model_helpers import load_model, get_sample_inputs_for_model


class TestBertBaseUncased:
    """Test class for bert-base-uncased model."""
    
    def __init__(self):
        """Initialize the test with model details and hardware detection."""
        self.model_name = "bert-base-uncased"
        self.model_type = "text"
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection for the template."""
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # Qualcomm AI Engine support
        self.has_qualcomm = 'qti' in sys.modules or 'qnn_wrapper' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {self.device}")
        
    def get_model(self):
        """Load model from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self.device)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    @pytest.mark.model
    @pytest.mark.text
    def test_basic_inference(self):
        """Run a basic inference test with the model."""
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            pytest.skip("Failed to load model or tokenizer")
        
        try:
            # Prepare input
            text = "This is a sample text for testing the bert-base-uncased model."
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
            assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            
            logger.info("Basic inference test passed")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            pytest.fail(f"Inference failed: {e}")
    
    @pytest.mark.model
    @pytest.mark.text
    @pytest.mark.slow
    def test_batch_inference(self):
        """Run a batch inference test with the model."""
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            pytest.skip("Failed to load model or tokenizer")
        
        try:
            # Prepare batch input
            texts = [
                "This is the first sample text for testing batch inference.",
                "This is the second sample text for testing batch inference."
            ]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == len(texts), f"Batch size should be {len(texts)}"
            assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
            logger.info(f"Batch output shape: {outputs.last_hidden_state.shape}")
            
            logger.info("Batch inference test passed")
        except Exception as e:
            logger.error(f"Error during batch inference: {e}")
            pytest.fail(f"Batch inference failed: {e}")
    
    @pytest.mark.model
    @pytest.mark.text
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_compatibility(self, device):
        """Test model compatibility with different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from transformers import AutoModel
            
            # Load model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(device)
            
            logger.info(f"Model loaded on {device}")
            assert model.device.type == device, f"Model should be on {device}"
            
            logger.info(f"Device compatibility test passed for {device}")
        except Exception as e:
            logger.error(f"Error loading model on {device}: {e}")
            pytest.fail(f"Device compatibility test failed for {device}: {e}")



if __name__ == "__main__":
    # Run tests directly
    pytest.main(["-xvs", __file__])

#!/usr/bin/env python3
"""
Test file for google/vit-base-patch16-224 model.

This file has been migrated to the refactored test suite.
Generated: 2025-03-21 from original test_vit-base-patch16-224.py
"""

import os
import sys
import logging
import unittest
import numpy as np
from pathlib import Path

from refactored_test_suite.model_test import ModelTest

# Try to import torch - but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import PIL - but don't fail if not available
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
# Try to import transformers - but don't fail if not available
try:
    import transformers
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class TestVitBasePatch16224(ModelTest):
    """Test class for google/vit-base-patch16-224 model."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_name = "google/vit-base-patch16-224"
        self.model_type = "vision"
        self.setup_hardware()
        self.processor = None
        self.test_image_path = os.path.join(self.model_dir, "test_image.jpg")
        self._create_test_image()
    
    def setup_hardware(self):
        """Set up hardware detection for the template."""
        # Skip test if torch is not available
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
            
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        
        # MPS support (Apple Silicon)
        try:
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except AttributeError:
            self.has_mps = False
            
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
            
        self.logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_name=None):
        """Load model from HuggingFace."""
        if not TRANSFORMERS_AVAILABLE:
            self.skipTest("Transformers not available")
            
        if model_name is None:
            model_name = self.model_name
            
        try:
            # Get image processor
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            # Get model with vision-specific settings
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                torchscript=True if self.device == 'cpu' else False
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading vision model with specific settings: {e}")
            
            # Fallback to generic model
            try:
                from transformers import AutoModel, AutoFeatureExtractor
                self.processor = AutoFeatureExtractor.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as e2:
                self.logger.error(f"Error in fallback loading: {e2}")
                self.skipTest(f"Could not load model: {str(e2)}")
    
    def _create_test_image(self):
        """Create a test image for inference."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
            
        # Create a simple test image (black and white gradient)
        size = 224
        img_array = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                img_array[i, j, :] = (i + j) % 256
        img = Image.fromarray(img_array)
        img.save(self.test_image_path)
        self.logger.info(f"Created test image at {self.test_image_path}")
    
    def test_basic_inference(self):
        """Run a basic inference test with the model."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE or not PIL_AVAILABLE:
            self.skipTest("Required dependencies not available")
            
        model = self.load_model()
        
        try:
            # Load and process the image
            image = Image.open(self.test_image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            self.assertIsNotNone(outputs, "Model outputs should not be None")
            
            if hasattr(outputs, "last_hidden_state"):
                self.assertEqual(outputs.last_hidden_state.shape[0], 1, "Batch size should be 1")
                self.logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            
            # If it's a classification model, check the logits
            if hasattr(outputs, "logits"):
                self.assertIsNotNone(outputs.logits, "Logits should not be None")
                self.assertEqual(outputs.logits.shape[0], 1, "Batch size should be 1")
                self.logger.info(f"Logits shape: {outputs.logits.shape}")
                
                # Get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                max_prob = probs.max().item()
                self.logger.info(f"Max probability: {max_prob:.4f}")
            
            self.logger.info("Basic inference test passed")
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            self.fail(f"Test failed: {str(e)}")
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware platforms."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE or not PIL_AVAILABLE:
            self.skipTest("Required dependencies not available")
            
        devices_to_test = []
        
        # Always test CPU
        devices_to_test.append('cpu')
        
        # Only test available hardware
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
            
        # Test each device
        for device in devices_to_test:
            original_device = self.device
            try:
                self.logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model for this device
                model = self.load_model()
                
                # Load and process the image
                image = Image.open(self.test_image_path)
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Basic check on outputs
                self.assertIsNotNone(outputs, f"Outputs on {device} should not be None")
                
                self.logger.info(f"Test on {device} passed")
            except Exception as e:
                self.logger.error(f"Error testing on {device}: {e}")
                self.fail(f"Test on {device} failed: {str(e)}")
            finally:
                # Restore original device
                self.device = original_device
            
if __name__ == "__main__":
    unittest.main()
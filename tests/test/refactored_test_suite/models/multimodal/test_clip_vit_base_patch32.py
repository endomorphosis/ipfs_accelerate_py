#!/usr/bin/env python3
"""Test file for openai/clip-vit-base-patch32 model."""

import os
import sys
import logging
import tempfile
from pathlib import Path
import time
import datetime
import unittest
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import test base classes
from refactored_test_suite.model_test import ModelTest

# Check for dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    from unittest.mock import MagicMock
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    from unittest.mock import MagicMock
    Image = MagicMock()
    HAS_PIL = False
    logger.warning("PIL.Image not available, using mock")

class TestClipVitBasePatch32(ModelTest):
    """Test class for openai/clip-vit-base-patch32 model."""
    
    def setUp(self):
        """Set up resources for each test method."""
        super().setUp()
        self.model_id = "openai/clip-vit-base-patch32"
        
        # Define model parameters
        self.model_type = "clip"
        self.task = "zero-shot-image-classification"
        self.class_name = "CLIPModel"
        self.description = "CLIP model with ViT base patch32 encoder"
        self.image_size = 224
        self.processor_class = "CLIPProcessor"
        
        # Define test inputs
        self.test_image_path = self.create_test_image()
        self.test_text = ["a photo of a cat", "a photo of a dog", "a photo of a person"]
        
        # Configure hardware preference
        self.preferred_device = self.detect_preferred_device()
    
    def create_test_image(self):
        """Create a test image for multimodal testing."""
        test_image_candidates = [
            "test.jpg", 
            "test.png", 
            "test_image.jpg", 
            "test_image.png"
        ]
        
        for path in test_image_candidates:
            if os.path.exists(path):
                return path
        
        # Create a dummy image if no test image is found
        if HAS_PIL:
            dummy_path = os.path.join(self.model_dir, "test_dummy.jpg")
            img = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            img.save(dummy_path)
            return dummy_path
        
        return None
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
        if not HAS_TORCH:
            return "cpu"
        
        # Check for CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            return "mps"
        
        # Fallback to CPU
        return "cpu"
    
    def test_model_loading(self):
        """Test basic model and processor loading."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
            
        # Test processor loading
        try:
            processor_class = getattr(transformers, self.processor_class)
            processor = processor_class.from_pretrained(self.model_id)
            self.assertIsNotNone(processor, "Processor loading failed")
        except Exception as e:
            self.fail(f"Processor loading failed: {e}")
        
        # Test model loading
        try:
            model_class = getattr(transformers, self.class_name)
            model = model_class.from_pretrained(self.model_id)
            self.assertIsNotNone(model, "Model loading failed")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_pipeline(self):
        """Test using the model with the transformers pipeline API."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
        if not HAS_PIL:
            self.skipTest("PIL library not available")
            
        # Skip if we don't have a test image
        if self.test_image_path is None:
            self.skipTest("No test image available")
        
        # Create pipeline with appropriate parameters
        try:
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": self.preferred_device
            }
            
            pipeline = transformers.pipeline(**pipeline_kwargs)
            self.assertIsNotNone(pipeline, "Pipeline creation failed")
            
            # For CLIP models
            pipeline_input = self.test_image_path
            pipeline_kwargs = {"candidate_labels": self.test_text}
            output = pipeline(pipeline_input, **pipeline_kwargs)
            
            # Verify we got output
            self.assertIsNotNone(output, "Pipeline produced no output")
            
        except Exception as e:
            self.fail(f"Pipeline test failed: {e}")
    
    def test_from_pretrained(self):
        """Test the model using direct from_pretrained loading."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
        if not HAS_PIL:
            self.skipTest("PIL library not available")
        if not HAS_TORCH:
            self.skipTest("PyTorch not available")
        
        # Skip if we don't have a test image
        if self.test_image_path is None:
            self.skipTest("No test image available")
        
        try:
            # Load processor
            processor_class = getattr(transformers, self.processor_class)
            processor = processor_class.from_pretrained(self.model_id)
            
            # Load model
            model_class = getattr(transformers, self.class_name)
            model = model_class.from_pretrained(self.model_id)
            
            # Move model to preferred device
            if self.preferred_device != "cpu":
                model = model.to(self.preferred_device)
            
            # Prepare image
            image = Image.open(self.test_image_path)
            
            # Process inputs - For CLIP models
            inputs = processor(text=self.test_text, images=image, return_tensors="pt", padding=True)
            
            # Move inputs to device
            if self.preferred_device != "cpu":
                inputs = {key: val.to(self.preferred_device) for key, val in inputs.items()}
            
            # Run inference - For CLIP, just forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs, "Model produced no outputs")
            
            # Process CLIP-specific outputs
            if hasattr(outputs, "logits_per_image"):
                logits_per_image = outputs.logits_per_image
                self.assertIsNotNone(logits_per_image, "No logits_per_image in outputs")
                probs = torch.softmax(logits_per_image, dim=1)
                self.assertEqual(probs.shape[1], len(self.test_text), 
                              "Output probabilities don't match number of test texts")
            
        except Exception as e:
            self.fail(f"Direct from_pretrained test failed: {e}")
    
    def test_with_openvino(self):
        """Test the model using OpenVINO integration."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
        if not HAS_PIL:
            self.skipTest("PIL library not available")
        
        # Check for OpenVINO
        try:
            import openvino
        except ImportError:
            self.skipTest("OpenVINO not available")
        
        # Skip if we don't have a test image
        if self.test_image_path is None:
            self.skipTest("No test image available")
        
        try:
            # Import OpenVINO optimum utilities
            from optimum.intel import OVModelForImageClassification
            
            # Load processor
            processor_class = getattr(transformers, self.processor_class)
            processor = processor_class.from_pretrained(self.model_id)
            
            # Load model with OpenVINO
            model = OVModelForImageClassification.from_pretrained(
                self.model_id,
                export=True,
                provider="CPU"
            )
            
            # Prepare image
            image = Image.open(self.test_image_path)
            
            # Process inputs for CLIP
            inputs = processor(text=self.test_text, images=image, return_tensors="pt", padding=True)
            
            # Run inference
            outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs, "OpenVINO model produced no outputs")
            
        except Exception as e:
            self.fail(f"OpenVINO integration test failed: {e}")
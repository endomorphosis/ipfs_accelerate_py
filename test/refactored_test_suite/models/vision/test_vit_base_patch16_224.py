#!/usr/bin/env python3
"""
Test file for ViT models using the refactored test suite structure.
"""

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestVitModel(ModelTest):
    """Test class for vision transformer models."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "google/vit-base-patch16-224"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "image-classification"
        
        # Define test inputs
        self.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    def tearDown(self):
        """Clean up resources after the test."""
        super().tearDown()
    
    def load_model(self):
        """Load the model for testing."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Load processor and model
            processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = AutoModelForImageClassification.from_pretrained(self.model_id)
            model = model.to(self.device)
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model_components = self.load_model()
        
        # Verify model and processor
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["processor"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        """Test basic inference with the model."""
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Create dummy image for testing
        from PIL import Image
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        # Process image
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, "logits"))
        logger.info(f"Inference successful: {outputs.logits.shape}")
    
    def test_hardware_compatibility(self):
        """Test model compatibility across hardware platforms."""
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        
        results = {}
        original_device = self.device
        
        for device in available_devices:
            try:
                self.device = device
                model_components = self.load_model()
                model = model_components["model"]
                
                # Basic verification
                self.assertIsNotNone(model)
                results[device] = True
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed on {device}: {e}")
                results[device] = False
            finally:
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()))

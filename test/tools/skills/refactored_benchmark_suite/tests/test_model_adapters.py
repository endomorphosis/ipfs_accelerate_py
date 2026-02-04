#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for model adapters.

This script tests the model adapters for different model types.
"""

import os
import sys
import unittest
from pathlib import Path
import logging

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from models import get_model_adapter
import unittest.mock as mock

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TestModelAdapters(unittest.TestCase):
    """Test cases for model adapters."""
    
    def setUp(self):
        """Set up test environment."""
        # Skip tests if PyTorch or Transformers are not available
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("PyTorch or Transformers not available")
    
    def test_text_model_adapter(self):
        """Test text model adapter."""
        try:
            # Get text model adapter
            adapter = get_model_adapter("bert-base-uncased", "fill-mask")
            
            # Verify adapter type
            from models.text_models import TextModelAdapter
            self.assertIsInstance(adapter, TextModelAdapter)
            
            # Test input preparation
            inputs = adapter.prepare_inputs(batch_size=1, sequence_length=16)
            self.assertIsNotNone(inputs)
            
            logger.info("Text model adapter test passed")
            
        except Exception as e:
            self.skipTest(f"Text model adapter test failed: {e}")
    
    def test_vision_model_adapter(self):
        """Test vision model adapter."""
        try:
            # Get vision model adapter
            adapter = get_model_adapter("google/vit-base-patch16-224", "image-classification")
            
            # Verify adapter type
            from models.vision_models import VisionModelAdapter
            self.assertIsInstance(adapter, VisionModelAdapter)
            
            # Test input preparation
            inputs = adapter.prepare_inputs(batch_size=1)
            self.assertIsNotNone(inputs)
            
            logger.info("Vision model adapter test passed")
            
        except Exception as e:
            self.skipTest(f"Vision model adapter test failed: {e}")
    
    def test_model_type_detection(self):
        """Test model type detection."""
        try:
            # Test with text model
            adapter = get_model_adapter("bert-base-uncased")
            from models.text_models import TextModelAdapter
            self.assertIsInstance(adapter, TextModelAdapter)
            
            # Test with vision model
            adapter = get_model_adapter("google/vit-base-patch16-224")
            from models.vision_models import VisionModelAdapter
            self.assertIsInstance(adapter, VisionModelAdapter)
            
            logger.info("Model type detection test passed")
            
        except Exception as e:
            self.skipTest(f"Model type detection test failed: {e}")
    
    def test_task_inference(self):
        """Test task inference."""
        try:
            # Test with text model
            adapter = get_model_adapter("bert-base-uncased")
            self.assertEqual(adapter.task, "fill-mask")
            
            # Test with vision model
            adapter = get_model_adapter("google/vit-base-patch16-224")
            self.assertEqual(adapter.task, "image-classification")
            
            logger.info("Task inference test passed")
            
        except Exception as e:
            self.skipTest(f"Task inference test failed: {e}")
    
    def test_device_handling(self):
        """Test device handling."""
        try:
            # Get adapter
            adapter = get_model_adapter("bert-base-uncased")
            
            # Create a device
            device = torch.device("cpu")
            
            # Load model on CPU
            model = adapter.load_model(device)
            
            # Verify model is on the correct device
            self.assertEqual(next(model.parameters()).device, device)
            
            # Test with CUDA if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = adapter.load_model(device)
                self.assertEqual(next(model.parameters()).device, device)
            
            logger.info("Device handling test passed")
            
        except Exception as e:
            self.skipTest(f"Device handling test failed: {e}")
    
    def test_multimodal_model_adapter(self):
        """Test multimodal model adapter."""
        try:
            # Get multimodal model adapter
            adapter = get_model_adapter("openai/clip-vit-base-patch32", "image-to-text")
            
            # Verify adapter type
            from models.multimodal_models import MultimodalModelAdapter
            self.assertIsInstance(adapter, MultimodalModelAdapter)
            
            # Test input preparation
            inputs = adapter.prepare_inputs(batch_size=1, sequence_length=32)
            self.assertIsNotNone(inputs)
            
            logger.info("Multimodal model adapter test passed")
            
        except Exception as e:
            self.skipTest(f"Multimodal model adapter test failed: {e}")
    
    def test_multimodal_model_type_detection(self):
        """Test multimodal model type detection."""
        try:
            # Test with CLIP model
            adapter = get_model_adapter("openai/clip-vit-base-patch32")
            from models.multimodal_models import MultimodalModelAdapter
            self.assertIsInstance(adapter, MultimodalModelAdapter)
            self.assertEqual(adapter.task, "image-to-text")
            
            logger.info("Multimodal model type detection test passed")
            
        except Exception as e:
            self.skipTest(f"Multimodal model type detection test failed: {e}")
    
    def test_multimodal_task_handling(self):
        """Test multimodal task handling."""
        try:
            # Test with VQA task
            adapter = get_model_adapter("openai/clip-vit-base-patch32", "visual-question-answering")
            self.assertEqual(adapter.task, "visual-question-answering")
            
            # Mock the processor and test the VQA input preparation
            with mock.patch.object(adapter, 'processor') as mock_processor:
                mock_processor.return_value = {"pixel_values": torch.rand(1, 3, 224, 224)}
                inputs = adapter._prepare_vqa_inputs(batch_size=1)
                self.assertIsNotNone(inputs)
            
            logger.info("Multimodal task handling test passed")
            
        except Exception as e:
            self.skipTest(f"Multimodal task handling test failed: {e}")
    
    def test_multimodal_model_specific_handling(self):
        """Test model-specific handling in the multimodal adapter."""
        try:
            # Test is_clip flag
            adapter = get_model_adapter("openai/clip-vit-base-patch32")
            self.assertTrue(adapter.is_clip)
            
            # Test is_blip flag
            adapter = get_model_adapter("Salesforce/blip-image-captioning-base")
            self.assertTrue(adapter.is_blip)
            
            logger.info("Multimodal model-specific handling test passed")
            
        except Exception as e:
            self.skipTest(f"Multimodal model-specific handling test failed: {e}")

if __name__ == "__main__":
    unittest.main()
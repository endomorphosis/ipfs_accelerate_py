#!/usr/bin/env python3
"""
Tests for the WebNN/WebGPU integration with IPFS acceleration.

These tests verify that the integration between WebNN/WebGPU and
IPFS acceleration works correctly.
"""

import os
import sys
import unittest
import logging
from unittest import mock

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_webnn_webgpu")

# Import package to test
from ipfs_accelerate_py.webnn_webgpu_integration import (
    WebNNWebGPUAccelerator,
    get_accelerator,
    accelerate_with_browser
)

class TestWebNNWebGPUIntegration(unittest.TestCase):
    """Tests for WebNN/WebGPU integration."""
    
    def test_accelerator_initialization(self):
        """Test that the accelerator initializes correctly."""
        accelerator = WebNNWebGPUAccelerator()
        self.assertIsNotNone(accelerator)
        self.assertFalse(accelerator.initialized)
        
        # Initialize should work
        result = accelerator.initialize()
        self.assertTrue(result)
        self.assertTrue(accelerator.initialized)
    
    def test_singleton_pattern(self):
        """Test that the singleton pattern works correctly."""
        # First call should create a new accelerator
        acc1 = get_accelerator()
        self.assertIsNotNone(acc1)
        
        # Second call should return the same accelerator
        acc2 = get_accelerator()
        self.assertIs(acc1, acc2)
    
    def test_model_type_detection(self):
        """Test model type detection."""
        accelerator = WebNNWebGPUAccelerator()
        
        # Test with known model types
        self.assertEqual(accelerator._determine_model_type("bert-base-uncased"), "text_embedding")
        self.assertEqual(accelerator._determine_model_type("llama-7b"), "text_generation")
        self.assertEqual(accelerator._determine_model_type("vit-base-patch16-224"), "vision")
        self.assertEqual(accelerator._determine_model_type("whisper-small"), "audio")
        self.assertEqual(accelerator._determine_model_type("llava-13b"), "multimodal")
        
        # Test with explicit type
        self.assertEqual(accelerator._determine_model_type("custom-model", "vision"), "vision")
    
    def test_optimal_browser_selection(self):
        """Test optimal browser selection."""
        accelerator = WebNNWebGPUAccelerator()
        
        # Test with known best browsers for model types
        self.assertEqual(accelerator._get_optimal_browser("audio", "webgpu"), "firefox")
        self.assertEqual(accelerator._get_optimal_browser("text_embedding", "webnn"), "edge")
        self.assertEqual(accelerator._get_optimal_browser("vision", "webgpu"), "chrome")
        self.assertEqual(accelerator._get_optimal_browser("multimodal", "webgpu"), "chrome")
        
        # Test default behaviors
        self.assertEqual(accelerator._get_optimal_browser("unknown", "webnn"), "edge")
        self.assertEqual(accelerator._get_optimal_browser("unknown", "webgpu"), "chrome")
    
    @mock.patch('anyio.run')
    @mock.patch('ipfs_accelerate_py.webnn_webgpu_integration._accelerate_async')
    def test_accelerate_with_browser(self, mock_accelerate_async, mock_run):
        """Test the main accelerate_with_browser function."""
        # Setup mock to return a success result
        mock_result = {
            "status": "success",
            "model_name": "bert-base-uncased",
            "model_type": "text_embedding",
            "output": {"embedding": [0.1, 0.2, 0.3]}
        }
        mock_run.return_value = mock_result
        
        # Call accelerate_with_browser
        inputs = {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}
        result = accelerate_with_browser(
            model_name="bert-base-uncased",
            inputs=inputs,
            platform="webgpu",
            browser="firefox"
        )
        
        # Verify the result
        self.assertEqual(result, mock_result)
        
        # Verify that _accelerate_async was called with the right arguments
        mock_accelerate_async.assert_called_once()
        call_args = mock_accelerate_async.call_args[1]
        self.assertEqual(call_args["model_name"], "bert-base-uncased")
        self.assertEqual(call_args["inputs"], inputs)
        self.assertEqual(call_args["platform"], "webgpu")
        self.assertEqual(call_args["browser"], "firefox")
    
    @mock.patch('ipfs_accelerate_py.webnn_webgpu_integration.WebNNWebGPUAccelerator.accelerate_with_browser')
    def test_mock_output_generation(self, mock_accelerate):
        """Test mock output generation for different model types."""
        accelerator = WebNNWebGPUAccelerator()
        
        # Test text embedding mock output
        text_embedding_output = accelerator._generate_mock_output(
            "bert-base-uncased", "text_embedding", {})
        self.assertIn("embedding", text_embedding_output)
        self.assertEqual(len(text_embedding_output["embedding"]), 768)
        
        # Test text generation mock output
        text_gen_output = accelerator._generate_mock_output(
            "llama-7b", "text_generation", {})
        self.assertIn("generated_text", text_gen_output)
        self.assertIn("tokens", text_gen_output)
        
        # Test vision mock output
        vision_output = accelerator._generate_mock_output(
            "vit-base", "vision", {})
        self.assertIn("image_embedding", vision_output)
        
        # Test audio mock output
        audio_output = accelerator._generate_mock_output(
            "whisper-small", "audio", {})
        self.assertTrue("text" in audio_output or "audio_embedding" in audio_output)
        
        # Test multimodal mock output
        multimodal_output = accelerator._generate_mock_output(
            "llava-13b", "multimodal", {})
        self.assertIn("text", multimodal_output)

if __name__ == '__main__':
    unittest.main()
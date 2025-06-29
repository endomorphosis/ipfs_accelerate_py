#!/usr/bin/env python3
"""
Test file for T5 models.

This file has been migrated to the refactored test suite.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional

from refactored_test_suite.model_test import ModelTest

# Try to import required packages with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False

try:
    import transformers
    from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    AutoTokenizer = MagicMock()
    T5ForConditionalGeneration = MagicMock()
    pipeline = MagicMock()
    HAS_TRANSFORMERS = False

class TestT5Models(ModelTest):
    """Test class for T5 models."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "t5-small"
        self.task = "translation_en_to_fr"
        self.class_name = "T5ForConditionalGeneration"
        self.description = "T5 base model"
        self.test_text = "translate English to French: Hello, how are you?"
        
        # Check hardware
        self.setup_hardware()
        
        # Results storage
        self.results = {}
    
    def setup_hardware(self):
        """Set up hardware detection for the test."""
        # Skip test if torch is not available
        if not HAS_TORCH:
            self.skipTest("PyTorch not available")
            
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        
        # MPS support (Apple Silicon)
        try:
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except AttributeError:
            self.has_mps = False
            
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        self.logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_id=None):
        """Load T5 model from HuggingFace."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
            
        model_id = model_id or self.model_id
            
        try:
            # Load model
            self.logger.info(f"Loading T5 model: {model_id}")
            model = T5ForConditionalGeneration.from_pretrained(model_id)
            model = model.to(self.device)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Store tokenizer for later use
            self.tokenizer = tokenizer
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.skipTest(f"Could not load model: {str(e)}")
    
    def test_pipeline_api(self):
        """Test the model using HuggingFace pipeline API."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
        
        self.logger.info(f"Testing {self.model_id} with pipeline() on {self.device}...")
        
        try:
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": self.device
            }
            
            # Create pipeline
            model_pipeline = pipeline(**pipeline_kwargs)
            
            # Run inference
            output = model_pipeline(self.test_text)
            
            # Verify output
            self.assertIsNotNone(output, "Pipeline output should not be None")
            
            if isinstance(output, list):
                self.assertGreater(len(output), 0, "Pipeline should return at least one result")
                self.logger.info(f"Pipeline output: {output[0]}")
            else:
                self.logger.info(f"Pipeline output: {output}")
                
            self.logger.info("Pipeline API test succeeded")
            
        except Exception as e:
            self.logger.error(f"Error testing pipeline: {e}")
            self.fail(f"Pipeline API test failed: {str(e)}")
    
    def test_direct_model_inference(self):
        """Test the model using direct model inference."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
        
        self.logger.info(f"Testing {self.model_id} with direct inference on {self.device}...")
        
        try:
            # Load model and tokenizer
            model = self.load_model()
            tokenizer = self.tokenizer
            
            # Tokenize input
            inputs = tokenizer(self.test_text, return_tensors="pt").to(self.device)
            
            # Generate output
            with torch.no_grad():
                outputs = model.generate(**inputs)
            
            # Decode output
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify output
            self.assertIsNotNone(decoded_output, "Model output should not be None")
            self.assertGreater(len(decoded_output), 0, "Model output should not be empty")
            
            self.logger.info(f"Model output: {decoded_output}")
            self.logger.info("Direct model inference test succeeded")
            
        except Exception as e:
            self.logger.error(f"Error in direct inference: {e}")
            self.fail(f"Direct model inference test failed: {str(e)}")
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware platforms."""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
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
                
                # Create pipeline with appropriate parameters
                pipeline_kwargs = {
                    "task": self.task,
                    "model": self.model_id,
                    "device": device
                }
                
                # Create pipeline
                model_pipeline = pipeline(**pipeline_kwargs)
                
                # Run inference
                output = model_pipeline(self.test_text)
                
                # Verify output
                self.assertIsNotNone(output, f"Pipeline output on {device} should not be None")
                
                self.logger.info(f"Test on {device} passed")
            except Exception as e:
                self.logger.error(f"Error testing on {device}: {e}")
                self.fail(f"Test on {device} failed: {str(e)}")
            finally:
                # Restore original device
                self.device = original_device



    def test_model_loading(self):
        # Test basic model loading
        if not hasattr(self, 'model_id') or not self.model_id:
            self.skipTest("No model_id specified")
        
        try:
            # Import the appropriate library
            if 'bert' in self.model_id.lower() or 'gpt' in self.model_id.lower() or 't5' in self.model_id.lower():
                import transformers
                model = transformers.AutoModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'clip' in self.model_id.lower():
                import transformers
                model = transformers.CLIPModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'whisper' in self.model_id.lower():
                import transformers
                model = transformers.WhisperModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'wav2vec2' in self.model_id.lower():
                import transformers
                model = transformers.Wav2Vec2Model.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            else:
                # Generic loading
                try:
                    import transformers
                    model = transformers.AutoModel.from_pretrained(self.model_id)
                    self.assertIsNotNone(model, "Model loading failed")
                except:
                    self.skipTest(f"Could not load model {self.model_id} with AutoModel")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")



    def detect_preferred_device(self):
        # Detect available hardware and choose the preferred device
        try:
            import torch
        
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
        
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
        
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"


if __name__ == "__main__":
    unittest.main()
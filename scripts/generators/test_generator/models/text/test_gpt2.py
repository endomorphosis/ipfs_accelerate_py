#!/usr/bin/env python3
"""
Test file for GPT models using the refactored test suite structure.
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

class TestGptModel(ModelTest):
    """Test class for GPT decoder-only models."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "gpt2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "text-generation"
        self.max_new_tokens = 20
        
        # Define test inputs
        self.test_text = "Once upon a time in a galaxy far, far away,"
    
    def tearDown(self):
        """Clean up resources after the test."""
        super().tearDown()
    
    def load_model(self):
        """Load the model for testing."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
            model = model.to(self.device)
            
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model_components = self.load_model()
        
        # Verify model and tokenizer
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["tokenizer"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        """Test basic inference with the model."""
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        
        # Prepare input
        inputs = tokenizer(self.test_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check that the output contains the input and has been extended
        self.assertTrue(self.test_text in generated_text)
        self.assertGreater(len(generated_text), len(self.test_text))
        
        logger.info(f"Generated text: {generated_text}")
        logger.info("Inference successful")
    
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
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output."""
        # This is a generic implementation - specific test classes should override this
        try:
            # For transformers models, we can try to run a forward pass
            import torch
            
            # Convert input data to tensors if needed
            if isinstance(input_data, dict):
                # Input is already a dict of tensors (e.g., from tokenizer)
                inputs = input_data
            elif isinstance(input_data, str):
                # If input is a string, we need to tokenize it
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
                inputs = tokenizer(input_data, return_tensors="pt")
            else:
                # Otherwise, assume input is already properly formatted
                inputs = input_data
                
            # Move inputs to the right device if model is on a specific device
            if hasattr(model, 'device') and model.device.type != 'cpu':
                inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, 'to')}
                
            # Run model
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Verify outputs
            self.assertIsNotNone(outputs, "Model output should not be None")
            
            # If expected output is provided, verify it matches
            if expected_output is not None:
                self.assertEqual(outputs, expected_output)
                
            return outputs
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            raise
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
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

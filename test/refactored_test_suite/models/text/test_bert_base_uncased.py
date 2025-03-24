#!/usr/bin/env python3
"""
Test file for BERT models using the refactored test suite structure.
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

class TestBertModel(ModelTest):
    """Test class for BERT encoder-only models."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "bert-base-uncased"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "fill-mask"
        
        # Define test inputs
        self.test_text = "The quick brown fox jumps over the [MASK] dog."
    
    def tearDown(self):
        """Clean up resources after the test."""
        super().tearDown()
    
    def load_model(self):
        """Load the model for testing."""
        try:
            from transformers import AutoTokenizer, BertForMaskedLM
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = BertForMaskedLM.from_pretrained(self.model_id)
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
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, "logits"))
        
        # Get the mask token prediction
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        top_tokens_words = [tokenizer.decode([token]).strip() for token in top_tokens]
        
        logger.info(f"Top predictions: {', '.join(top_tokens_words)}")
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

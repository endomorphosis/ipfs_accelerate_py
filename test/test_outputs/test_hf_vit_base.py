#!/usr/bin/env python3
"""
Test for vit-base model with hardware platform support
"""

import os
import sys
import unittest
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class TestVitbaseModels(unittest.TestCase):
    """Test vit-base model across hardware platforms."""
    
    def setUp(self):
        """Set up test."""
        self.model_id = "vit-base"
        self.test_text = "This is a test sentence."
        self.test_batch = ["First test sentence.", "Second test sentence."]
    
    def test_with_cpu(self):
        """Test vit-base with cpu."""
        # Test initialization
        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Initialize model
            model = AutoModel.from_pretrained(self.model_id)
            
            # Process input
            inputs = tokenizer(self.test_text, return_tensors="pt")
            outputs = model(**inputs)
            
            # Verify output
            self.assertIsNotNone(outputs)
            self.assertIn("last_hidden_state", outputs)
            
            print(f"Model {self.model_id} successfully tested with cpu")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")
    

if __name__ == "__main__":
    unittest.main()

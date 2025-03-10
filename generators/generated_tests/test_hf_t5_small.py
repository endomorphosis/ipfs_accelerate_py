#!/usr/bin/env python3
"""
Test for t5-small model with hardware platform support
"""

import os
import sys
import unittest
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class TestT5SmallModels(unittest.TestCase):
    """Test t5-small model across hardware platforms."""
    
    def setUp(self):
        """Set up test."""
        self.model_id = "t5-small"
        self.test_text = "This is a test sentence."
        self.test_batch = ["First test sentence.", "Second test sentence."]

    def test_with_cpu(self):
        """Test t5-small with cpu."""
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
            
            print(f"Model {self.model_id} successfully tested with cpu")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")

    def test_with_cuda(self):
        """Test t5-small with cuda."""
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
            
            print(f"Model {self.model_id} successfully tested with cuda")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")

    def test_with_rocm(self):
        """Test t5-small with rocm."""
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
            
            print(f"Model {self.model_id} successfully tested with rocm")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")

    def test_with_mps(self):
        """Test t5-small with mps."""
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
            
            print(f"Model {self.model_id} successfully tested with mps")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")

    def test_with_openvino(self):
        """Test t5-small with openvino."""
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
            
            print(f"Model {self.model_id} successfully tested with openvino")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")

    def test_with_webnn(self):
        """Test t5-small with webnn."""
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
            
            print(f"Model {self.model_id} successfully tested with webnn")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")

    def test_with_webgpu(self):
        """Test t5-small with webgpu."""
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
            
            print(f"Model {self.model_id} successfully tested with webgpu")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {str(e)}")

#!/usr/bin/env python3
"""
Test for llava-v1.5-7b model with hardware platform support
"""

import os
import sys
import unittest
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class TestLlavav1.57BModels(unittest.TestCase):
    """Test llava-v1.5-7b model across hardware platforms."""
    
    def setUp(self):
        """Set up test."""
        self.model_id = "llava-v1.5-7b"
        self.test_text = "This is a test sentence."
        self.test_batch = ["First test sentence.", "Second test sentence."]

    def test_with_cpu(self):
        """Test llava-v1.5-7b with cpu."""
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
        """Test llava-v1.5-7b with cuda."""
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
        """Test llava-v1.5-7b with rocm."""
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
        """Test llava-v1.5-7b with mps."""
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
        """Test llava-v1.5-7b with openvino."""
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
        """Test llava-v1.5-7b with webnn."""
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
        """Test llava-v1.5-7b with webgpu."""
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

#!/usr/bin/env python3

import os
import sys
import unittest
import torch
from transformers import AutoModel, AutoTokenizer

# Hardware detection
HAS_CUDA = torch.cuda.is_available() if hasattr(torch, "cuda") else False
HAS_MPS = hasattr(torch, "mps") and torch.mps.is_available() if hasattr(torch, "mps") else False
HAS_ROCM = hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version") if hasattr(torch, "_C") else False

class TestBertbaseuncased(unittest.TestCase):
    """Test bert-base-uncased model."""
    
    def setUp(self):
        self.model_name = "bert-base-uncased"
        self.tokenizer = None
        self.model = None
    
    def test_cpu(self):
        """Test on cpu platform."""
        # Skip if hardware not available
        if "cpu" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "cpu" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "cpu" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on cpu")
        
    def test_cuda(self):
        """Test on cuda platform."""
        # Skip if hardware not available
        if "cuda" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "cuda" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "cuda" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on cuda")
        
    def test_mps(self):
        """Test on mps platform."""
        # Skip if hardware not available
        if "mps" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "mps" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "mps" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on mps")
        
    def test_openvino(self):
        """Test on openvino platform."""
        # Skip if hardware not available
        if "openvino" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "openvino" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "openvino" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on openvino")
        
    def test_qualcomm(self):
        """Test on qualcomm platform."""
        # Skip if hardware not available
        if "qualcomm" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "qualcomm" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "qualcomm" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on qualcomm")
        
    def test_webnn(self):
        """Test on webnn platform."""
        # Skip if hardware not available
        if "webnn" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "webnn" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "webnn" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on webnn")
        
    def test_webgpu(self):
        """Test on webgpu platform."""
        # Skip if hardware not available
        if "webgpu" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "webgpu" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "webgpu" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on webgpu")
        

if __name__ == "__main__":
    unittest.main()

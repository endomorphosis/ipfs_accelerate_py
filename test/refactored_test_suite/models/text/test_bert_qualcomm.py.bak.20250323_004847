"""Migrated to refactored test suite on 2025-03-21

Test file for bert with cross-platform hardware support
"""

import os
import sys
import unittest
import importlib.util
import logging
import torch
from transformers import AutoModel, AutoTokenizer

from refactored_test_suite.hardware_test import HardwareTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
HAS_CUDA = torch.cuda.is_available() if hasattr(torch, "cuda") else False
HAS_MPS = False
if hasattr(torch, "mps"):
    try:
        HAS_MPS = torch.mps.is_available()
    except:
        pass
HAS_ROCM = (hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version")) if hasattr(torch, "_C") else False
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
HAS_QUALCOMM = (
    importlib.util.find_spec("qnn_wrapper") is not None or
    importlib.util.find_spec("qti") is not None or
    "QUALCOMM_SDK" in os.environ
)
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

class TestBertQualcomm(HardwareTest):
    """Test bert model with hardware platform support."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model_name = "bert-base-uncased"
        self.tokenizer = None
        self.model = None
        self.device = self.get_device()

    def get_device(self):
        """Get the appropriate device based on available hardware."""
        if HAS_QUALCOMM:
            return "cpu"  # Qualcomm uses CPU for PyTorch API
        elif HAS_CUDA:
            return "cuda"
        elif HAS_MPS:
            return "mps"
        else:
            return "cpu"

    def test_qualcomm(self):
        """Test bert on qualcomm platform."""
        # Skip if hardware not available:
        if not HAS_QUALCOMM: 
            self.skipTest("Qualcomm AI Engine not available")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to device if needed
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            # Test basic functionality
            inputs = self.tokenizer("Hello, world!", return_tensors="pt")
            
            # Move inputs to device if needed
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            self.assertIsNotNone(outputs.last_hidden_state)
            self.assertEqual(outputs.last_hidden_state.shape[0], 1)
            
            # Log success
            self.logger.info(f"Successfully tested {self.model_name} on qualcomm")
            
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            raise

if __name__ == "__main__":
    unittest.main()
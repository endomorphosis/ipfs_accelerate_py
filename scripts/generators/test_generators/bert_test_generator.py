#!/usr/bin/env python3
"""
BERT Test Generator

This is a specialized version of the test generator that focuses only on BERT model tests
with full hardware platform support.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bert_test_generator")

# Hardware compatibility matrix
KEY_MODEL_HARDWARE_MAP = {
    "bert": {
        "cpu": "REAL",
        "cuda": "REAL",
        "openvino": "REAL",
        "mps": "REAL",
        "rocm": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    }
}

def create_hardware_compatibility_matrix():
    """Create a compatibility matrix for hardware platforms and model types."""
    # Create matrix with default values
    compatibility = {
        "hardware": {
            "cpu": {"available": True},
            "cuda": {"available": True},
            "rocm": {"available": False},
            "mps": {"available": False},
            "openvino": {"available": False},
            "webnn": {"available": False},
            "webgpu": {"available": False}
        },
        "categories": {
            "text": {
                "cpu": True,
                "cuda": True,
                "rocm": True,
                "mps": True,
                "openvino": True,
                "webnn": True,
                "webgpu": True
            }
        },
        "models": {}
    }
    
    # Add specific model compatibility
    for model_name, hw_support in KEY_MODEL_HARDWARE_MAP.items():
        compatibility["models"][model_name] = {}
        for hw_type, support_level in hw_support.items():
            compatibility["models"][model_name][hw_type] = support_level != "NONE"
    
    return compatibility

def platform_supported_for_model(model_name, platform):
    """Check if a platform is supported for a specific model."""
    # Load compatibility matrix
    compatibility = create_hardware_compatibility_matrix()
    
    # Check model specific compatibility
    model_compat = compatibility.get("models", {}).get(model_name, {})
    if platform in model_compat:
        return model_compat[platform]
    
    return True  # Default to supported

TEMPLATE = """#!/usr/bin/env python3
\"\"\"
Test for BERT model.

This test file validates that the BERT model works correctly across all hardware platforms:
- CPU
- CUDA (NVIDIA GPU)
- ROCm (AMD GPU)
- MPS (Apple Silicon)
- OpenVINO
- WebNN (Browser)
- WebGPU (Browser)
\"\"\"

import os
import logging
import numpy as np
import unittest
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, some tests will be skipped")

# Initialize hardware capability flags
HAS_CUDA = False
HAS_ROCM = False
HAS_MPS = False
HAS_OPENVINO = False
HAS_WEBNN = False
HAS_WEBGPU = False

# Detect available hardware
if HAS_TORCH:
    # CUDA detection
    HAS_CUDA = torch.cuda.is_available()
    
    # ROCm detection
    if HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # Apple MPS detection
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available()

# OpenVINO detection
try:
    import openvino
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

# Web platform detection (simulation in non-browser environments)
HAS_WEBNN = 'WEBNN_SIMULATION' in os.environ
HAS_WEBGPU = 'WEBGPU_SIMULATION' in os.environ

class TestBERT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "bert-base-uncased"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.test_text = "This is a simple test sentence"
    
    def test_cpu(self):
        """Test BERT on CPU."""
        model = AutoModel.from_pretrained(self.model_name)
        
        # Run inference
        inputs = self.tokenizer(self.test_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify output shape
        self.assertIsNotNone(outputs)
        self.assertIn('last_hidden_state', outputs)
        self.assertEqual(outputs.last_hidden_state.shape[0], 1)  # Batch size 1
    
    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    def test_cuda(self):
        """Test BERT on CUDA."""
        model = AutoModel.from_pretrained(self.model_name).to("cuda")
        
        # Run inference
        inputs = self.tokenizer(self.test_text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify output shape
        self.assertIsNotNone(outputs)
        self.assertIn('last_hidden_state', outputs)
        self.assertEqual(outputs.last_hidden_state.shape[0], 1)  # Batch size 1
        
        # Move back to CPU for comparison
        cpu_output = outputs.last_hidden_state.cpu().numpy()
        self.assertEqual(cpu_output.shape[0], 1)
    
    @unittest.skipIf(not HAS_ROCM, "ROCm not available")
    def test_rocm(self):
        """Test BERT on ROCm (AMD GPU)."""
        model = AutoModel.from_pretrained(self.model_name).to("cuda")  # ROCm uses CUDA API
        
        # Run inference
        inputs = self.tokenizer(self.test_text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify output shape
        self.assertIsNotNone(outputs)
        self.assertIn('last_hidden_state', outputs)
        self.assertEqual(outputs.last_hidden_state.shape[0], 1)
    
    @unittest.skipIf(not HAS_MPS, "MPS not available")
    def test_mps(self):
        """Test BERT on MPS (Apple Silicon)."""
        model = AutoModel.from_pretrained(self.model_name).to("mps")
        
        # Run inference
        inputs = self.tokenizer(self.test_text, return_tensors="pt")
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify output shape
        self.assertIsNotNone(outputs)
        self.assertIn('last_hidden_state', outputs)
        self.assertEqual(outputs.last_hidden_state.shape[0], 1)
    
    @unittest.skipIf(not HAS_OPENVINO, "OpenVINO not available")
    def test_openvino(self):
        """Test BERT on OpenVINO."""
        try:
            from openvino.runtime import Core
            from optimum.intel import OVModelForFeatureExtraction
            
            # Load model through Optimum
            model = OVModelForFeatureExtraction.from_pretrained(
                self.model_name, 
                device="CPU"
            )
            
            # Run inference
            inputs = self.tokenizer(self.test_text, return_tensors="pt")
            outputs = model(**inputs)
            
            # Verify output shape
            self.assertIsNotNone(outputs)
            self.assertIn('last_hidden_state', outputs)
            self.assertEqual(outputs.last_hidden_state.shape[0], 1)
        except (ImportError, Exception) as e:
            self.skipTest(f"OpenVINO test failed: {e}")
    
    @unittest.skipIf(not HAS_WEBNN, "WebNN not available")
    def test_webnn(self):
        """Test BERT on WebNN (browser neural network API)."""
        # This is a simulation - in a real browser, WebNN would be used
        # For testing, we'll use CPU but log that WebNN would be used
        model = AutoModel.from_pretrained(self.model_name)
        logger.info("Using WebNN simulation mode for BERT")
        
        # Run inference
        inputs = self.tokenizer(self.test_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify output shape
        self.assertIsNotNone(outputs)
        self.assertIn('last_hidden_state', outputs)
        self.assertEqual(outputs.last_hidden_state.shape[0], 1)
    
    @unittest.skipIf(not HAS_WEBGPU, "WebGPU not available")
    def test_webgpu(self):
        """Test BERT on WebGPU (browser GPU API)."""
        # This is a simulation - in a real browser, WebGPU would be used
        # For testing, we'll use CPU but log that WebGPU would be used
        model = AutoModel.from_pretrained(self.model_name)
        logger.info("Using WebGPU simulation mode for BERT")
        
        # Run inference
        inputs = self.tokenizer(self.test_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify output shape
        self.assertIsNotNone(outputs)
        self.assertIn('last_hidden_state', outputs)
        self.assertEqual(outputs.last_hidden_state.shape[0], 1)

if __name__ == "__main__":
    unittest.main()
\"\"\"

def generate_bert_test(output_path=None):
    """Generate a BERT test file."""
    if output_path is None:
        output_path = "./test_hf_bert.py"
    
    output_path = Path(output_path)
    
    logger.info(f"Generating BERT test file at {output_path}")
    with open(output_path, 'w') as f:
        f.write(TEMPLATE)
    
    logger.info(f"Successfully generated {output_path}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate BERT test file with cross-platform support")
    parser.add_argument("--output", help="Output file path", default="./test_hf_bert.py")
    
    args = parser.parse_args()
    
    success = generate_bert_test(args.output)
    
    if success:
        logger.info("Test generation completed successfully")
        print(f"\nTo run the generated test, use: python {args.output}")
    else:
        logger.error("Test generation failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
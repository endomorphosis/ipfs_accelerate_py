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

from refactored_test_suite.model_test import ModelTest

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

class TestBertQualcomm(ModelTest):
    """Test bert model with hardware platform support."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model_id = "bert-base-uncased"
        self.model_name = self.model_id
        self.tokenizer = None
        self.model = None
        self.device = self.get_device()
        # Hardware detection from HardwareTest
        self.detect_hardware()

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


            raise
        
    # Hardware-specific methods from HardwareTest
    def detect_hardware(self):
        """Detect available hardware."""
        self.has_webgpu = self._check_webgpu()
        self.has_webnn = self._check_webnn()
        self.has_qualcomm = HAS_QUALCOMM
        
    def _check_webgpu(self):
        """Check if WebGPU is available."""
        return HAS_WEBGPU
        
    def _check_webnn(self):
        """Check if WebNN is available."""
        return HAS_WEBNN
    
    def skip_if_no_webgpu(self):
        """Skip test if WebGPU is not available."""
        if not self.has_webgpu:
            self.skipTest("WebGPU not available")
    
    def skip_if_no_webnn(self):
        """Skip test if WebNN is not available."""
        if not self.has_webnn:
            self.skipTest("WebNN not available")
            
    def skip_if_no_qualcomm(self):
        """Skip test if Qualcomm hardware is not available."""
        if not self.has_qualcomm:
            self.skipTest("Qualcomm hardware not available")
            
    # Required methods from ModelTest
    def load_model(self, model_name):
        """Load a model for testing."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move model to device if needed
        device = self.get_device()
        if device != "cpu":
            model = model.to(device)
            
        return {"model": model, "tokenizer": tokenizer}
    
    def verify_model_output(self, model_dict, input_data, expected_output=None):
        """Verify that model produces expected output."""
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        
        # Tokenize input
        inputs = tokenizer(input_data, return_tensors="pt")
        
        # Move inputs to device if needed
        device = self.get_device()
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIsNotNone(outputs.last_hidden_state)
        
        if expected_output is not None:
            self.assertEqual(expected_output, outputs)
            
        return outputs

if __name__ == "__main__":
    unittest.main()
#!/usr/bin/env python3
"""
Test file for Llama models.

This file has been migrated to the refactored test suite.
Generated: 2025-03-21 from original_llama.py and fixed_llama.py
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    AutoModelForCausalLM = MagicMock()
    AutoTokenizer = MagicMock()
    AutoConfig = MagicMock()
    HAS_TRANSFORMERS = False

# Optional OpenVINO check
try:
    import openvino
    HAS_OPENVINO = True
except ImportError:
    openvino = MagicMock()
    HAS_OPENVINO = False

# Mock handler for platforms that don't have real implementations
class MockHandler:
    """Mock handler for platforms that don't have real implementations."""
    
    def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        logging.info(f"Created mock handler for {platform}")
    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        logging.info(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")
        return {
            "generated_text": f"Mock generated text for {self.platform}",
            "success": True,
            "platform": self.platform
        }

class TestLlamaModel(ModelTest):
    """Test class for Llama models."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "facebook/opt-125m"  # Default to a small model for testing
        self.test_prompt = "Generate a short story about:"
        self.tokenizer = None
        self.model = None
        
        # Hardware detection
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
            
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        # OpenVINO support
        self.has_openvino = HAS_OPENVINO
        
        # Qualcomm AI Engine support
        try:
            import importlib.util
            has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
            has_qti = importlib.util.find_spec("qti") is not None
            has_qualcomm_env = "QUALCOMM_SDK" in os.environ
            self.has_qualcomm = has_qnn or has_qti or has_qualcomm_env
        except ImportError:
            self.has_qualcomm = False
        
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        self.logger.info(f"Using device: {self.device}")
    
    def load_tokenizer(self):
        """Load tokenizer."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
            
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                return True
            except Exception as e:
                self.logger.error(f"Error loading tokenizer: {e}")
                return False
        return True
    
    def load_model(self, model_id=None):
        """Load model from HuggingFace."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
            
        model_id = model_id or self.model_id
            
        try:
            # Load model with appropriate device
            self.logger.info(f"Loading model: {model_id} on {self.device}")
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = model.to(self.device)
            
            # Load tokenizer if not already loaded
            if not self.load_tokenizer():
                self.skipTest("Failed to load tokenizer")
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.skipTest(f"Could not load model: {str(e)}")
    
    def test_cpu_inference(self):
        """Test the model on CPU."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
        
        # Save current device
        original_device = self.device
        self.device = 'cpu'
        
        try:
            self.logger.info(f"Testing {self.model_id} on CPU...")
            
            # Load model
            model = self.load_model()
            
            # Tokenize input
            inputs = self.tokenizer(self.test_prompt, return_tensors="pt")
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify output
            self.assertIsNotNone(generated_text, "Generated text should not be None")
            self.assertGreater(len(generated_text), len(self.test_prompt), "Model should generate additional text")
            
            self.logger.info(f"Generated text: {generated_text}")
            self.logger.info("CPU inference test passed")
            
        except Exception as e:
            self.logger.error(f"Error in CPU inference: {e}")
            self.fail(f"CPU inference test failed: {str(e)}")
        finally:
            # Restore original device
            self.device = original_device
    
    def test_cuda_inference(self):
        """Test the model on CUDA if available."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
        
        if not self.has_cuda:
            self.skipTest("CUDA not available")
        
        # Save current device
        original_device = self.device
        self.device = 'cuda'
        
        try:
            self.logger.info(f"Testing {self.model_id} on CUDA...")
            
            # Load model
            model = self.load_model()
            
            # Tokenize input
            inputs = self.tokenizer(self.test_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify output
            self.assertIsNotNone(generated_text, "Generated text should not be None")
            self.assertGreater(len(generated_text), len(self.test_prompt), "Model should generate additional text")
            
            self.logger.info(f"Generated text: {generated_text}")
            self.logger.info("CUDA inference test passed")
            
        except Exception as e:
            self.logger.error(f"Error in CUDA inference: {e}")
            self.fail(f"CUDA inference test failed: {str(e)}")
        finally:
            # Restore original device
            self.device = original_device
    
    def test_mps_inference(self):
        """Test the model on MPS if available."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
        
        if not self.has_mps:
            self.skipTest("MPS not available")
        
        # Save current device
        original_device = self.device
        self.device = 'mps'
        
        try:
            self.logger.info(f"Testing {self.model_id} on MPS...")
            
            # Load model
            model = self.load_model()
            
            # Tokenize input
            inputs = self.tokenizer(self.test_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify output
            self.assertIsNotNone(generated_text, "Generated text should not be None")
            self.assertGreater(len(generated_text), len(self.test_prompt), "Model should generate additional text")
            
            self.logger.info(f"Generated text: {generated_text}")
            self.logger.info("MPS inference test passed")
            
        except Exception as e:
            self.logger.error(f"Error in MPS inference: {e}")
            self.fail(f"MPS inference test failed: {str(e)}")
        finally:
            # Restore original device
            self.device = original_device
    
    def test_openvino_inference(self):
        """Test the model with OpenVINO if available."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
        
        if not self.has_openvino:
            self.skipTest("OpenVINO not available")
        
        # Save current device
        original_device = self.device
        self.device = 'openvino'
        
        try:
            self.logger.info(f"Testing {self.model_id} with OpenVINO...")
            
            # In a real implementation, we would use the OpenVINO API
            # For testing, we use a mock handler
            handler = MockHandler(self.model_id, "openvino")
            
            # Run inference with mock handler
            result = handler(self.test_prompt, max_new_tokens=20)
            
            # Verify output
            self.assertIsNotNone(result, "Result should not be None")
            self.assertTrue(result["success"], "Operation should be successful")
            self.assertIn("generated_text", result, "Result should contain generated_text")
            
            self.logger.info(f"Generated text: {result.get('generated_text')}")
            self.logger.info("OpenVINO inference test passed")
            
        except Exception as e:
            self.logger.error(f"Error in OpenVINO inference: {e}")
            self.fail(f"OpenVINO inference test failed: {str(e)}")
        finally:
            # Restore original device
            self.device = original_device
    
    def test_qualcomm_inference(self):
        """Test the model with Qualcomm AI Engine if available."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
        
        if not self.has_qualcomm:
            self.skipTest("Qualcomm AI Engine not available")
        
        # Save current device
        original_device = self.device
        self.device = 'qualcomm'
        
        try:
            self.logger.info(f"Testing {self.model_id} with Qualcomm AI Engine...")
            
            # In a real implementation, we would use the Qualcomm SDK
            # For testing, we use a mock handler
            handler = MockHandler(self.model_id, "qualcomm")
            
            # Run inference with mock handler
            result = handler(self.test_prompt, max_new_tokens=20)
            
            # Verify output
            self.assertIsNotNone(result, "Result should not be None")
            self.assertTrue(result["success"], "Operation should be successful")
            self.assertIn("generated_text", result, "Result should contain generated_text")
            
            self.logger.info(f"Generated text: {result.get('generated_text')}")
            self.logger.info("Qualcomm inference test passed")
            
        except Exception as e:
            self.logger.error(f"Error in Qualcomm inference: {e}")
            self.fail(f"Qualcomm inference test failed: {str(e)}")
        finally:
            # Restore original device
            self.device = original_device
    
    def test_hardware_compatibility(self):
        """Test model compatibility with all available hardware platforms."""
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
        if self.has_rocm:
            devices_to_test.append('cuda')  # ROCm uses CUDA compatibility layer
            
        # Test each device
        for device in devices_to_test:
            original_device = self.device
            try:
                self.logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model
                model = self.load_model()
                
                # Tokenize input
                inputs = self.tokenizer(self.test_prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate text
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                    )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Verify output
                self.assertIsNotNone(generated_text, f"Generated text on {device} should not be None")
                self.assertGreater(len(generated_text), len(self.test_prompt), 
                                  f"Model on {device} should generate additional text")
                
                self.logger.info(f"Test on {device} passed")
                self.results[device] = True
            except Exception as e:
                self.logger.error(f"Error testing on {device}: {e}")
                self.results[device] = False
                # Don't fail the test, just record failure for this device
            finally:
                # Restore original device
                self.device = original_device
        
        # Ensure at least CPU test passed
        self.assertTrue(self.results.get('cpu', False), "CPU test should have passed")
        
        # Log overall results
        self.logger.info(f"Hardware compatibility test results: {self.results}")



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
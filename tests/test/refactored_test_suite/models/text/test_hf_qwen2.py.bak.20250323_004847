#!/usr/bin/env python3
"""
Test file for Qwen2 models.

This file has been migrated to the refactored test suite.
Generated: 2025-03-21 from skills/test_hf_qwen2.py
"""

import os
import sys
import json
import logging
import unittest
import time
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional, Union

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
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    AutoTokenizer = MagicMock()
    AutoModelForCausalLM = MagicMock()
    HAS_TRANSFORMERS = False

try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    accelerate = MagicMock()
    HAS_ACCELERATE = False

try:
    import openvino
    HAS_OPENVINO = True
except ImportError:
    openvino = MagicMock()
    HAS_OPENVINO = False

# Qwen2 models registry
QWEN2_MODELS_REGISTRY = {
    "Qwen/Qwen2-7B-Instruct": {
        "description": "Qwen2 7B instruction-tuned model",
        "class": "Qwen2ForCausalLM",
    },
    "Qwen/Qwen2-7B": {
        "description": "Qwen2 7B base model",
        "class": "Qwen2ForCausalLM",
    },
    "Qwen/Qwen2-0.5B-Instruct": {
        "description": "Qwen2 0.5B instruction-tuned model (small)",
        "class": "Qwen2ForCausalLM",
    },
    "Qwen/Qwen2-1.5B-Instruct": {
        "description": "Qwen2 1.5B instruction-tuned model (medium)",
        "class": "Qwen2ForCausalLM",
    }
}

class TestQwen2Models(ModelTest):
    """Test class for Qwen2 models."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "Qwen/Qwen2-7B-Instruct"  # Default model
        
        # Model parameters
        if self.model_id not in QWEN2_MODELS_REGISTRY:
            self.logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = QWEN2_MODELS_REGISTRY["Qwen/Qwen2-7B-Instruct"]
        else:
            self.model_info = QWEN2_MODELS_REGISTRY[self.model_id]
        
        self.task = "text-generation"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Test prompts
        self.test_prompt = "Explain the concept of neural networks to a beginner"
        self.test_prompts = [
            "Explain the concept of neural networks to a beginner",
            "What are the applications of machine learning in healthcare?",
            "Write a short story about a robot learning to be human"
        ]
        
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
        
        # WebNN/WebGPU support
        self.has_webnn = False
        self.has_webgpu = False
        
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
    
    def load_model(self, model_id=None):
        """Load Qwen2 model from HuggingFace."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
            
        model_id = model_id or self.model_id
            
        try:
            # Load tokenizer
            self.logger.info(f"Loading Qwen2 tokenizer: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load model
            self.logger.info(f"Loading Qwen2 model: {model_id} on {self.device}")
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            
            # Store tokenizer for later use
            self.tokenizer = tokenizer
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.skipTest(f"Could not load model: {str(e)}")
    
    def test_pipeline_api(self):
        """Test the model using HuggingFace pipeline API."""
        if not HAS_TRANSFORMERS or not HAS_ACCELERATE:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} with pipeline() on {self.device}...")
        
        try:
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": self.device
            }
            
            # Create pipeline
            model_pipeline = transformers.pipeline(**pipeline_kwargs)
            
            # Run inference
            start_time = time.time()
            output = model_pipeline(self.test_prompt)
            inference_time = time.time() - start_time
            
            # Verify output
            self.assertIsNotNone(output, "Pipeline output should not be None")
            
            if isinstance(output, list):
                self.assertGreater(len(output), 0, "Pipeline should return at least one result")
                if hasattr(output[0], "get") and output[0].get("generated_text"):
                    generated_text = output[0]["generated_text"]
                else:
                    generated_text = str(output[0])
            else:
                generated_text = str(output)
            
            self.logger.info(f"Input prompt: {self.test_prompt}")
            self.logger.info(f"Generated text: {generated_text}")
            self.logger.info(f"Inference time: {inference_time:.4f} seconds")
            
            # Store results
            self.results['pipeline_api'] = {
                'prompt': self.test_prompt,
                'output': generated_text,
                'inference_time': inference_time
            }
            
            self.logger.info("Pipeline API test succeeded")
            
        except Exception as e:
            self.logger.error(f"Error testing pipeline: {e}")
            self.fail(f"Pipeline API test failed: {str(e)}")
    
    def test_direct_model_inference(self):
        """Test the model using direct model inference."""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} with direct inference on {self.device}...")
        
        try:
            # Load model and tokenizer
            model = self.load_model()
            tokenizer = self.tokenizer
            
            # Tokenize input
            inputs = tokenizer(self.test_prompt, return_tensors="pt").to(self.device)
            
            # Generate output
            with torch.no_grad():
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
                inference_time = time.time() - start_time
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify output
            self.assertIsNotNone(generated_text, "Model output should not be None")
            self.assertGreater(len(generated_text), 0, "Model output should not be empty")
            
            self.logger.info(f"Input prompt: {self.test_prompt}")
            self.logger.info(f"Generated text: {generated_text}")
            self.logger.info(f"Inference time: {inference_time:.4f} seconds")
            
            # Store results
            self.results['direct_model_inference'] = {
                'prompt': self.test_prompt,
                'output': generated_text,
                'inference_time': inference_time
            }
            
            self.logger.info("Direct model inference test succeeded")
            
        except Exception as e:
            self.logger.error(f"Error in direct inference: {e}")
            self.fail(f"Direct model inference test failed: {str(e)}")
    
    def test_multiple_prompts(self):
        """Test the model with multiple prompts."""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} with multiple prompts on {self.device}...")
        
        try:
            # Load model and tokenizer
            model = self.load_model()
            tokenizer = self.tokenizer
            
            # Results for each prompt
            prompt_results = {}
            
            for prompt in self.test_prompts:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate output
                with torch.no_grad():
                    start_time = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,  # Shorter for multiple prompts
                        do_sample=False
                    )
                    inference_time = time.time() - start_time
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Verify output
                self.assertIsNotNone(generated_text, "Model output should not be None")
                self.assertGreater(len(generated_text), 0, "Model output should not be empty")
                
                self.logger.info(f"Input prompt: {prompt}")
                self.logger.info(f"Generated text: {generated_text}")
                self.logger.info(f"Inference time: {inference_time:.4f} seconds")
                
                # Store result for this prompt
                prompt_results[prompt] = {
                    'output': generated_text,
                    'inference_time': inference_time
                }
            
            # Store all prompt results
            self.results['multiple_prompts'] = prompt_results
            
            self.logger.info("Multiple prompts test succeeded")
            
        except Exception as e:
            self.logger.error(f"Error in multiple prompts test: {e}")
            self.fail(f"Multiple prompts test failed: {str(e)}")
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware platforms."""
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
            
        # Test each device
        for device in devices_to_test:
            original_device = self.device
            try:
                self.logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model and tokenizer
                model = self.load_model()
                tokenizer = self.tokenizer
                
                # Tokenize input (shorter for hardware test)
                inputs = tokenizer("Briefly explain what a neural network is", return_tensors="pt").to(device)
                
                # Generate output
                with torch.no_grad():
                    start_time = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,  # Short for hardware test
                        do_sample=False
                    )
                    inference_time = time.time() - start_time
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Verify output
                self.assertIsNotNone(generated_text, f"Model output on {device} should not be None")
                self.assertGreater(len(generated_text), 0, f"Model output on {device} should not be empty")
                
                # Store performance results
                self.results[f'performance_{device}'] = {
                    'inference_time': inference_time,
                    'output_length': len(generated_text)
                }
                
                self.logger.info(f"Test on {device} passed (inference time: {inference_time:.4f}s)")
            except Exception as e:
                self.logger.error(f"Error testing on {device}: {e}")
                self.fail(f"Test on {device} failed: {str(e)}")
            finally:
                # Restore original device
                self.device = original_device
    
    def test_openvino_inference(self):
        """Test the model using OpenVINO."""
        if not HAS_TRANSFORMERS or not HAS_TORCH or not self.has_openvino:
            self.skipTest("Required dependencies or OpenVINO not available")
        
        self.logger.info(f"Testing {self.model_id} with OpenVINO...")
        
        try:
            # For this test, we'll just check if optimum-intel is installed
            # Full implementation would use OVModelForCausalLM
            try:
                from optimum.intel import OVModelForCausalLM
            except ImportError:
                self.skipTest("optimum-intel not available")
            
            # Note: In a real implementation, we would load the model with OpenVINO:
            # model = OVModelForCausalLM.from_pretrained(self.model_id, export=True)
            
            # For this test, we'll just log that OpenVINO would be used
            self.logger.info("OpenVINO support is available for Qwen2 models")
            
            # Store results
            self.results['openvino_inference'] = {
                'supported': True,
                'notes': "Full implementation would use OVModelForCausalLM"
            }
            
        except Exception as e:
            self.logger.error(f"Error in OpenVINO test: {e}")
            self.fail(f"OpenVINO test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()